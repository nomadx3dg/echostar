"""Local web UI for playing an Echo bot, with chat-style commentary.

    python echo.py ui --style style.npz

Standard library only. The server binds to 127.0.0.1, serves the page (ui/index.html, styles.css, app.js) and a small
JSON API, and keeps one game in memory. It does not need to be exposed to a network and refuses requests that come from other web pages.
"""
import hashlib
import hmac
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import chess
import chess.engine

import commentary
import review

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BODY = 4096
EVAL_DEPTH = 10

# A small hidden extra: typing the right word on the page unlocks one more opponent. Only the hash is stored here, so the
# word is not sitting in the source for a casual grep. It is an easter egg, not a security feature.
UNLOCK_SHA256 = "c0cff7dcdcb5dc4d00b2270b77e0d4a873c7aa2f1045479a28d5516017edfe94"
UNLOCK_MIN, UNLOCK_MAX = 4, 24


def unlock_matches(typed):
    """True if the end of `typed` is the unlock word (case-insensitive, letters and digits only)."""
    text = "".join(ch for ch in str(typed).lower() if ch.isalnum())[-UNLOCK_MAX:]
    for n in range(UNLOCK_MIN, len(text) + 1):
        digest = hashlib.sha256(text[-n:].encode()).hexdigest()
        if hmac.compare_digest(digest, UNLOCK_SHA256):
            return True
    return False


class Bot:
    def __init__(self, bot_id, name, blurb, persona, brain, dial, available=True, note=""):
        self.id, self.name, self.blurb, self.persona = bot_id, name, blurb, persona
        self.brain, self.dial, self.available, self.note = brain, dial, available, note

    def public(self):
        return {"id": self.id, "name": self.name, "blurb": self.blurb, "available": self.available,
                "note": self.note, "persona": self.persona}


def analyse_position(engine, board, depth=EVAL_DEPTH):
    """One shallow search: {"cp": White's-view centipawns, "best": best move (UCI), "second": score of the runner-up}.

    The runner-up score is what lets practice mode tell an only move from one of several good ones.
    """
    if board.is_checkmate():
        return {"cp": -commentary.MATE if board.turn == chess.WHITE else commentary.MATE, "best": None, "second": None}
    if board.is_game_over(claim_draw=True):
        return {"cp": 0, "best": None, "second": None}
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
    infos = infos if isinstance(infos, list) else [infos]
    score = lambda info: info["score"].white().score(mate_score=commentary.MATE)  # noqa: E731
    top = infos[0]
    return {"cp": score(top), "best": top["pv"][0].uci() if top.get("pv") else None,
            "second": score(infos[1]) if len(infos) > 1 else None}


def eval_white(engine, board, depth=EVAL_DEPTH):
    """Engine score in centipawns from White's point of view."""
    return analyse_position(engine, board, depth)["cp"]


class GameApp:
    """The state behind the UI. All public methods return the JSON-ready state and are safe to call from threads."""

    def __init__(self, engine, echo_factory, bots, save_dir="games", save=True, movetime=None, seed=None,
                 save_fn=None, openings=None):
        self.engine, self.echo_factory, self.bots = engine, echo_factory, bots
        self.openings = openings          # openings.Openings or None: names and "Book" signs
        self.practice = False             # practice mode: move signs, hints and the take-back prompt
        self.save_dir, self.save, self.movetime = save_dir, save, movetime
        self.save_fn = save_fn            # echo.save_game, passed in so this module never imports echo.py
        self.unlocked = False
        self.lock = threading.RLock()
        self.rng = random.Random(seed)
        self.new_game("echo", "white", "balanced")

    # -- bots -------------------------------------------------------------------------------------------------------
    def bot_list(self):
        return [b.public() for b in self.bots.values() if b.id == "echo" or self.unlocked]

    def unlock(self, typed):
        with self.lock:
            if unlock_matches(typed):
                self.unlocked = True
            return {"unlocked": self.unlocked, "bots": self.bot_list()}

    # -- game flow --------------------------------------------------------------------------------------------------
    def new_game(self, bot_id="echo", color="white", dial=None):
        with self.lock:
            bot = self.bots.get(bot_id)
            if bot is None or (bot_id != "echo" and not self.unlocked):
                raise ValueError("unknown opponent")
            if not bot.available:
                raise ValueError(bot.note or "that opponent is not available")
            if color not in ("white", "black"):
                raise ValueError("color must be white or black")
            self.bot = bot
            self.human = chess.WHITE if color == "white" else chess.BLACK
            self.echo = self.echo_factory(bot, dial or bot.dial)
            self.dial_name = dial or bot.dial
            self.board = chess.Board()
            self.talk = commentary.Commentator(bot.persona, random.Random(self.rng.random()))
            self.chat, self.result, self.resigned, self.saved_path = [], None, False, None
            self.analysis = [analyse_position(self.engine, self.board)]   # analysis[i]: the position before ply i
            self.reviews, self.trail = [], []     # per ply: the move's sign, and the latest opening name reached
            self.chat.append(self._line(0, self.talk.start(self.human == chess.BLACK)))
            return self.state()                    # if the bot moves first, state()["pending"] is true: call reply()

    def move(self, uci):
        with self.lock:
            if self.result or self.board.turn != self.human:
                raise ValueError("it is not your turn")
            try:
                mv = chess.Move.from_uci(str(uci))
            except ValueError:
                raise ValueError("unreadable move") from None
            if mv not in self.board.legal_moves:
                raise ValueError("illegal move")
            self._play(mv, mover_is_bot=False)
            return self.state()                    # the bot's answer is a separate call, so your move shows at once

    def reply(self):
        """Play the bot's move if it is the bot's turn (no-op otherwise, so a repeated call is harmless)."""
        with self.lock:
            if not self.result and self.board.turn != self.human:
                self._bot_move()
            return self.state()

    def undo(self):
        with self.lock:
            if self.result and not self.resigned:
                raise ValueError("the game is over; start a new one")
            take = 2 if self.board.turn == self.human else 1
            if len(self.board.move_stack) < take:
                raise ValueError("nothing to undo yet")
            for _ in range(take):
                self.board.pop()
                self.analysis.pop()
                self.reviews.pop()
                self.trail.pop()
            keep = len(self.board.move_stack)
            self.chat = [c for c in self.chat if c["ply"] <= keep]
            self.result, self.resigned, self.saved_path = None, False, None
            return self.state()

    def set_practice(self, on):
        with self.lock:
            self.practice = bool(on)
            return self.state()

    def resign(self):
        with self.lock:
            if self.result:
                raise ValueError("the game is over")
            self.result = "0-1" if self.human == chess.WHITE else "1-0"
            self.resigned = True
            self._close(commentary_line=self.talk.on_result(self.result, self.human == chess.BLACK, resigned=True))
            return self.state()

    # -- internals --------------------------------------------------------------------------------------------------
    def _line(self, ply, text, who="bot"):
        return {"ply": ply, "who": who, "text": text}

    def _play(self, mv, mover_is_bot):
        before = self.board.copy(stack=True)
        a_before = self.analysis[-1]
        self.board.push(mv)
        a_after = analyse_position(self.engine, self.board)
        self.analysis.append(a_after)
        book = bool(self.openings and self.openings.is_book(self.board))
        self.reviews.append(review.classify(before, mv, a_before, a_after, book))
        named = self.openings.lookup(self.board) if self.openings else None
        self.trail.append(named or (self.trail[-1] if self.trail else None))
        text = self.talk.on_move(before, mv, mover_is_bot, a_before["cp"], a_after["cp"])
        if text:
            self.chat.append(self._line(len(self.board.move_stack), text, "bot"))
        self._finished()

    def _bot_move(self):
        try:
            mv = self.echo.choose(self.board, self.movetime)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            raise RuntimeError(f"the engine stopped responding ({exc})") from exc
        self._play(mv, mover_is_bot=True)

    def _finished(self):
        if self.result:
            return True
        if self.board.is_game_over(claim_draw=True):
            self.result = self.board.result(claim_draw=True)
            last = self.chat[-1]["text"] if self.chat else ""
            line = None if self.board.is_checkmate() else self.talk.on_result(self.result, self.human == chess.BLACK)
            self._close(commentary_line=line, already=last)
            return True
        return False

    def _close(self, commentary_line=None, already=""):
        if commentary_line and commentary_line != already:
            self.chat.append(self._line(len(self.board.move_stack), commentary_line))
        if self.save and self.save_fn and self.board.move_stack and self.saved_path is None:
            # Games against the plain Echo bot are labelled with the profile's player name so `learn --resume` finds them.
            # The Fischer bot's games say "You": they are not Fischer's games and must not be fed back as his.
            me = self.bot.brain.player if self.bot.id == "echo" else "You"
            white, black = (me, self.bot.name) if self.human == chess.WHITE else (self.bot.name, me)
            self.saved_path = self.save_fn(self.board, white, black, self.save_dir, self.result)

    def state(self):
        b = self.board
        stack = b.move_stack
        replay, sans = chess.Board(), []
        fens = [replay.board_fen()]              # piece placement after 0, 1, 2 ... plies, for stepping through the game
        for mv in stack:
            sans.append(replay.san(mv))
            replay.push(mv)
            fens.append(replay.board_fen())
        last = stack[-1].uci() if stack else None
        check_sq = chess.square_name(b.king(b.turn)) if b.is_check() else None
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        captured = {"w": [], "b": []}            # pieces each side has taken
        start = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
        for color, key in ((chess.WHITE, "b"), (chess.BLACK, "w")):
            for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
                missing = start[pt] - len(b.pieces(pt, color))
                captured[key] += [chess.piece_symbol(pt)] * max(0, missing)
        mat = sum(values[pt] * (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK))) for pt in values)
        best = self.analysis[-1]["best"] if self.practice and not self.result and b.turn == self.human else None
        hint = {"uci": best, "san": b.san(chess.Move.from_uci(best))} if best else None
        eco, name = self.trail[-1] if self.trail and self.trail[-1] else (None, None)
        return {
            "practice": self.practice, "hint": hint, "reviews": self.reviews, "fens": fens,
            "ucis": [m.uci() for m in stack], "opening": {"eco": eco, "name": name} if name else None,
            "fen": b.fen(), "turn": "w" if b.turn else "b", "human": "w" if self.human else "b",
            "legal": [m.uci() for m in b.legal_moves] if not self.result and b.turn == self.human else [],
            "last": last, "check": check_sq, "san": sans, "captured": captured, "material": mat,
            "pending": not self.result and b.turn != self.human,
            "result": self.result, "resigned": self.resigned, "chat": self.chat, "saved": self.saved_path,
            "bot": self.bot.public(), "dial": self.dial_name, "bots": self.bot_list(), "unlocked": self.unlocked,
        }


# ------------------------------------------------------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------------------------------------------------------
# The only files the server will send: a fixed allow-list, so no request path can reach anything else on disk.
ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
# Everything is served from this origin: no inline script or style, no outside requests, no framing.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def make_handler(app):
    files = {}
    for url, (name, ctype) in ASSETS.items():
        with open(os.path.join(HERE, "ui", name), "rb") as fh:
            files[url] = (fh.read(), ctype)

    class Handler(BaseHTTPRequestHandler):
        server_version = "echostar-ui"

        def log_message(self, fmt, *args):      # keep the terminal quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype != "application/json" else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in files:
                body, ctype = files[path]
                self._send(200, body, ctype)
            elif path == "/api/state":
                with app.lock:
                    self._send(200, app.state())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            port = self.server.server_address[1]
            if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self._send(403, {"error": "request from another site refused"})
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._send(415, {"error": "send JSON"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                return self._send(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, {"error": "unreadable request"})
            path = urlparse(self.path).path
            try:
                if path == "/api/new":
                    out = app.new_game(str(body.get("bot", "echo")), str(body.get("color", "white")),
                                       body.get("dial") if body.get("dial") in ("faithful", "balanced", "sharp")
                                       else None)
                elif path == "/api/move":
                    out = app.move(body.get("uci", ""))
                elif path == "/api/reply":
                    out = app.reply()
                elif path == "/api/practice":
                    out = app.set_practice(body.get("on") is True)
                elif path == "/api/undo":
                    out = app.undo()
                elif path == "/api/resign":
                    out = app.resign()
                elif path == "/api/unlock":
                    out = app.unlock(body.get("typed", ""))
                else:
                    return self._send(404, {"error": "not found"})
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
            except RuntimeError as exc:
                return self._send(502, {"error": str(exc)})
            self._send(200, out)

    return Handler


def serve(app, port=8765, open_browser=True):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Echo UI running at {url}  (Ctrl-C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
