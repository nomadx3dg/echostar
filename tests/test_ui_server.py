import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import chess
import chess.engine
import pytest

import echo
import openings
import ui_server

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
CODE = "reykjavik72"


def material(board):
    return sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in VALUES.items())


class MaterialEngine:
    """Stands in for Stockfish: scores a position by material, one ply deep. Good enough to drive the app."""
    def analyse(self, board, limit, multipv=None):
        rows = []
        for mv in list(board.legal_moves):
            board.push(mv)
            white_cp = material(board) * 100
            board.pop()
            rows.append((mv, white_cp))
        rows.sort(key=lambda r: -r[1] if board.turn == chess.WHITE else r[1])
        score = lambda cp: chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)  # noqa: E731
        if multipv:
            return [{"pv": [mv], "score": score(cp)} for mv, cp in rows[:multipv]]
        return {"score": score(material(board) * 100)}


class Brain:
    def __init__(self, player="Tester"):
        self.player = player

    def scores(self, board, moves):
        return [0.0] * len(moves)


_OPENINGS = []


def shared_openings():
    """The opening data takes a couple of seconds to load; load it once for the whole test run."""
    if not _OPENINGS:
        _OPENINGS.append(openings.Openings())
    return _OPENINGS[0]


def make_app(tmp_path, fischer_available=True, save=False, with_openings=False):
    engine = MaterialEngine()
    bots = {
        "echo": ui_server.Bot("echo", "Echo", "style bot", "echo", Brain("Tester"), "balanced"),
        "fischer": ui_server.Bot("fischer", "Fischer", "style model", "fischer", Brain("Fischer, Robert James"),
                                 "sharp", available=fischer_available, note="Train it first."),
    }
    return ui_server.GameApp(engine, lambda bot, dial: echo.Echo(engine, bot.brain, dial, depth=1), bots,
                             save_dir=str(tmp_path), save=save, seed=1, save_fn=echo.save_game,
                             openings=shared_openings() if with_openings else None)


def test_unlock_word_matching():
    assert ui_server.unlock_matches(CODE)
    assert ui_server.unlock_matches("zzz REYKJAVIK-72")           # case, punctuation and leading noise are ignored
    assert ui_server.unlock_matches("hello" + CODE)
    assert not ui_server.unlock_matches("reykjavik")
    assert not ui_server.unlock_matches("reykjavik73")
    assert not ui_server.unlock_matches(CODE + "x")                # must be the end of what was typed
    assert not ui_server.unlock_matches("")


def test_source_does_not_contain_the_unlock_word():
    with open(ui_server.__file__, encoding="utf-8") as fh:
        assert CODE not in fh.read().lower()


def test_new_game_and_move_flow(tmp_path):
    app = make_app(tmp_path)
    st = app.state()
    assert st["turn"] == "w" and "e2e4" in st["legal"] and not st["pending"]
    assert [b["id"] for b in st["bots"]] == ["echo"]               # the extra opponent is hidden
    st = app.move("e2e4")
    assert st["pending"] and st["legal"] == [] and st["san"] == ["e4"]
    st = app.reply()
    assert not st["pending"] and len(st["san"]) == 2 and st["legal"]
    assert app.reply()["san"] == st["san"]                        # a repeated reply() does not play twice
    assert st["chat"][0]["ply"] == 0 and st["chat"][0]["text"]


def test_rejects_bad_moves_and_wrong_turn(tmp_path):
    app = make_app(tmp_path)
    for bad in ("e2e5", "zzzz", "", "e7e5"):
        with pytest.raises(ValueError):
            app.move(bad)
    app.move("e2e4")
    with pytest.raises(ValueError):                                # bot's turn
        app.move("d2d4")


def test_undo_takes_back_a_full_move_and_trims_chat(tmp_path):
    app = make_app(tmp_path)
    with pytest.raises(ValueError):
        app.undo()
    app.move("e2e4")
    app.reply()
    st = app.undo()
    assert st["san"] == [] and st["legal"] and all(c["ply"] == 0 for c in st["chat"])


def test_bot_plays_first_when_human_is_black(tmp_path):
    app = make_app(tmp_path)
    st = app.new_game("echo", "black")
    assert st["human"] == "b" and st["pending"]
    st = app.reply()
    assert len(st["san"]) == 1 and not st["pending"] and st["legal"]


def test_resign_ends_game_and_saves_pgn_with_profile_name(tmp_path):
    app = make_app(tmp_path, save=True)
    app.move("e2e4")
    app.reply()
    st = app.resign()
    assert st["result"] == "0-1" and st["resigned"] and st["saved"]
    text = open(st["saved"]).read()
    assert '[White "Tester"]' in text and '[Black "Echo"]' in text
    with pytest.raises(ValueError):
        app.move("d2d4")


def test_unlock_reveals_fischer_and_its_games_are_not_labelled_as_his(tmp_path):
    app = make_app(tmp_path, save=True)
    with pytest.raises(ValueError):
        app.new_game("fischer")                                    # locked
    assert app.unlock("nothing to see")["unlocked"] is False
    assert app.unlock(CODE)["unlocked"] is True
    assert [b["id"] for b in app.bot_list()] == ["echo", "fischer"]
    st = app.new_game("fischer", "white")
    assert st["bot"]["persona"] == "fischer" and st["dial"] == "sharp"
    app.move("e2e4")
    app.reply()
    text = open(app.resign()["saved"]).read()
    assert '[White "You"]' in text and '[Black "Fischer"]' in text
    assert "Fischer, Robert James" not in text          # these are not Fischer's games


def test_unavailable_fischer_model_gives_a_helpful_error(tmp_path):
    app = make_app(tmp_path, fischer_available=False)
    app.unlock(CODE)
    with pytest.raises(ValueError, match="Train it first"):
        app.new_game("fischer")


def test_checkmate_sets_result_and_ends_the_turn(tmp_path):
    app = make_app(tmp_path)
    app.new_game("echo", "white")
    for san in ["f3", "e5"]:
        app.board.push_san(san)
    app.analysis = [{"cp": 0, "best": None, "second": None}] * (len(app.board.move_stack) + 1)
    app.reviews, app.trail = [None] * len(app.board.move_stack), [None] * len(app.board.move_stack)
    app.move("g2g4")
    app.board.push_san("Qh4#")                          # Black mates
    assert app._finished() and app.result == "0-1"
    st = app.state()
    assert st["result"] == "0-1" and st["legal"] == [] and not st["pending"]


@pytest.fixture
def server(tmp_path):
    app = make_app(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui_server.make_handler(app))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1], app
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    hdrs = {"Content-Type": "application/json"} if method == "POST" else {}
    hdrs.update(headers or {})
    payload = raw if raw is not None else (json.dumps(body) if body is not None else None)
    conn.request(method, path, body=payload, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_http_page_and_api(server):
    port, _ = server
    status, page = call(port, "GET", "/")
    assert status == 200 and b"<title>Echo" in page
    assert b'href="/styles.css"' in page and b'src="/app.js"' in page
    assert b"<style" not in page and b"<script>" not in page          # nothing inline, so the strict CSP can hold
    status, data = call(port, "POST", "/api/move", {"uci": "e2e4"})
    assert status == 200 and json.loads(data)["pending"]
    status, data = call(port, "POST", "/api/reply", {})
    assert status == 200 and len(json.loads(data)["san"]) == 2
    status, data = call(port, "POST", "/api/move", {"uci": "a1a8"})
    assert status == 400 and "error" in json.loads(data)
    assert call(port, "GET", "/nope")[0] == 404
    assert call(port, "POST", "/api/nope", {})[0] == 404


def test_http_refuses_other_sites_wrong_types_and_big_bodies(server):
    port, _ = server
    assert call(port, "POST", "/api/new", {}, headers={"Origin": "http://evil.example"})[0] == 403
    assert call(port, "POST", "/api/new", {}, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
    assert call(port, "POST", "/api/new", {}, headers={"Content-Type": "text/plain"})[0] == 415
    assert call(port, "POST", "/api/new", raw="x" * 10000)[0] == 413
    assert call(port, "POST", "/api/new", raw="not json")[0] == 400
    assert call(port, "POST", "/api/new", raw="[1,2]")[0] == 400


def test_http_unlock_roundtrip(server):
    port, _ = server
    _, data = call(port, "POST", "/api/unlock", {"typed": "abc"})
    assert json.loads(data)["unlocked"] is False
    _, data = call(port, "POST", "/api/unlock", {"typed": CODE})
    out = json.loads(data)
    assert out["unlocked"] and [b["id"] for b in out["bots"]] == ["echo", "fischer"]


def test_http_serves_only_the_listed_assets_with_a_strict_csp(server):
    port, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    for path, ctype in (("/styles.css", "text/css"), ("/app.js", "text/javascript"), ("/", "text/html")):
        conn.request("GET", path)
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 200 and resp.getheader("Content-Type").startswith(ctype), path
        csp = resp.getheader("Content-Security-Policy")
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp
        assert resp.getheader("X-Content-Type-Options") == "nosniff"
    conn.close()
    for path in ("/ui_server.py", "/echo.py", "/../ui_server.py", "/%2e%2e/echo.py", "/ui/app.js", "/README.md",
                 "/styles.css/../echo.py"):
        assert call(port, "GET", path)[0] == 404, path


def test_state_carries_history_signs_and_hint_only_in_practice_mode(tmp_path):
    app = make_app(tmp_path)
    st = app.state()
    assert st["practice"] is False and st["hint"] is None and st["reviews"] == [] and len(st["fens"]) == 1
    app.move("e2e4")
    st = app.reply()
    plies = len(st["san"])
    assert plies == 2 and len(st["fens"]) == plies + 1 and len(st["ucis"]) == plies and len(st["reviews"]) == plies
    assert st["fens"][0].startswith("rnbqkbnr/pppppppp") and st["ucis"][0] == "e2e4"
    labels = {"brilliant", "great", "best", "excellent", "good", "book", "forced", "inaccuracy", "mistake", "blunder"}
    assert all(r["label"] in labels for r in st["reviews"])
    assert all({"label", "loss", "best", "best_san"} <= set(r) for r in st["reviews"])
    assert st["hint"] is None                                     # practice mode is off
    st = app.set_practice(True)
    assert st["practice"] is True and st["hint"]["uci"] in st["legal"] and st["hint"]["san"]
    assert app.set_practice(False)["hint"] is None


def test_hint_is_withheld_when_it_is_not_your_move_or_the_game_is_over(tmp_path):
    app = make_app(tmp_path)
    app.set_practice(True)
    app.move("e2e4")
    assert app.state()["pending"] and app.state()["hint"] is None
    app.reply()
    assert app.state()["hint"] is not None
    assert app.resign()["hint"] is None


def test_undo_trims_history_and_signs_together(tmp_path):
    app = make_app(tmp_path)
    app.move("e2e4")
    app.reply()
    app.move("d2d4")
    app.reply()
    st = app.undo()
    assert len(st["san"]) == 2 and len(st["reviews"]) == 2 and len(st["fens"]) == 3 and len(st["ucis"]) == 2
    st = app.undo()
    assert st["reviews"] == [] and st["fens"] == [chess.STARTING_BOARD_FEN.split(" ")[0]]


def test_opening_name_and_book_sign_come_from_the_opening_data(tmp_path):
    app = make_app(tmp_path, with_openings=True)
    assert app.state()["opening"] is None
    st = app.move("e2e4")
    assert st["opening"] and st["opening"]["name"] and st["opening"]["eco"]
    assert st["reviews"][0]["label"] == "book"                    # 1.e4 is on a named line, and loses nothing
    app.reply()
    assert app.state()["opening"]["name"]                          # the name sticks until a deeper one replaces it
    assert make_app(tmp_path).state()["opening"] is None           # no data file, no names, and nothing breaks


def test_new_game_starts_a_fresh_history_but_keeps_practice_mode(tmp_path):
    app = make_app(tmp_path)
    app.set_practice(True)
    app.move("e2e4")
    app.reply()
    st = app.new_game("echo", "white")
    assert st["practice"] is True and st["reviews"] == [] and st["fens"] == [st["fens"][0]] and st["ucis"] == []


def test_http_practice_toggle(server):
    port, _ = server
    _, data = call(port, "POST", "/api/practice", {"on": True})
    assert json.loads(data)["practice"] is True
    _, data = call(port, "POST", "/api/practice", {"on": "yes"})        # only a real true turns it on
    assert json.loads(data)["practice"] is False
