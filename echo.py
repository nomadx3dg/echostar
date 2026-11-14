#!/usr/bin/env python3
"""
Echo v2: a chess engine that learns how YOU play, then plays your style correctly.

Layers
  1. Style Brain : small neural net (+ hand-made style features) trained to predict YOUR moves
  2. Truth Engine: Stockfish (UCI) gives evals for the top candidate moves
  3. Chooser     : picks the most "you" move inside a safety band, then a blunder gate re-checks it

Setup
  pip install python-chess numpy
  Install Stockfish (apt install stockfish / brew install stockfish) or pass --stockfish PATH

Usage
  python echo.py learn  my_games.pgn --player MyLichessName --out style.npz
  python echo.py profile style.npz
  python echo.py play --style style.npz --dial balanced --color white --show
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter

import chess
import chess.engine
import chess.pgn
import numpy as np

# ----------------------------------------------------------------------------
# Hand-made style features (readable; also feed the net as a prior)
# ----------------------------------------------------------------------------
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FEATURES = [
    "capture", "check", "castle", "promotion",
    "moves_pawn", "moves_knight", "moves_bishop", "moves_rook", "moves_queen", "moves_king",
    "to_center", "to_extended_center", "pawn_advance", "retreat",
    "even_or_winning_trade", "kingside_pawn_push", "queenside_pawn_push",
    "early_queen", "develops_minor", "moves_into_attack", "quiet_move",
]
PIECE_IDX = {chess.PAWN: 4, chess.KNIGHT: 5, chess.BISHOP: 6, chess.ROOK: 7, chess.QUEEN: 8, chess.KING: 9}
CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
EXT_CENTER = {chess.square(f, r) for f in range(2, 6) for r in range(2, 6)}


def features(board: chess.Board, move: chess.Move) -> np.ndarray:
    f = np.zeros(len(FEATURES))
    piece = board.piece_at(move.from_square)
    color = board.turn
    is_capture = board.is_capture(move)
    gives_check = board.gives_check(move)

    f[0] = is_capture
    f[1] = gives_check
    f[2] = board.is_castling(move)
    f[3] = move.promotion is not None
    f[PIECE_IDX[piece.piece_type]] = 1
    f[10] = move.to_square in CENTER
    f[11] = move.to_square in EXT_CENTER

    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    forward = (to_rank - from_rank) if color == chess.WHITE else (from_rank - to_rank)
    rel_to_rank = to_rank if color == chess.WHITE else 7 - to_rank
    f[12] = piece.piece_type == chess.PAWN and rel_to_rank >= 3
    f[13] = forward < 0

    if is_capture:
        victim = board.piece_at(move.to_square)
        victim_val = VALUE[victim.piece_type] if victim else 1  # en passant
        f[14] = victim_val >= VALUE[piece.piece_type]

    if piece.piece_type == chess.PAWN:
        file = chess.square_file(move.to_square)
        f[15] = file >= 5
        f[16] = file <= 2

    f[17] = piece.piece_type == chess.QUEEN and board.fullmove_number <= 10
    back_rank = 0 if color == chess.WHITE else 7
    f[18] = piece.piece_type in (chess.KNIGHT, chess.BISHOP) and from_rank == back_rank
    f[19] = board.is_attacked_by(not color, move.to_square) and piece.piece_type != chess.PAWN
    f[20] = not is_capture and not gives_check
    return f


# ----------------------------------------------------------------------------
# Style net: board (768 inputs, side-to-move perspective) -> hidden -> logit per from-to move
# Final move logit = net logit + feature logit, trained jointly with softmax over legal moves.
# ----------------------------------------------------------------------------
HIDDEN = 128


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def encode(board: chess.Board):
    """Indices of active input units, always from the mover's point of view."""
    turn = board.turn
    idx = []
    for sq, p in board.piece_map().items():
        s = sq if turn == chess.WHITE else chess.square_mirror(sq)
        idx.append((p.piece_type - 1 + (0 if p.color == turn else 6)) * 64 + s)
    return np.array(idx, dtype=np.int64)


def move_index(board: chess.Board, move: chess.Move) -> int:
    f, t = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t


class StyleNet:
    def __init__(self, rng=None, hidden=HIDDEN):
        rng = rng or np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.05, (768, hidden))
        self.b1 = np.zeros(hidden)
        self.W2T = rng.normal(0, 0.02, (4096, hidden))   # one row per from-to move
        self.b2 = np.zeros(4096)
        self.w = np.zeros(len(FEATURES))

    def logits(self, board, moves, F=None):
        act = encode(board)
        h = np.maximum(self.W1[act].sum(0) + self.b1, 0)
        midx = np.array([move_index(board, m) for m in moves])
        if F is None:
            F = np.array([features(board, m) for m in moves])
        return self.W2T[midx] @ h + self.b2[midx] + F @ self.w

    def step(self, sample, lr):
        act, midx, F, t = sample
        pre = self.W1[act].sum(0) + self.b1
        h = np.maximum(pre, 0)
        p = softmax(self.W2T[midx] @ h + self.b2[midx] + F @ self.w)
        d = p.copy()
        d[t] -= 1.0
        dh = d @ self.W2T[midx]
        np.add.at(self.W2T, midx, -lr * np.outer(d, h))
        np.add.at(self.b2, midx, -lr * d)
        self.w -= lr * (F.T @ d)
        dpre = dh * (pre > 0)
        self.W1[act] -= lr * dpre
        self.b1 -= lr * dpre
        return -math.log(p[t] + 1e-12)

    def evaluate(self, samples):
        if not samples:
            return None
        top1 = top3 = rand = loss = 0
        for act, midx, F, t in samples:
            lg = self.W2T[midx] @ np.maximum(self.W1[act].sum(0) + self.b1, 0) + self.b2[midx] + F @ self.w
            order = np.argsort(-lg)
            top1 += order[0] == t
            top3 += t in order[:3]
            rand += 1 / len(midx)
            loss -= math.log(softmax(lg)[t] + 1e-12)
        n = len(samples)
        return {"top1": top1 / n, "top3": top3 / n, "random_top1": rand / n, "loss": loss / n}

    def params(self):
        return {k: getattr(self, k).copy() for k in ("W1", "b1", "W2T", "b2", "w")}

    def load_params(self, p):
        for k, v in p.items():
            setattr(self, k, v.copy())


BOOK_PLIES = 16


def book_key(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


# ----------------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------------
def classify_speed(time_control):
    """Lichess-style speed class from a TimeControl header such as '300+3'."""
    tc = str(time_control)
    if "/" in tc:                      # daily / correspondence, e.g. "1/86400"
        return "classical"
    parts = tc.split("+")
    try:
        est = int(parts[0]) + 40 * (int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return None
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


def read_games(pgn_path, player, speeds=None, min_opp_rating=None):
    """Return ([(game, your_color)], skipped_count) for games played by `player`."""
    name = player.lower()
    games, skipped = [], 0
    with open(pgn_path, encoding="utf-8", errors="ignore") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            h = game.headers
            white, black = h.get("White", "").lower(), h.get("Black", "").lower()
            if name == white:
                me = chess.WHITE
            elif name == black:
                me = chess.BLACK
            else:
                continue
            if h.get("Variant", "Standard") not in ("Standard", "From Position"):
                skipped += 1   # chess960, crazyhouse, etc. would break move parsing
                continue
            if speeds and classify_speed(h.get("TimeControl")) not in speeds:
                skipped += 1
                continue
            if min_opp_rating:
                opp = h.get("BlackElo" if me == chess.WHITE else "WhiteElo", "")
                if not opp.isdigit() or int(opp) < min_opp_rating:
                    skipped += 1
                    continue
            games.append((game, me))
    return games, skipped


def learn(args):
    if not os.path.exists(args.pgn):
        sys.exit(f"PGN file not found: {args.pgn}")
    speeds = set(args.speed.split(",")) if args.speed else None
    games, skipped = read_games(args.pgn, args.player, speeds, args.min_opp_rating)
    if not games:
        sys.exit(f"No usable games found for player '{args.player}'. "
                 f"Check the name matches the PGN headers (and your filters).")
    print(f"Found {len(games)} games ({skipped} skipped by filters). Reading your positions...")

    per_game, book = [], {}
    played_sum = np.zeros(len(FEATURES))
    base_sum = np.zeros(len(FEATURES))
    n_pos = 0
    for game, me in games:
        board, samples = game.board(), []
        for ply, move in enumerate(game.mainline_moves()):
            if board.turn == me:
                moves = list(board.legal_moves)
                if move in moves:
                    F = np.array([features(board, m) for m in moves])
                    t = moves.index(move)
                    samples.append((encode(board), np.array([move_index(board, m) for m in moves]), F, t))
                    played_sum += F[t]
                    base_sum += F.mean(0)
                    n_pos += 1
                if ply < BOOK_PLIES:
                    book.setdefault(book_key(board), Counter())[move.uci()] += 1
            board.push(move)
        per_game.append(samples)

    # Hold out the last 10% of games to measure whether the model really predicts you
    n_val = max(1, len(per_game) // 10) if len(per_game) >= 10 else 0
    train = [s for g in per_game[:len(per_game) - n_val] for s in g]
    val = [s for g in per_game[len(per_game) - n_val:] for s in g] if n_val else []
    print(f"{n_pos} positions ({len(train)} train / {len(val)} held-out).")
    if len(train) < 2000 and not args.resume:
        print("Note: under ~2000 positions (roughly 60+ games) the net will mostly memorize. More games help.")

    old = None
    if args.resume:
        net, old = load_model(args.resume)
        if old["player"].lower() != args.player.lower():
            sys.exit(f"{args.resume} was trained for '{old['player']}', not '{args.player}'.")
        lr0 = args.lr * 0.5
        print(f"Resuming from {args.resume} ({old['games']} games, {old['positions']} positions).")
    else:
        net, lr0 = StyleNet(), args.lr

    ev0 = net.evaluate(val) if (old and val) else None
    best = ev0["loss"] if ev0 else float("inf")
    best_params = net.params()
    for epoch in range(args.epochs):
        random.shuffle(train)
        lr = lr0 / (1 + 0.15 * epoch)
        tl = sum(net.step(s, lr) for s in train) / max(1, len(train))
        ev = net.evaluate(val)
        score = ev["loss"] if ev else tl
        msg = f"  epoch {epoch + 1:>2}/{args.epochs}  train loss {tl:.3f}"
        if ev:
            msg += f"  | held-out loss {ev['loss']:.3f}  top-1 {ev['top1']:.1%}"
        print(msg)
        if score < best:
            best, best_params = score, net.params()
    net.load_params(best_params)

    # Merge with the old model's statistics when resuming
    total = n_pos
    if old:
        total += old["positions"]
        played_sum = played_sum + np.array([old["played_rate"][n] for n in FEATURES]) * old["positions"]
        base_sum = base_sum + np.array([old["baseline_rate"][n] for n in FEATURES]) * old["positions"]
        for key, counts in old["book"].items():
            for uci, c in counts.items():
                book.setdefault(key, Counter())[uci] += c

    meta = {
        "player": args.player,
        "games": len(games) + (old["games"] if old else 0),
        "positions": total,
        "played_rate": dict(zip(FEATURES, (played_sum / total).tolist())),
        "baseline_rate": dict(zip(FEATURES, (base_sum / total).tolist())),
        "book": {k: dict(v) for k, v in book.items()},
        "validation": net.evaluate(val) or (old or {}).get("validation"),
    }
    np.savez_compressed(args.out, meta=json.dumps(meta), **net.params())
    print(f"\nSaved style model to {args.out}")
    print_profile(meta)


def load_model(path):
    if not os.path.exists(path):
        sys.exit(f"Style file not found: {path}. Create one first with: python echo.py learn <games.pgn> --player <name>")
    d = np.load(path, allow_pickle=False)
    net = StyleNet(hidden=d["W1"].shape[1])
    net.load_params({k: d[k] for k in ("W1", "b1", "W2T", "b2", "w")})
    return net, json.loads(str(d["meta"]))


# ----------------------------------------------------------------------------
# Portable model format for the Rust port (see docs/FORMAT.md)
# ----------------------------------------------------------------------------
FORMAT_MAGIC = b"ECHOSTYL"
FORMAT_VERSION = 1


def export_echo(net, meta, path):
    """Write weights + metadata in a dependency-free binary format: magic, version, JSON, f32 tensors."""
    header = dict(meta)
    header["dims"] = {"input": 768, "hidden": int(net.W1.shape[1]), "moves": 4096, "features": len(FEATURES)}
    header["feature_names"] = FEATURES
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(FORMAT_MAGIC)
        fh.write(np.array([FORMAT_VERSION, len(blob)], dtype="<u4").tobytes())
        fh.write(blob)
        for name in ("W1", "b1", "W2T", "b2", "w"):
            fh.write(np.ascontiguousarray(getattr(net, name), dtype="<f4").tobytes())


def export_cmd(args):
    net, meta = load_model(args.style)
    export_echo(net, meta, args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) for the Rust engine.")


def tendency_rows(meta):
    rows = [(meta["played_rate"][n] - meta["baseline_rate"][n], n) for n in FEATURES]
    return sorted(rows, key=lambda r: -abs(r[0]))


def repertoire(meta):
    """Most common opening moves from the book: [(label, 'Nf3 (4x), d4 (2x)')]."""
    book = meta.get("book", {})

    def top(board, n=3):
        counts = book.get(book_key(board), {})
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:n]
        return ", ".join(f"{board.san(chess.Move.from_uci(u))} ({c}x)" for u, c in ranked)

    lines = []
    first = top(chess.Board())
    if first:
        lines.append(("As White, first move", first))
    for uci in ("e2e4", "d2d4", "c2c4", "g1f3"):
        board = chess.Board()
        mv = chess.Move.from_uci(uci)
        label = board.san(mv)
        board.push(mv)
        reply = top(board)
        if reply:
            lines.append((f"As Black vs {label}", reply))
    return lines


def print_profile(meta):
    print(f"\nStyle profile for {meta['player']} ({meta['games']} games, {meta['positions']} positions)")
    print("Tendency = how much more (or less) you pick this kind of move than a random legal move.\n")
    for diff, n in tendency_rows(meta)[:10]:
        print(f"  {'+' if diff > 0 else '-'} {n:<24} you {meta['played_rate'][n]:5.1%}  "
              f"vs random {meta['baseline_rate'][n]:5.1%}")
    reps = repertoire(meta)
    print(f"\n  Opening book: {len(meta['book'])} positions from your games")
    for label, text in reps:
        print(f"    {label}: {text}")
    v = meta.get("validation")
    if v:
        print(f"\n  Accuracy on games it never trained on:")
        print(f"    predicts your exact move {v['top1']:.1%} of the time (random guessing: {v['random_top1']:.1%})")
        print(f"    your move is in its top 3 {v['top3']:.1%} of the time")
    else:
        print("\n  Accuracy: not measured (need 10+ games to hold some out).")


def export_markdown(meta):
    out = [f"# Style profile: {meta['player']}", "",
           f"{meta['games']} games, {meta['positions']} positions.", "",
           "## Tendencies", "",
           "| Move type | You | Random legal move | Difference |", "|---|---|---|---|"]
    for diff, n in tendency_rows(meta):
        out.append(f"| {n} | {meta['played_rate'][n]:.1%} | {meta['baseline_rate'][n]:.1%} | {diff:+.1%} |")
    out += ["", "## Opening repertoire", ""]
    out += [f"- **{label}:** {text}" for label, text in repertoire(meta)] or ["No opening data."]
    v = meta.get("validation")
    out += ["", "## Accuracy", ""]
    if v:
        out += [f"- Exact move predicted: {v['top1']:.1%} (random guessing {v['random_top1']:.1%})",
                f"- Your move in top 3: {v['top3']:.1%}"]
    else:
        out.append("Not measured (needs 10+ games to hold some out).")
    return "\n".join(out) + "\n"


def profile(args):
    meta = load_model(args.style)[1]
    print_profile(meta)
    if args.export:
        with open(args.export, "w") as fh:
            fh.write(export_markdown(meta))
        print(f"\nReadable summary written to {args.export}")


class StyleBrain:
    def __init__(self, path=None):
        self.net, self.book, self.player = None, {}, "You"
        if path:
            self.net, meta = load_model(path)
            self.book = meta.get("book", {})
            self.player = meta["player"]

    def scores(self, board, moves):
        if self.net is None:
            return [0.0] * len(moves)
        # Score each move on its own raw logit (comparable within one position)
        lg = self.net.logits(board, moves)
        bonus = [1.5 * math.log1p(self.book.get(book_key(board), {}).get(m.uci(), 0)) for m in moves]
        return [float(a + b) for a, b in zip(lg, bonus)]


# ----------------------------------------------------------------------------
# Truth Engine + Chooser
# ----------------------------------------------------------------------------
DIALS = {
    # band: allowed centipawn drop from the best move; temp: 0 = always the most "you" safe move
    "faithful": {"band": 120, "temp": 0.6},
    "balanced": {"band": 50, "temp": 0.3},
    "sharp": {"band": 20, "temp": 0.0},
}
GATE_SLACK = 40
MAX_GATE_CHECKS = 3   # re-verify at most this many style picks before trusting the engine
MATE = 100000


class Echo:
    def __init__(self, engine, brain, dial="balanced", depth=14, multipv=8, verbose=False):
        self.engine, self.brain = engine, brain
        self.dial = dict(DIALS[dial]) if isinstance(dial, str) else dict(dial)
        self.depth, self.multipv, self.verbose = depth, multipv, verbose
        self.history = []   # (matched engine best?, centipawns given up) per move

    def summary(self):
        if not self.history:
            return ""
        n = len(self.history)
        matched = sum(1 for m, _ in self.history if m)
        cost = sum(c for _, c in self.history) / n
        return (f"Echo matched the engine's top move {matched}/{n} times and gave up "
                f"{cost:.0f} centipawns per move on average to stay in your style.")

    def candidates(self, board, movetime=None):
        limit = chess.engine.Limit(time=movetime * 0.5) if movetime else chess.engine.Limit(depth=self.depth)
        infos = self.engine.analyse(board, limit, multipv=self.multipv)
        return [(i["pv"][0], i["score"].pov(board.turn).score(mate_score=MATE)) for i in infos if "pv" in i]

    def verify(self, board, move, movetime=None):
        """Blunder gate: deeper search after the move, from the mover's point of view."""
        mover = board.turn
        limit = (chess.engine.Limit(time=max(0.02, movetime * 0.15)) if movetime
                 else chess.engine.Limit(depth=self.depth + 6))
        board.push(move)
        try:
            info = self.engine.analyse(board, limit)
            return info["score"].pov(mover).score(mate_score=MATE)
        finally:
            board.pop()

    def choose(self, board, movetime=None):
        """Pick a move. `movetime` (seconds) switches from fixed depth to a time budget."""
        cands = self.candidates(board, movetime)
        best_move, best_cp = cands[0]
        band = self.dial["band"]
        if best_cp >= MATE - 1000:
            band = 0                      # forced mate: take it
        elif abs(best_cp) > 500:
            band = min(band, 30)          # decided position: stop experimenting
        safe = [(m, cp) for m, cp in cands if cp >= best_cp - band]

        style = self.brain.scores(board, [m for m, _ in safe])
        scored = [(s, m, cp) for s, (m, cp) in zip(style, safe)]
        temp = self.dial["temp"]
        if temp > 0 and len(scored) > 1:
            probs = softmax(np.array([s for s, _, _ in scored]) / temp)
            order = np.random.choice(len(scored), size=len(scored), replace=False, p=probs)
            ranked = [scored[i] for i in order]
        else:
            ranked = sorted(scored, key=lambda t: -t[0])

        checks = 0
        for _, move, cp in ranked:
            if move != best_move:
                if checks >= MAX_GATE_CHECKS:
                    break                  # out of budget: fall back to the engine's best
                checks += 1
            if move == best_move or self.verify(board, move, movetime) >= best_cp - band - GATE_SLACK:
                self.history.append((move == best_move, best_cp - cp))
                if self.verbose:
                    print(f"   [coach] {len(safe)}/{len(cands)} moves were safe | "
                          f"engine best {board.san(best_move)} ({best_cp / 100:+.2f}) | "
                          f"Echo plays {board.san(move)} ({cp / 100:+.2f})")
                return move
        self.history.append((True, 0))
        return best_move


# ----------------------------------------------------------------------------
# Board rendering
# ----------------------------------------------------------------------------
GLYPH = {chess.PAWN: "\u265f\ufe0e", chess.KNIGHT: "\u265e", chess.BISHOP: "\u265d",
         chess.ROOK: "\u265c", chess.QUEEN: "\u265b", chess.KING: "\u265a"}
LIGHT, DARK = 180, 95            # square colors (256-color palette)
LAST_LIGHT, LAST_DARK = 186, 142  # last-move highlight
CHECK_BG = 160


def material_diff(board):
    score = 0
    for p in board.piece_map().values():
        score += VALUE[p.piece_type] * (1 if p.color == chess.WHITE else -1)
    return score


def render(board, perspective=chess.WHITE, last=None, color=True):
    ranks = range(7, -1, -1) if perspective == chess.WHITE else range(8)
    files = range(8) if perspective == chess.WHITE else range(7, -1, -1)
    last_sqs = {last.from_square, last.to_square} if last else set()
    king_in_check = board.king(board.turn) if board.is_check() else None
    out = []
    for r in ranks:
        row = f" {r + 1} "
        for f in files:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            light = (f + r) % 2 == 1
            if color:
                bg = CHECK_BG if sq == king_in_check else (
                    (LAST_LIGHT if light else LAST_DARK) if sq in last_sqs else (LIGHT if light else DARK))
                fg = 231 if piece and piece.color == chess.WHITE else 16
                glyph = GLYPH[piece.piece_type] if piece else " "
                row += f"\033[48;5;{bg}m\033[38;5;{fg}m {glyph} \033[0m"
            else:
                row += f" {piece.symbol() if piece else ('.' if not light else ' ')} "
        out.append(row)
    letters = "abcdefgh" if perspective == chess.WHITE else "hgfedcba"
    out.append("    " + "  ".join(letters))
    diff = material_diff(board)
    who = "White" if diff > 0 else "Black"
    status = f"Material: {who} +{abs(diff)}" if diff else "Material: even"
    if board.is_check():
        status += "   CHECK"
    out.append(f"\n  Move {board.fullmove_number}, {'White' if board.turn else 'Black'} to move   {status}")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Play in the terminal
# ----------------------------------------------------------------------------
def find_stockfish(path):
    found = path or shutil.which("stockfish")
    if not found:
        sys.exit("Stockfish not found. Install it or pass --stockfish /path/to/stockfish")
    return found


def save_game(board, white, black, directory, result=None):
    """Write the finished (or abandoned) game as PGN so `learn --resume` can study it."""
    import datetime
    game = chess.pgn.Game.from_board(board)
    game.headers.update({"Event": "Echo practice game", "Site": "Local",
                         "Date": datetime.date.today().strftime("%Y.%m.%d"),
                         "White": white, "Black": black,
                         "Result": result or board.result(claim_draw=True)})
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, time.strftime("echo_%Y%m%d_%H%M%S.pgn"))
    with open(path, "w") as fh:
        print(game, file=fh, end="\n\n")
    return path


def play(args):
    brain = StyleBrain(args.style)
    if not args.style:
        print("No style file given: Echo will play as a plain engine.")
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish(args.stockfish))
    opts = {"Threads": args.threads, "Hash": args.hash}
    engine.configure({k: v for k, v in opts.items() if v and k in engine.options})
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    dial = dict(DIALS[args.dial])
    if args.band is not None:
        dial["band"] = args.band
    if args.temp is not None:
        dial["temp"] = args.temp
    echo = Echo(engine, brain, dial, args.depth, verbose=args.show)
    human = chess.WHITE if args.color == "white" else chess.BLACK
    board, last, result = chess.Board(), None, None
    print(f"Echo ({args.dial}, band {dial['band']} cp) vs you. Moves in SAN (e4, Nf3) or UCI (e2e4).")
    print("Commands: undo, hint, resign, help, quit.\n")

    try:
        while not board.is_game_over(claim_draw=True):
            print(render(board, human, last, color=not args.plain), "\n")
            if board.turn == human:
                text = input("Your move: ").strip()
                if text in ("quit", "q"):
                    break
                if text == "help":
                    print("Enter a move like e4, Nf3, O-O, e8=Q (or e2e4). undo = take back your last move,\n"
                          "hint = ask the engine for its best move, resign = give up, quit = stop and save.\n")
                    continue
                if text == "resign":
                    result = "0-1" if human == chess.WHITE else "1-0"
                    print("You resigned.")
                    break
                if text == "hint":
                    info = engine.analyse(board, chess.engine.Limit(depth=min(args.depth, 12)))
                    print(f"Engine suggests {board.san(info['pv'][0])} "
                          f"({info['score'].pov(board.turn).score(mate_score=MATE) / 100:+.2f})\n")
                    continue
                if text == "undo" and len(board.move_stack) >= 2:
                    board.pop(); board.pop()
                    last = board.peek() if board.move_stack else None
                    continue
                try:
                    try:
                        move = board.parse_san(text)
                    except ValueError:
                        move = chess.Move.from_uci(text)
                        if move not in board.legal_moves:
                            raise ValueError
                except ValueError:
                    print("Illegal or unreadable move, try again.\n")
                    continue
                board.push(move)
                last = move
            else:
                try:
                    move = echo.choose(board, args.movetime)
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
                    print(f"The engine stopped responding ({exc}). Saving the game so far.")
                    break
                print(f"Echo plays {board.san(move)}\n")
                board.push(move)
                last = move
        print(render(board, human, last, color=not args.plain), "\nResult:",
              result or board.result(claim_draw=True))
    finally:
        if args.show and echo.summary():
            print(echo.summary())
        if not args.no_save and board.move_stack:
            white, black = (brain.player, "Echo") if human == chess.WHITE else ("Echo", brain.player)
            print("Game saved to", save_game(board, white, black, args.save_dir, result))
        engine.quit()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Download your games (Lichess / Chess.com public APIs, no account token needed)
# ----------------------------------------------------------------------------
LICHESS_URL = "https://lichess.org"
CHESSCOM_URL = "https://api.chess.com"
USER_AGENT = "echostar (https://github.com/nomadx3dg/echostar)"


def http_get(url, accept="*/*", retries=3):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                print("Rate limited by the server, waiting 60 seconds...")
                time.sleep(60)
                continue
            if exc.code == 404:
                sys.exit(f"Not found: {url}\nCheck the username spelling.")
            sys.exit(f"The server answered HTTP {exc.code} for {url}")
        except urllib.error.URLError as exc:
            sys.exit(f"Network error: {exc.reason}")


def fetch_lichess(user, max_games, speeds=None, rated=False, base=LICHESS_URL):
    query = f"max={max_games}&tags=true&clocks=false&evals=false&opening=false"
    if speeds:
        names = {"bullet": "bullet", "blitz": "blitz", "rapid": "rapid", "classical": "classical"}
        query += "&perfType=" + ",".join(names[s] for s in speeds)
    if rated:
        query += "&rated=true"
    return http_get(f"{base}/api/games/user/{user}?{query}", accept="application/x-chess-pgn")


def fetch_chesscom(user, max_games, base=CHESSCOM_URL):
    import json as _json
    archives = _json.loads(http_get(f"{base}/pub/player/{user}/games/archives", "application/json")).get("archives", [])
    chunks, total = [], 0
    for url in reversed(archives):             # newest month first
        text = http_get(url + "/pgn", "application/x-chess-pgn")
        chunks.append(text)
        total += text.count("[Event ")
        if total >= max_games:
            break
    return "\n\n".join(chunks)


def fetch_cmd(args):
    speeds = [x for x in args.speed.split(",")] if args.speed else None
    for sp in speeds or []:
        if sp not in ("bullet", "blitz", "rapid", "classical"):
            sys.exit(f"Unknown speed '{sp}'. Use bullet, blitz, rapid or classical.")
    if args.site == "lichess":
        text = fetch_lichess(args.user, args.max, speeds, args.rated)
    else:
        text = fetch_chesscom(args.user, args.max)
    count = text.count("[Event ")
    if count == 0:
        sys.exit("No games came back. Check the username, and loosen --speed / --rated if you used them.")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Saved {count} games to {args.out}")
    print(f"Next: python echo.py learn {args.out} --player {args.user}")
    if args.site == "chesscom" and speeds:
        print("Note: Chess.com downloads aren't filtered by speed here; use `learn --speed` to filter while training.")


def main():
    p = argparse.ArgumentParser(description="Echo: a chess engine that plays your style, correctly.")
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("learn", help="train a style model from your PGN games")
    l.add_argument("pgn")
    l.add_argument("--player", required=True, help="your name exactly as in the PGN headers")
    l.add_argument("--out", default="style.npz")
    l.add_argument("--epochs", type=int, default=12)
    l.add_argument("--lr", type=float, default=0.02)
    l.add_argument("--speed", help="only use these speeds, comma separated: bullet,blitz,rapid,classical")
    l.add_argument("--min-opp-rating", type=int, help="only use games against opponents rated at least this")
    l.add_argument("--resume", help="continue training an existing style.npz with new games")
    l.set_defaults(fn=learn)

    f = sub.add_parser("fetch", help="download your games from Lichess or Chess.com")
    f.add_argument("user", help="your username on that site")
    f.add_argument("--site", choices=["lichess", "chesscom"], default="lichess")
    f.add_argument("--max", type=int, default=500, help="how many games (newest first), default 500")
    f.add_argument("--speed", help="Lichess only: bullet,blitz,rapid,classical (comma separated)")
    f.add_argument("--rated", action="store_true", help="Lichess only: rated games only")
    f.add_argument("--out", default="games.pgn")
    f.set_defaults(fn=fetch_cmd)

    pr = sub.add_parser("profile", help="show your style profile and accuracy")
    pr.add_argument("style")
    pr.add_argument("--export", metavar="FILE", help="also write a readable markdown summary")
    pr.set_defaults(fn=profile)

    g = sub.add_parser("play", help="play a game against Echo")
    g.add_argument("--style", help="style.npz from the learn step")
    g.add_argument("--dial", choices=list(DIALS), default="balanced")
    g.add_argument("--color", choices=["white", "black"], default="white", help="your color")
    g.add_argument("--depth", type=int, default=14)
    g.add_argument("--movetime", type=float, help="seconds per move (overrides --depth)")
    g.add_argument("--band", type=int, help="custom safety band in centipawns (overrides the dial)")
    g.add_argument("--temp", type=float, help="custom style temperature, 0 = always the most 'you' safe move")
    g.add_argument("--seed", type=int, help="random seed, for reproducible games")
    g.add_argument("--stockfish", help="path to the Stockfish binary")
    g.add_argument("--show", action="store_true", help="coach view: engine best vs Echo's pick")
    g.add_argument("--plain", action="store_true", help="plain ASCII board (no colors/unicode)")
    g.add_argument("--threads", type=int, help="Stockfish threads")
    g.add_argument("--hash", type=int, help="Stockfish hash size in MB")
    g.add_argument("--save-dir", default="games", help="where finished games are saved as PGN")
    g.add_argument("--no-save", action="store_true", help="don't save the game")
    g.set_defaults(fn=play)

    ex = sub.add_parser("export", help="convert a style.npz to the .echo format used by the Rust engine")
    ex.add_argument("style")
    ex.add_argument("--out", default="style.echo")
    ex.set_defaults(fn=export_cmd)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
