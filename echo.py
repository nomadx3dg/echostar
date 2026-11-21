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
from urllib.parse import quote

import chess
import chess.engine
import chess.pgn
import numpy as np

import personality

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


# Output units. Units 0..4095 are from-to squares (v1). Queen promotions keep their v1 unit, so ordinary moves and
# queen promotions score exactly as before. Under-promotions (knight, bishop, rook) get their own units:
# 3 pieces x 3 directions (capture left, straight, capture right) x 8 from-files = 72 extra units.
N_BASE_MOVES = 4096
UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
N_MOVES = N_BASE_MOVES + len(UNDERPROMO) * 24
PROFILE_VERSION = 2


def underpromo_sources():
    """For each under-promotion unit, the v1 from-to unit (the queen promotion) it was split off from."""
    src = []
    for _piece in UNDERPROMO:
        for drift in (-1, 0, 1):
            for file in range(8):
                to_file = min(7, max(0, file + drift))     # off-board directions never occur; any valid source works
                src.append((48 + file) * 64 + 56 + to_file)   # from the 7th rank to the 8th, in the mover's view
    return src


def move_index(board: chess.Board, move: chess.Move) -> int:
    f, t = move.from_square, move.to_square
    if move.promotion in UNDERPROMO:
        drift = chess.square_file(t) - chess.square_file(f)    # -1, 0 or +1; files are unaffected by mirroring
        return N_BASE_MOVES + UNDERPROMO[move.promotion] * 24 + (drift + 1) * 8 + chess.square_file(f)
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t


class StyleNet:
    def __init__(self, rng=None, hidden=HIDDEN):
        rng = rng or np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.05, (768, hidden))
        self.b1 = np.zeros(hidden)
        # one row per from-to move, plus the under-promotion units (they start as copies of the queen promotion)
        widened, _ = migrate_params({"W2T": rng.normal(0, 0.02, (N_BASE_MOVES, hidden)), "b2": np.zeros(N_BASE_MOVES)})
        self.W2T, self.b2 = widened["W2T"], widened["b2"]
        self.w = np.zeros(len(FEATURES))

    def logits(self, board, moves, F=None):
        act = encode(board)
        h = np.maximum(self.W1[act].sum(0) + self.b1, 0)
        midx = np.array([move_index(board, m) for m in moves])
        if F is None:
            F = np.array([features(board, m) for m in moves])
        return self.W2T[midx] @ h + self.b2[midx] + F @ self.w

    def step(self, sample, lr, mix=0.0):
        """One SGD step. `sample` is (active inputs, move units, features, index of the move played) and may carry a
        fifth element: a teacher distribution over the same legal moves (see maia_teacher.py). With `mix` > 0 the
        training target becomes (1 - mix) * the move played + mix * the teacher; with `mix` = 0 (the default) the
        teacher is ignored and this is the original update. The returned loss is always the plain cross-entropy
        on the move played, so it stays comparable across runs."""
        act, midx, F, t = sample[:4]
        q = sample[4] if len(sample) > 4 else None
        pre = self.W1[act].sum(0) + self.b1
        h = np.maximum(pre, 0)
        p = softmax(self.W2T[midx] @ h + self.b2[midx] + F @ self.w)
        d = p.copy()
        if q is not None and mix > 0:
            d -= mix * q
            d[t] -= 1.0 - mix
        else:
            d[t] -= 1.0
        dh = d @ self.W2T[midx]
        # Legal moves map to distinct units, so plain fancy indexing equals np.add.at here (bit-identical, faster).
        self.W2T[midx] -= lr * np.outer(d, h)
        self.b2[midx] -= lr * d
        self.w -= lr * (F.T @ d)
        dpre = dh * (pre > 0)
        self.W1[act] -= lr * dpre
        self.b1 -= lr * dpre
        return -math.log(p[t] + 1e-12)

    def evaluate(self, samples):
        if not samples:
            return None
        top1 = top3 = rand = loss = 0
        for sample in samples:
            act, midx, F, t = sample[:4]
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


def migrate_params(p):
    """Profile v1 -> v2: widen W2T and b2 from 4096 to N_MOVES rows. Returns (params, migrated?).

    v1 scored every promotion with the queen-promotion unit. Each new under-promotion unit starts as a copy of that
    unit, so a migrated profile scores every move exactly as it did before; only further training can tell the
    promotion pieces apart."""
    rows = p["W2T"].shape[0]
    if rows == N_MOVES:
        return p, False
    if rows != N_BASE_MOVES:
        raise ValueError(f"unsupported profile: {rows} move units (expected {N_BASE_MOVES} or {N_MOVES})")
    src = underpromo_sources()
    out = dict(p)
    out["W2T"] = np.vstack([p["W2T"], p["W2T"][src]])
    out["b2"] = np.concatenate([p["b2"], p["b2"][src]])
    return out, True


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


def build_teacher(args):
    """A Maia teacher from --maia-weights (+ --lc0) and/or --maia-cache, or None when neither was given."""
    weights, cache = getattr(args, "maia_weights", None), getattr(args, "maia_cache", None)
    if not weights and not cache:
        return None
    import maia_teacher
    try:
        inner = None
        if weights:
            lc0 = getattr(args, "lc0", None) or shutil.which("lc0")
            if not lc0:
                sys.exit("lc0 not found. Install it or pass --lc0 /path/to/lc0 (Maia-1 weights run on lc0).")
            inner = maia_teacher.Lc0Teacher(lc0, weights)
        return maia_teacher.CachingTeacher(cache, inner) if cache else inner
    except maia_teacher.TeacherError as exc:
        sys.exit(f"Maia teacher: {exc}")


def open_teacher(args):
    """(teacher, mix) from the optional Maia flags of `learn`; (None, 0.0) when none are given.

    Maia is only a training-time helper: without these flags nothing here runs and maia_teacher is never imported."""
    mix = float(getattr(args, "maia_mix", 0.0) or 0.0)
    given = getattr(args, "maia_weights", None) or getattr(args, "maia_cache", None)
    if mix == 0 and not given:
        return None, 0.0
    if not 0 <= mix < 1:
        sys.exit("--maia-mix must be at least 0 and below 1 (0 = ignore Maia).")
    if mix == 0:
        sys.exit("Maia options were given but --maia-mix is 0, so Maia would have no effect. Try --maia-mix 0.25.")
    teacher = build_teacher(args)
    if teacher is None:
        sys.exit("--maia-mix needs a teacher: give --maia-weights (and lc0) or a --maia-cache file.")
    return teacher, mix


def learn(args):
    if not os.path.exists(args.pgn):
        sys.exit(f"PGN file not found: {args.pgn}")
    speeds = set(args.speed.split(",")) if args.speed else None
    games, skipped = read_games(args.pgn, args.player, speeds, args.min_opp_rating)
    if not games:
        sys.exit(f"No usable games found for player '{args.player}'. "
                 f"Check the name matches the PGN headers (and your filters).")
    print(f"Found {len(games)} games ({skipped} skipped by filters). Reading your positions...")
    teacher, mix = open_teacher(args)
    if teacher is not None:
        print(f"Maia teacher {getattr(teacher, 'name', '?')}: soft-target weight {mix} "
              f"(your own moves still carry weight {1 - mix:g}).")

    per_game, book = [], {}
    played_sum = np.zeros(len(FEATURES))
    base_sum = np.zeros(len(FEATURES))
    n_pos = 0
    tally = personality.new_tally()
    try:
        for game_no, (game, me) in enumerate(games):
            board, samples = game.board(), []
            for ply, move in enumerate(game.mainline_moves()):
                if board.turn == me:
                    moves = list(board.legal_moves)
                    if move in moves:
                        personality.observe(board, move, moves, tally, half=game_no % 2)
                        F = np.array([features(board, m) for m in moves])
                        t = moves.index(move)
                        sample = (encode(board), np.array([move_index(board, m) for m in moves]), F, t)
                        if teacher is not None:
                            sample += (teacher.probs(board, moves),)
                        samples.append(sample)
                        played_sum += F[t]
                        base_sum += F.mean(0)
                        n_pos += 1
                    if ply < BOOK_PLIES:
                        book.setdefault(book_key(board), Counter())[move.uci()] += 1
                board.push(move)
            per_game.append(samples)
    except Exception as exc:                       # a teacher failure is a clear message, not a traceback
        if teacher is None or exc.__class__.__name__ != "TeacherError":
            raise
        sys.exit(f"Maia teacher: {exc}")
    finally:
        if teacher is not None and hasattr(teacher, "close"):
            teacher.close()                        # also flushes the cache file

    if n_pos == 0:
        sys.exit(f"Found {len(games)} games for '{args.player}', but none contain a move by them to learn from.")

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
        tl = sum(net.step(s, lr, mix) for s in train) / max(1, len(train))
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
    dim_games = len(games)
    if old:
        old_tally, old_dim_games = personality.from_meta(old.get("dimensions"))
        if old_tally:
            tally, dim_games = personality.merge_tally(tally, old_tally), dim_games + old_dim_games
        else:
            print("Note: the old profile has no behavior measurements, so they cover only the games added now.")
        total += old["positions"]
        played_sum = played_sum + np.array([old["played_rate"][n] for n in FEATURES]) * old["positions"]
        base_sum = base_sum + np.array([old["baseline_rate"][n] for n in FEATURES]) * old["positions"]
        for key, counts in old["book"].items():
            for uci, c in counts.items():
                book.setdefault(key, Counter())[uci] += c

    meta = {
        "profile_version": PROFILE_VERSION,
        "player": args.player,
        "games": len(games) + (old["games"] if old else 0),
        "positions": total,
        "played_rate": dict(zip(FEATURES, (played_sum / total).tolist())),
        "baseline_rate": dict(zip(FEATURES, (base_sum / total).tolist())),
        "book": {k: dict(v) for k, v in book.items()},
        "validation": net.evaluate(val) or (old or {}).get("validation"),
        "dimensions": personality.to_meta(tally, dim_games),
    }
    if teacher is not None:
        # Provenance only: no tensor or feature changed, so the format version stays (see docs/FORMAT.md).
        meta["teacher"] = {"version": 1, "name": getattr(teacher, "name", None), "elo": getattr(teacher, "elo", None),
                           "mix": mix, "games": len(games)}
    elif old and old.get("teacher"):
        meta["teacher"] = old["teacher"]
    np.savez_compressed(args.out, meta=json.dumps(meta), **net.params())
    print(f"\nSaved style model to {args.out}")
    print_profile(meta)


def load_model(path):
    if not os.path.exists(path):
        sys.exit(f"Style file not found: {path}. Create one first with: python echo.py learn <games.pgn> --player <name>")
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    version = meta.get("profile_version", 1)          # profiles written before versioning have no field: v1
    if version > PROFILE_VERSION:
        sys.exit(f"{path} is profile version {version}, but this build reads up to {PROFILE_VERSION}. Update echostar.")
    params, migrated = migrate_params({k: d[k] for k in ("W1", "b1", "W2T", "b2", "w")})
    net = StyleNet(hidden=params["W1"].shape[1])
    net.load_params(params)
    meta["profile_version"] = PROFILE_VERSION
    if migrated:
        meta["migrated_from"] = version
        print(f"Note: {path} is a v{version} profile; migrated in memory to v{PROFILE_VERSION} "
              f"(promotion-aware). Scores are unchanged. The file is not modified: `learn --resume` or `export` "
              f"will write the new version.", file=sys.stderr)
    return net, meta


# ----------------------------------------------------------------------------
# Portable model format for the Rust port (see docs/FORMAT.md)
# ----------------------------------------------------------------------------
FORMAT_MAGIC = b"ECHOSTYL"
FORMAT_VERSION = 2


def export_echo(net, meta, path, version=FORMAT_VERSION):
    """Write weights + metadata in a dependency-free binary format: magic, version, JSON, f32 tensors.

    version=1 writes the legacy 4096-unit layout (under-promotion units dropped) for engine builds that predate
    promotion-aware profiles; current builds read both."""
    if version not in (1, FORMAT_VERSION):
        raise ValueError(f"cannot write format version {version}")
    moves = N_BASE_MOVES if version == 1 else N_MOVES
    header = dict(meta)
    header["dims"] = {"input": 768, "hidden": int(net.W1.shape[1]), "moves": moves, "features": len(FEATURES)}
    header["feature_names"] = FEATURES
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(FORMAT_MAGIC)
        fh.write(np.array([version, len(blob)], dtype="<u4").tobytes())
        fh.write(blob)
        tensors = {"W1": net.W1, "b1": net.b1, "W2T": net.W2T[:moves], "b2": net.b2[:moves], "w": net.w}
        for name in ("W1", "b1", "W2T", "b2", "w"):
            fh.write(np.ascontiguousarray(tensors[name], dtype="<f4").tobytes())


def export_cmd(args):
    net, meta = load_model(args.style)
    export_echo(net, meta, args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) for the Rust engine.")


def tendency_rows(meta):
    rows = [(meta["played_rate"][n] - meta["baseline_rate"][n], n) for n in FEATURES]
    return sorted(rows, key=lambda r: -abs(r[0]))


def dimension_rows(meta):
    """[(Dimension, summary)] for the behavior dimensions, or None when the profile has none."""
    tally, _ = personality.from_meta(meta.get("dimensions"))
    if tally is None:
        return None
    summary = personality.summarize(tally)
    return [(d, summary[d.name]) for d in personality.DIMENSIONS]


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
    rows = dimension_rows(meta)
    if rows is None:
        print("\n  Behavior: not measured (this profile predates it; `learn --resume` adds it for the new games).")
    else:
        games = meta["dimensions"]["games"]
        print(f"\n  Behavior, measured from {games} games against a random legal move "
              f"(z: 0 = random, |z| >= 3 = clear):")
        for dim, s in rows:
            print(f"    {dim.label:<27} you {s['observed_rate']:5.1%} vs random {s['expected_rate']:5.1%}  "
                  f"z {s['z']:+5.1f}  {s['verdict']}")
    reps = repertoire(meta)
    print(f"\n  Opening book: {len(meta['book'])} positions from your games")
    for label, text in reps:
        print(f"    {label}: {text}")
    v = meta.get("validation")
    if v:
        print("\n  Accuracy on games it never trained on:")
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
    rows = dimension_rows(meta)
    out += ["", "## Behavior", ""]
    if rows is None:
        out.append("Not measured: this profile predates behavior measurement.")
    else:
        out += ["Measured against a uniformly random legal move (z = 0 means no difference from random). "
                "Compare profiles with each other; do not read one profile as 'high' or 'low' on its own.", "",
                "| Dimension | What is measured | You | Random | z | Reading |", "|---|---|---|---|---|---|"]
        for dim, s in rows:
            out.append(f"| {dim.label} | {dim.description} | {s['observed_rate']:.1%} | {s['expected_rate']:.1%} "
                       f"| {s['z']:+.1f} | {s['verdict']} |")
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


def maia_compare(args):
    """How close are your moves to what Maia predicts a typical player would choose? (Optional; needs a teacher.)

    Reads your games only; trains nothing and writes nothing except the optional cache file."""
    if not os.path.exists(args.pgn):
        sys.exit(f"PGN file not found: {args.pgn}")
    teacher = build_teacher(args)
    if teacher is None:
        sys.exit("maia-compare needs a Maia teacher: give --maia-weights (and lc0) or a --maia-cache file.")
    speeds = set(args.speed.split(",")) if args.speed else None
    games, _ = read_games(args.pgn, args.player, speeds)
    if not games:
        sys.exit(f"No usable games found for player '{args.player}'.")
    brain = StyleBrain(args.style) if args.style else None
    import maia_teacher
    rows = []                                   # (maia prob of your move, your move rank, style rank, san, maia top san, p top)
    try:
        for game, me in games:
            board = game.board()
            for move in game.mainline_moves():
                if board.turn == me:
                    moves = list(board.legal_moves)
                    if move in moves and len(moves) > 1:
                        q = teacher.probs(board, moves)
                        t = moves.index(move)
                        top = int(np.argmax(q))
                        srank = None
                        if brain is not None:
                            order = np.argsort(-np.array(brain.scores(board, moves)))
                            srank = int(np.where(order == t)[0][0])
                            style_top_is_maia = int(order[0]) == top
                        else:
                            style_top_is_maia = None
                        rows.append((float(q[t]), int((q > q[t]).sum()), srank, style_top_is_maia,
                                     board.san(move), board.san(moves[top]), float(q[top]), board.fen()))
                board.push(move)
    except maia_teacher.TeacherError as exc:
        sys.exit(f"Maia teacher: {exc}")
    finally:
        teacher.close()
    if not rows:
        sys.exit("No positions to compare.")
    n = len(rows)
    top1 = sum(r[1] == 0 for r in rows) / n
    top3 = sum(r[1] < 3 for r in rows) / n
    nll = -sum(math.log(max(r[0], 1e-4)) for r in rows) / n
    print(f"\nMaia comparison for {args.player}: {len(games)} games, {n} positions "
          f"(teacher {getattr(teacher, 'name', '?')})")
    print(f"  Maia's top move is the one you played: {top1:.1%}   (in its top 3: {top3:.1%})")
    print(f"  Average probability Maia gave your move: {sum(r[0] for r in rows) / n:.1%}   "
          f"(log-loss {nll:.2f})")
    if brain is not None:
        s1 = sum(r[2] == 0 for r in rows) / n
        agree = sum(bool(r[3]) for r in rows) / n
        print(f"  Your style model's top move is the one you played: {s1:.1%} "
              f"(this is on all your games, so it includes training positions)")
        print(f"  Your style model and Maia pick the same top move: {agree:.1%}")
    print("\n  Moves you played that Maia found least typical (your move | Maia's favorite):")
    for r in sorted(rows, key=lambda r: r[0])[:max(0, args.examples)]:
        print(f"    {r[4]:<8} Maia gave it {r[0]:5.1%} | {r[5]} {r[6]:5.1%}   {r[7]}")
    print("\n  Reading this: Maia models typical players at one rating band. A low match can mean an unusual style, "
          "a rating far from the model's, or just a game full of forcing moves; it is not a verdict on your play. "
          "It is also measured on your training games, not held out.")


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
            if board.is_checkmate():
                return MATE              # nothing to search, and the Rust port does the same
            info = self.engine.analyse(board, limit)
            return info["score"].pov(mover).score(mate_score=MATE)
        finally:
            board.pop()

    def choose(self, board, movetime=None):
        """Pick a move. `movetime` (seconds) switches from fixed depth to a time budget."""
        cands = self.candidates(board, movetime)
        if not cands:
            # The engine gave nothing usable: play any legal move rather than crash (same as the Rust port).
            self.history.append((False, 0))
            return next(iter(board.legal_moves))
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
        if self.verbose:
            print(f"   [coach] {len(safe)}/{len(cands)} moves were safe | "
                  f"engine best {board.san(best_move)} ({best_cp / 100:+.2f}) | "
                  f"Echo falls back to the engine's best move")
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


def parse_move(board, text):
    """Read a typed move: SAN, UCI, or a forgiving variant. Returns a legal Move or None.

    Accepts a lowercase piece letter (bf3, nxe5), 0-0 / o-o for castling, and e2-e4 style input.
    The lowercase fix is only tried after the literal text failed, so a real pawn move always wins.
    """
    text = text.strip()
    if not text:
        return None
    castle = text.lower().replace("0", "o")
    if castle in ("o-o", "o-o-o"):
        text = castle.upper()
    candidates = [text, text.replace("-", "")]
    if text[0] in "nbrqk":
        candidates.append(text[0].upper() + text[1:])
    for cand in candidates:
        try:
            return board.parse_san(cand)
        except ValueError:
            pass
        try:
            move = chess.Move.from_uci(cand)
        except ValueError:
            continue
        if move in board.legal_moves:
            return move
    return None


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
                try:
                    text = input("Your move: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()              # Ctrl-D / Ctrl-C: stop cleanly, the game is still saved below
                    break
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
                if text == "undo":
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        last = board.peek() if board.move_stack else None
                    else:
                        print("Nothing to undo yet.\n")
                    continue
                move = parse_move(board, text)
                if move is None:
                    print("Illegal or unreadable move, try again. (Examples: e4, Nf3, O-O, e8=Q, e2e4)\n")
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
        try:
            engine.quit()
        except chess.engine.EngineError:
            pass                         # the engine already died; don't hide the real error behind this one


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
        except OSError as exc:           # timeouts and dropped connections while reading the body
            sys.exit(f"Network error: {exc}")


def fetch_lichess(user, max_games, speeds=None, rated=False, base=LICHESS_URL):
    query = f"max={max_games}&tags=true&clocks=false&evals=false&opening=false"
    if speeds:
        names = {"bullet": "bullet", "blitz": "blitz", "rapid": "rapid", "classical": "classical"}
        query += "&perfType=" + ",".join(names[s] for s in speeds)
    if rated:
        query += "&rated=true"
    return http_get(f"{base}/api/games/user/{quote(user, safe='')}?{query}", accept="application/x-chess-pgn")


def split_games(pgn_text):
    """Split a PGN string into one string per game."""
    import re
    parts = re.split(r"(?m)^(?=\[Event )", pgn_text)
    return [p.strip() for p in parts if p.strip().startswith("[Event ")]


def game_speed(game_text):
    import re
    m = re.search(r'\[TimeControl "([^"]*)"\]', game_text)
    return classify_speed(m.group(1)) if m else None


def fetch_chesscom(user, max_games, speeds=None, base=CHESSCOM_URL):
    """Newest `max_games` games, walking Chess.com's monthly archives from the latest month backwards."""
    import json as _json
    archives = _json.loads(http_get(f"{base}/pub/player/{quote(user, safe='')}/games/archives",
                                    "application/json")).get("archives", [])
    games = []
    for url in reversed(archives):                       # newest month first
        month = split_games(http_get(url + "/pgn", "application/x-chess-pgn"))
        month.reverse()                                  # each month's file lists oldest game first
        if speeds:
            month = [g for g in month if game_speed(g) in speeds]
        games.extend(month)
        if len(games) >= max_games:
            break
    return "\n\n".join(games[:max_games]) + "\n"


def fetch_cmd(args):
    speeds = [x for x in args.speed.split(",")] if args.speed else None
    for sp in speeds or []:
        if sp not in ("bullet", "blitz", "rapid", "classical"):
            sys.exit(f"Unknown speed '{sp}'. Use bullet, blitz, rapid or classical.")
    if args.site == "lichess":
        text = fetch_lichess(args.user, args.max, speeds, args.rated)
    else:
        text = fetch_chesscom(args.user, args.max, set(speeds) if speeds else None)
    count = text.count("[Event ")
    if count == 0:
        sys.exit("No games came back. Check the username, and loosen --speed / --rated if you used them.")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Saved {count} games to {args.out}")
    print(f"Next: python echo.py learn {args.out} --player {args.user}")


def main():
    p = argparse.ArgumentParser(description="Echo: a chess engine that plays your style, correctly.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ln = sub.add_parser("learn", help="train a style model from your PGN games")
    ln.add_argument("pgn")
    ln.add_argument("--player", required=True, help="your name exactly as in the PGN headers")
    ln.add_argument("--out", default="style.npz")
    ln.add_argument("--epochs", type=int, default=12)
    ln.add_argument("--lr", type=float, default=0.02)
    ln.add_argument("--speed", help="only use these speeds, comma separated: bullet,blitz,rapid,classical")
    ln.add_argument("--min-opp-rating", type=int, help="only use games against opponents rated at least this")
    ln.add_argument("--resume", help="continue training an existing style.npz with new games")
    ln.add_argument("--maia-mix", type=float, default=0.0, metavar="W",
                   help="optional: blend a Maia teacher into the training target, 0 (default, off) to <1; "
                        "needs --maia-weights with lc0, or --maia-cache")
    ln.add_argument("--maia-weights", metavar="FILE", help="Maia-1 weights (maia-1500.pb.gz); run through lc0")
    ln.add_argument("--lc0", metavar="PATH", help="path to the lc0 binary (default: lc0 on PATH)")
    ln.add_argument("--maia-cache", metavar="FILE",
                   help="JSON-lines cache of Maia probabilities; filled when lc0 is given, replayed when it is not")
    ln.set_defaults(fn=learn)

    mc = sub.add_parser("maia-compare", help="compare your moves with what a Maia model predicts (needs lc0 + weights or a cache)")
    mc.add_argument("pgn")
    mc.add_argument("--player", required=True, help="your name exactly as in the PGN headers")
    mc.add_argument("--maia-weights", metavar="FILE")
    mc.add_argument("--lc0", metavar="PATH")
    mc.add_argument("--maia-cache", metavar="FILE")
    mc.add_argument("--style", help="also compare your style.npz with Maia")
    mc.add_argument("--speed", help="only these speeds, comma separated")
    mc.add_argument("--examples", type=int, default=8, help="how many of your most Maia-atypical moves to list")
    mc.set_defaults(fn=maia_compare)

    f = sub.add_parser("fetch", help="download your games from Lichess or Chess.com")
    f.add_argument("user", help="your username on that site")
    f.add_argument("--site", choices=["lichess", "chesscom"], default="lichess")
    f.add_argument("--max", type=int, default=500, help="how many games (newest first), default 500")
    f.add_argument("--speed", help="only these speeds: bullet,blitz,rapid,classical (comma separated)")
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
