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
import random
import shutil
import sys
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
    def __init__(self, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.05, (768, HIDDEN))
        self.b1 = np.zeros(HIDDEN)
        self.W2T = rng.normal(0, 0.02, (4096, HIDDEN))   # one row per from-to move
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
def read_games(pgn_path, player):
    name = player.lower()
    games = []
    with open(pgn_path, encoding="utf-8", errors="ignore") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            white, black = game.headers.get("White", "").lower(), game.headers.get("Black", "").lower()
            if name == white:
                games.append((game, chess.WHITE))
            elif name == black:
                games.append((game, chess.BLACK))
    return games


def learn(args):
    games = read_games(args.pgn, args.player)
    if not games:
        sys.exit(f"No games found for player '{args.player}'. Check the name matches the PGN headers.")
    print(f"Found {len(games)} games. Reading your positions...")

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
    if len(train) < 2000:
        print("Note: under ~2000 positions (roughly 60+ games) the net will mostly memorize. More games help.")

    net = StyleNet()
    best, best_params = float("inf"), net.params()
    for epoch in range(args.epochs):
        random.shuffle(train)
        lr = args.lr / (1 + 0.15 * epoch)
        tl = sum(net.step(s, lr) for s in train) / len(train)
        ev = net.evaluate(val)
        score = ev["loss"] if ev else tl
        msg = f"  epoch {epoch + 1:>2}/{args.epochs}  train loss {tl:.3f}"
        if ev:
            msg += f"  | held-out loss {ev['loss']:.3f}  top-1 {ev['top1']:.1%}"
        print(msg)
        if score < best:
            best, best_params = score, net.params()
    net.load_params(best_params)

    final = net.evaluate(val)
    meta = {
        "player": args.player, "games": len(games), "positions": n_pos,
        "played_rate": dict(zip(FEATURES, (played_sum / n_pos).tolist())),
        "baseline_rate": dict(zip(FEATURES, (base_sum / n_pos).tolist())),
        "book": {k: dict(v) for k, v in book.items()},
        "validation": final,
    }
    np.savez_compressed(args.out, meta=json.dumps(meta), **net.params())
    print(f"\nSaved style model to {args.out}")
    print_profile(meta)


def load_model(path):
    d = np.load(path, allow_pickle=False)
    net = StyleNet()
    net.load_params({k: d[k] for k in ("W1", "b1", "W2T", "b2", "w")})
    return net, json.loads(str(d["meta"]))


def print_profile(meta):
    print(f"\nStyle profile for {meta['player']} ({meta['games']} games, {meta['positions']} positions)")
    print("Tendency = how much more (or less) you pick this kind of move than a random legal move.\n")
    rows = [(meta["played_rate"][n] - meta["baseline_rate"][n], n) for n in FEATURES]
    rows.sort(key=lambda r: -abs(r[0]))
    for diff, n in rows[:10]:
        print(f"  {'+' if diff > 0 else '-'} {n:<24} you {meta['played_rate'][n]:5.1%}  "
              f"vs random {meta['baseline_rate'][n]:5.1%}")
    print(f"\n  Opening book: {len(meta['book'])} positions from your games")
    v = meta.get("validation")
    if v:
        print(f"\n  Accuracy on games it never trained on:")
        print(f"    predicts your exact move {v['top1']:.1%} of the time (random guessing: {v['random_top1']:.1%})")
        print(f"    your move is in its top 3 {v['top3']:.1%} of the time")
    else:
        print("\n  Accuracy: not measured (need 10+ games to hold some out).")


def profile(args):
    print_profile(load_model(args.style)[1])


class StyleBrain:
    def __init__(self, path=None):
        self.net, self.book = None, {}
        if path:
            self.net, meta = load_model(path)
            self.book = meta.get("book", {})

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
MATE = 100000


class Echo:
    def __init__(self, engine, brain, dial="balanced", depth=14, multipv=8, verbose=False):
        self.engine, self.brain = engine, brain
        self.dial, self.depth, self.multipv, self.verbose = DIALS[dial], depth, multipv, verbose

    def candidates(self, board):
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=self.multipv)
        return [(i["pv"][0], i["score"].pov(board.turn).score(mate_score=MATE)) for i in infos if "pv" in i]

    def verify(self, board, move):
        """Blunder gate: deeper search after the move, from the mover's point of view."""
        mover = board.turn
        board.push(move)
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth + 6))
            return info["score"].pov(mover).score(mate_score=MATE)
        finally:
            board.pop()

    def choose(self, board):
        cands = self.candidates(board)
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

        for _, move, cp in ranked:
            if move == best_move or self.verify(board, move) >= best_cp - band - GATE_SLACK:
                if self.verbose:
                    print(f"   [coach] {len(safe)}/{len(cands)} moves were safe | "
                          f"engine best {board.san(best_move)} ({best_cp / 100:+.2f}) | "
                          f"Echo plays {board.san(move)} ({cp / 100:+.2f})")
                return move
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


def play(args):
    brain = StyleBrain(args.style)
    if not args.style:
        print("No style file given: Echo will play as a plain engine.")
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish(args.stockfish))
    echo = Echo(engine, brain, args.dial, args.depth, verbose=args.show)
    human = chess.WHITE if args.color == "white" else chess.BLACK
    board, last = chess.Board(), None
    print(f"Echo ({args.dial}) vs you. Moves in SAN (e4, Nf3) or UCI (e2e4). 'undo' takes back a move, 'quit' stops.\n")

    try:
        while not board.is_game_over():
            print(render(board, human, last, color=not args.plain), "\n")
            if board.turn == human:
                text = input("Your move: ").strip()
                if text in ("quit", "q"):
                    break
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
                move = echo.choose(board)
                print(f"Echo plays {board.san(move)}\n")
                board.push(move)
                last = move
        print(render(board, human, last, color=not args.plain), "\nResult:", board.result())
    finally:
        engine.quit()


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Echo: a chess engine that plays your style, correctly.")
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("learn", help="train a style model from your PGN games")
    l.add_argument("pgn")
    l.add_argument("--player", required=True, help="your name exactly as in the PGN headers")
    l.add_argument("--out", default="style.npz")
    l.add_argument("--epochs", type=int, default=12)
    l.add_argument("--lr", type=float, default=0.02)
    l.set_defaults(fn=learn)

    pr = sub.add_parser("profile", help="show your style profile and accuracy")
    pr.add_argument("style")
    pr.set_defaults(fn=profile)

    g = sub.add_parser("play", help="play a game against Echo")
    g.add_argument("--style", help="style.npz from the learn step")
    g.add_argument("--dial", choices=list(DIALS), default="balanced")
    g.add_argument("--color", choices=["white", "black"], default="white", help="your color")
    g.add_argument("--depth", type=int, default=14)
    g.add_argument("--stockfish", help="path to the Stockfish binary")
    g.add_argument("--show", action="store_true", help="coach view: engine best vs Echo's pick")
    g.add_argument("--plain", action="store_true", help="plain ASCII board (no colors/unicode)")
    g.set_defaults(fn=play)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
