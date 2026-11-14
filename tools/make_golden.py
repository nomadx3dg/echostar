#!/usr/bin/env python3
"""Generate golden test vectors so the Rust engine can be checked against the Python reference.

Writes rust/tests/data/golden.echo (a tiny random model, hidden=8) and golden.json (positions with the
features, input units, move indices, logits and style scores that Python computes for every legal move).
"""
import json
import math
import os
import random
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import echo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rust", "tests", "data")

SPECIAL = [
    chess.STARTING_FEN,
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",                      # both castles
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",                      # same, black to move
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",           # en passant available
    "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",                                       # promotions both sides
    "4k3/P6P/8/8/8/8/p6p/4K3 b - - 0 1",
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",           # white in check
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",     # early queen, scholar's setup
]


def positions(n_random=24, seed=7):
    rng = random.Random(seed)
    fens = list(SPECIAL)
    while len(fens) < len(SPECIAL) + n_random:
        board = chess.Board()
        for _ in range(rng.randint(2, 70)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            fens.append(board.fen())
    return fens


def main():
    rng = np.random.default_rng(3)
    net = echo.StyleNet(rng=rng, hidden=8)
    net.W1 = rng.normal(0, 0.6, net.W1.shape)
    net.b1 = rng.normal(0, 0.3, net.b1.shape)
    net.W2T = rng.normal(0, 0.6, net.W2T.shape)
    net.b2 = rng.normal(0, 0.2, net.b2.shape)
    net.w = rng.normal(0, 0.8, net.w.shape)
    # round-trip through float32 so Python and Rust use identical weights
    for name in ("W1", "b1", "W2T", "b2", "w"):
        setattr(net, name, getattr(net, name).astype("<f4").astype(float))

    fens = positions()
    book = {}
    for fen in fens:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if moves:
            book[echo.book_key(board)] = {m.uci(): rng.integers(1, 6).item() for m in moves[:3]}
    meta = {"player": "Golden", "games": 1, "positions": 1, "played_rate": {}, "baseline_rate": {},
            "book": book, "validation": None}

    os.makedirs(OUT, exist_ok=True)
    echo.export_echo(net, meta, os.path.join(OUT, "golden.echo"))

    brain = echo.StyleBrain()
    brain.net, brain.book = net, book
    cases = []
    for fen in fens:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if not moves:
            continue
        cases.append({
            "fen": fen,
            "book_key": echo.book_key(board),
            "active": sorted(int(i) for i in echo.encode(board)),
            "moves": [m.uci() for m in moves],
            "san": [board.san(m) for m in moves],
            "features": [echo.features(board, m).tolist() for m in moves],
            "move_index": [echo.move_index(board, m) for m in moves],
            "logits": net.logits(board, moves).tolist(),
            "scores": brain.scores(board, moves),
        })
    with open(os.path.join(OUT, "golden.json"), "w") as fh:
        json.dump({"feature_names": echo.FEATURES, "cases": cases}, fh)
    print(f"{len(cases)} positions, {sum(len(c['moves']) for c in cases)} moves -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
