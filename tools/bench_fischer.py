#!/usr/bin/env python3
"""Cross-validated accuracy of a style model on one player's real games.

echo.py's own held-out line takes the *last* 10% of games and counts opening moves, which favours players who repeat
their openings. This tool does k-fold cross-validation by game (seeded shuffle) and reports the opening (ply <
BOOK_PLIES) and the rest of the game separately, scoring moves the way the chooser does (net logit + book bonus built
from the training folds only). Needs no Stockfish.

    python tools/bench_fischer.py games.pgn --player "Fischer, Robert James" --folds 5
"""
import argparse
import math
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import echo  # noqa: E402


def build(pgn, player, cache):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    games, _ = echo.read_games(pgn, player)
    out = []
    for game, me in games:
        board, rows = game.board(), []
        for ply, move in enumerate(game.mainline_moves()):
            if board.turn == me:
                moves = list(board.legal_moves)
                if move in moves and len(moves) > 1:
                    F = np.array([echo.features(board, m) for m in moves])
                    sample = (echo.encode(board), np.array([echo.move_index(board, m) for m in moves]), F,
                              moves.index(move))
                    rows.append((ply, echo.book_key(board), [m.uci() for m in moves], sample))
            board.push(move)
        out.append(rows)
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(out, f)
    return out


def scores(net, book, bonus_w, key, ucis, sample):
    act, midx, F, _ = sample
    lg = net.W2T[midx] @ np.maximum(net.W1[act].sum(0) + net.b1, 0) + net.b2[midx] + F @ net.w
    counts = book.get(key, {})
    return lg + np.array([bonus_w * math.log1p(counts.get(u, 0)) for u in ucis])


def run(games, folds, seed, epochs, lr, bonus_w, book_plies, train_fn=None):
    order = list(range(len(games)))
    random.Random(seed).shuffle(order)
    acc = {"open": [0, 0, 0.0, 0], "rest": [0, 0, 0.0, 0]}
    for k in range(folds):
        test_ids = set(order[k::folds])
        train = [s for i in order if i not in test_ids for s in games[i]]
        book = {}
        for r in train:
            if r[0] < book_plies:
                book.setdefault(r[1], Counter())[r[2][r[3][3]]] += 1
        net = echo.StyleNet()
        samples = [r[3] for r in train]
        rng = random.Random(seed + k)
        (train_fn or default_train)(net, samples, epochs, lr, rng, [i for i in order if i not in test_ids])
        for i in test_ids:
            for ply, key, ucis, sample in games[i]:
                sc = scores(net, book, bonus_w, key, ucis, sample)
                t = sample[3]
                order_ = np.argsort(-sc)
                p = echo.softmax(sc)
                a = acc["open" if ply < echo.BOOK_PLIES else "rest"]
                a[0] += order_[0] == t
                a[1] += t in order_[:3]
                a[2] -= math.log(p[t] + 1e-12)
                a[3] += 1
    return {name: {"n": a[3], "top1": a[0] / a[3], "top3": a[1] / a[3], "loss": a[2] / a[3]} for name, a in acc.items()}


def default_train(net, samples, epochs, lr, rng, train_ids=None):
    for e in range(epochs):
        rng.shuffle(samples)
        for s in samples:
            net.step(s, lr / (1 + 0.15 * e))


def report(name, res):
    print(f"{name:34s}", "  ".join(f"{k}: top1 {v['top1']:.1%} top3 {v['top3']:.1%} loss {v['loss']:.3f} (n={v['n']})"
                                   for k, v in res.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--player", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--cache", help="pickle file for the extracted positions (speeds up repeated runs)")
    a = ap.parse_args()
    games = build(a.pgn, a.player, a.cache)
    print(f"{len(games)} games, {sum(len(g) for g in games)} positions, {a.folds}-fold CV by game")
    report("current defaults", run(games, a.folds, a.seed, a.epochs, a.lr, 1.5, echo.BOOK_PLIES))


if __name__ == "__main__":
    main()
