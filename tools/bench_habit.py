#!/usr/bin/env python3
"""
Does each system reproduce the planted habit of a synthetic player (tools/synth_games.py)?

Exact move agreement is a blunt personalization metric: when several moves are about equally good the player's pick
among them is partly luck. The planted habit (queenside / light / rim) is a clean, known property of the player, so the
question "how often does the system's move have the habit?" measures personalization directly. This only works on
synthetic data, where the habit is known by construction; there is no equivalent for a real person.

For the player's held-out test positions (split 0 of tools/bench_maia.py) it reports the habit rate of:
  the player's own moves, a random safe move (the no-style null: any move inside the same safety band),
  Stockfish's best move, and the real chooser (band + blunder gate, temperature 0) driven by each brain.

    python tools/bench_habit.py --pgn synth/rim.pgn --player SynthRim --habit rim --lc0 ... --maia-weights ... --maia-cache ...
"""
import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess  # noqa: E402
import chess.engine  # noqa: E402
import numpy as np  # noqa: E402

import bench_maia as bm  # noqa: E402
import echo  # noqa: E402
import maia_teacher as mt  # noqa: E402
from synth_games import SIGNALS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--player", required=True)
    ap.add_argument("--habit", required=True, choices=list(SIGNALS))
    ap.add_argument("--lc0")
    ap.add_argument("--maia-weights")
    ap.add_argument("--maia-cache")
    ap.add_argument("--stockfish")
    ap.add_argument("--mix", type=float, default=0.25)
    ap.add_argument("--mu", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--band", type=int, default=50)
    ap.add_argument("--splits", type=int, default=2)
    args = ap.parse_args()
    has_habit = SIGNALS[args.habit]

    per_game = bm.load_positions(args.pgn, args.player)
    teacher = echo.build_teacher(argparse.Namespace(maia_weights=args.maia_weights, lc0=args.lc0, maia_cache=args.maia_cache))
    for g in per_game:
        for p in g:
            p.q = teacher.probs(p.board, p.moves)
    sf = args.stockfish or shutil.which("stockfish") or "/usr/games/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({"Threads": 1, "Hash": 32})
    eng = bm.TimedEngine(engine)

    rows = {}                                      # system -> [habit rates per split]
    chance = []
    for seed in range(args.splits):
        tr, va, te = bm.split_games(len(per_game), seed)
        train, val, test = bm.flat(per_game, tr), bm.flat(per_game, va), bm.flat(per_game, te)
        net0 = bm.fit(train, val, 0.0, args.epochs, args.lr, seed)
        netM = bm.fit(train, val, args.mix, args.epochs, args.lr, seed)
        brains = {"echostar (existing)": bm.NetBrain(net0),
                  f"echostar+maia training teacher (mix {args.mix:g})": bm.NetBrain(netM),
                  f"echostar~maia runtime blend (mu {args.mu:g})": mt.MaiaBlendBrain(bm.NetBrain(net0), teacher, args.mu)}
        cands = [bm.candidates(eng, p.board, args.depth) for p in test]
        rng = random.Random(seed)

        def rate(moves, test=test):
            return float(np.mean([has_habit(p.board, m) for p, m in zip(test, moves)]))

        rows.setdefault("player's own moves", []).append(rate([p.move for p in test]))
        rows.setdefault("stockfish best move", []).append(rate([c[0][0] for c in cands]))
        safe_pick, safe_rate = [], []
        for p, c in zip(test, cands):
            best = c[0][1]
            safe = [m for m, cp in c if cp >= best - args.band]
            safe_pick.append(rng.choice(safe))
            safe_rate.append(np.mean([has_habit(p.board, m) for m in safe]))
        rows.setdefault(f"random safe move (band {args.band}, no style)", []).append(rate(safe_pick))
        chance.append(float(np.mean(safe_rate)))
        for name, brain in brains.items():
            e = echo.Echo(eng, brain, {"band": args.band, "temp": 0.0}, depth=args.depth)
            rows.setdefault(f"chooser + {name}", []).append(rate([e.choose(p.board.copy()) for p in test]))
            raw = [p.moves[int(np.argmax(brain.scores(p.board, p.moves)))] for p in test]
            rows.setdefault(f"raw argmax, {name}", []).append(rate(raw))
        print(f"split {seed}: {len(test)} test positions", flush=True)
    engine.quit()
    teacher.close()

    print(f"\nHabit '{args.habit}' for {args.player} (SYNTHETIC; habit planted at beta 2.0). Test positions only, "
          f"{args.splits} splits.\n")
    print("| system | share of moves with the habit |")
    print("|---|---|")
    for name, v in rows.items():
        print(f"| {name} | {100 * np.mean(v):.1f}% (splits: {', '.join(f'{100 * x:.1f}' for x in v)}) |")
    print(f"\nChance for the chooser: the average share of safe candidates that have the habit is {100 * np.mean(chance):.1f}%. "
          "A system with no style knowledge should sit near that line; a system that learned the habit sits above it, "
          "toward the player's own rate.")


if __name__ == "__main__":
    main()
