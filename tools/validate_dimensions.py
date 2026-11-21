#!/usr/bin/env python3
"""Do the behavior dimensions measure what they claim to?

Real players have no ground truth, so this uses synthetic players whose preference is known because we planted it: a
planted player picks each move with probability proportional to exp(beta * signal(move)) for one dimension, and
uniformly otherwise. Four checks:

  1. Null calibration: players who choose uniformly at random must score z ~ N(0, 1), and rarely get a verdict.
  2. Direction: beta > 0 must give z > 0 on the planted dimension, beta < 0 must give z < 0.
  3. Dose response and reliability: across players with different beta, the measured effect must rise with beta
     (Spearman) and agree between two halves of the same player's games (split-half correlation).
  4. Cross-talk: how much does planting one dimension move the others? (This shows which dimensions overlap.)

Usage: python tools/validate_dimensions.py [--quick] [--seed N] [--json out.json]
"""
import argparse
import json
import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import personality  # noqa: E402

# Endgame starting positions (non-pawn material <= 24), so the endgame dimension gets data.
ENDGAME_FENS = [
    "8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 1",
    "r3k3/pp3ppp/8/8/8/8/PP3PPP/4K2R w - - 0 1",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
    "2r3k1/5ppp/8/8/8/8/5PPP/2R3K1 b - - 0 1",
    "3qk3/5ppp/8/8/8/8/5PPP/3QK3 w - - 0 1",
    "4k3/2b2ppp/8/8/8/8/2N2PPP/4K3 w - - 0 1",
    "8/2p2k2/8/3P4/8/5K2/8/4B3 w - - 0 1",
    "6k1/5p2/8/8/8/2N5/5PP1/6K1 w - - 0 1",
    "4r1k1/5ppp/8/8/8/8/PPP2PPP/4R1K1 w - - 0 1",
    "8/pp3k2/2p5/8/8/2P2K2/PP6/8 w - - 0 1",
]


def play_games(rng, n_games, plies, dim=None, beta=0.0):
    """Games where the player's moves follow the planted preference (uniform if dim is None) and the opponent plays
    uniformly at random. Even games start from the standard position, odd ones from an endgame. Returns a tally."""
    tally = personality.new_tally()
    for g in range(n_games):
        board = chess.Board() if g % 2 == 0 else chess.Board(ENDGAME_FENS[int(rng.integers(len(ENDGAME_FENS)))])
        pair = g // 2                     # games 2k and 2k+1 (standard, endgame) share a colour and a half
        me = chess.WHITE if pair % 2 == 0 else chess.BLACK
        half = (pair // 2) % 2
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            if board.turn == me:
                if dim is None or beta == 0.0:
                    move = moves[int(rng.integers(len(moves)))]
                else:
                    w = np.array([np.exp(beta * personality.signal_of(dim, board, m))
                                  if personality.in_scope(dim, board, m) else 1.0 for m in moves])
                    move = moves[int(rng.choice(len(moves), p=w / w.sum()))]
                personality.observe(board, move, moves, tally, half=half)
            else:
                move = moves[int(rng.integers(len(moves)))]
            board.push(move)
            if board.is_game_over():
                break
    return tally


def effect(summary_row):
    return summary_row["observed_rate"] - summary_row["expected_rate"]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_null(n_players, n_games, plies, seed):
    rng = np.random.default_rng(seed)
    z = {n: [] for n in personality.NAMES}
    flagged = {n: {"weak_or_clear": 0, "clear": 0, "eligible": 0} for n in personality.NAMES}
    any_clear = 0
    for _ in range(n_players):
        s = personality.summarize(play_games(rng, n_games, plies))
        hit = False
        for n in personality.NAMES:
            if s[n]["verdict"] == "too little data":
                continue
            z[n].append(s[n]["z"])
            flagged[n]["eligible"] += 1
            flagged[n]["weak_or_clear"] += s[n]["verdict"].startswith(("weak", "clear"))
            flagged[n]["clear"] += s[n]["verdict"].startswith("clear")
            hit |= s[n]["verdict"].startswith("clear")
        any_clear += hit
    return {"z": z, "flagged": flagged, "players": n_players, "players_with_any_clear": any_clear}


def run_planted(n_betas, replicates, n_games, plies, seed, beta_max=2.0):
    rng = np.random.default_rng(seed)
    betas = np.linspace(-beta_max, beta_max, n_betas)
    out = {}
    for dim in personality.NAMES:
        rows = []
        for beta in betas:
            for _ in range(replicates):
                tally = play_games(rng, n_games, plies, dim, float(beta))
                s = personality.summarize(tally)
                halves = []
                for h in (0, 1):
                    n, o, e, _v = tally[dim][h]
                    halves.append((o - e) / n if n else 0.0)
                rows.append({"beta": float(beta), "summary": s, "halves": halves})
        out[dim] = rows
    return out


def report(null, planted, out=sys.stdout):
    def p(*a):
        print(*a, file=out)

    names = personality.NAMES

    p("## 1. Null calibration (players who choose uniformly at random)\n")
    p(f"{null['players']} random players. A calibrated z has mean 0, sd 1, |z|>=2 about 4.6%, |z|>=3 about 0.3%.\n")
    p("| Dimension | players | mean z | sd z | share with a weak or clear verdict | share with a clear verdict |")
    p("|---|---|---|---|---|---|")
    for n in names:
        z = np.array(null["z"][n])
        f = null["flagged"][n]
        if len(z) == 0:
            p(f"| {n} | 0 | - | - | - | - |")
            continue
        p(f"| {n} | {len(z)} | {z.mean():+.2f} | {z.std():.2f} | {f['weak_or_clear'] / f['eligible']:.1%} "
          f"| {f['clear'] / f['eligible']:.1%} |")
    p(f"\nRandom players given at least one 'clear' verdict on any dimension: "
      f"{null['players_with_any_clear']} of {null['players']}.\n")

    p("## 2. Direction and dose response (planted preference of known strength)\n")
    p("beta > 0 = prefers the behavior, beta < 0 = avoids it. `z(+2)` / `z(-2)` are the mean z of the planted "
      "dimension at the extreme betas. Spearman is between beta and the measured effect over all planted players; "
      "split-half is the correlation between the effect measured on two halves of each player's games.\n")
    p("| Dimension | z at beta=+2 | z at beta=-2 | Spearman(beta, effect) | split-half r |")
    p("|---|---|---|---|---|")
    for dim in names:
        rows = planted[dim]
        betas = np.array([r["beta"] for r in rows])
        eff = np.array([effect(r["summary"][dim]) for r in rows])
        zs = np.array([r["summary"][dim]["z"] for r in rows])
        ha = np.array([r["halves"][0] for r in rows])
        hb = np.array([r["halves"][1] for r in rows])
        top, bottom = zs[betas == betas.max()].mean(), zs[betas == betas.min()].mean()
        half_r = float(np.corrcoef(ha, hb)[0, 1]) if ha.std() > 0 and hb.std() > 0 else float("nan")
        p(f"| {dim} | {top:+.1f} | {bottom:+.1f} | {spearman(betas, eff):+.2f} | {half_r:+.2f} |")

    p("\n## 3. Cross-talk (mean z of each dimension, when one dimension is planted at beta=+2)\n")
    p("| planted \\ measured | " + " | ".join(names) + " |")
    p("|---|" + "---|" * len(names))
    for dim in names:
        top = [r for r in planted[dim] if r["beta"] == max(x["beta"] for x in planted[dim])]
        cells = [np.mean([r["summary"][m]["z"] for r in top]) for m in names]
        p(f"| **{dim}** | " + " | ".join(f"{'**' if m == dim else ''}{c:+.1f}{'**' if m == dim else ''}"
                                         for m, c in zip(names, cells)) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.quick:
        null = run_null(20, 8, 40, a.seed)
        planted = run_planted(3, 2, 8, 40, a.seed + 1)
    else:
        null = run_null(100, 12, 40, a.seed)
        planted = run_planted(9, 2, 14, 40, a.seed + 1)
    report(null, planted)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"null": null, "planted": planted}, fh)


if __name__ == "__main__":
    main()
