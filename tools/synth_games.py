#!/usr/bin/env python3
"""
Synthetic "competent but quirky" players, for testing the pipeline. NOT real human behavior.

Each move, Stockfish proposes its top candidates. The player picks one with probability proportional to
exp(cp / tau + beta * signal): a noisy but tactically sensible mover (tau sets how sloppy) with one planted preference
(beta sets how strong). The preferences are chosen so they are NOT among EchoStar's 21 hand-made features:

  queenside : the move ends on files a-c
  light     : the move ends on a light square
  rim       : the move ends on the board edge (file a/h or rank 1/8)

This is more realistic than uniformly random games (those have no sense at all, so a human-move model is
meaningless on them) but it is still a machine's idea of a quirky player: the planted habit is a clean function of the
move, real habits are not, and nothing here has a human's blind spots. Use it to check that code works and to compare
systems on identical data; do not read the numbers as evidence about real players.

    python tools/synth_games.py --out synth --games 100 --seed 1
"""
import argparse
import math
import os
import random
import shutil
import sys

import chess
import chess.engine
import chess.pgn

SIGNALS = {
    "queenside": lambda b, m: chess.square_file(m.to_square) <= 2,
    "light": lambda b, m: (chess.square_file(m.to_square) + chess.square_rank(m.to_square)) % 2 == 1,
    "rim": lambda b, m: chess.square_file(m.to_square) in (0, 7) or chess.square_rank(m.to_square) in (0, 7),
}
CP_CLIP = 1000


def pick(engine, board, rng, tau, beta, signal, depth, multipv, opening_plies):
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    moves, logits = [], []
    t = tau * (2.5 if board.ply() < opening_plies else 1.0)          # more variety in the opening
    for info in infos:
        if "pv" not in info:
            continue
        cp = max(-CP_CLIP, min(CP_CLIP, info["score"].pov(board.turn).score(mate_score=CP_CLIP)))
        move = info["pv"][0]
        moves.append(move)
        logits.append(cp / t + (beta * float(signal(board, move)) if signal else 0.0))
    top = max(logits)
    weights = [math.exp(x - top) for x in logits]
    return rng.choices(moves, weights)[0]


def play_game(engine, rng, name, opponent, player_white, signal, beta, tau, opp_tau, depth, max_plies, multipv):
    board, game = chess.Board(), chess.pgn.Game()
    game.headers.update({"Event": "synthetic", "Site": "local", "White": name if player_white else opponent,
                         "Black": opponent if player_white else name, "TimeControl": "600+0",
                         "WhiteElo": "1500", "BlackElo": "1500"})
    node = game
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mine = (board.turn == chess.WHITE) == player_white
        move = pick(engine, board, rng, tau if mine else opp_tau, beta if mine else 0.0, signal if mine else None,
                    depth, multipv, opening_plies=8)
        board.push(move)
        node = node.add_variation(move)
    game.headers["Result"] = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    return game


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synth", help="directory for the PGN files")
    ap.add_argument("--players", default=",".join(SIGNALS), help="comma separated: " + ",".join(SIGNALS))
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--beta", type=float, default=2.0, help="strength of the planted preference (0 = none)")
    ap.add_argument("--tau", type=float, default=70.0, help="player sloppiness in centipawns")
    ap.add_argument("--opp-tau", type=float, default=120.0)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--multipv", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=90)
    ap.add_argument("--stockfish")
    args = ap.parse_args()

    sf = args.stockfish or shutil.which("stockfish") or "/usr/games/stockfish"
    if not os.path.exists(sf):
        sys.exit("Stockfish not found; pass --stockfish")
    os.makedirs(args.out, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({"Threads": 1, "Hash": 16})
    try:
        for name in args.players.split(","):
            if name not in SIGNALS:
                sys.exit(f"unknown player '{name}'")
            rng = random.Random(f"{args.seed}:{name}")
            label = f"Synth{name.capitalize()}"
            path = os.path.join(args.out, f"{name}.pgn")
            with open(path, "w") as fh:
                for g in range(args.games):
                    engine.protocol.send_line("ucinewgame")
                    game = play_game(engine, rng, label, "SynthOpp", g % 2 == 0, SIGNALS[name], args.beta, args.tau,
                                     args.opp_tau, args.depth, args.max_plies, args.multipv)
                    print(game, file=fh, end="\n\n")
            print(f"wrote {path} ({args.games} games, player '{label}', beta {args.beta}, tau {args.tau})", flush=True)
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
