#!/usr/bin/env python3
"""
Benchmark EchoStar with and without a Maia teacher, on identical held-out positions.

Systems compared (all scored on the same test positions of the same games):
  random          a uniformly random legal move (the floor)
  stockfish-only  the engine's best move at --depth (no style at all)
  maia            Maia's most likely move (a generic human model, rating-conditioned, not personalized)
  echostar        the existing Style Brain, trained as `echo.py learn` trains it
  echostar+maia   the same Style Brain trained with a Maia soft target (training-time teacher only; same file format)
  echostar~maia   the existing Style Brain plus a Maia term at play time (needs Maia at runtime)

Honest-evaluation rules this script follows
  * Split by GAME into train / validation / test. The epoch and the Maia weights (mix, blend) are chosen on validation;
    everything reported comes from test positions the model never trained on and never selected on.
  * Repeated over several random splits (--seeds); the test sets of different splits overlap, so the spread understates
    the real uncertainty. A single split of a few dozen games is noisy; read differences smaller than the spread as ties.
  * --data-kind must be given and is printed on every report: results on synthetic games say nothing about real players.

Safety / strength section (--safety, needs Stockfish): the same test positions are run through the real
`echo.Echo.choose` (safety band + blunder gate) with each brain, and the move each system would play is scored by a deeper
reference search. "Blunder" = the pick is >= 200 cp worse than the engine's best. Playing strength (Elo) is NOT measured.

    python tools/bench_maia.py --pgn synth/queenside.pgn --player SynthQueenside --data-kind synthetic \\
        --lc0 /path/to/lc0 --maia-weights maia-1500.pb.gz --maia-cache maia.jsonl --stockfish /usr/games/stockfish --safety
"""
import argparse
import json
import math
import os
import random
import resource
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess  # noqa: E402
import chess.engine  # noqa: E402
import numpy as np  # noqa: E402

import echo  # noqa: E402
import maia_teacher as mt  # noqa: E402

CP_CLIP = 1000
BLUNDER_CP = 200
DIAL_BANDS = {"faithful": 120, "balanced": 50, "sharp": 20}


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
class Pos:
    """One position where the player moved: everything the model and the metrics need."""
    __slots__ = ("board", "move", "moves", "t", "act", "midx", "F", "q", "game")

    def sample(self):
        return (self.act, self.midx, self.F, self.t, self.q)


def load_positions(pgn, player, max_games=None):
    games, _ = echo.read_games(pgn, player)
    if max_games:
        games = games[:max_games]
    per_game = []
    for gi, (game, me) in enumerate(games):
        board, rows = game.board(), []
        for move in game.mainline_moves():
            if board.turn == me:
                moves = list(board.legal_moves)
                if move in moves and len(moves) > 1:           # a forced move says nothing about style
                    p = Pos()
                    p.board, p.move, p.moves, p.t, p.game = board.copy(), move, moves, moves.index(move), gi
                    p.act = echo.encode(board)
                    p.midx = np.array([echo.move_index(board, m) for m in moves])
                    p.F = np.array([echo.features(board, m) for m in moves])
                    p.q = None
                    rows.append(p)
            board.push(move)
        per_game.append(rows)
    return per_game


def split_games(n, seed, frac=0.15):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    k = max(1, round(n * frac))
    return order[2 * k:], order[k:2 * k], order[:k]         # train, validation, test


def flat(per_game, idx):
    return [p for i in idx for p in per_game[i]]


# ----------------------------------------------------------------------------------------------
# models and metrics
# ----------------------------------------------------------------------------------------------
def fit(train, val, mix, epochs, lr, seed):
    """Train a Style Brain the way `learn` does (SGD, lr schedule, best epoch by validation loss)."""
    net, rng = echo.StyleNet(), random.Random(seed)
    train = list(train)
    val_samples = [p.sample() for p in val]
    best, best_params = float("inf"), net.params()
    for epoch in range(epochs):
        rng.shuffle(train)
        step_lr = lr / (1 + 0.15 * epoch)
        for p in train:
            net.step(p.sample(), step_lr, mix)
        loss = net.evaluate(val_samples)["loss"]
        if loss < best:
            best, best_params = loss, net.params()
    net.load_params(best_params)
    return net


def logits(net, p):
    h = np.maximum(net.W1[p.act].sum(0) + net.b1, 0)
    return net.W2T[p.midx] @ h + net.b2[p.midx] + p.F @ net.w


def score_dists(dists, ps):
    """top1 / top3 / mean negative log-likelihood of the move played, from one probability vector per position."""
    top1 = top3 = nll = 0.0
    for d, p in zip(dists, ps):
        order = np.argsort(-d)
        top1 += order[0] == p.t
        top3 += p.t in order[:3]
        nll -= math.log(max(d[p.t], 1e-12))
    n = len(ps)
    return {"top1": top1 / n, "top3": top3 / n, "nll": nll / n}


def style_dist(net, ps):
    return [echo.softmax(logits(net, p)) for p in ps]


def blend_dist(net, ps, mu):
    return [echo.softmax(logits(net, p) + mu * np.log(np.maximum(p.q, mt.PROB_FLOOR))) for p in ps]


class NetBrain:
    """The Style Brain's net alone (no opening book), as `learn` validates it."""
    def __init__(self, net):
        self.net = net

    def scores(self, board, moves):
        return [float(x) for x in self.net.logits(board, moves)]


# ----------------------------------------------------------------------------------------------
# Stockfish helpers
# ----------------------------------------------------------------------------------------------
class TimedEngine:
    """Memoizing, timing wrapper with the interface `echo.Echo` uses; identical positions are searched once."""
    def __init__(self, engine):
        self.engine, self.memo, self.t_multi, self.n_multi, self.t_single, self.n_single = engine, {}, 0.0, 0, 0.0, 0

    def analyse(self, board, limit, multipv=None):
        key = (board.fen(), limit.depth, multipv)
        if key not in self.memo:
            t = time.perf_counter()
            self.memo[key] = self.engine.analyse(board, limit, multipv=multipv) if multipv else \
                self.engine.analyse(board, limit)
            dt = time.perf_counter() - t
            if multipv:
                self.t_multi, self.n_multi = self.t_multi + dt, self.n_multi + 1
            else:
                self.t_single, self.n_single = self.t_single + dt, self.n_single + 1
        return self.memo[key]


def candidates(eng, board, depth):
    infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=8)
    return [(i["pv"][0], i["score"].pov(board.turn).score(mate_score=CP_CLIP)) for i in infos if "pv" in i]


def ref_eval(eng, board, move, depth):
    """Evaluation of `move` by a reference search after it is played, from the mover's side, clipped to +-1000 cp."""
    mover = board.turn
    b = board.copy()
    b.push(move)
    if b.is_checkmate():
        return CP_CLIP
    info = eng.analyse(b, chess.engine.Limit(depth=depth))
    return max(-CP_CLIP, min(CP_CLIP, info["score"].pov(mover).score(mate_score=CP_CLIP)))


# ----------------------------------------------------------------------------------------------
# measurements of resources
# ----------------------------------------------------------------------------------------------
def proc_mem_mb(pid):
    """(current, peak) resident memory of a process in MB from /proc, or (None, None)."""
    try:
        cur = peak = None
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 1024
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) / 1024
        return cur, peak
    except OSError:
        return None, None


def engine_pid(engine):
    try:
        return engine.transport.get_pid()
    except Exception:
        return None


def time_teacher(lc0, weights, ps, n=200):
    """Per-position latency of a live Maia (lc0 `go nodes 1`, BLAS CPU) on `n` real positions, plus its memory."""
    sample = random.Random(0).sample(ps, min(n, len(ps)))
    with mt.Lc0Teacher(lc0, weights) as t:
        for p in sample[:5]:
            t.probs(p.board, p.moves)                       # warm-up
        times = []
        for p in sample:
            fresh = p.board.copy()
            fresh.push(chess.Move.null()) if False else None
            t0 = time.perf_counter()
            t.probs(fresh, p.moves)
            times.append((time.perf_counter() - t0) * 1000)
        cur, peak = proc_mem_mb(t.pid())
    times.sort()
    return {"n": len(times), "median_ms": times[len(times) // 2], "p90_ms": times[int(len(times) * 0.9)],
            "mean_ms": sum(times) / len(times), "rss_mb": cur, "peak_rss_mb": peak,
            "note": "positions may be served from lc0's internal cache only if repeated; each call here is distinct"}


# ----------------------------------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------------------------------
def mean_sd(xs):
    a = np.array(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def run_prediction(per_game, args, sf=None):
    """Held-out prediction for every system, over several splits. Returns {system: {metric: [per split]}} and chosen knobs."""
    systems = {}
    chosen = []
    times = {"train_s_per_model": []}

    def add(name, scores, split_id):
        d = systems.setdefault(name, {})
        for k, v in scores.items():
            d.setdefault(k, []).append(v)

    nets_for_safety = None
    for seed in range(args.seeds):
        tr_i, va_i, te_i = split_games(len(per_game), seed)
        train, val, test = flat(per_game, tr_i), flat(per_game, va_i), flat(per_game, te_i)
        if args.train_games:
            keep = tr_i[:args.train_games]
            train = flat(per_game, keep)
        nets, val_nll = {}, {}
        for mix in args.mixes:
            t0 = time.perf_counter()
            nets[mix] = fit(train, val, mix, args.epochs, args.lr, seed)
            times["train_s_per_model"].append(time.perf_counter() - t0)
            val_nll[mix] = score_dists(style_dist(nets[mix], val), val)["nll"]
        best_mix = min((m for m in args.mixes if m > 0), key=lambda m: val_nll[m]) if any(m > 0 for m in args.mixes) else 0.0
        net0 = nets[0.0]
        mu_nll = {mu: score_dists(blend_dist(net0, val, mu), val)["nll"] for mu in args.blends}
        best_mu = min((m for m in args.blends if m > 0), key=lambda m: mu_nll[m]) if any(m > 0 for m in args.blends) else 0.0

        add("random", {"top1": sum(1 / len(p.moves) for p in test) / len(test), "top3": float("nan"),
                       "nll": sum(math.log(len(p.moves)) for p in test) / len(test)}, seed)
        add("maia", score_dists([p.q for p in test], test), seed)
        add("echostar (existing)", score_dists(style_dist(net0, test), test), seed)
        add("echostar+maia (training teacher)", score_dists(style_dist(nets[best_mix], test), test), seed)
        add("echostar~maia (runtime blend)", score_dists(blend_dist(net0, test, best_mu), test), seed)
        for mix in args.mixes:
            add(f"  grid: echostar mix={mix:g}", score_dists(style_dist(nets[mix], test), test), seed)
        for mu in args.blends:
            add(f"  grid: blend mu={mu:g}", score_dists(blend_dist(net0, test, mu), test), seed)
        chosen.append({"seed": seed, "games": [len(tr_i), len(va_i), len(te_i)], "positions": [len(train), len(val), len(test)],
                       "best_mix": best_mix, "best_mu": best_mu,
                       "val_nll_by_mix": {str(k): v for k, v in val_nll.items()},
                       "val_nll_by_mu": {str(k): v for k, v in mu_nll.items()}})
        if seed == 0:
            nets_for_safety = (nets, best_mix, best_mu, test)
        print(f"  split {seed}: train {len(train)} / val {len(val)} / test {len(test)} positions; "
              f"chosen mix {best_mix:g}, blend {best_mu:g}", flush=True)
    return systems, chosen, times, nets_for_safety


def run_safety(nets_for_safety, teacher, args):
    """Engine-referenced behavior of each system on split 0's test positions."""
    nets, best_mix, best_mu, test = nets_for_safety
    sf_path = args.stockfish or shutil.which("stockfish") or "/usr/games/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": 1, "Hash": 32})
    eng = TimedEngine(engine)
    ref = TimedEngine(engine)                              # separate memo so reference timings stay separate
    D, R = args.depth, args.ref_depth
    net0, netM = nets[0.0], nets[best_mix]
    brains = {"echostar (existing)": NetBrain(net0),
              "echostar+maia (training teacher)": NetBrain(netM),
              "echostar~maia (runtime blend)": mt.MaiaBlendBrain(NetBrain(net0), teacher, best_mu)}
    out = {"depth": D, "ref_depth": R, "positions": len(test), "systems": {}}

    base = [candidates(eng, p.board, D) for p in test]
    best_ref = [ref_eval(ref, p.board, c[0][0], R) if c else 0 for p, c in zip(test, base)]

    def loss_of(i, move):
        return max(0, best_ref[i] - ref_eval(ref, test[i].board, move, R))

    def summarize(name, picks, extra=None):
        losses = [loss_of(i, m) for i, m in enumerate(picks)]
        agree = sum(m == p.move for m, p in zip(picks, test)) / len(test)
        row = {"top1_vs_player": agree, "acpl": float(np.mean(losses)), "median_loss": float(np.median(losses)),
               "blunder_rate": float(np.mean([x >= BLUNDER_CP for x in losses])),
               "loss_ge_100": float(np.mean([x >= 100 for x in losses]))}
        row.update(extra or {})
        out["systems"][name] = row

    # raw picks: what each system would play with NO safety layer (argmax over every legal move)
    rng = random.Random(0)
    summarize("random legal move (raw)", [rng.choice(p.moves) for p in test])
    summarize("stockfish-only", [c[0][0] if c else p.moves[0] for p, c in zip(test, base)])
    summarize("maia (raw argmax)", [p.moves[int(np.argmax(p.q))] for p in test])
    for name, brain in brains.items():
        picks = []
        for p in test:
            s = brain.scores(p.board, p.moves)
            picks.append(p.moves[int(np.argmax(s))])
        summarize(f"{name} (raw argmax)", picks)

    # the real chooser: safety band + blunder gate, deterministic (temperature 0) at each dial's band
    ceilings = {}
    for dial, band in DIAL_BANDS.items():
        safe_player = []
        for p, c in zip(test, base):
            best_cp = c[0][1]
            eff = 0 if best_cp >= echo.MATE - 1000 else (min(band, 30) if abs(best_cp) > 500 else band)
            safe_player.append(any(m == p.move and cp >= best_cp - eff for m, cp in c))
        ceilings[dial] = float(np.mean(safe_player))
        for name, brain in brains.items():
            e = echo.Echo(eng, brain, {"band": band, "temp": 0.0}, depth=D)
            picks = [e.choose(p.board.copy()) for p in test]
            summarize(f"{name} via chooser [{dial}, band {band}]", picks)
    out["player_move_inside_band"] = ceilings
    cur, peak = proc_mem_mb(engine_pid(engine))
    out["stockfish"] = {"multipv8_ms_per_position": 1000 * eng.t_multi / max(1, eng.n_multi),
                        "single_pv_ms_per_search": 1000 * (eng.t_single + ref.t_single) / max(1, eng.n_single + ref.n_single),
                        "multipv_searches": eng.n_multi, "single_searches": eng.n_single + ref.n_single,
                        "rss_mb": cur, "peak_rss_mb": peak}
    engine.quit()
    return out


def fmt(xs, pct=True, digits=1):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "n/a"
    m, s = mean_sd(xs)
    return f"{100 * m:.{digits}f}% ± {100 * s:.{digits}f}" if pct else f"{m:.3f} ± {s:.3f}"


def report_markdown(res):
    L = []
    kind = res["data_kind"].upper()
    L += [f"# EchoStar x Maia benchmark ({kind} DATA)", ""]
    if res["data_kind"] == "synthetic":
        L += ["> **Synthetic games.** The player is a noisy engine with a planted habit (tools/synth_games.py). "
              "These numbers test the pipeline and compare systems on identical data; they say nothing about whether "
              "EchoStar or Maia works on a real person.", ""]
    L += [f"Player `{res['player']}`: {res['games']} games, {res['positions']} positions with a real choice. "
          f"Teacher: {res['teacher']}. {res['seeds']} random game-level splits "
          f"(train/val/test games per split: {res['splits'][0]['games']}).", ""]
    L += ["## Held-out move prediction (test games only)", "",
          "| system | top-1 | top-3 | log-loss (nats) |", "|---|---|---|---|"]
    for name, d in res["prediction"].items():
        L.append(f"| {name} | {fmt(d['top1'])} | {fmt(d['top3'])} | {fmt(d['nll'], pct=False)} |")
    L += ["", "± is the standard deviation over splits; test sets overlap between splits, so it understates uncertainty. "
          "Log-loss is the average of -ln(probability given to the move actually played); lower is better, "
          "random guessing is the first row. Stockfish-only has no probabilities, so its top-1 is in the next table.", ""]
    s = res.get("safety")
    if s:
        L += [f"## Engine-referenced behavior (split 0 test positions: {s['positions']}; Stockfish depth {s['depth']} to "
              f"decide, depth {s['ref_depth']} to judge)", "",
              "| system | matches player's move | avg cp lost | blunders (>=200 cp) | >=100 cp lost |",
              "|---|---|---|---|---|"]
        for name, r in s["systems"].items():
            L.append(f"| {name} | {100 * r['top1_vs_player']:.1f}% | {r['acpl']:.0f} | {100 * r['blunder_rate']:.1f}% "
                     f"| {100 * r['loss_ge_100']:.1f}% |")
        L += ["", "Share of positions where the player's own move was inside each dial's safety band "
              "(the ceiling on how often the chooser can match the player): "
              + ", ".join(f"{d} {100 * v:.0f}%" for d, v in s["player_move_inside_band"].items()) + ".",
              "Average centipawns lost is a *proxy* for strength, not an Elo; playing strength was not measured.", ""]
    rr = res.get("resources")
    if rr:
        L += ["## Resources", "", "| item | value |", "|---|---|"]
        for k, v in rr.items():
            L.append(f"| {k} | {v} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--player", required=True)
    ap.add_argument("--data-kind", required=True, choices=["synthetic", "real"],
                    help="say what the games are; it is printed on every report")
    ap.add_argument("--lc0")
    ap.add_argument("--maia-weights")
    ap.add_argument("--maia-cache", help="JSON-lines cache of Maia probabilities (filled if lc0 is given)")
    ap.add_argument("--stockfish")
    ap.add_argument("--seeds", type=int, default=3, help="number of random game-level train/val/test splits")
    ap.add_argument("--mixes", default="0,0.25,0.5,0.75", help="Maia soft-target weights to try (0 = existing EchoStar)")
    ap.add_argument("--blends", default="0,0.25,0.5,1,2", help="runtime blend weights to try")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--train-games", type=int, help="cap the training games per split (learning-curve runs)")
    ap.add_argument("--max-games", type=int)
    ap.add_argument("--safety", action="store_true", help="also run the Stockfish/chooser section")
    ap.add_argument("--depth", type=int, default=8, help="Stockfish depth for candidates (chooser gate adds 6)")
    ap.add_argument("--ref-depth", type=int, default=12, help="Stockfish depth used to judge the picks")
    ap.add_argument("--out", help="write the full results as JSON")
    ap.add_argument("--md", help="write a Markdown report")
    args = ap.parse_args()
    args.mixes = sorted({float(x) for x in args.mixes.split(",")} | {0.0})
    args.blends = sorted({float(x) for x in args.blends.split(",")} | {0.0})

    t_start = time.time()
    per_game = load_positions(args.pgn, args.player, args.max_games)
    n_pos = sum(len(g) for g in per_game)
    if len(per_game) < 10:
        sys.exit(f"need at least 10 games for a train/validation/test split, found {len(per_game)}")
    print(f"{len(per_game)} games, {n_pos} positions ({args.data_kind} data)", flush=True)

    # ---- Maia probabilities for every position, once (live lc0 fills the cache; the cache alone replays it)
    ns = argparse.Namespace(maia_weights=args.maia_weights, lc0=args.lc0, maia_cache=args.maia_cache)
    teacher = echo.build_teacher(ns)
    if teacher is None:
        sys.exit("a Maia teacher is needed: --maia-weights with --lc0, or --maia-cache")
    t0 = time.perf_counter()
    try:
        for g in per_game:
            for p in g:
                p.q = teacher.probs(p.board, p.moves)
    except mt.TeacherError as exc:
        sys.exit(f"Maia teacher: {exc}")
    fill_s = time.perf_counter() - t0
    teacher.flush() if hasattr(teacher, "flush") else None
    print(f"Maia probabilities ready ({getattr(teacher, 'name', '?')}, {fill_s:.1f}s)", flush=True)

    systems, chosen, times, nets_for_safety = run_prediction(per_game, args)
    res = {"data_kind": args.data_kind, "player": args.player, "games": len(per_game), "positions": n_pos,
           "teacher": getattr(teacher, "name", "?"), "teacher_elo": getattr(teacher, "elo", None), "seeds": args.seeds,
           "splits": chosen, "prediction": systems, "args": {k: v for k, v in vars(args).items() if k != "out"}}

    resources = {"python peak RSS (MB)": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}",
                 "training time per model (s, mean)": f"{np.mean(times['train_s_per_model']):.1f}"}
    maia_sizes = {}
    if args.maia_weights and os.path.exists(args.maia_weights):
        maia_sizes["Maia weights file (MB)"] = f"{os.path.getsize(args.maia_weights) / 1e6:.2f}"
    if args.lc0 and os.path.exists(args.lc0):
        maia_sizes["lc0 binary (MB)"] = f"{os.path.getsize(args.lc0) / 1e6:.1f}"
        lat = time_teacher(args.lc0, args.maia_weights, [p for g in per_game for p in g]) if args.maia_weights else None
        if lat:
            res["maia_latency"] = lat
            resources["Maia (lc0, BLAS CPU) median / p90 latency per position (ms)"] = f"{lat['median_ms']:.1f} / {lat['p90_ms']:.1f}"
            resources["Maia lc0 process RSS / peak (MB)"] = f"{lat['rss_mb']:.0f} / {lat['peak_rss_mb']:.0f}"
    resources.update(maia_sizes)

    if args.safety:
        print("Stockfish section...", flush=True)
        res["safety"] = run_safety(nets_for_safety, teacher, args)
        sf = res["safety"]["stockfish"]
        resources[f"Stockfish MultiPV 8 at depth {args.depth} (ms per position)"] = f"{sf['multipv8_ms_per_position']:.0f}"
        resources["Stockfish process RSS / peak (MB)"] = f"{sf['rss_mb']:.0f} / {sf['peak_rss_mb']:.0f}"
        sfp = args.stockfish or shutil.which("stockfish") or "/usr/games/stockfish"
        if os.path.exists(sfp):
            resources["Stockfish binary (MB)"] = f"{os.path.getsize(sfp) / 1e6:.1f}"

    # model file sizes (trained on split 0)
    nets, best_mix, _, _ = nets_for_safety
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for label, net in (("existing", nets[0.0]), (f"with Maia teacher (mix {best_mix:g})", nets[best_mix])):
            meta = {"player": args.player, "games": len(per_game), "positions": n_pos, "played_rate": {}, "baseline_rate": {},
                    "book": {}, "validation": None}
            path = os.path.join(tmp, "m.echo")
            echo.export_echo(net, meta, path)
            resources[f".echo size, {label} (MB)"] = f"{os.path.getsize(path) / 1e6:.2f}"
    if args.maia_cache and os.path.exists(args.maia_cache):
        resources["Maia probability cache (MB)"] = f"{os.path.getsize(args.maia_cache) / 1e6:.2f}"
    res["resources"] = resources
    res["wall_seconds"] = time.time() - t_start

    teacher.close()
    md = report_markdown(res)
    print("\n" + md)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(md)


if __name__ == "__main__":
    main()
