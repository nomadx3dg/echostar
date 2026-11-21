# Maia and EchoStar: study, integration and results

**Short answer.** Maia can be attached to EchoStar as an optional teacher and yardstick, and that is implemented. On the
synthetic players used here it did **not** make EchoStar better at reproducing an individual's style, and in the
cases measured it made it slightly worse at the thing EchoStar is for. Recommendation: keep the current architecture,
leave `--maia-mix` off, and keep `maia-compare` as an optional diagnostic. Every number below is from **synthetic
players**; nothing here shows how EchoStar or Maia behaves on a real person (see [Limits](#limits)).

## 1. The Maia versions

| | Maia-1 (`CSSLab/maia-chess`) | Maia-3 (`CSSLab/maia3`) |
|---|---|---|
| What it is | 9 separate nets, one per rating band 1100-1900 | One Chessformer (transformer) model conditioned on Elo, with optional move history |
| Size | 6 residual blocks x 64 filters; each net 1.2-1.3 MB (measured, `maia_weights/`) | "5M", "23M", "79M" variants; the 5M repo on Hugging Face stores about 21 MB (a `.pt` file) |
| Runtime | `lc0` (C++), weights in lc0 format; no PyTorch | Python + PyTorch (`torch`, `numpy`, `python-chess`, `huggingface-hub`), UCI wrapper `maia3-uci`; CPU via `--device cpu` |
| Output | Policy over moves (one forward pass, `go nodes 1`) | Likely human moves (as `MultiPV`) plus a win/draw/loss head |
| Published speed | none that I found | none in the README |
| Measured here | yes (below) | **no**: the weights host (Hugging Face) and the PyTorch index are blocked from this sandbox |
| Code license | GPL-3.0 (repo `LICENSE`; README: "available under the GPL License") | **AGPL-3.0** (repo `LICENSE`) |
| Weights license | none separate in the README; shipped inside the GPL repo | **none stated**: no license field on the Hugging Face model, none in the README |

Maia-1 is the suitable version for EchoStar: it is tiny, needs no ML framework, is deterministic, and its policy is
exactly "which move would a player of about this rating choose". Maia-3 is likely more accurate and adds Elo conditioning,
but it needs PyTorch and a 20+ MB checkpoint, its weights have no stated license, and I could not run it. Maia-2 was not
evaluated.

**Mobile feasibility (estimate, not measured on a phone).** The Maia-1 net is about 60 MFLOP per position
(6 blocks x 2 convolutions x 64x64x9 x 64 squares x 2). It ran at **5.4-5.9 ms median / 7.8-8.3 ms p90 per position** on this
server's 2-vCPU x86 CPU through lc0's BLAS backend, with the lc0 process at **about 124 MB resident**. That memory is lc0's
generic runtime, not the net; a purpose-written Rust forward pass over 1.3 MB of weights would need a few MB. I did not test
lc0 on Android, and I did not write that port. Maia-3 would need PyTorch on the device, which is not realistic for the
Rust/Termux design. Conclusion: use Maia **offline during training and development**, which is how it is integrated.

## 2. What was built

Everything is optional. Without Maia installed, `echo.py` behaves exactly as before (the original 86 Python tests pass
unchanged; the Rust runtime has no Maia code).

| Piece | Where | Purpose |
|---|---|---|
| `Lc0Teacher` | `maia_teacher.py` | Runs Maia-1 through a local `lc0` (`go nodes 1`, raw policy temperature 1.0) and returns probabilities over the legal moves |
| `CachingTeacher` | `maia_teacher.py` | JSON-lines cache keyed by position; refuses to mix teachers; replays without lc0 so training is reproducible |
| `MaiaBlendBrain` | `maia_teacher.py` | Runtime option: style score + `weight * ln(Maia probability)`. Only reorders moves inside the safe set |
| `learn --maia-mix W` (+ `--maia-weights`, `--lc0`, `--maia-cache`) | `echo.py` | Soft training target `(1-W)*player's move + W*Maia`; `W=0` is the original update, bit for bit |
| `maia-compare` | `echo.py` | How typical are your moves for Maia? (no training, no writes) |
| `teacher` header block | [FORMAT.md](FORMAT.md) | Provenance only; no tensor changed; format stays version 2; Rust ignores it (tested) |
| `tools/synth_games.py`, `bench_maia.py`, `bench_habit.py` | `tools/` | Synthetic players and the benchmarks below |

Safety is unchanged: Maia never touches candidate generation, the safety band, or the blunder gate. Tests prove a
teacher that is certain of a blunder cannot be chosen outside the band, cannot pass the gate, and cannot override a
forced mate (`tests/test_maia_teacher.py`). The style dials mean what they did.

**A bug found by the checks.** lc0 prints castling as king-takes-rook (`e1h1`); python-chess says `e1g1`. My first version
silently gave every castling move the floor probability. It surfaced as a "policy covers only 45%" failure on a real
position where Maia gives castling 66%. Fixed, with a regression test against real lc0, and the teacher now raises an error
whenever less than 95% of the probability mass maps to legal moves. Without that check the training signal would have been
quietly wrong.

## 3. Method

* **Data (synthetic).** `tools/synth_games.py` plays three "quirky" players, 200 games each: Stockfish proposes its top
  moves and the player picks among them with probability proportional to `exp(cp/70 + 2*habit)`, where the habit is one of
  `queenside` (ends on files a-c), `light` (ends on a light square) or `rim` (ends on the board edge). None of these is one
  of EchoStar's 21 features. These are machines with a planted habit, not people.
* **Split.** By game: 70% train / 15% validation / 15% test, 3 random splits per player. The epoch, the Maia mix
  (0, .25, .5, .75) and the runtime blend (0, .25, .5, 1, 2) are chosen on validation only; all numbers are test positions.
  The test sets of different splits overlap, so the +/- understates uncertainty.
* **Positions with a forced move are dropped** (a single legal move says nothing about style).
* **Teacher.** Maia-1 at rating 1500, whatever the synthetic player's strength.
* **Not measured:** Elo / playing strength. Average centipawns lost against a depth-12 search is reported as a proxy only.

## 4. Results (synthetic data)

### 4.1 Held-out move prediction (test games, mean of 3 splits)

Log-loss is -ln of the probability given to the move actually played (lower is better; random is about 3.4).

| player | system | top-1 | top-3 | log-loss |
|---|---|---|---|---|
| queenside | Maia-1 alone (not personalized) | 22.5% | 47.0% | 2.987 |
| | **existing EchoStar** | 19.9% | 39.8% | 2.886 |
| | EchoStar + Maia training teacher | 19.6% | 39.7% | 2.887 |
| | EchoStar + Maia at runtime (blend) | 23.9% | 47.8% | 2.649 |
| light | Maia-1 alone | 23.0% | 45.3% | 3.045 |
| | **existing EchoStar** | 18.4% | 37.1% | 2.991 |
| | EchoStar + Maia training teacher | 18.6% | 38.3% | 2.985 |
| | EchoStar + Maia at runtime | 21.7% | 45.4% | 2.785 |
| rim | Maia-1 alone | 19.3% | 39.6% | 3.350 |
| | **existing EchoStar** | 16.6% | 34.7% | 2.995 |
| | EchoStar + Maia training teacher | 16.3% | 35.0% | 3.011 |
| | EchoStar + Maia at runtime | 19.2% | 42.0% | 2.805 |

* The **training teacher does nothing measurable**: differences of 0.3 points and 0.01 nats are inside the split-to-split
  spread, and the validation grid chose the smallest non-zero mix every time (higher mixes were worse).
* That stays true with little data. Queenside, log-loss with 15 / 40 / about 120 training games: existing 3.009 / 2.953 /
  2.886, with teacher 3.008 / 2.951 / 2.887. A likely reason is that the compact net (a board encoding plus 21 features)
  cannot represent Maia's policy, so it has little to absorb. I did not test a richer student.
* The **runtime blend** improves held-out likelihood clearly (about 0.2 nats), but it needs Maia on the device, and it is
  the weaker result than it looks: on `rim`, Maia alone is *worse* than EchoStar alone (3.350 vs 2.995) because Maia does not
  know the habit.

### 4.2 Does the system reproduce the player's habit? (the personalization test)

Exact agreement with the player's move is blunt (among equally good moves the choice is partly luck), so I also measured how
often each system's move has the planted habit, through the real chooser (balanced band 50, blunder gate on, temperature 0).
"Chance" is the share of safe candidates that have the habit, i.e. what a style-blind chooser would show.

| player | player's own moves | chance (no style) | existing EchoStar | + Maia training teacher | + Maia runtime blend |
|---|---|---|---|---|---|
| queenside | 59.4% | 30.4% | **44.8%** | 40.0% | 39.7% |
| light | 68.9% | 37.9% | **50.5%** | 47.5% | 48.4% |
| rim | 54.2% | 22.6% | **33.5%** | 29.5% | 26.9% |

The existing EchoStar learns the planted habit on all three players. Both Maia variants give back part of it (2 to 7 points
of the 11 to 14 point gain) while leaving move quality unchanged. This is the core finding: Maia models the *typical* player, and
pulling toward it dilutes an *individual*. It is consistent in direction across three players and two splits each, but it is
a small sample of synthetic players; I would not call it settled.

### 4.3 Safety and strength proxy (split 0 test positions, about 1,300 per player)

Same positions for every system; "blunder" = at least 200 cp worse than the best move by a depth-12 search.

| system | queenside | light | rim | avg cp lost (queenside) | blunders (queenside) |
|---|---|---|---|---|---|
| Stockfish-only, matches player's move | 27.0% | 27.9% | 23.4% | 0 | 0.0% |
| Maia raw argmax | 21.7% | 23.4% | 19.1% | 45 | 6.4% |
| existing EchoStar raw argmax (no safety layer) | 18.8% | 19.3% | 17.5% | 115 | 19.0% |
| existing EchoStar via chooser (balanced) | 29.4% | 29.4% | 25.4% | 10 | 0.0% |
| + Maia training teacher via chooser | 27.6% | 29.4% | 24.3% | 8 | 0.1% |
| + Maia runtime blend via chooser | 29.1% | 30.0% | 24.0% | 8 | 0.0% |

* The first three columns are "agreement with the player's actual move". With about 1,300 positions the standard error is
  about 1.2 points, so the three chooser rows are **statistically indistinguishable** on this metric; only the habit test
  (4.2) separates them.
* The player's own move lies inside the balanced band only 57-63% of the time (faithful 72-76%, sharp 41-46%), which caps
  how often any chooser can match.
* **Maia cannot replace the truth engine**: its raw picks blunder 4.5-6.4% of the time; EchoStar's raw picks blunder 13-19%.
  Through the chooser every system's blunder rate is 0.0-0.5% at all dials.
* Chooser rows use temperature 0 at each dial's band (not the dials' sampling temperatures), to isolate the brain's ranking.

### 4.4 Resources (this server: 2 vCPU x86, Linux; not a phone)

| item | value |
|---|---|
| Maia-1 weights / lc0 binary | 1.26 MB / 2.8 MB (excluding the OpenBLAS library) |
| Maia latency, lc0 BLAS, 1 thread | 5.4-5.9 ms median, 7.8-8.3 ms p90 per position |
| lc0 + Maia process memory | about 124 MB resident |
| Teacher cost during training | 8,752 positions in 48.9 s; cache 7.9-8.2 MB per player |
| Stockfish MultiPV 8, one thread | depth 8: about 30 ms; depth 12: 374 ms; depth 14: 1.3 s per position |
| Stockfish memory / binary | 166 MB resident (Hash 32 MB) / 40.4 MB |
| `.echo` model, existing and with teacher | 2.55 MB both (identical size) |
| Training time | 6.9 s per model on about 6,100 positions (pure NumPy) |
| `echo-rs` (Rust runtime) | 1.6 MB binary; loading a model took 4 ms and peaked at 13.1 MB resident |
| Python benchmark process | about 315 MB peak |

Maia at runtime costs about 6 ms and over 100 MB next to a Stockfish search that costs 30-1300 ms, so the latency is not the
obstacle; the dependency (lc0 or PyTorch on the device), the weights' license and the lack of benefit are.

## 5. Licensing

I did not copy or bundle any Maia code, weights or lc0 binary. The repository contains only a client that calls an
`lc0` the user installs, and a header field naming the teacher.

| Component | License | Notes |
|---|---|---|
| EchoStar | **GPL-3.0-only** (`LICENSE`) | Chosen to match its GPL dependencies (python-chess, shakmaty) and Stockfish. Before this change the repo had no license |
| Maia-1 code and weights | GPL-3.0 (repo LICENSE; README says "GPL License") | Weights are distributed in the GPL repo; whether the weights are covered by the GPL as "software" is not spelled out |
| Maia-3 code | AGPL-3.0 | Network-use clause; copyleft |
| Maia-3 weights | **no license stated** (none on Hugging Face, none in README) | Not the same as the code license, and "no license" means no redistribution right. Do not bundle. Ask the authors (University of Toronto CSSLab) if you need to |
| lc0 | GPL-3.0 (COPYING; contains an "Additional permissions" section for GPU vendor libraries) | Run as a separate process only |
| Stockfish 16 | GPL-3.0 (Debian package copyright) | Shipping its binary with EchoStar requires GPL-compatible terms |
| python-chess | GPL-3.0+ | EchoStar imports it |
| shakmaty (Rust) | GPL-3.0+ | `echo-rs` links it |
| numpy, serde, serde_json | BSD-3 / MIT-or-Apache | Permissive |

Practical reading, not legal advice: EchoStar depends on GPL-3.0+ libraries (python-chess, shakmaty) and expects Stockfish,
so a distributed build had to be GPL-compatible; it is now GPL-3.0-only (`LICENSE`). Keeping Maia as a separate, user-installed program, used offline, avoids adding anything new. A model trained with
Maia as a teacher may carry questions about derived works (its targets came from Maia's outputs); the Maia authors do not
address this in the READMEs I read, and I have not obtained a legal opinion. Maia-3 should not be used for distillation or
shipped until its weights have a stated license. If you cite results that used Maia, cite the authors' papers (Maia-1: "Aligning Superhuman AI with Human Behavior: Chess as a Model System", KDD 2020;
Maia-3 README asks for "Chessformer: A Unified Architecture for Chess Modeling", ICLR 2026).

## 6. Limits

* **Nothing here is real-player validation.** The sandbox could not reach Lichess or Chess.com, so there are no real human
  games. The synthetic players are Stockfish plus noise plus one clean planted habit; real habits are messier and real
  players have blind spots Stockfish-based players lack. Those differences could favor or hurt Maia in ways I cannot
  predict. Roadmap M2 (real games) is the test that matters.
* Three synthetic players, three splits each, overlapping test sets; differences under about 2 points are ties.
* Only Maia-1 at rating 1500 was run. Maia-3 and Maia-2 were not run. Rating-matched Maia nets were not tried.
* The mix and blend grids were small and the learning rate and epochs fixed. A richer student, a different loss, or using Maia
  only on positions where the player's own data is thin might behave differently. I did not try them, and the task asked not to
  grow the model just because Maia is large.
* The habit test needs a known planted habit, so it cannot be run on a real person.
* Playing strength (Elo) was not measured; centipawn loss is a proxy.
* Not tested on a phone or Android; the mobile numbers above are estimates labeled as such.

## 7. How to reproduce

```bash
python tools/synth_games.py --out synth --games 200 --seed 1          # needs Stockfish
python tools/bench_maia.py --pgn synth/queenside.pgn --player SynthQueenside --data-kind synthetic \
    --lc0 /path/to/lc0 --maia-weights maia-1500.pb.gz --maia-cache maia.jsonl --safety
python tools/bench_habit.py --pgn synth/queenside.pgn --player SynthQueenside --habit queenside \
    --lc0 /path/to/lc0 --maia-weights maia-1500.pb.gz --maia-cache maia.jsonl
# with real games, the first question to ask:
python echo.py maia-compare my_games.pgn --player MyName --maia-weights maia-1500.pb.gz
ECHOSTAR_LC0=/path/to/lc0 ECHOSTAR_MAIA_WEIGHTS=maia-1500.pb.gz python -m pytest   # also runs the real-lc0 tests
```

Raw outputs of the runs reported here are in `docs/maia_results/`.
