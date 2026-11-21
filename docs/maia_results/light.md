# EchoStar x Maia benchmark (SYNTHETIC DATA)

> **Synthetic games.** The player is a noisy engine with a planted habit (tools/synth_games.py). These numbers test the pipeline and compare systems on identical data; they say nothing about whether EchoStar or Maia works on a real person.

Player `SynthLight`: 200 games, 8819 positions with a real choice. Teacher: lc0:maia-1500.pb.gz. 3 random game-level splits (train/val/test games per split: [140, 30, 30]).

## Held-out move prediction (test games only)

| system | top-1 | top-3 | log-loss (nats) |
|---|---|---|---|
| random | 4.0% ± 0.2 | n/a | 3.444 ± 0.014 |
| maia | 23.0% ± 0.8 | 45.3% ± 1.1 | 3.045 ± 0.028 |
| echostar (existing) | 18.4% ± 0.8 | 37.1% ± 1.3 | 2.991 ± 0.022 |
| echostar+maia (training teacher) | 18.6% ± 0.6 | 38.3% ± 1.2 | 2.985 ± 0.023 |
| echostar~maia (runtime blend) | 21.7% ± 0.3 | 45.4% ± 1.0 | 2.785 ± 0.039 |
|   grid: echostar mix=0 | 18.4% ± 0.8 | 37.1% ± 1.3 | 2.991 ± 0.022 |
|   grid: echostar mix=0.25 | 18.6% ± 0.6 | 38.3% ± 1.2 | 2.985 ± 0.023 |
|   grid: echostar mix=0.5 | 18.1% ± 0.5 | 37.3% ± 0.8 | 3.003 ± 0.042 |
|   grid: echostar mix=0.75 | 16.9% ± 1.0 | 36.0% ± 0.7 | 3.074 ± 0.045 |
|   grid: blend mu=0 | 18.4% ± 0.8 | 37.1% ± 1.3 | 2.991 ± 0.022 |
|   grid: blend mu=0.25 | 21.7% ± 0.3 | 45.4% ± 1.0 | 2.785 ± 0.039 |
|   grid: blend mu=0.5 | 23.1% ± 0.9 | 46.7% ± 0.6 | 2.863 ± 0.055 |
|   grid: blend mu=1 | 23.8% ± 0.9 | 47.4% ± 1.2 | 3.471 ± 0.080 |
|   grid: blend mu=2 | 24.3% ± 0.7 | 47.2% ± 0.7 | 5.298 ± 0.112 |

± is the standard deviation over splits; test sets overlap between splits, so it understates uncertainty. Log-loss is the average of -ln(probability given to the move actually played); lower is better, random guessing is the first row. Stockfish-only has no probabilities, so its top-1 is in the next table.

## Engine-referenced behavior (split 0 test positions: 1335; Stockfish depth 8 to decide, depth 12 to judge)

| system | matches player's move | avg cp lost | blunders (>=200 cp) | >=100 cp lost |
|---|---|---|---|---|
| random legal move (raw) | 3.9% | 242 | 37.8% | 55.8% |
| stockfish-only | 27.9% | 0 | 0.0% | 0.0% |
| maia (raw argmax) | 23.4% | 38 | 4.7% | 11.5% |
| echostar (existing) (raw argmax) | 19.3% | 91 | 13.5% | 24.8% |
| echostar+maia (training teacher) (raw argmax) | 19.3% | 86 | 12.9% | 22.7% |
| echostar~maia (runtime blend) (raw argmax) | 21.6% | 63 | 8.8% | 18.0% |
| echostar (existing) via chooser [faithful, band 120] | 27.7% | 19 | 0.4% | 4.1% |
| echostar+maia (training teacher) via chooser [faithful, band 120] | 28.2% | 18 | 0.4% | 3.8% |
| echostar~maia (runtime blend) via chooser [faithful, band 120] | 28.2% | 16 | 0.4% | 3.4% |
| echostar (existing) via chooser [balanced, band 50] | 29.4% | 9 | 0.0% | 0.7% |
| echostar+maia (training teacher) via chooser [balanced, band 50] | 29.4% | 8 | 0.0% | 0.4% |
| echostar~maia (runtime blend) via chooser [balanced, band 50] | 30.0% | 8 | 0.0% | 0.5% |
| echostar (existing) via chooser [sharp, band 20] | 29.6% | 3 | 0.0% | 0.3% |
| echostar+maia (training teacher) via chooser [sharp, band 20] | 29.0% | 3 | 0.0% | 0.2% |
| echostar~maia (runtime blend) via chooser [sharp, band 20] | 29.2% | 3 | 0.0% | 0.3% |

Share of positions where the player's own move was inside each dial's safety band (the ceiling on how often the chooser can match the player): faithful 75%, balanced 63%, sharp 46%.
Average centipawns lost is a *proxy* for strength, not an Elo; playing strength was not measured.

## Resources

| item | value |
|---|---|
| python peak RSS (MB) | 320 |
| training time per model (s, mean) | 7.1 |
| Maia (lc0, BLAS CPU) median / p90 latency per position (ms) | 5.4 / 7.9 |
| Maia lc0 process RSS / peak (MB) | 123 / 123 |
| Maia weights file (MB) | 1.26 |
| lc0 binary (MB) | 2.8 |
| Stockfish MultiPV 8 at depth 8 (ms per position) | 31 |
| Stockfish process RSS / peak (MB) | 166 / 166 |
| Stockfish binary (MB) | 40.4 |
| .echo size, existing (MB) | 2.55 |
| .echo size, with Maia teacher (mix 0.25) (MB) | 2.55 |
| Maia probability cache (MB) | 8.42 |
