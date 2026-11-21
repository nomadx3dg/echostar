# EchoStar x Maia benchmark (SYNTHETIC DATA)

> **Synthetic games.** The player is a noisy engine with a planted habit (tools/synth_games.py). These numbers test the pipeline and compare systems on identical data; they say nothing about whether EchoStar or Maia works on a real person.

Player `SynthQueenside`: 200 games, 8752 positions with a real choice. Teacher: lc0:maia-1500.pb.gz. 3 random game-level splits (train/val/test games per split: [140, 30, 30]).

## Held-out move prediction (test games only)

| system | top-1 | top-3 | log-loss (nats) |
|---|---|---|---|
| random | 4.4% ± 0.1 | n/a | 3.376 ± 0.014 |
| maia | 22.5% ± 0.7 | 47.0% ± 0.4 | 2.987 ± 0.022 |
| echostar (existing) | 19.9% ± 1.3 | 39.8% ± 0.8 | 2.886 ± 0.025 |
| echostar+maia (training teacher) | 19.6% ± 0.6 | 39.7% ± 1.5 | 2.887 ± 0.025 |
| echostar~maia (runtime blend) | 23.9% ± 0.9 | 47.8% ± 0.8 | 2.649 ± 0.030 |
|   grid: echostar mix=0 | 19.9% ± 1.3 | 39.8% ± 0.8 | 2.886 ± 0.025 |
|   grid: echostar mix=0.25 | 19.6% ± 0.6 | 39.7% ± 1.5 | 2.887 ± 0.025 |
|   grid: echostar mix=0.5 | 18.7% ± 0.8 | 38.8% ± 0.9 | 2.917 ± 0.021 |
|   grid: echostar mix=0.75 | 17.5% ± 0.7 | 37.8% ± 1.5 | 2.991 ± 0.021 |
|   grid: blend mu=0 | 19.9% ± 1.3 | 39.8% ± 0.8 | 2.886 ± 0.025 |
|   grid: blend mu=0.25 | 23.9% ± 0.9 | 47.8% ± 0.8 | 2.649 ± 0.030 |
|   grid: blend mu=0.5 | 24.6% ± 0.6 | 49.9% ± 0.7 | 2.707 ± 0.035 |
|   grid: blend mu=1 | 24.6% ± 0.6 | 50.0% ± 0.4 | 3.302 ± 0.047 |
|   grid: blend mu=2 | 24.3% ± 0.6 | 49.4% ± 0.5 | 5.115 ± 0.076 |

± is the standard deviation over splits; test sets overlap between splits, so it understates uncertainty. Log-loss is the average of -ln(probability given to the move actually played); lower is better, random guessing is the first row. Stockfish-only has no probabilities, so its top-1 is in the next table.

## Engine-referenced behavior (split 0 test positions: 1337; Stockfish depth 8 to decide, depth 12 to judge)

| system | matches player's move | avg cp lost | blunders (>=200 cp) | >=100 cp lost |
|---|---|---|---|---|
| random legal move (raw) | 5.3% | 259 | 42.1% | 59.2% |
| stockfish-only | 27.0% | 0 | 0.0% | 0.0% |
| maia (raw argmax) | 21.7% | 45 | 6.4% | 13.5% |
| echostar (existing) (raw argmax) | 18.8% | 115 | 19.0% | 34.7% |
| echostar+maia (training teacher) (raw argmax) | 18.9% | 118 | 20.5% | 32.6% |
| echostar~maia (runtime blend) (raw argmax) | 23.1% | 79 | 12.6% | 23.9% |
| echostar (existing) via chooser [faithful, band 120] | 29.9% | 22 | 0.4% | 6.2% |
| echostar+maia (training teacher) via chooser [faithful, band 120] | 28.9% | 20 | 0.5% | 5.1% |
| echostar~maia (runtime blend) via chooser [faithful, band 120] | 30.3% | 18 | 0.4% | 4.7% |
| echostar (existing) via chooser [balanced, band 50] | 29.4% | 10 | 0.0% | 1.0% |
| echostar+maia (training teacher) via chooser [balanced, band 50] | 27.6% | 8 | 0.1% | 0.7% |
| echostar~maia (runtime blend) via chooser [balanced, band 50] | 29.1% | 8 | 0.0% | 0.8% |
| echostar (existing) via chooser [sharp, band 20] | 28.9% | 4 | 0.0% | 0.3% |
| echostar+maia (training teacher) via chooser [sharp, band 20] | 28.3% | 3 | 0.0% | 0.2% |
| echostar~maia (runtime blend) via chooser [sharp, band 20] | 28.8% | 3 | 0.0% | 0.1% |

Share of positions where the player's own move was inside each dial's safety band (the ceiling on how often the chooser can match the player): faithful 76%, balanced 57%, sharp 41%.
Average centipawns lost is a *proxy* for strength, not an Elo; playing strength was not measured.

## Resources

| item | value |
|---|---|
| python peak RSS (MB) | 319 |
| training time per model (s, mean) | 6.9 |
| Maia (lc0, BLAS CPU) median / p90 latency per position (ms) | 5.9 / 7.8 |
| Maia lc0 process RSS / peak (MB) | 124 / 124 |
| Maia weights file (MB) | 1.26 |
| lc0 binary (MB) | 2.8 |
| Stockfish MultiPV 8 at depth 8 (ms per position) | 29 |
| Stockfish process RSS / peak (MB) | 166 / 166 |
| Stockfish binary (MB) | 40.4 |
| .echo size, existing (MB) | 2.55 |
| .echo size, with Maia teacher (mix 0.25) (MB) | 2.55 |
| Maia probability cache (MB) | 8.15 |
