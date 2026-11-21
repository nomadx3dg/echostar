# EchoStar x Maia benchmark (SYNTHETIC DATA)

> **Synthetic games.** The player is a noisy engine with a planted habit (tools/synth_games.py). These numbers test the pipeline and compare systems on identical data; they say nothing about whether EchoStar or Maia works on a real person.

Player `SynthQueenside`: 200 games, 8752 positions with a real choice. Teacher: lc0:maia-1500.pb.gz. 3 random game-level splits (train/val/test games per split: [140, 30, 30]).

## Held-out move prediction (test games only)

| system | top-1 | top-3 | log-loss (nats) |
|---|---|---|---|
| random | 4.4% ± 0.1 | n/a | 3.376 ± 0.014 |
| maia | 22.5% ± 0.7 | 47.0% ± 0.4 | 2.987 ± 0.022 |
| echostar (existing) | 18.9% ± 1.4 | 37.8% ± 1.6 | 2.953 ± 0.039 |
| echostar+maia (training teacher) | 18.3% ± 0.8 | 38.1% ± 1.9 | 2.951 ± 0.032 |
| echostar~maia (runtime blend) | 22.9% ± 1.0 | 46.2% ± 2.0 | 2.708 ± 0.025 |
|   grid: echostar mix=0 | 18.9% ± 1.4 | 37.8% ± 1.6 | 2.953 ± 0.039 |
|   grid: echostar mix=0.25 | 18.3% ± 0.8 | 38.1% ± 1.9 | 2.951 ± 0.032 |
|   grid: echostar mix=0.5 | 18.1% ± 0.9 | 37.3% ± 0.9 | 2.977 ± 0.034 |
|   grid: echostar mix=0.75 | 17.0% ± 1.1 | 34.6% ± 1.9 | 3.038 ± 0.032 |
|   grid: blend mu=0 | 18.9% ± 1.4 | 37.8% ± 1.6 | 2.953 ± 0.039 |
|   grid: blend mu=0.25 | 22.9% ± 1.0 | 46.2% ± 2.0 | 2.708 ± 0.025 |
|   grid: blend mu=0.5 | 24.0% ± 1.4 | 48.3% ± 1.3 | 2.763 ± 0.024 |
|   grid: blend mu=1 | 24.2% ± 0.5 | 49.1% ± 0.7 | 3.357 ± 0.038 |
|   grid: blend mu=2 | 23.5% ± 1.2 | 49.0% ± 0.6 | 5.170 ± 0.075 |

± is the standard deviation over splits; test sets overlap between splits, so it understates uncertainty. Log-loss is the average of -ln(probability given to the move actually played); lower is better, random guessing is the first row. Stockfish-only has no probabilities, so its top-1 is in the next table.

## Resources

| item | value |
|---|---|
| python peak RSS (MB) | 312 |
| training time per model (s, mean) | 2.4 |
| Maia (lc0, BLAS CPU) median / p90 latency per position (ms) | 5.7 / 7.9 |
| Maia lc0 process RSS / peak (MB) | 124 / 124 |
| Maia weights file (MB) | 1.26 |
| lc0 binary (MB) | 2.8 |
| .echo size, existing (MB) | 2.55 |
| .echo size, with Maia teacher (mix 0.25) (MB) | 2.55 |
| Maia probability cache (MB) | 8.15 |
