# EchoStar x Maia benchmark (SYNTHETIC DATA)

> **Synthetic games.** The player is a noisy engine with a planted habit (tools/synth_games.py). These numbers test the pipeline and compare systems on identical data; they say nothing about whether EchoStar or Maia works on a real person.

Player `SynthQueenside`: 200 games, 8752 positions with a real choice. Teacher: lc0:maia-1500.pb.gz. 3 random game-level splits (train/val/test games per split: [140, 30, 30]).

## Held-out move prediction (test games only)

| system | top-1 | top-3 | log-loss (nats) |
|---|---|---|---|
| random | 4.4% ± 0.1 | n/a | 3.376 ± 0.014 |
| maia | 22.5% ± 0.7 | 47.0% ± 0.4 | 2.987 ± 0.022 |
| echostar (existing) | 16.0% ± 2.0 | 35.7% ± 1.0 | 3.009 ± 0.030 |
| echostar+maia (training teacher) | 16.7% ± 0.8 | 35.4% ± 0.8 | 3.008 ± 0.026 |
| echostar~maia (runtime blend) | 22.4% ± 0.6 | 45.6% ± 0.8 | 2.682 ± 0.020 |
|   grid: echostar mix=0 | 16.0% ± 2.0 | 35.7% ± 1.0 | 3.009 ± 0.030 |
|   grid: echostar mix=0.25 | 16.7% ± 0.8 | 35.4% ± 0.8 | 3.008 ± 0.026 |
|   grid: echostar mix=0.5 | 15.8% ± 0.9 | 34.0% ± 0.4 | 3.034 ± 0.021 |
|   grid: echostar mix=0.75 | 15.0% ± 0.6 | 31.4% ± 0.5 | 3.093 ± 0.019 |
|   grid: blend mu=0 | 16.0% ± 2.0 | 35.7% ± 1.0 | 3.009 ± 0.030 |
|   grid: blend mu=0.25 | 22.4% ± 0.6 | 45.6% ± 0.8 | 2.682 ± 0.020 |
|   grid: blend mu=0.5 | 23.6% ± 0.5 | 48.1% ± 0.8 | 2.688 ± 0.041 |
|   grid: blend mu=1 | 23.6% ± 1.0 | 48.8% ± 0.9 | 3.254 ± 0.048 |
|   grid: blend mu=2 | 22.9% ± 0.9 | 48.4% ± 0.7 | 5.068 ± 0.051 |

± is the standard deviation over splits; test sets overlap between splits, so it understates uncertainty. Log-loss is the average of -ln(probability given to the move actually played); lower is better, random guessing is the first row. Stockfish-only has no probabilities, so its top-1 is in the next table.

## Resources

| item | value |
|---|---|
| python peak RSS (MB) | 311 |
| training time per model (s, mean) | 1.0 |
| Maia (lc0, BLAS CPU) median / p90 latency per position (ms) | 5.3 / 7.2 |
| Maia lc0 process RSS / peak (MB) | 122 / 122 |
| Maia weights file (MB) | 1.26 |
| lc0 binary (MB) | 2.8 |
| .echo size, existing (MB) | 2.55 |
| .echo size, with Maia teacher (mix 0.25) (MB) | 2.55 |
| Maia probability cache (MB) | 8.15 |
