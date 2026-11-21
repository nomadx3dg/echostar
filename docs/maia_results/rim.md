# EchoStar x Maia benchmark (SYNTHETIC DATA)

> **Synthetic games.** The player is a noisy engine with a planted habit (tools/synth_games.py). These numbers test the pipeline and compare systems on identical data; they say nothing about whether EchoStar or Maia works on a real person.

Player `SynthRim`: 200 games, 8725 positions with a real choice. Teacher: lc0:maia-1500.pb.gz. 3 random game-level splits (train/val/test games per split: [140, 30, 30]).

## Held-out move prediction (test games only)

| system | top-1 | top-3 | log-loss (nats) |
|---|---|---|---|
| random | 4.1% ± 0.2 | n/a | 3.383 ± 0.023 |
| maia | 19.3% ± 0.5 | 39.6% ± 0.5 | 3.350 ± 0.019 |
| echostar (existing) | 16.6% ± 0.9 | 34.7% ± 1.0 | 2.995 ± 0.019 |
| echostar+maia (training teacher) | 16.3% ± 1.1 | 35.0% ± 1.1 | 3.011 ± 0.023 |
| echostar~maia (runtime blend) | 19.2% ± 0.3 | 42.0% ± 1.7 | 2.805 ± 0.034 |
|   grid: echostar mix=0 | 16.6% ± 0.9 | 34.7% ± 1.0 | 2.995 ± 0.019 |
|   grid: echostar mix=0.25 | 16.3% ± 1.1 | 35.0% ± 1.1 | 3.011 ± 0.023 |
|   grid: echostar mix=0.5 | 15.6% ± 0.6 | 34.4% ± 0.5 | 3.044 ± 0.033 |
|   grid: echostar mix=0.75 | 14.2% ± 0.9 | 31.6% ± 1.1 | 3.143 ± 0.033 |
|   grid: blend mu=0 | 16.6% ± 0.9 | 34.7% ± 1.0 | 2.995 ± 0.019 |
|   grid: blend mu=0.25 | 19.2% ± 0.3 | 42.0% ± 1.7 | 2.805 ± 0.034 |
|   grid: blend mu=0.5 | 19.3% ± 0.6 | 42.5% ± 0.4 | 2.923 ± 0.045 |
|   grid: blend mu=1 | 19.0% ± 0.1 | 42.5% ± 0.4 | 3.657 ± 0.055 |
|   grid: blend mu=2 | 19.2% ± 0.2 | 42.0% ± 0.3 | 5.761 ± 0.057 |

± is the standard deviation over splits; test sets overlap between splits, so it understates uncertainty. Log-loss is the average of -ln(probability given to the move actually played); lower is better, random guessing is the first row. Stockfish-only has no probabilities, so its top-1 is in the next table.

## Engine-referenced behavior (split 0 test positions: 1294; Stockfish depth 8 to decide, depth 12 to judge)

| system | matches player's move | avg cp lost | blunders (>=200 cp) | >=100 cp lost |
|---|---|---|---|---|
| random legal move (raw) | 4.0% | 237 | 39.9% | 60.4% |
| stockfish-only | 23.4% | 0 | 0.0% | 0.0% |
| maia (raw argmax) | 19.1% | 35 | 4.5% | 10.9% |
| echostar (existing) (raw argmax) | 17.5% | 92 | 13.3% | 28.4% |
| echostar+maia (training teacher) (raw argmax) | 17.5% | 94 | 14.5% | 26.7% |
| echostar~maia (runtime blend) (raw argmax) | 19.3% | 61 | 8.8% | 18.2% |
| echostar (existing) via chooser [faithful, band 120] | 25.8% | 23 | 0.7% | 6.0% |
| echostar+maia (training teacher) via chooser [faithful, band 120] | 24.4% | 21 | 0.6% | 4.9% |
| echostar~maia (runtime blend) via chooser [faithful, band 120] | 24.2% | 18 | 0.6% | 4.2% |
| echostar (existing) via chooser [balanced, band 50] | 25.4% | 13 | 0.2% | 1.8% |
| echostar+maia (training teacher) via chooser [balanced, band 50] | 24.3% | 12 | 0.2% | 1.6% |
| echostar~maia (runtime blend) via chooser [balanced, band 50] | 24.0% | 11 | 0.3% | 1.5% |
| echostar (existing) via chooser [sharp, band 20] | 24.6% | 5 | 0.1% | 0.4% |
| echostar+maia (training teacher) via chooser [sharp, band 20] | 24.3% | 4 | 0.1% | 0.4% |
| echostar~maia (runtime blend) via chooser [sharp, band 20] | 24.0% | 4 | 0.2% | 0.3% |

Share of positions where the player's own move was inside each dial's safety band (the ceiling on how often the chooser can match the player): faithful 72%, balanced 57%, sharp 41%.
Average centipawns lost is a *proxy* for strength, not an Elo; playing strength was not measured.

## Resources

| item | value |
|---|---|
| python peak RSS (MB) | 311 |
| training time per model (s, mean) | 6.9 |
| Maia (lc0, BLAS CPU) median / p90 latency per position (ms) | 5.9 / 8.3 |
| Maia lc0 process RSS / peak (MB) | 123 / 123 |
| Maia weights file (MB) | 1.26 |
| lc0 binary (MB) | 2.8 |
| Stockfish MultiPV 8 at depth 8 (ms per position) | 34 |
| Stockfish process RSS / peak (MB) | 166 / 166 |
| Stockfish binary (MB) | 40.4 |
| .echo size, existing (MB) | 2.55 |
| .echo size, with Maia teacher (mix 0.25) (MB) | 2.55 |
| Maia probability cache (MB) | 7.92 |
