# Technical audit

Scope: the repository as of commit `d04e24f`, plus the fixes in the first review series. Everything below was checked
by reading the code and, where a number is given, by running it. The architecture (Style Brain, Truth Engine, Chooser with
blunder gate) is sound and is kept as is.

## Architecture in one paragraph
A small net (768 inputs from the mover's view, 128 ReLU units, one logit per move unit) plus a linear term over 21
move-local features scores each legal move. Stockfish returns the top 8 moves; moves inside a centipawn band of the best
are "safe"; the safe move the brain likes most (or a temperature sample) is re-searched deeper and played if it holds up.
Training is plain SGD with a softmax over legal moves, in Python; the Rust port only runs the model and is checked against
Python with golden vectors.

## What the tests cover (before and after)
- Python: encoding and move-index symmetry, features, PGN filters, learn/resume round trip, profile export, chooser logic with
  a fake engine (band, gate, mate, gate budget), game saving, `.echo` layout, game download against mock servers.
- Rust: UCI parsing, time plan, chooser logic with a fake engine, PGN dates, board rendering, 968-move parity with Python.
- Not covered anywhere: a real Stockfish process (not installed in the audit environment), strength, and any real player's games.

## Baseline (measured in this environment)
| Measurement | Result |
|---|---|
| Python test suite before this work | 33 passed |
| `learn`, 60 random games, 12 epochs, 1789 positions | 2.7 s |
| Held-out top-1 on those games | 2.8 to 7.8% across epochs (random guessing 4.6%) |

The games are random, so there is no style to learn: the numbers only show the pipeline runs and that held-out loss rises
while training loss falls from epoch 2, which the best-epoch rule handles. They say nothing about real players. Stockfish
cost (MultiPV, gate, nodes per move) was **not measured**: Stockfish is not installed here (roadmap Phase 10).

## Findings

| # | Finding | Status |
|---|---|---|
| 1 | Move unit `from * 64 + to` gave all four promotion pieces one unit, so the model could only tell them apart through the `check` and `quiet_move` features (knight and bishop promotions were indistinguishable). Measured: `a7a8q/r/b/n` all mapped to unit 3128. | Fixed in profile v2, with explicit migration |
| 2 | Python and Rust disagreed on edge cases (no engine lines, mate in the gate, negative clocks). | Fixed (first series) |
| 3 | `moves_into_attack` looks at the position before the move: it ignores that the moving piece changes which pieces attack the square (x-rays, a vacated blocker) and ignores defenders, so a defended square counts as "attacked". | Open: needs a versioned feature set (changes scores) |
| 4 | `even_or_winning_trade` compares victim and mover values only; there is no recapture or exchange evaluation. | Open: same |
| 5 | Features are move-local. Nothing describes game phase, material balance, king safety or pawn structure, so the model cannot make personality depend on the situation. | Open: roadmap Phases 3 and 4 |
| 6 | The epoch with the best held-out loss is kept and that same held-out set is then reported, so the reported accuracy is slightly optimistic. Splitting is by game (no leakage between train and held-out), but there is no third set. | Open: Phase 7 |
| 7 | Held-out metrics are reported once there are 10 games, however few positions that is. | Open: Phase 7 |
| 8 | The blunder gate compares a deeper evaluation of the chosen move with the shallower evaluation of the best move, which biases the check by the depth difference. Fixing it needs measurements. | Open: Phase 5 and 10 |
| 9 | With no style file, `balanced` and `faithful` pick randomly among safe moves rather than the engine's best, which differs from the "plain engine" wording. | Open |
| 10 | Everything was validated on synthetic data only. | Open: roadmap M2 |

## Remaining roadmap phases not started
3 (personality dimensions), 4 (position-aware features), 5 (adaptive gate), 6 (feature corrections, see findings 3 and 4),
7 (metrics), 8 (coach), 9 (multi-profile management), 10 (benchmarks and self-play).
