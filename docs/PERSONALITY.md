# Behavior dimensions

Phase 3 of the roadmap. EchoStar now measures seven behavior dimensions from the moves a player actually chose. They are
shown by `python echo.py profile` (and `echo-rs profile`) and stored in the profile. The learned move-selection model is
unchanged and remains the thing that plays; the dimensions describe the player, they do not steer the model.

## How a dimension is measured

No dimension is a hand-set score. Each has a yes/no **signal** on a single move, and a **scope** (the moves, and the
positions, where the signal is meaningful). At every position where the player chose a move in scope, the choice is compared
with a uniformly random choice among the legal moves in scope:

- observed `O` = signal of the chosen move
- expected `E` = mean signal over the in-scope moves
- variance `V` = variance of the signal over the in-scope moves

Over all positions, **z = (O - E) / sqrt(V)** (sums over positions). It is about 0 for a player who picks at random, positive
when the player picks the property more often than chance would, negative when less. Positions where every in-scope move
has the same signal carry no information and are skipped. The profile also shows the rates: how often the player did it
against how often a random mover would have.

| Dimension | Signal (what is measured) | Scope |
|---|---|---|
| Aggression | plays a capture or a check | all moves |
| Risk tolerance | after the move, the moved piece can be taken for free or by a cheaper piece (computed on the position *after* the move, defenders counted) | non-captures, not the king |
| Tactical appetite | the moved piece newly attacks an enemy piece that is undefended or worth more | non-captures |
| Simplification preference | captures a piece of equal value (piece for piece) | all moves |
| Initiative preference | advances a pawn to its 4th rank or beyond without capturing (space gaining; a proxy) | all moves |
| Positional preference | castles, develops a minor piece from the back rank, or occupies d4/e4/d5/e5 | quiet moves (no capture, no check) |
| Endgame king activity | moves the king closer to the centre | endgames only (non-pawn material <= 24; the start position has 62) |

**Reading a result.** The verdict needs at least 50 informative positions. `clear` means |z| >= 3 *and* both halves of the
games (alternate games) point the same way as the whole; `weak` means |z| >= 2 with the same consistency; otherwise "not
distinguishable from random". The threshold is strict on purpose: there are seven dimensions, so some will cross |z| >= 2 by
chance.

## What the numbers do and do not mean

- **The reference is a random legal move, not a typical player.** Almost any sensible player hangs pieces less than a random
  mover, so a negative risk z is expected of everyone. Compare two profiles with each other, not one profile with zero.
- **The signals are move-local proxies.** They cannot see recaptures or longer tactics. "Simplification" means offering a
  piece-for-piece capture, not that a trade happened. "Initiative" is space-gaining pawn advances, not a full notion of
  initiative.
- **No context yet.** Each number averages over every kind of position, so "aggressive when ahead, defensive when behind"
  is invisible. That is roadmap Phase 4.
- **The dimensions overlap** (see cross-talk below). They are not independent axes.
- **Simplification needs a lot of data.** Equal-piece captures are rare, so it often reports "too little data".
- **Descriptive, not predictive.** The dimensions use all of a player's games; they are not a held-out accuracy claim.

## Does it measure what it claims? (validation)

Real players have no ground truth for "aggression", so the check uses synthetic players whose preference is **known
because it was planted**: a planted player picks each move with probability proportional to `exp(beta * signal)` for one
dimension and uniformly otherwise; the opponent plays uniformly. Games alternate between the standard start and ten endgame
positions so the endgame dimension gets data. Reproduce with:

```
python tools/validate_dimensions.py --seed 7        # about 40 seconds
```

Results of that exact command (100 random players of 12 games each; 126 planted players of 14 games each):

### 1. Null calibration (players who choose uniformly at random)

100 random players. A calibrated z has mean 0, sd 1, |z|>=2 about 4.6%, |z|>=3 about 0.3%.

| Dimension | players | mean z | sd z | share with a weak or clear verdict | share with a clear verdict |
|---|---|---|---|---|---|
| aggression | 100 | +0.10 | 0.95 | 1.0% | 0.0% |
| risk_tolerance | 100 | -0.05 | 1.09 | 7.0% | 0.0% |
| tactical_appetite | 100 | +0.07 | 1.03 | 6.0% | 0.0% |
| simplification | 10 | -0.40 | 0.81 | 0.0% | 0.0% |
| initiative | 100 | +0.08 | 0.98 | 6.0% | 0.0% |
| positional | 100 | +0.11 | 1.03 | 4.0% | 1.0% |
| endgame_activity | 100 | +0.00 | 0.92 | 3.0% | 0.0% |

Random players given at least one 'clear' verdict on any dimension: 1 of 100.

### 2. Direction and dose response (planted preference of known strength)

beta > 0 = prefers the behavior, beta < 0 = avoids it. `z(+2)` / `z(-2)` are the mean z of the planted dimension at the extreme betas. Spearman is between beta and the measured effect over all planted players; split-half is the correlation between the effect measured on two halves of each player's games.

| Dimension | z at beta=+2 | z at beta=-2 | Spearman(beta, effect) | split-half r |
|---|---|---|---|---|
| aggression | +14.9 | -4.8 | +0.98 | +0.95 |
| risk_tolerance | +12.6 | -5.1 | +0.98 | +0.97 |
| tactical_appetite | +13.6 | -5.2 | +0.98 | +0.95 |
| simplification | +6.8 | -1.6 | +0.86 | +0.71 |
| initiative | +14.4 | -7.0 | +0.99 | +0.99 |
| positional | +13.4 | -6.9 | +1.00 | +0.95 |
| endgame_activity | +11.0 | -4.2 | +0.97 | +0.95 |

### 3. Cross-talk (mean z of each dimension, when one dimension is planted at beta=+2)

| planted \ measured | aggression | risk_tolerance | tactical_appetite | simplification | initiative | positional | endgame_activity |
|---|---|---|---|---|---|---|---|
| **aggression** | **+14.9** | +1.5 | -0.4 | +3.9 | -1.7 | -0.6 | -0.1 |
| **risk_tolerance** | -0.7 | **+12.6** | +3.2 | -1.0 | +3.8 | +0.5 | -0.5 |
| **tactical_appetite** | -1.2 | +3.1 | **+13.6** | -0.6 | +0.5 | +1.5 | -0.3 |
| **simplification** | +1.8 | -0.1 | -1.0 | **+6.8** | -1.1 | +0.5 | +0.4 |
| **initiative** | -1.0 | +0.3 | -0.1 | +0.2 | **+14.4** | -0.7 | -0.4 |
| **positional** | -2.2 | +1.3 | -1.2 | -0.3 | +1.2 | **+13.4** | +1.5 |
| **endgame_activity** | -0.2 | -0.3 | +1.2 | -0.5 | -1.1 | +2.8 | **+11.0** |

### What these results show, and what they do not

- **Calibrated.** Random players have mean z near 0 and sd 0.9 to 1.1. The share given a `weak` or `clear` verdict by chance
  is 0 to 7% per dimension, and `clear` 0 to 1%; one random player in 100 got a clear verdict somewhere, about what seven
  dimensions at |z| >= 3 predict.
- **Right direction, graded response.** Every dimension moves the right way when a preference is planted, and the measured
  effect rises with the planted strength (Spearman 0.86 to 1.00). Avoiding a behavior is detected more weakly than seeking it
  (smaller |z| at beta = -2): a behavior that is already rare has little room to go lower.
- **Reliable.** The effect measured on two halves of a player's games agrees (split-half r 0.95 to 0.99), except
  simplification (0.71) because it has little data per player.
- **Overlap is real.** Planting aggression also lifts simplification (z +3.9, because piece trades are captures). Planting
  risk also lifts tactical appetite and initiative (z +3.2 and +3.8: pieces that step into danger often attack something or
  advance). Planting endgame king activity lifts positional (z +2.8: both reward central squares). Treat each dimension as a
  separate reading of the same moves, not as independent personality factors.
- **Not shown.** This proves the statistic is calibrated and responds to a planted single-feature preference. It does **not**
  show that a real person's "aggression" matches the label, or that the signals separate real players. That needs real games
  from players with an independent label, which this repository does not have.

## Profile storage

The measurements are stored in the profile metadata under `dimensions` (see [FORMAT.md](FORMAT.md)). The model weights and
profile version are unchanged, so this needs no migration. Older profiles have no `dimensions`; `profile` says so, and
`learn --resume` adds measurements for the newly learned games only (and says that they cover only those games).

## Cost

Measuring adds one pass over the legal moves of each position. On 60 random games (1789 positions) `learn` went from 2.7 s to
4.5 s on this machine, about 1 ms per position. Stockfish was not involved.
