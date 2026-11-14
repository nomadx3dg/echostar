# Architecture

## Design goal
Strength and style are separate dials. A strong engine keeps play correct; a style model steers *which* of the
near-equal moves gets played. Nothing in the style model can make Echo play a move the engine considers a blunder.

## Components

| Piece | Python (`echo.py`) | Rust (`rust/src/`) |
|---|---|---|
| Style features (21) | `features()` | `features.rs` |
| Board encoding, move index, book key | `encode()`, `move_index()`, `book_key()` | `features.rs` |
| Style net + scoring | `StyleNet`, `StyleBrain` | `model.rs` |
| Truth engine client | python-chess `SimpleEngine` | `engine.rs` (hand-written UCI client) |
| Chooser | `Echo.choose()` | `chooser.rs` |
| Terminal board | `render()` | `render.rs` |
| Game saving | `save_game()` | `pgn.rs` |
| UCI engine mode | n/a | `uci.rs` |
| Training | `learn()` | n/a (training stays in Python) |

## The style model
- **Input:** 768 units (12 piece types x 64 squares), always from the mover's point of view (black positions are mirrored).
- **Hidden layer:** ReLU, 128 units by default.
- **Output:** one logit per move unit (4168: from-to squares, plus separate units for knight, bishop and rook
  promotions, see [FORMAT.md](FORMAT.md)), plus a linear term over the 21 hand-made features.
- **Training:** softmax over the legal moves, cross-entropy on the move the player actually chose, plain SGD, the best epoch
  by held-out loss is kept.
- **Opening book:** counts of the moves the player chose in the first 16 plies; added to the score as `1.5 * ln(1 + count)`.

## Choosing a move
1. Stockfish analyses with MultiPV 8 (fixed depth, or a time budget) and returns the top moves with scores.
2. Safety band: keep moves within `band` centipawns of the best. The band shrinks to 30 cp when the position is clearly
   decided (more than 500 cp) and to 0 when there is a forced mate.
3. Rank the safe moves by style score. With temperature 0 the top one wins; otherwise sample without replacement,
   weighted by softmax(score / temperature).
4. Blunder gate: re-search the position after the pick, deeper (or with a time slice). If it scores worse than
   `best - band - 40` cp, try the next move. After 3 failed checks, play the engine's best move.

## Python and Rust must agree
The Rust engine only runs the model, so both implementations must produce identical features and scores.
`tools/make_golden.py` writes a small random model plus the expected values for ~940 moves; `rust/tests/parity.rs`
checks them. Change a feature in one place and the parity test fails until the other matches and the vectors are regenerated
(`make golden`).

## Files
- `style.npz`: Python training output (numpy arrays plus JSON metadata).
- `style.echo`: portable format for Rust, see [FORMAT.md](FORMAT.md).
- `games/*.pgn`: games played against Echo, readable by `learn --resume`.
