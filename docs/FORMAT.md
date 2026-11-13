# `.echo` model format (version 1)

Training happens in Python (`echo.py learn`); the Rust engine only runs the model. `python echo.py export style.npz --out style.echo`
converts a trained model into this dependency-free format.

All integers and floats are little-endian.

| Offset | Size | Content |
|---|---|---|
| 0 | 8 | magic bytes `ECHOSTYL` |
| 8 | 4 | `u32` format version (currently `1`) |
| 12 | 4 | `u32` length `N` of the JSON header |
| 16 | `N` | UTF-8 JSON header |
| 16+N | ... | five `f32` tensors, row-major, in the order below |

**JSON header:** `player`, `games`, `positions`, `played_rate`, `baseline_rate`, `book` (position key -> UCI move -> count),
`validation`, `feature_names` (must match the engine's feature list exactly), and
`dims` = `{"input": 768, "hidden": H, "moves": 4096, "features": 21}`.

**Tensors:** `W1` (768 x H), `b1` (H), `W2T` (4096 x H), `b2` (4096), `w` (21).

**Scoring a move** (everything from the mover's point of view; for black, squares are mirrored with `sq ^ 56` and piece colors swapped):

1. Input: for each piece, set unit `(role - 1 + (0 if own piece else 6)) * 64 + square`.
2. `h = relu(sum of W1 rows for active units + b1)`
3. For move `from * 64 + to`: `logit = W2T[move] . h + b2[move] + features(move) . w`
4. Style score = `logit + 1.5 * ln(1 + book_count)` where the book count comes from `book[position_key][uci]`.
   `position_key` is the first four fields of the FEN (board, turn, castling, en passant).

`rust/tests/` holds golden vectors generated from Python so both implementations can be checked against each other
(`python tools/make_golden.py`).
