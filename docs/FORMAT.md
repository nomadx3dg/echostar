# `.echo` model format (version 2)

Training happens in Python (`echo.py learn`); the Rust engine only runs the model. `python echo.py export style.npz --out style.echo`
converts a trained model into this dependency-free format.

All integers and floats are little-endian.

| Offset | Size | Content |
|---|---|---|
| 0 | 8 | magic bytes `ECHOSTYL` |
| 8 | 4 | `u32` format version (currently `2`; `1` is still readable) |
| 12 | 4 | `u32` length `N` of the JSON header |
| 16 | `N` | UTF-8 JSON header |
| 16+N | ... | five `f32` tensors, row-major, in the order below |

**JSON header:** `player`, `games`, `positions`, `played_rate`, `baseline_rate`, `book` (position key -> UCI move -> count),
`validation`, `feature_names` (must match the engine's feature list exactly), and
`dims` = `{"input": 768, "hidden": H, "moves": 4168, "features": 21}` (`"moves": 4096` in a v1 file).

**Tensors:** `W1` (768 x H), `b1` (H), `W2T` (4168 x H), `b2` (4168), `w` (21). A v1 file has 4096 rows in `W2T` and `b2`.

**Scoring a move** (everything from the mover's point of view; for black, squares are mirrored with `sq ^ 56` and piece colors swapped):

1. Input: for each piece, set unit `(role - 1 + (0 if own piece else 6)) * 64 + square`.
2. `h = relu(sum of W1 rows for active units + b1)`
3. For the move's output unit `u` (see below): `logit = W2T[u] . h + b2[u] + features(move) . w`
4. Style score = `logit + 1.5 * ln(1 + book_count)` where the book count comes from `book[position_key][uci]`.
   `position_key` is the first four fields of the FEN (board, turn, castling, en passant).

`rust/tests/` holds golden vectors generated from Python so both implementations can be checked against each other
(`python tools/make_golden.py`).

## Move units

Units `0..4095` are `from * 64 + to` (mirrored for black). Queen promotions and every non-promotion move use these.
Under-promotions use 72 extra units: `4096 + piece * 24 + (drift + 1) * 8 + from_file`, where `piece` is 0 knight,
1 bishop, 2 rook, and `drift = to_file - from_file` is -1, 0 or +1 (a capture left, a push, a capture right). Files are
not affected by the mirroring used for black.

## Versions and migration

| Version | Move units | Notes |
|---|---|---|
| 1 | 4096 | all four promotion pieces shared the queen-promotion unit |
| 2 | 4168 | knight, bishop and rook promotions have their own units |

Migration v1 to v2 is explicit and lossless. Each new unit starts as a copy of the v1 unit it was split from (the queen
promotion for the same from/to squares), so a migrated model scores every move exactly as before; only more training can
tell the promotion pieces apart.

- **Python (`style.npz`):** the metadata field `profile_version` is 2 for files written by this build; files without the field are
  v1. `load_model` migrates in memory, prints a note to stderr, and sets `migrated_from` in the metadata. The file on disk
  is never rewritten by loading; `learn --resume` or `export` writes v2. A file with a newer `profile_version` than the build
  knows is refused.
- **Rust (`.echo`):** reads v1 and v2. A v1 file is migrated in memory the same way (`Model::migrated_from` is `Some(1)`,
  and `echo-rs profile` says so).
- **Older engine builds** cannot read v2. `export_echo(..., version=1)` writes the legacy layout (promotion units dropped).

`tools/make_golden.py` writes `golden.echo` (v2), `golden_v1.echo` (the same weights in the v1 layout) and the expected
scores for both, so `cargo test` checks the Rust migration against Python.

## Optional metadata: behavior dimensions

Both the Python profile (`profile_version` 2) and the `.echo` header can carry a `dimensions` block. It is additive: it
does not change any tensor, so it needs no new version, and readers that do not know it ignore it.

```json
"dimensions": {"version": 1, "games": 60,
               "tally": {"aggression": [[n, observed, expected, variance], [n, observed, expected, variance]], "...": []}}
```

Each dimension has two cells (alternate games go to alternate cells) so a reader can check that both halves agree. A cell is
`[positions, sum observed, sum expected, sum variance]`; cells add when profiles are resumed. `games` is how many games the
tally covers, which can be fewer than the profile's `games` after resuming an older profile. The definitions and how to
read z are in [PERSONALITY.md](PERSONALITY.md). A reader must treat a missing block, or a `version` it does not know, as
"not measured". `echo-rs` does so and never fails to load a model because of this block.

## Optional metadata: teacher provenance

A profile trained with the optional Maia teacher (`echo.py learn --maia-mix W ...`, see [MAIA.md](MAIA.md)) records where its
soft training targets came from. This is provenance only: no tensor, feature or dimension changed, so the format version stays
`2`, a taught model has exactly the same size as an untaught one, and no new reader code is needed.

```json
"teacher": {"version": 1, "name": "lc0:maia-1500.pb.gz", "elo": 1500, "mix": 0.25, "games": 60}
```

`mix` is the weight of the teacher in the training target (`1 - mix` stays on the move the player actually made). The block is
absent for models trained without a teacher, and is kept when such a model is later resumed without one. A reader must ignore
it if missing or unknown; `echo-rs` does (`rust/tests/teacher_header.rs`). It contains no Maia weights or probabilities.

