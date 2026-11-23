# Experiment: training Echo on Bobby Fischer's games

First run on real (not synthetic) games, toward ROADMAP milestone M2. The games are **not included** in this repository;
use your own Fischer PGN and pass the name as it appears in the headers to `--player`. A large public source exists: the
Figshare "Chess Database" (A. L. Schaigorodsky, CC BY 4.0, https://figshare.com/articles/dataset/Chess_Database/4276523).
Credit it if you use games from it.

Data used: 833 unique games (836 records; 3 exact duplicates by move sequence removed), empty and `?`-only PGN tags
stripped, player name standardised to `Fischer, Robert James`; 33,981 Fischer positions with more than one legal move.

## What the built-in held-out score hides

`learn` holds out the **last 10% of games in file order** (here the latest games of the file), not a random 10%, and its
headline accuracy counts opening moves. Fischer repeated opening lines, so the headline (34%) is mostly the opening.
`learn` now also prints the same held-out accuracy **after the opening** (from move 9 on).

`tools/bench_fischer.py` measures this properly: seeded 5-fold cross-validation by game, opening and the rest reported
separately, scored the way the chooser scores moves (net logit plus book bonus, with the book built from training folds
only). It needs no Stockfish.

```bash
python tools/bench_fischer.py fischer.pgn --player "Fischer, Robert James" --cache fischer.pkl
```

## Results (5-fold CV by game, seed 0; positions from move 9 on, n = 27,320)

| setting | top-1 | top-3 | log-loss |
|---|---|---|---|
| old defaults: 12 epochs, lr 0.02, no early stopping | 23.5% | 43.0% | 3.458 |
| 3 epochs, lr 0.01 | 26.6% | 47.1% | 2.679 |
| 3 epochs, lr 0.01, opponent warm-up (2 epochs) | 27.0% | 48.2% | 2.663 |
| 3 epochs, lr 0.01, book depth 32 plies (instead of 16) | 27.2% | 47.5% | 2.660 |

Random guessing is about 5% top-1 and a log-loss near 3.5. The opening (first 8 moves) is 70% top-1 or higher in every
row: it is mostly memorised repertoire. The combination of all three changes was not benchmarked together.

End to end with the real `learn` command on its own held-out split (last 83 games, after the opening):

| command | exact move | top-3 | epochs run |
|---|---|---|---|
| old: `learn` with all 12 epochs | 28.2% | 48.9% | 12 |
| `--lr 0.01 --book-plies 32 --pretrain-opponents 2` | 28.3% | 49.5% | 7 (stopped early) |

## What changed in the code

- `learn --patience N` (default 3): stop when the held-out score has not improved for N epochs. `learn` already restored
  the best epoch, so this mostly saves time; the 12-epoch row above shows what happens without any early stopping.
- `learn --book-plies N` (default 16, unchanged): opening-book depth. 32 suits players who repeat long theory lines. Only
  exact repeated positions are affected. The runtime scoring formula is unchanged.
- `learn --pretrain-opponents EPOCHS` (default 0, off): warm up the net on the opponents' moves from the training games
  first. Only sensible when the opponents played well (a master against masters); do not use it on games against much
  weaker players.
- `learn` prints and stores `validation_after_book` next to `validation`.
- Model format and Rust runtime are untouched: no tensor shape or feature changed, and the new metadata key is ignored by
  older readers.

## Honest limits

- One seed, one dataset. Differences of about one point are within noise; the early-stopping effect is the only large one,
  and `learn` already captured most of it.
- After the opening, this style model tops out near 28% exact-move accuracy. It has 21 hand-made features and no notion
  of tactics, so it cannot tell why a move is good. The chooser relies on Stockfish for that. Closing the gap needs richer
  features (hanging pieces, exchange evaluation, threats), which changes the model format and the Rust parity vectors.
- Stockfish was not available where this was run, so playing strength and the "can you tell it from an engine" blind test
  are still unmeasured (ROADMAP M2, M4).
