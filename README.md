# echostar

Echo: a chess engine that learns how you play, then plays your style while staying correct.

- **Style Brain**: small neural net + hand-made style features trained on your PGN games
- **Truth Engine**: Stockfish (UCI) evaluates the top candidate moves
- **Chooser**: picks the most "you" move inside a safety band, then a blunder gate re-checks it

## Setup
```
pip install -r requirements.txt
# install Stockfish: apt install stockfish / brew install stockfish / pkg install stockfish (Termux)
```

## Use
```
python echo.py learn my_games.pgn --player YourName --out style.npz
python echo.py profile style.npz --export profile.md
python echo.py play --style style.npz --dial balanced --color white --show
```

`--player` is your username exactly as it appears in the PGN `[White]`/`[Black]` headers (case doesn't matter).

### Dials
| Dial | Safety band | Behavior |
|---|---|---|
| `faithful` | 120 cp | plays most like you, accepts small inaccuracies |
| `balanced` | 50 cp | your style with fewer slips |
| `sharp` | 20 cp | near-perfect play, style as tiebreaker |

The band tightens automatically in decided positions, and a forced mate is always taken.

### Learning options
```
--speed blitz,rapid        only train on these speeds (bullet, blitz, rapid, classical)
--min-opp-rating 1500      only train on games against opponents at least this rated
--resume style.npz         keep training an existing model with new games
```
Chess variants (Chess960, Crazyhouse, ...) are skipped automatically.

### Learning loop
`play` saves every game to `games/` as a PGN (use `--no-save` to disable). Feed them back in:
```
python echo.py learn games/*.pgn --player YourName --resume style.npz --out style.npz
```
(or concatenate several PGNs into one file first).

### Other options
`play` also takes `--depth`, `--threads`, `--hash`, `--plain` (ASCII board), and `--show` (coach view, plus an end-of-game summary of how much strength Echo traded for style). Type `undo` to take back a move.

## Accuracy
`learn` holds out 10% of your games and reports how often the model predicts your exact move, next to what random guessing would score. With fewer than ~60 games the model mostly memorizes; use more games for a meaningful profile. `echo_sample_training.pgn` is only a 2-game smoke test.

## Tests
```
pip install -r requirements-dev.txt
python -m pytest
```
The chooser tests use a fake engine, so Stockfish isn't needed to run them.
