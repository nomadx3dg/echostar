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
python echo.py profile style.npz
python echo.py play --style style.npz --dial balanced --color white --show
```

`--player` is your username exactly as it appears in the PGN `[White]`/`[Black]` headers.
Dials: `faithful` (most like you), `balanced`, `sharp` (near-perfect play, style as tiebreaker).
Use 60+ games for a useful style model; `echo_sample_training.pgn` is only a 2-game smoke test.
