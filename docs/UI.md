# Web UI and commentary

```bash
python echo.py ui --style style.npz        # opens http://127.0.0.1:8765/
```

A local page (`ui/index.html`, `styles.css`, `app.js`; no build step, no outside requests) with a board, an opponent card, a chat feed where the bot comments on the game like a chess-site bot, and a
scoresheet. Click a piece, then a square. Needs Stockfish, like `play`.

- **Opponent, strength, color:** chosen on the page; they apply to the next new game. Strength uses the same three dials
  as `play` (faithful, balanced, sharp).
- **Commentary:** the bot reacts to captures, checks, castling, promotions, mate, and to swings in the evaluation
  (your blunders and your good moves). It takes one shallow engine check per move to know which is which. The lines live
  in `commentary.py`; add your own persona there.
- **Games are saved** to `games/` as PGN, as in `play`. Games against the plain Echo opponent carry your profile's player
  name, so `learn --resume` can learn from them. Games against any other opponent say "You" and are never labelled as the
  profile's player.
- **Terminal:** `python echo.py play --style style.npz --chat` gives the same commentary in the terminal.

## Practice mode

The **Practice** switch (next to the opponent's name) turns on coaching for the current game; it takes effect at once.

| sign | meaning |
|---|---|
| `!!` brilliant | a piece sacrifice that is as good as the engine's best and keeps a sound position |
| `!` great | the best move when the next-best is much worse (an only move) |
| star best | the engine's top choice |
| check excellent, good | gave up almost nothing (under 2, under 5 points of winning chance) |
| `B` book | the position lies on a named opening line (see below); not an engine verdict |
| arrow forced | the only legal move |
| `?!` inaccuracy, `?` mistake, `??` blunder | gave up 5, 10 and 15+ points of winning chance (Lichess's cut-offs) |

Signs appear on the square of the last move, next to moves in the scoresheet, and in a prompt after your own mistakes
with the better move and a **Take it back** button. **Hint** draws the engine's best move as a green arrow. Opening names
are shown above the board.

**Stepping through the game:** the arrow bar under the board (first, previous, next, back to the game), the left and
right arrow keys, Home and End, and a tap on any move in the scoresheet all show earlier positions without changing the
game. While you step back, a move that was an inaccuracy, mistake or blunder shows the better move as an arrow, the way
Lichess's analysis board does.

How it works and what it cannot do: every position gets one shallow engine search (depth 10, two lines) as you play, so
signs are only as good as that search; at that depth a few labels will be debatable. Brilliant and great are deliberately
hard to earn. "Book" comes from `data/openings.tsv` (see `data/README.md`), which covers named openings, not all theory.
Logic is in `review.py` and `openings.py`; it is pure and tested without an engine.

## Hidden extra

There is one hidden extra opponent that you unlock on the page. On a computer, type the code anywhere on the page. On a
phone, tap the opponent's portrait five times and enter the code in the box that opens.

Its model can come in either of two forms:

- `profiles/fischer.npz` (change with `--fischer-style`), a plain profile you trained yourself. `*.npz` is git-ignored.
- `data/echo-extra.bin`, the same profile sealed under the code (`python tools/seal.py profiles/fischer.npz
  data/echo-extra.bin`). This one is safe to commit: it is unreadable until the code is typed, and the server only
  decrypts it in memory at that moment. The plain file wins if both exist.

If neither exists, the opponent shows the training command instead of starting a game. It is a style model trained on that
player's games; it is not the player, and its lines are written for this program, not quotes. Train it as described in
[FISCHER.md](FISCHER.md).

## Safety notes

- The server binds to `127.0.0.1` only and refuses POST requests that carry another site's `Origin`, so a web page you
  have open elsewhere cannot make moves in your game.
- The unlock word is stored only as a SHA-256 hash, and the sealed model is encrypted under it. That keeps both out of a
  casual look at the repository; it is an easter egg, not access control. The code is short, so someone determined could
  still guess it offline.
- One game is kept in memory; restarting the server starts fresh.

## API (for tests and tinkering)

`GET /api/state`; `POST /api/new {bot, color, dial}`, `/api/move {uci}`, `/api/reply`, `/api/undo`, `/api/resign`,
`/api/practice {on}`, `/api/unlock {typed}`. `/api/move` plays your move and returns at once; `/api/reply` plays the bot's answer.
