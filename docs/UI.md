# Web UI and commentary

```bash
python echo.py ui --style style.npz        # opens http://127.0.0.1:8765/
```

A local page with a board, an opponent card, a chat feed where the bot comments on the game like a chess-site bot, and a
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

## Hidden extra

There is one hidden extra opponent that you unlock on the page. It plays from `profiles/fischer.npz` (change with
`--fischer-style`). That file is not in the repository (`*.npz` is ignored): train it from your own copy of a player's
games, see [FISCHER.md](FISCHER.md). Until it exists, the opponent shows a message with the training command instead of
starting a game. It is a style model trained on that player's games; it is not the player, and its lines are written for
this program, not quotes.

## Safety notes

- The server binds to `127.0.0.1` only and refuses POST requests that carry another site's `Origin`, so a web page you
  have open elsewhere cannot make moves in your game.
- The unlock word is stored only as a SHA-256 hash. That keeps it out of a casual search of the source; it is an easter egg,
  not access control.
- One game is kept in memory; restarting the server starts fresh.

## API (for tests and tinkering)

`GET /api/state`; `POST /api/new {bot, color, dial}`, `/api/move {uci}`, `/api/reply`, `/api/undo`, `/api/resign`,
`/api/unlock {typed}`. `/api/move` plays your move and returns at once; `/api/reply` plays the bot's answer.
