# Roadmap

Goal: **Echo plays like you, correctly, on a phone-sized footprint, and is playable on Lichess.**

Status key: done / next / later. Each milestone has an exit test so "done" is checkable.

## M0. Python prototype (done)
Style net + hand-made features, safety band, blunder gate, terminal play, learning loop (`learn --resume`),
profile export, 23 tests.
**Exit test:** `python -m pytest` passes; a full game against Stockfish works.

## M1. Rust port (done)
`rust/` reproduces the Python scoring exactly (942 golden moves: features, encoding, move indices, logits, SAN),
talks to Stockfish over UCI, plays in the terminal, and runs as a UCI engine (`echo-rs uci`).
**Exit test:** `cargo test` passes, including `tests/parity.rs`.

## M2. Real data and measured accuracy (next)
Everything so far was tested on synthetic games. This milestone finds out whether the idea works on a real player.
- Export 300+ real games (Lichess/Chess.com PGN), train, read the held-out accuracy in `profile`.
- Compare against a baseline (random guessing is already reported; add "always the most common move type").
- Tune: epochs, learning rate, hidden size, feature set. Keep what improves held-out accuracy, drop what doesn't.
- Blind test: play 10 games against Echo; can you tell it from a generic engine at the same strength?

**Exit test:** held-out top-1 accuracy clearly above the baseline on at least two different players' games.
**Risk:** with a few hundred games the net may memorize. Fallback: lean on the 21 features, shrink the net.

## M3. Engine hardening (next)
Make `echo-rs uci` safe to run unattended.
- Honor `stop` and `ponderhit` (the search is currently synchronous).
- Time management: tune the clock formula in `uci::plan_time`; never flag on increment-only or sudden-death clocks.
- Recover if Stockfish dies mid-game (restart it, replay the position).
- Handle illegal or odd positions without crashing; always answer `bestmove`.
- Release build, binaries for Linux x86_64 / aarch64 (Raspberry Pi) and Termux.

**Exit test:** 200 games against Stockfish limited to a weak level in a GUI harness (for example cutechess-cli):
zero illegal moves, zero time forfeits, zero crashes.

## M4. Strength calibration (next)
Know how strong each dial really is, so the bot's rating label is honest.
- Play `faithful`, `balanced`, `sharp` against Stockfish at fixed Elo settings (`UCI_LimitStrength` / `UCI_Elo`).
- Record a rating estimate per dial and per time control; write it to `docs/STRENGTH.md`.

**Exit test:** a table of dial vs estimated Elo with the number of games behind each figure.

## M5. Lichess bot, local run (later)
Lichess supports bot accounts through its Bot API; `lichess-bot` is the standard bridge.
1. Create a **new** Lichess account for the bot. An account that has already played games cannot become a bot.
2. Create an API token with the `bot:play` scope. It is shown once, so store it safely (never commit it).
3. Upgrade the account to a BOT account. **This is irreversible.**
4. Point `lichess-bot` at Echo with the UCI protocol. In its `config.yml`, roughly:
   ```yaml
   token: "<your token, or set LICHESS_BOT_TOKEN>"
   engine:
     dir: "./engines/"
     name: "echo-rs"
     protocol: "uci"
     uci_options:
       StylePath: "/path/to/style.echo"
       StockfishPath: "/usr/games/stockfish"
       Dial: "balanced"
   ```
   Check the field names against the `config.yml.default` shipped with the current `lichess-bot` release.
5. Start with casual games only, one game at a time, slower time controls (rapid and up), and a restricted
   rating range for challengers.
6. Write the steps into `docs/LICHESS.md`.

**Exit test:** the bot accepts a challenge, plays a full game, and finishes without a crash, forfeit, or illegal move.
**Risks:** read Lichess's current bot rules before step 3. Name and profile must make clear it is a bot.
Stockfish is GPL: if you ship its binary together with Echo, pick a compatible license and comply with it.

## M6. Lichess bot, running unattended (later)
- Run on an always-on machine (a Raspberry Pi, a cheap VPS, or a spare phone in Termux) under a service manager
  (systemd) with automatic restart.
- Log every game; track rating and result per dial; alert on crashes.
- Rate limiting and queueing so several simultaneous challenges can't exhaust the CPU.

**Exit test:** 7 days of continuous uptime with no manual restarts.

## M7. Play-like-me products (later)
- **Play like me bot:** the bot is trained on one player's games (yours), under its own Lichess account.
- **Opponent prep:** download an opponent's public games, train a profile, practice against it.
- **Coach view in the app:** show your move vs the style-consistent best move vs the engine's best, with a plain
  explanation of the gap.
- Respect privacy: only train on games from the player who asks, or on public games for opponent prep.

## M8. A lighter, stronger Echo (later)
The original "Wisp" idea: reduce the dependence on a full Stockfish process.
- Distill a small quantized evaluator (1 to 2 MB int8) and a learned pruner; run the truth engine in Rust.
- Replace the hand-made features with a stronger, Maia-style move predictor conditioned on rating and clock time.
- Mobile app (Flutter or native) calling the Rust engine through FFI; WebAssembly build for the browser.

**Exit test:** equal-nodes or equal-energy comparison against Stockfish; see M4 for the method.

## Not planned
Using Echo to pick moves in your own live games on Lichess or Chess.com. That is engine assistance and gets
accounts banned. Echo is a separate opponent, never a co-pilot.
