//! Echo (Rust port): a chess engine that learns how you play, then plays your style correctly.
//!
//! * `model`    - loads the `.echo` style model exported by the Python trainer and scores moves
//! * `features` - hand-made style features, board encoding and book keys (must match `echo.py` exactly)
//! * `engine`   - minimal UCI client used to talk to Stockfish (the "truth engine")
//! * `chooser`  - safety band, style ranking and blunder gate
//! * `render`   - terminal board
//! * `pgn`      - saves games so the Python trainer can learn from them
//! * `uci`      - run Echo itself as a UCI engine (for GUIs and lichess-bot)

pub mod chooser;
pub mod engine;
pub mod features;
pub mod model;
pub mod pgn;
pub mod render;
pub mod uci;
