//! Run Echo itself as a UCI engine, so chess GUIs and `lichess-bot` can use it.
//!
//! Echo asks Stockfish (the truth engine) for evaluations internally, so Stockfish must be installed.
//! Options: StylePath, StockfishPath, Dial, Band, Depth, Threads, Hash.

use crate::chooser::{Dial, Echo, Rng};
use crate::engine::Engine;
use crate::model::{Brain, Model};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Position};
use std::io::{self, BufRead, Write};

#[derive(Clone, Debug)]
pub struct Config {
    pub style: Option<String>,
    pub stockfish: Option<String>,
    pub dial: String,
    pub band: Option<i32>,
    pub depth: u32,
    pub threads: Option<u32>,
    pub hash: Option<u32>,
    pub seed: Option<u64>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            style: None,
            stockfish: None,
            dial: "balanced".into(),
            band: None,
            depth: 14,
            threads: None,
            hash: None,
            seed: None,
        }
    }
}

pub fn find_in_path(name: &str) -> Option<String> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
        .map(|p| p.display().to_string())
}

pub fn build_echo(cfg: &Config) -> Result<Echo<Engine>, String> {
    let brain = match &cfg.style {
        Some(path) => Brain::with_model(Model::load(path)?),
        None => Brain::none(),
    };
    let sf = cfg
        .stockfish
        .clone()
        .or_else(|| find_in_path("stockfish"))
        .ok_or("Stockfish not found. Install it or pass --stockfish /path/to/stockfish")?;
    let mut opts: Vec<(&str, String)> = vec![];
    if let Some(t) = cfg.threads {
        opts.push(("Threads", t.to_string()));
    }
    if let Some(h) = cfg.hash {
        opts.push(("Hash", h.to_string()));
    }
    let engine = Engine::start(&sf, &opts)?;
    let mut dial = Dial::named(&cfg.dial).ok_or_else(|| format!("unknown dial '{}'", cfg.dial))?;
    if let Some(b) = cfg.band {
        dial.band = b;
    }
    let rng = cfg.seed.map(Rng::new).unwrap_or_else(Rng::from_clock);
    Ok(Echo::new(engine, brain, dial, cfg.depth, rng))
}

#[derive(Default, Debug, PartialEq)]
pub struct Go {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u64>,
    pub movetime: Option<u64>,
    pub depth: Option<u32>,
}

pub fn parse_go(tokens: &[&str]) -> Go {
    let mut g = Go::default();
    let mut i = 0;
    while i < tokens.len() {
        let val = tokens.get(i + 1).and_then(|v| v.parse::<u64>().ok());
        match tokens[i] {
            "wtime" => g.wtime = val,
            "btime" => g.btime = val,
            "winc" => g.winc = val,
            "binc" => g.binc = val,
            "movestogo" => g.movestogo = val,
            "movetime" => g.movetime = val,
            "depth" => g.depth = val.map(|v| v as u32),
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    g
}

/// Seconds Echo may spend on this move, or `None` to search to a fixed depth instead.
/// Worst case the chooser uses ~95% of the budget (candidates + three gate checks), so we plan with 80%.
pub fn plan_time(g: &Go, side: Color) -> Option<f64> {
    if let Some(ms) = g.movetime {
        return Some((ms as f64 / 1000.0 * 0.8).max(0.05));
    }
    let (time, inc) = if side == Color::White { (g.wtime, g.winc) } else { (g.btime, g.binc) };
    let t = time? as f64 / 1000.0;
    let inc = inc.unwrap_or(0) as f64 / 1000.0;
    let moves = g.movestogo.unwrap_or(30).max(1) as f64;
    let budget = (t / moves + inc * 0.7).min(t * 0.25);
    Some((budget * 0.8).max(0.05))
}

pub fn parse_position(tokens: &[&str]) -> Result<Chess, String> {
    let i;
    let mut pos = Chess::default();
    match tokens.first() {
        Some(&"startpos") => i = 1,
        Some(&"fen") => {
            let end = tokens.iter().position(|t| *t == "moves").unwrap_or(tokens.len());
            let fen: Fen = tokens[1..end].join(" ").parse().map_err(|e| format!("bad FEN: {e}"))?;
            pos = fen.into_position(CastlingMode::Standard).map_err(|e| format!("bad position: {e}"))?;
            i = end;
        }
        _ => return Err("position needs startpos or fen".into()),
    }
    if tokens.get(i) == Some(&"moves") {
        for t in &tokens[i + 1..] {
            let uci: UciMove = t.parse().map_err(|_| format!("bad move '{t}'"))?;
            let m = uci.to_move(&pos).map_err(|_| format!("illegal move '{t}'"))?;
            pos.play_unchecked(&m);
        }
    }
    Ok(pos)
}

pub fn run(mut cfg: Config) -> Result<(), String> {
    let stdin = io::stdin();
    let mut out = io::stdout();
    let mut say = |s: &str| {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    };
    let mut pos = Chess::default();
    let mut echo: Option<Echo<Engine>> = None;

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("uci") => {
                say("id name Echo 0.1");
                say("id author nomadx3dg");
                say("option name StylePath type string default <empty>");
                say("option name StockfishPath type string default <empty>");
                say("option name Dial type combo default balanced var faithful var balanced var sharp");
                say("option name Band type spin default -1 min -1 max 500");
                say("option name Depth type spin default 14 min 1 max 40");
                say("option name Threads type spin default 1 min 1 max 512");
                say("option name Hash type spin default 16 min 1 max 33554432");
                say("uciok");
            }
            Some("isready") => {
                if echo.is_none() {
                    match build_echo(&cfg) {
                        Ok(e) => echo = Some(e),
                        Err(e) => say(&format!("info string error: {e}")),
                    }
                }
                say("readyok");
            }
            Some("setoption") => {
                let name_at = tokens.iter().position(|t| *t == "name");
                let value_at = tokens.iter().position(|t| *t == "value");
                if let Some(n) = name_at {
                    let name = tokens[n + 1..value_at.unwrap_or(tokens.len())].join(" ");
                    let value = value_at.map(|v| tokens[v + 1..].join(" ")).unwrap_or_default();
                    let empty = value.is_empty() || value == "<empty>";
                    match name.as_str() {
                        "StylePath" => cfg.style = if empty { None } else { Some(value) },
                        "StockfishPath" => cfg.stockfish = if empty { None } else { Some(value) },
                        "Dial" => cfg.dial = value,
                        "Band" => cfg.band = value.parse::<i32>().ok().filter(|b| *b >= 0),
                        "Depth" => cfg.depth = value.parse().unwrap_or(cfg.depth),
                        "Threads" => cfg.threads = value.parse().ok(),
                        "Hash" => cfg.hash = value.parse().ok(),
                        _ => {} // options meant for other engines (Move Overhead, Ponder, ...) are ignored
                    }
                    echo = None; // rebuild with the new settings at the next isready/go
                }
            }
            Some("ucinewgame") => pos = Chess::default(),
            Some("position") => match parse_position(&tokens[1..]) {
                Ok(p) => pos = p,
                Err(e) => say(&format!("info string error: {e}")),
            },
            Some("go") => {
                if echo.is_none() {
                    match build_echo(&cfg) {
                        Ok(e) => echo = Some(e),
                        Err(e) => say(&format!("info string error: {e}")),
                    }
                }
                let go = parse_go(&tokens[1..]);
                let reply = match echo.as_mut() {
                    Some(e) => {
                        let saved_depth = e.depth;
                        if let Some(d) = go.depth {
                            e.depth = d;
                        }
                        let movetime = plan_time(&go, pos.turn());
                        let result = e.choose(&pos, movetime);
                        e.depth = saved_depth;
                        match result {
                            Ok(m) => {
                                if let Some(note) = e.note.take() {
                                    say(&format!("info string {note}"));
                                }
                                UciMove::from_standard(&m).to_string()
                            }
                            Err(err) => {
                                say(&format!("info string error: {err}"));
                                "0000".into()
                            }
                        }
                    }
                    None => "0000".into(),
                };
                say(&format!("bestmove {reply}"));
            }
            Some("quit") => break,
            _ => {} // stop, ponderhit, debug, unknown: nothing to do (search is synchronous)
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_go_commands() {
        let g = parse_go(&["wtime", "60000", "btime", "58000", "winc", "1000", "binc", "1000"]);
        assert_eq!((g.wtime, g.btime, g.winc, g.binc), (Some(60000), Some(58000), Some(1000), Some(1000)));
        assert_eq!(parse_go(&["movetime", "500"]).movetime, Some(500));
        assert_eq!(parse_go(&["depth", "9"]).depth, Some(9));
        assert_eq!(parse_go(&["infinite"]), Go::default());
    }

    #[test]
    fn time_plan_is_sane() {
        let g = parse_go(&["wtime", "60000", "btime", "60000", "winc", "0", "binc", "0"]);
        let t = plan_time(&g, Color::White).unwrap();
        assert!(t > 1.0 && t < 3.0, "60s clock should give ~1.6s, got {t}");
        let low = parse_go(&["wtime", "400", "btime", "400"]);
        assert!(plan_time(&low, Color::White).unwrap() <= 0.1);
        assert_eq!(plan_time(&parse_go(&["depth", "8"]), Color::White), None);
        assert!((plan_time(&parse_go(&["movetime", "1000"]), Color::Black).unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn parses_positions() {
        let pos = parse_position(&["startpos", "moves", "e2e4", "e7e5", "g1f3"]).unwrap();
        assert_eq!(pos.turn(), Color::Black);
        let fen = parse_position(&["fen", "4k3/8/8/8/8/8/8/4K2R", "w", "K", "-", "0", "1", "moves", "e1g1"]).unwrap();
        assert_eq!(fen.turn(), Color::Black);
        assert!(parse_position(&["startpos", "moves", "e2e5"]).is_err());
    }
}
