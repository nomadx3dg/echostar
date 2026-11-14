use echo_rs::chooser::{fen_of, Echo};
use echo_rs::engine::{Analyser, Engine, Limit};
use echo_rs::features::{book_key, from_to, FEATURE_NAMES};
use echo_rs::model::Model;
use echo_rs::uci::{build_echo, Config};
use echo_rs::{pgn, render, uci};
use shakmaty::san::{San, SanPlus};
use shakmaty::uci::UciMove;
use shakmaty::{Chess, Color, Move, Position};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

const HELP: &str = "echo-rs: a chess engine that plays your style, correctly.

USAGE
  echo-rs play    [--style style.echo] [--dial faithful|balanced|sharp] [--color white|black]
                  [--depth N | --movetime SECONDS] [--band CP] [--seed N] [--stockfish PATH]
                  [--threads N] [--hash MB] [--show] [--plain] [--save-dir DIR] [--no-save]
  echo-rs profile style.echo
  echo-rs uci     [--style style.echo] [--dial ...] [--stockfish PATH]   (run as a UCI engine)

Train in Python (`python echo.py learn ...`), then export with `python echo.py export style.npz --out style.echo`.";

struct Args {
    flags: HashMap<String, String>,
    switches: Vec<String>,
    positional: Vec<String>,
}

const SWITCHES: [&str; 4] = ["show", "plain", "no-save", "help"];
const VALUE_FLAGS: [&str; 11] =
    ["style", "stockfish", "dial", "band", "depth", "threads", "hash", "seed", "movetime", "color", "save-dir"];

/// Unknown options and flags without a value are errors: a typo like `--dail sharp` must not silently play on the
/// default dial.
fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut a = Args { flags: HashMap::new(), switches: vec![], positional: vec![] };
    let mut i = 0;
    while i < raw.len() {
        if let Some(name) = raw[i].strip_prefix("--") {
            if SWITCHES.contains(&name) {
                a.switches.push(name.to_string());
            } else if VALUE_FLAGS.contains(&name) {
                let v = raw.get(i + 1).ok_or_else(|| format!("--{name} needs a value"))?;
                a.flags.insert(name.to_string(), v.clone());
                i += 1;
            } else {
                return Err(format!("unknown option --{name} (see `echo-rs help`)"));
            }
        } else {
            a.positional.push(raw[i].clone());
        }
        i += 1;
    }
    Ok(a)
}

impl Args {
    fn get(&self, k: &str) -> Option<&String> {
        self.flags.get(k)
    }
    fn on(&self, k: &str) -> bool {
        self.switches.iter().any(|s| s == k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        match self.get(k) {
            Some(v) => v.parse().map(Some).map_err(|_| format!("--{k} needs a number, got '{v}'")),
            None => Ok(None),
        }
    }
}

fn config_from(a: &Args) -> Result<Config, String> {
    let defaults = Config::default();
    Ok(Config {
        style: a.get("style").cloned(),
        stockfish: a.get("stockfish").cloned(),
        dial: a.get("dial").cloned().unwrap_or(defaults.dial),
        band: a.num("band")?,
        depth: a.num("depth")?.unwrap_or(defaults.depth),
        threads: a.num("threads")?,
        hash: a.num("hash")?,
        seed: a.num("seed")?,
    })
}

fn print_profile(m: &Model) {
    println!("\nStyle profile for {} ({} games, {} positions)", m.player, m.games, m.positions);
    println!("Tendency = how much more (or less) you pick this kind of move than a random legal move.\n");
    let rate = |map: &HashMap<String, f64>, n: &str| map.get(n).copied().unwrap_or(0.0);
    let mut rows: Vec<(f64, &str)> = FEATURE_NAMES
        .iter()
        .map(|n| (rate(&m.played_rate, n) - rate(&m.baseline_rate, n), *n))
        .collect();
    rows.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));
    for (diff, n) in rows.iter().take(10) {
        println!(
            "  {} {:<24} you {:5.1}%  vs random {:5.1}%",
            if *diff > 0.0 { "+" } else { "-" },
            n,
            rate(&m.played_rate, n) * 100.0,
            rate(&m.baseline_rate, n) * 100.0
        );
    }
    println!("\n  Opening book: {} positions from your games", m.book.len());
    match &m.validation {
        Some(v) => {
            println!("\n  Accuracy on games it never trained on:");
            println!(
                "    predicts your exact move {:.1}% of the time (random guessing: {:.1}%)",
                v.top1 * 100.0,
                v.random_top1 * 100.0
            );
            println!("    your move is in its top 3 {:.1}% of the time", v.top3 * 100.0);
        }
        None => println!("\n  Accuracy: not measured (need 10+ games to hold some out)."),
    }
}

fn parse_move(pos: &Chess, text: &str) -> Option<Move> {
    if let Ok(san) = text.parse::<San>() {
        if let Ok(m) = san.to_move(pos) {
            return Some(m);
        }
    }
    text.parse::<UciMove>().ok()?.to_move(pos).ok()
}

fn play(a: &Args) -> Result<(), String> {
    let cfg = config_from(a)?;
    let movetime: Option<f64> = a.num("movetime")?;
    let human = if a.get("color").map(String::as_str) == Some("black") { Color::Black } else { Color::White };
    let color_output = !a.on("plain");
    let show = a.on("show");
    if cfg.style.is_none() {
        println!("No style file given: Echo will play as a plain engine.");
    }
    let mut echo: Echo<Engine> = build_echo(&cfg)?;
    let player = echo.brain.player();
    println!("Echo ({}, band {} cp) vs you. Moves in SAN (e4, Nf3) or UCI (e2e4).", cfg.dial, echo.dial.band);
    println!("Commands: undo, hint, resign, help, quit.\n");

    let mut pos = Chess::default();
    let mut history: Vec<(Chess, Move, String)> = vec![]; // position before the move, the move, its SAN
    let mut seen: HashMap<String, u32> = HashMap::new();
    *seen.entry(book_key(&pos)).or_insert(0) += 1;
    let mut result: Option<String> = None;
    let stdin = io::stdin();

    loop {
        let last = history.last().map(|(_, m, _)| from_to(m));
        println!("{}\n", render::render(&pos, human, last, color_output));
        if let Some(outcome) = pos.outcome() {
            result = Some(outcome.to_string());
            break;
        }
        if seen.values().any(|c| *c >= 3) || pos.halfmoves() >= 100 {
            println!("Draw claimed (repetition or fifty-move rule).");
            result = Some("1/2-1/2".into());
            break;
        }

        if pos.turn() == human {
            print!("Your move: ");
            io::stdout().flush().ok();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                break;
            }
            let text = line.trim();
            match text {
                "quit" | "q" => break,
                "help" => {
                    println!("Enter a move like e4, Nf3, O-O, e8=Q (or e2e4). undo = take back your last move,\nhint = ask the engine for its best move, resign = give up, quit = stop and save.\n");
                    continue;
                }
                "resign" => {
                    result = Some(if human == Color::White { "0-1" } else { "1-0" }.into());
                    println!("You resigned.");
                    break;
                }
                "hint" => {
                    let lines = echo.engine.analyse(&fen_of(&pos), Limit::Depth(cfg.depth.min(12)), 1)?;
                    if let Some(l) = lines.first() {
                        if let Some(m) = parse_move(&pos, &l.mv) {
                            println!(
                                "Engine suggests {} ({:+.2})\n",
                                SanPlus::from_move(pos.clone(), &m),
                                l.cp as f64 / 100.0
                            );
                        }
                    }
                    continue;
                }
                "undo" => {
                    if history.len() >= 2 {
                        for _ in 0..2 {
                            let (before, _, _) = history.pop().unwrap();
                            if let Some(c) = seen.get_mut(&book_key(&pos)) {
                                *c -= 1;
                            }
                            pos = before;
                        }
                    }
                    continue;
                }
                _ => {}
            }
            match parse_move(&pos, text) {
                Some(m) => {
                    let san = SanPlus::from_move(pos.clone(), &m).to_string();
                    let before = pos.clone();
                    pos.play_unchecked(&m);
                    *seen.entry(book_key(&pos)).or_insert(0) += 1;
                    history.push((before, m, san));
                }
                None => println!("Illegal or unreadable move, try again.\n"),
            }
        } else {
            match echo.choose(&pos, movetime) {
                Ok(m) => {
                    if show {
                        if let Some(note) = echo.note.take() {
                            println!("   {note}");
                        }
                    }
                    let san = SanPlus::from_move(pos.clone(), &m).to_string();
                    println!("Echo plays {san}\n");
                    let before = pos.clone();
                    pos.play_unchecked(&m);
                    *seen.entry(book_key(&pos)).or_insert(0) += 1;
                    history.push((before, m, san));
                }
                Err(e) => {
                    println!("The engine stopped responding ({e}). Saving the game so far.");
                    break;
                }
            }
        }
    }

    let result = result.unwrap_or_else(|| "*".into());
    println!("Result: {result}");
    if show && !echo.summary().is_empty() {
        println!("{}", echo.summary());
    }
    if !a.on("no-save") && !history.is_empty() {
        let sans: Vec<String> = history.iter().map(|(_, _, s)| s.clone()).collect();
        let (w, b) = if human == Color::White { (player.as_str(), "Echo") } else { ("Echo", player.as_str()) };
        let dir = a.get("save-dir").map(String::as_str).unwrap_or("games");
        println!("Game saved to {}", pgn::save_game(&sans, w, b, &result, dir)?);
    }
    Ok(())
}

fn real_main() -> Result<(), String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        println!("{HELP}");
        return Ok(());
    };
    let args = parse_args(&raw[1..])?;
    if args.on("help") {
        println!("{HELP}");
        return Ok(());
    }
    match cmd.as_str() {
        "play" => play(&args),
        "profile" => {
            let path = args.positional.first().ok_or("usage: echo-rs profile style.echo")?;
            print_profile(&Model::load(path)?);
            Ok(())
        }
        "uci" => uci::run(config_from(&args)?),
        "help" | "--help" | "-h" => {
            println!("{HELP}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{HELP}")),
    }
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_flags_switches_and_positionals() {
        let a = parse_args(&s(&["style.echo", "--dial", "sharp", "--show"])).unwrap();
        assert_eq!(a.positional, vec!["style.echo"]);
        assert_eq!(a.get("dial").map(String::as_str), Some("sharp"));
        assert!(a.on("show") && !a.on("plain"));
    }

    #[test]
    fn rejects_unknown_flags_and_missing_values() {
        assert!(parse_args(&s(&["--dail", "sharp"])).err().unwrap().contains("--dail"));
        assert!(parse_args(&s(&["--depth"])).err().unwrap().contains("needs a value"));
    }

    #[test]
    fn config_uses_defaults_and_overrides() {
        let cfg = config_from(&parse_args(&s(&["--band", "35"])).unwrap()).unwrap();
        assert_eq!((cfg.dial.as_str(), cfg.band, cfg.depth), ("balanced", Some(35), 14));
        assert!(config_from(&parse_args(&s(&["--depth", "deep"])).unwrap()).is_err());
    }
}
