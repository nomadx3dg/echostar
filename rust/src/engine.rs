//! Minimal UCI client for the truth engine (Stockfish), plus a trait so the chooser can be tested with a fake.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub const MATE: i32 = 100_000;

#[derive(Clone, Copy, Debug)]
pub enum Limit {
    Depth(u32),
    MoveTimeMs(u64),
}

/// One analysed line, scored from the side to move's point of view.
#[derive(Clone, Debug)]
pub struct Line {
    pub mv: String, // UCI move, empty if unknown
    pub cp: i32,
}

pub trait Analyser {
    fn analyse(&mut self, fen: &str, limit: Limit, multipv: usize) -> Result<Vec<Line>, String>;
}

pub struct Engine {
    child: Child,
    stdin: ChildStdin,
    out: BufReader<ChildStdout>,
    multipv: usize,
}

impl Engine {
    pub fn start(path: &str, options: &[(&str, String)]) -> Result<Engine, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start engine '{path}': {e}"))?;
        let stdin = child.stdin.take().ok_or("no engine stdin")?;
        let out = BufReader::new(child.stdout.take().ok_or("no engine stdout")?);
        let mut e = Engine { child, stdin, out, multipv: 1 };
        e.send("uci")?;
        e.wait_for("uciok")?;
        for (name, value) in options {
            e.send(&format!("setoption name {name} value {value}"))?;
        }
        e.send("isready")?;
        e.wait_for("readyok")?;
        Ok(e)
    }

    fn send(&mut self, line: &str) -> Result<(), String> {
        writeln!(self.stdin, "{line}").map_err(|e| format!("engine write failed: {e}"))?;
        self.stdin.flush().map_err(|e| format!("engine write failed: {e}"))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let mut line = String::new();
        let n = self.out.read_line(&mut line).map_err(|e| format!("engine read failed: {e}"))?;
        if n == 0 {
            return Err("engine closed unexpectedly".into());
        }
        Ok(line.trim().to_string())
    }

    fn wait_for(&mut self, token: &str) -> Result<(), String> {
        loop {
            if self.read_line()? == token {
                return Ok(());
            }
        }
    }
}

/// Parse `info ... multipv K score cp X|mate Y ... pv m1 m2 ...` into (k, depth, cp, first move).
fn parse_info(line: &str) -> Option<(usize, u32, i32, String)> {
    let t: Vec<&str> = line.split_whitespace().collect();
    if t.first() != Some(&"info") || t.iter().any(|w| *w == "lowerbound" || *w == "upperbound") {
        return None;
    }
    let find = |key: &str| t.iter().position(|w| *w == key);
    let pv_at = find("pv")?;
    let depth = t.get(find("depth")? + 1)?.parse().ok()?;
    let k = find("multipv").and_then(|i| t.get(i + 1)?.parse().ok()).unwrap_or(1);
    let s = find("score")?;
    let value: i32 = t.get(s + 2)?.parse().ok()?;
    let cp = match *t.get(s + 1)? {
        "cp" => value,
        "mate" => {
            if value > 0 {
                MATE - value
            } else {
                -MATE - value
            }
        }
        _ => return None,
    };
    Some((k, depth, cp, t.get(pv_at + 1)?.to_string()))
}

impl Analyser for Engine {
    fn analyse(&mut self, fen: &str, limit: Limit, multipv: usize) -> Result<Vec<Line>, String> {
        if multipv != self.multipv {
            self.send(&format!("setoption name MultiPV value {multipv}"))?;
            self.multipv = multipv;
        }
        self.send(&format!("position fen {fen}"))?;
        match limit {
            Limit::Depth(d) => self.send(&format!("go depth {d}"))?,
            Limit::MoveTimeMs(ms) => self.send(&format!("go movetime {ms}"))?,
        }
        let mut best: BTreeMap<usize, (u32, i32, String)> = BTreeMap::new();
        loop {
            let line = self.read_line()?;
            if line.starts_with("bestmove") {
                break;
            }
            if let Some((k, depth, cp, mv)) = parse_info(&line) {
                if best.get(&k).is_none_or(|(d, _, _)| depth >= *d) {
                    best.insert(k, (depth, cp, mv));
                }
            }
        }
        Ok(best.into_values().map(|(_, cp, mv)| Line { mv, cp }).collect())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cp_and_mate_lines() {
        let (k, d, cp, mv) =
            parse_info("info depth 12 seldepth 18 multipv 2 score cp -34 nodes 1 pv e7e5 g1f3").unwrap();
        assert_eq!((k, d, cp, mv.as_str()), (2, 12, -34, "e7e5"));
        let (_, _, cp, _) = parse_info("info depth 9 multipv 1 score mate 3 pv d1h5").unwrap();
        assert_eq!(cp, MATE - 3);
        let (_, _, cp, _) = parse_info("info depth 9 multipv 1 score mate -2 pv a1a2").unwrap();
        assert_eq!(cp, -MATE + 2);
        assert!(parse_info("info depth 9 multipv 1 score cp 10 lowerbound pv e2e4").is_none());
        assert!(parse_info("bestmove e2e4").is_none());
    }
}
