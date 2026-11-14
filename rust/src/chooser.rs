//! The chooser: keep moves inside the safety band, rank them by style, re-check the pick with a deeper search.

use crate::engine::{Analyser, Limit, MATE};
use crate::model::Brain;
use shakmaty::fen::Fen;
use shakmaty::san::SanPlus;
use shakmaty::uci::UciMove;
use shakmaty::{Chess, EnPassantMode, Move, Position};
use std::collections::HashMap;

/// Extra centipawns of tolerance in the deeper blunder check.
pub const GATE_SLACK: i32 = 40;
/// Re-verify at most this many style picks before trusting the engine's best move.
pub const MAX_GATE_CHECKS: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct Dial {
    /// Allowed centipawn drop from the engine's best move.
    pub band: i32,
    /// 0 = always the most "you" safe move; higher = more varied.
    pub temp: f64,
}

impl Dial {
    pub fn named(name: &str) -> Option<Dial> {
        match name {
            "faithful" => Some(Dial { band: 120, temp: 0.6 }),
            "balanced" => Some(Dial { band: 50, temp: 0.3 }),
            "sharp" => Some(Dial { band: 20, temp: 0.0 }),
            _ => None,
        }
    }
}

/// Small xorshift generator so we don't need an external crate; seedable for reproducible games.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn from_clock() -> Rng {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1);
        Rng::new(nanos)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub fn fen_of(pos: &Chess) -> String {
    Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string()
}

/// Order indices by sampling without replacement, weighted by softmax(score / temp).
fn sample_order(scores: &[f64], temp: f64, rng: &mut Rng) -> Vec<usize> {
    let mut remaining: Vec<usize> = (0..scores.len()).collect();
    let mut order = Vec::with_capacity(scores.len());
    while !remaining.is_empty() {
        let max = remaining.iter().map(|&i| scores[i]).fold(f64::MIN, f64::max);
        let weights: Vec<f64> = remaining.iter().map(|&i| ((scores[i] - max) / temp).exp()).collect();
        let mut r = rng.next_f64() * weights.iter().sum::<f64>();
        let mut pick = remaining.len() - 1;
        for (j, w) in weights.iter().enumerate() {
            if r < *w {
                pick = j;
                break;
            }
            r -= w;
        }
        order.push(remaining.remove(pick));
    }
    order
}

pub struct Echo<A: Analyser> {
    pub engine: A,
    pub brain: Brain,
    pub dial: Dial,
    pub depth: u32,
    pub multipv: usize,
    pub rng: Rng,
    /// (matched the engine's best move?, centipawns given up) for every move played.
    pub history: Vec<(bool, i32)>,
    /// Human-readable note about the last decision (the CLI prints it in coach mode).
    pub note: Option<String>,
}

impl<A: Analyser> Echo<A> {
    pub fn new(engine: A, brain: Brain, dial: Dial, depth: u32, rng: Rng) -> Echo<A> {
        Echo { engine, brain, dial, depth, multipv: 8, rng, history: vec![], note: None }
    }

    pub fn summary(&self) -> String {
        if self.history.is_empty() {
            return String::new();
        }
        let n = self.history.len();
        let matched = self.history.iter().filter(|(m, _)| *m).count();
        let cost = self.history.iter().map(|(_, c)| *c as f64).sum::<f64>() / n as f64;
        format!(
            "Echo matched the engine's top move {matched}/{n} times and gave up {cost:.0} centipawns per move on average to stay in your style."
        )
    }

    fn candidate_limit(&self, movetime: Option<f64>) -> Limit {
        match movetime {
            Some(s) => Limit::MoveTimeMs((s * 500.0).max(10.0) as u64),
            None => Limit::Depth(self.depth),
        }
    }

    /// Blunder gate: search the position after `mv` and return the score from the mover's point of view.
    fn verify(&mut self, pos: &Chess, mv: &Move, movetime: Option<f64>) -> Result<i32, String> {
        let mut next = pos.clone();
        next.play_unchecked(mv);
        if next.is_checkmate() {
            return Ok(MATE);
        }
        let limit = match movetime {
            Some(s) => Limit::MoveTimeMs((s * 150.0).max(20.0) as u64),
            None => Limit::Depth(self.depth + 6),
        };
        let lines = self.engine.analyse(&fen_of(&next), limit, 1)?;
        Ok(lines.first().map(|l| -l.cp).unwrap_or(0))
    }

    /// Pick a move. `movetime` (seconds) switches from fixed depth to a time budget.
    pub fn choose(&mut self, pos: &Chess, movetime: Option<f64>) -> Result<Move, String> {
        let legal = pos.legal_moves();
        if legal.is_empty() {
            return Err("no legal moves".into());
        }
        let lines = self.engine.analyse(&fen_of(pos), self.candidate_limit(movetime), self.multipv)?;
        let by_uci: HashMap<String, Move> =
            legal.iter().map(|m| (UciMove::from_standard(m).to_string(), m.clone())).collect();
        let cands: Vec<(Move, i32)> =
            lines.iter().filter_map(|l| by_uci.get(&l.mv).map(|m| (m.clone(), l.cp))).collect();
        let Some((best_move, best_cp)) = cands.first().cloned() else {
            // The engine gave nothing usable: play any legal move rather than crash.
            self.history.push((false, 0));
            return Ok(legal[0].clone());
        };

        let mut band = self.dial.band;
        if best_cp >= MATE - 1000 {
            band = 0; // forced mate: take it
        } else if best_cp.abs() > 500 {
            band = band.min(30); // decided position: stop experimenting
        }
        let safe: Vec<(Move, i32)> = cands.iter().filter(|(_, cp)| *cp >= best_cp - band).cloned().collect();

        let safe_moves: Vec<Move> = safe.iter().map(|(m, _)| m.clone()).collect();
        let style = self.brain.scores(pos, &safe_moves);
        let order: Vec<usize> = if self.dial.temp > 0.0 && safe.len() > 1 {
            sample_order(&style, self.dial.temp, &mut self.rng)
        } else {
            let mut idx: Vec<usize> = (0..safe.len()).collect();
            idx.sort_by(|&a, &b| style[b].partial_cmp(&style[a]).unwrap_or(std::cmp::Ordering::Equal));
            idx
        };

        let mut checks = 0;
        let mut chosen: Option<(Move, i32)> = None;
        for i in order {
            let (mv, cp) = (&safe[i].0, safe[i].1);
            if *mv != best_move {
                if checks >= MAX_GATE_CHECKS {
                    break;
                }
                checks += 1;
                if self.verify(pos, mv, movetime)? < best_cp - band - GATE_SLACK {
                    continue;
                }
            }
            chosen = Some((mv.clone(), cp));
            break;
        }

        match chosen {
            Some((mv, cp)) => {
                self.history.push((mv == best_move, best_cp - cp));
                self.note = Some(format!(
                    "[coach] {}/{} moves were safe | engine best {} ({:+.2}) | Echo plays {} ({:+.2})",
                    safe.len(),
                    cands.len(),
                    SanPlus::from_move(pos.clone(), &best_move),
                    best_cp as f64 / 100.0,
                    SanPlus::from_move(pos.clone(), &mv),
                    cp as f64 / 100.0
                ));
                Ok(mv)
            }
            None => {
                self.history.push((true, 0));
                Ok(best_move)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Line;
    use crate::model::Brain;

    /// Canned evals: `cps` maps the first move (UCI) to centipawns; `deep` overrides the blunder-gate result.
    struct Fake {
        root: Chess,
        cps: Vec<(&'static str, i32)>,
        deep: HashMap<&'static str, i32>,
        verifies: usize,
    }

    impl Fake {
        fn new(cps: Vec<(&'static str, i32)>) -> Fake {
            Fake { root: Chess::default(), cps, deep: HashMap::new(), verifies: 0 }
        }
    }

    impl Analyser for Fake {
        fn analyse(&mut self, fen: &str, _limit: Limit, multipv: usize) -> Result<Vec<Line>, String> {
            if fen == fen_of(&self.root) {
                let mut lines: Vec<Line> =
                    self.cps.iter().map(|(u, cp)| Line { mv: u.to_string(), cp: *cp }).collect();
                lines.sort_by_key(|l| -l.cp);
                lines.truncate(multipv);
                return Ok(lines);
            }
            self.verifies += 1;
            for (u, cp) in &self.cps {
                let m = u.parse::<UciMove>().unwrap().to_move(&self.root).unwrap();
                let mut next = self.root.clone();
                next.play_unchecked(&m);
                if fen_of(&next) == fen {
                    let v = self.deep.get(u).copied().unwrap_or(*cp);
                    return Ok(vec![Line { mv: String::new(), cp: -v }]);
                }
            }
            Err(format!("unexpected position {fen}"))
        }
    }

    fn brain_with(prefs: &[(&str, f64)]) -> Brain {
        Brain::fixed(prefs.iter().map(|(u, v)| (u.to_string(), *v)).collect())
    }

    fn make(cps: Vec<(&'static str, i32)>, prefs: &[(&str, f64)], dial: Dial) -> Echo<Fake> {
        Echo::new(Fake::new(cps), brain_with(prefs), dial, 10, Rng::new(1))
    }

    /// Run one decision from the start position and return the chosen move in UCI.
    fn choose_with_prefs(e: &mut Echo<Fake>) -> String {
        let pos = e.engine.root.clone();
        let mv = e.choose(&pos, None).unwrap();
        UciMove::from_standard(&mv).to_string()
    }

    const SHARP: Dial = Dial { band: 20, temp: 0.0 };

    #[test]
    fn style_picks_among_safe_moves_only() {
        let mut e = make(vec![("e2e4", 50), ("d2d4", 45), ("a2a3", -200)], &[("d2d4", 5.0), ("a2a3", 9.0)], SHARP);
        assert_eq!(choose_with_prefs(&mut e), "d2d4"); // style loves a3, but a3 is a blunder
    }

    #[test]
    fn blunder_gate_falls_back_when_deep_search_disagrees() {
        let mut fake = Fake::new(vec![("e2e4", 50), ("d2d4", 45)]);
        fake.deep.insert("d2d4", -300);
        let mut e = Echo::new(fake, brain_with(&[("d2d4", 5.0)]), SHARP, 10, Rng::new(1));
        assert_eq!(choose_with_prefs(&mut e), "e2e4");
    }

    #[test]
    fn forced_mate_is_always_taken() {
        let faithful = Dial::named("faithful").unwrap();
        let mut e = make(vec![("e2e4", MATE - 1), ("d2d4", 100)], &[("d2d4", 9.0)], faithful);
        assert_eq!(choose_with_prefs(&mut e), "e2e4");
    }

    #[test]
    fn gate_checks_are_capped() {
        let cps = vec![("e2e4", 50), ("d2d4", 49), ("c2c4", 48), ("g1f3", 47), ("b1c3", 46)];
        let mut fake = Fake::new(cps);
        for u in ["d2d4", "c2c4", "g1f3", "b1c3"] {
            fake.deep.insert(u, -500);
        }
        let prefs = [("d2d4", 5.0), ("c2c4", 4.0), ("g1f3", 3.0), ("b1c3", 2.0)];
        let mut e = Echo::new(fake, brain_with(&prefs), Dial { band: 100, temp: 0.0 }, 10, Rng::new(1));
        assert_eq!(choose_with_prefs(&mut e), "e2e4");
        assert_eq!(e.engine.verifies, MAX_GATE_CHECKS);
    }

    #[test]
    fn summary_reports_cost() {
        let mut e = make(vec![("e2e4", 50), ("d2d4", 45)], &[("d2d4", 5.0)], SHARP);
        choose_with_prefs(&mut e);
        assert!(e.summary().contains("1 times") || e.summary().contains("/1"));
    }

    #[test]
    fn sampling_covers_every_index_once() {
        let mut rng = Rng::new(9);
        let mut order = sample_order(&[1.0, 2.0, 3.0, 0.5], 0.5, &mut rng);
        order.sort();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }
}
