//! Loads the `.echo` model (see docs/FORMAT.md) and scores moves.

use crate::features::{self, BASE_MOVES, FEATURE_NAMES, NUM_FEATURES, NUM_MOVES};
use serde::Deserialize;
use shakmaty::uci::UciMove;
use shakmaty::{Chess, Move};
use std::collections::HashMap;

const MAGIC: &[u8; 8] = b"ECHOSTYL";
const VERSION: u32 = 2;
/// Format v1 had 4096 move units and no promotion pieces; it is migrated in memory when loaded.
const LEGACY_VERSION: u32 = 1;

#[derive(Debug, Deserialize, Clone)]
pub struct Validation {
    pub top1: f64,
    pub top3: f64,
    pub random_top1: f64,
}

#[derive(Debug, Deserialize)]
struct Dims {
    input: usize,
    hidden: usize,
    moves: usize,
    features: usize,
}

#[derive(Debug, Deserialize)]
struct Header {
    player: String,
    games: u64,
    positions: u64,
    #[serde(default)]
    played_rate: HashMap<String, f64>,
    #[serde(default)]
    baseline_rate: HashMap<String, f64>,
    #[serde(default)]
    book: HashMap<String, HashMap<String, u32>>,
    #[serde(default)]
    validation: Option<Validation>,
    dims: Dims,
    feature_names: Vec<String>,
}

pub struct Model {
    pub player: String,
    pub games: u64,
    pub positions: u64,
    pub played_rate: HashMap<String, f64>,
    pub baseline_rate: HashMap<String, f64>,
    pub book: HashMap<String, HashMap<String, u32>>,
    pub validation: Option<Validation>,
    /// Set when the file was an older format that was migrated in memory (the file itself is never modified).
    pub migrated_from: Option<u32>,
    hidden: usize,
    w1: Vec<f64>,  // 768 x hidden
    b1: Vec<f64>,  // hidden
    w2t: Vec<f64>, // NUM_MOVES x hidden
    b2: Vec<f64>,  // NUM_MOVES
    w: Vec<f64>,   // features
}

fn take_f32(bytes: &[u8], pos: &mut usize, n: usize) -> Result<Vec<f64>, String> {
    let end = *pos + 4 * n;
    let chunk = bytes.get(*pos..end).ok_or("model file is truncated")?;
    *pos = end;
    Ok(chunk
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect())
}

impl Model {
    pub fn load(path: &str) -> Result<Model, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Model::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err("not an .echo file (run `python echo.py export style.npz` to make one)".into());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION && version != LEGACY_VERSION {
            return Err(format!("unsupported model version {version} (this build reads {LEGACY_VERSION} and {VERSION})"));
        }
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json = bytes.get(16..16 + json_len).ok_or("model header is truncated")?;
        let h: Header = serde_json::from_slice(json).map_err(|e| format!("bad model header: {e}"))?;

        let expected: Vec<&str> = FEATURE_NAMES.to_vec();
        if h.feature_names.iter().map(String::as_str).collect::<Vec<_>>() != expected
            || h.dims.features != NUM_FEATURES
            || h.dims.input != 768
            || h.dims.moves != if version == LEGACY_VERSION { BASE_MOVES } else { NUM_MOVES }
        {
            return Err("model was trained with a different feature set than this engine".into());
        }
        let hid = h.dims.hidden;
        let mut pos = 16 + json_len;
        let w1 = take_f32(bytes, &mut pos, 768 * hid)?;
        let b1 = take_f32(bytes, &mut pos, hid)?;
        let mut w2t = take_f32(bytes, &mut pos, h.dims.moves * hid)?;
        let mut b2 = take_f32(bytes, &mut pos, h.dims.moves)?;
        let w = take_f32(bytes, &mut pos, NUM_FEATURES)?;
        if version == LEGACY_VERSION {
            // Every under-promotion unit starts as a copy of the queen promotion it was split from, so a v1 model
            // scores every move exactly as it did before.
            for src in features::underpromo_sources() {
                let row = w2t[src * hid..(src + 1) * hid].to_vec();
                w2t.extend_from_slice(&row);
                b2.push(b2[src]);
            }
        }
        Ok(Model {
            player: h.player,
            games: h.games,
            positions: h.positions,
            played_rate: h.played_rate,
            baseline_rate: h.baseline_rate,
            book: h.book,
            validation: h.validation,
            migrated_from: (version == LEGACY_VERSION).then_some(version),
            hidden: hid,
            w1,
            b1,
            w2t,
            b2,
            w,
        })
    }

    /// Raw logit for each move (network output + feature term), same order as `moves`.
    pub fn logits(&self, pos: &Chess, moves: &[Move]) -> Vec<f64> {
        let hid = self.hidden;
        let mut pre = self.b1.clone();
        for unit in features::encode(pos) {
            let row = &self.w1[unit * hid..(unit + 1) * hid];
            for (p, r) in pre.iter_mut().zip(row) {
                *p += r;
            }
        }
        let h: Vec<f64> = pre.iter().map(|x| x.max(0.0)).collect();
        moves
            .iter()
            .map(|m| {
                let idx = features::move_index(pos, m);
                let row = &self.w2t[idx * hid..(idx + 1) * hid];
                let net: f64 = row.iter().zip(&h).map(|(a, b)| a * b).sum();
                let feat: f64 = features::features(pos, m).iter().zip(&self.w).map(|(a, b)| a * b).sum();
                net + self.b2[idx] + feat
            })
            .collect()
    }

    /// Style score: logit plus a bonus for moves from your own opening repertoire.
    pub fn scores(&self, pos: &Chess, moves: &[Move]) -> Vec<f64> {
        let key = features::book_key(pos);
        let counts = self.book.get(&key);
        self.logits(pos, moves)
            .into_iter()
            .zip(moves)
            .map(|(logit, m)| {
                let uci = UciMove::from_standard(m).to_string();
                let c = counts.and_then(|c| c.get(&uci)).copied().unwrap_or(0);
                logit + 1.5 * (1.0 + c as f64).ln()
            })
            .collect()
    }
}

/// The style brain: a model if one is loaded, otherwise neutral (plain engine play).
pub struct Brain {
    pub model: Option<Model>,
    fixed: Option<HashMap<String, f64>>,
}

impl Brain {
    pub fn none() -> Brain {
        Brain { model: None, fixed: None }
    }
    pub fn with_model(model: Model) -> Brain {
        Brain { model: Some(model), fixed: None }
    }
    /// Fixed per-move scores keyed by UCI string (for tests and experiments); unlisted moves score 0.
    pub fn fixed(scores: HashMap<String, f64>) -> Brain {
        Brain { model: None, fixed: Some(scores) }
    }
    pub fn player(&self) -> String {
        self.model.as_ref().map(|m| m.player.clone()).unwrap_or_else(|| "You".into())
    }
    pub fn scores(&self, pos: &Chess, moves: &[Move]) -> Vec<f64> {
        if let Some(fixed) = &self.fixed {
            return moves
                .iter()
                .map(|m| fixed.get(&UciMove::from_standard(m).to_string()).copied().unwrap_or(0.0))
                .collect();
        }
        match &self.model {
            Some(m) => m.scores(pos, moves),
            None => vec![0.0; moves.len()],
        }
    }
}
