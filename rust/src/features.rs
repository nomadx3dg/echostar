//! Style features, board encoding and book keys. These mirror `echo.py` one-to-one;
//! `tests/parity.rs` checks them against golden vectors generated from Python.

use shakmaty::fen::Fen;
use shakmaty::{Chess, Color, EnPassantMode, File, Move, Position, Role, Square};

pub const NUM_FEATURES: usize = 21;
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "capture",
    "check",
    "castle",
    "promotion",
    "moves_pawn",
    "moves_knight",
    "moves_bishop",
    "moves_rook",
    "moves_queen",
    "moves_king",
    "to_center",
    "to_extended_center",
    "pawn_advance",
    "retreat",
    "even_or_winning_trade",
    "kingside_pawn_push",
    "queenside_pawn_push",
    "early_queen",
    "develops_minor",
    "moves_into_attack",
    "quiet_move",
];

fn value(role: Role) -> u32 {
    match role {
        Role::Pawn => 1,
        Role::Knight | Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}

fn b(x: bool) -> f64 {
    if x {
        1.0
    } else {
        0.0
    }
}

/// (from, to) squares the way python-chess reports them: castling is king-from to king-to (e1g1).
pub fn from_to(m: &Move) -> (Square, Square) {
    match m {
        Move::Castle { king, rook } => {
            let file = if (rook.file() as u8) > (king.file() as u8) { File::G } else { File::C };
            (*king, Square::from_coords(file, king.rank()))
        }
        _ => (m.from().expect("a board move has a from-square"), m.to()),
    }
}

pub fn gives_check(pos: &Chess, m: &Move) -> bool {
    let mut next = pos.clone();
    next.play_unchecked(m);
    next.is_check()
}

pub fn features(pos: &Chess, m: &Move) -> [f64; NUM_FEATURES] {
    let mut f = [0.0; NUM_FEATURES];
    let color = pos.turn();
    let white = color == Color::White;
    let role = m.role();
    let (from, to) = from_to(m);
    let is_capture = m.is_capture();
    let check = gives_check(pos, m);

    f[0] = b(is_capture);
    f[1] = b(check);
    f[2] = b(m.is_castle());
    f[3] = b(m.promotion().is_some());
    f[3 + role as usize] = 1.0; // pawn=1 -> index 4 ... king=6 -> index 9

    let (tf, tr) = (to.file() as i32, to.rank() as i32);
    let center = (tf == 3 || tf == 4) && (tr == 3 || tr == 4);
    f[10] = b(center);
    f[11] = b((2..=5).contains(&tf) && (2..=5).contains(&tr));

    let from_rank = from.rank() as i32;
    let forward = if white { tr - from_rank } else { from_rank - tr };
    let rel_to_rank = if white { tr } else { 7 - tr };
    f[12] = b(role == Role::Pawn && rel_to_rank >= 3);
    f[13] = b(forward < 0);

    if is_capture {
        let victim = m.capture().map(value).unwrap_or(1);
        f[14] = b(victim >= value(role));
    }
    if role == Role::Pawn {
        f[15] = b(tf >= 5);
        f[16] = b(tf <= 2);
    }
    f[17] = b(role == Role::Queen && pos.fullmoves().get() <= 10);
    let back_rank = if white { 0 } else { 7 };
    f[18] = b((role == Role::Knight || role == Role::Bishop) && from_rank == back_rank);
    let attacked = pos.board().attacks_to(to, !color, pos.board().occupied()).any();
    f[19] = b(attacked && role != Role::Pawn);
    f[20] = b(!is_capture && !check);
    f
}

fn mirror(sq: Square, turn: Color) -> usize {
    let s = sq as usize;
    if turn == Color::White {
        s
    } else {
        s ^ 56
    }
}

/// Active input units (768 = 12 piece planes x 64), always from the mover's point of view.
pub fn encode(pos: &Chess) -> Vec<usize> {
    let turn = pos.turn();
    let board = pos.board();
    board
        .occupied()
        .into_iter()
        .map(|sq| {
            let piece = board.piece_at(sq).expect("occupied square has a piece");
            let plane = (piece.role as usize - 1) + if piece.color == turn { 0 } else { 6 };
            plane * 64 + mirror(sq, turn)
        })
        .collect()
}

/// Output unit for a move: `from * 64 + to`, mirrored for black.
pub fn move_index(pos: &Chess, m: &Move) -> usize {
    let (from, to) = from_to(m);
    mirror(from, pos.turn()) * 64 + mirror(to, pos.turn())
}

/// First four FEN fields (board, turn, castling, en passant): identifies a position for the opening book.
pub fn book_key(pos: &Chess) -> String {
    let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
    fen.split(' ').take(4).collect::<Vec<_>>().join(" ")
}
