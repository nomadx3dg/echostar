//! Terminal board: colored squares, unicode pieces, last-move highlight, check marker.

use shakmaty::{Chess, Color, Position, Role, Square};

const LIGHT: u8 = 180;
const DARK: u8 = 95;
const LAST_LIGHT: u8 = 186;
const LAST_DARK: u8 = 142;
const CHECK_BG: u8 = 160;

fn glyph(role: Role) -> &'static str {
    match role {
        Role::Pawn => "\u{265f}\u{fe0e}",
        Role::Knight => "\u{265e}",
        Role::Bishop => "\u{265d}",
        Role::Rook => "\u{265c}",
        Role::Queen => "\u{265b}",
        Role::King => "\u{265a}",
    }
}

fn letter(role: Role, color: Color) -> char {
    let c = match role {
        Role::Pawn => 'p',
        Role::Knight => 'n',
        Role::Bishop => 'b',
        Role::Rook => 'r',
        Role::Queen => 'q',
        Role::King => 'k',
    };
    if color == Color::White {
        c.to_ascii_uppercase()
    } else {
        c
    }
}

fn value(role: Role) -> i32 {
    match role {
        Role::Pawn => 1,
        Role::Knight | Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}

pub fn material_diff(pos: &Chess) -> i32 {
    let board = pos.board();
    board
        .occupied()
        .into_iter()
        .map(|sq| {
            let p = board.piece_at(sq).unwrap();
            value(p.role) * if p.color == Color::White { 1 } else { -1 }
        })
        .sum()
}

pub fn render(pos: &Chess, perspective: Color, last: Option<(Square, Square)>, color: bool) -> String {
    let ranks: Vec<i32> = if perspective == Color::White { (0..8).rev().collect() } else { (0..8).collect() };
    let files: Vec<i32> = if perspective == Color::White { (0..8).collect() } else { (0..8).rev().collect() };
    let king_in_check = if pos.is_check() { pos.board().king_of(pos.turn()) } else { None };
    let mut out = Vec::new();
    for &r in &ranks {
        let mut row = format!(" {} ", r + 1);
        for &f in &files {
            let sq = Square::from_coords(shakmaty::File::ALL[f as usize], shakmaty::Rank::ALL[r as usize]);
            let piece = pos.board().piece_at(sq);
            let light = (f + r) % 2 == 1;
            if color {
                let is_last = last.map_or(false, |(a, b)| a == sq || b == sq);
                let bg = if Some(sq) == king_in_check {
                    CHECK_BG
                } else if is_last {
                    if light { LAST_LIGHT } else { LAST_DARK }
                } else if light {
                    LIGHT
                } else {
                    DARK
                };
                let fg = if piece.map_or(false, |p| p.color == Color::White) { 231 } else { 16 };
                let g = piece.map_or(" ", |p| glyph(p.role));
                row += &format!("\x1b[48;5;{bg}m\x1b[38;5;{fg}m {g} \x1b[0m");
            } else {
                let ch = match piece {
                    Some(p) => letter(p.role, p.color),
                    None => if light { ' ' } else { '.' },
                };
                row += &format!(" {ch} ");
            }
        }
        out.push(row);
    }
    let letters: Vec<&str> = if perspective == Color::White {
        vec!["a", "b", "c", "d", "e", "f", "g", "h"]
    } else {
        vec!["h", "g", "f", "e", "d", "c", "b", "a"]
    };
    out.push(format!("    {}", letters.join("  ")));
    let diff = material_diff(pos);
    let mut status = if diff == 0 {
        "Material: even".to_string()
    } else {
        format!("Material: {} +{}", if diff > 0 { "White" } else { "Black" }, diff.abs())
    };
    if pos.is_check() {
        status += "   CHECK";
    }
    out.push(format!(
        "\n  Move {}, {} to move   {}",
        pos.fullmoves(),
        if pos.turn() == Color::White { "White" } else { "Black" },
        status
    ));
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_and_plain_mode() {
        let pos = Chess::default();
        let white = render(&pos, Color::White, None, false);
        let black = render(&pos, Color::Black, None, false);
        let wl: Vec<&str> = white.lines().collect();
        let bl: Vec<&str> = black.lines().collect();
        assert!(wl[0].starts_with(" 8 ") && bl[0].starts_with(" 1 "));
        assert!(wl[8].contains("a  b  c") && bl[8].contains("h  g  f"));
        assert!(white.contains("Material: even"));
    }
}
