//! Save games as PGN so the Python trainer can learn from them (`learn --resume`).

use std::path::Path;

/// (year, month, day, hour, minute, second) in UTC for a Unix timestamp.
pub fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400) as u32;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d, rem / 3600, (rem % 3600) / 60, rem % 60)
}

pub fn movetext(sans: &[String], result: &str) -> String {
    let mut out = String::new();
    for (i, san) in sans.iter().enumerate() {
        if i % 2 == 0 {
            out += &format!("{}. ", i / 2 + 1);
        }
        out += san;
        out.push(' ');
    }
    out + result
}

pub fn pgn_text(sans: &[String], white: &str, black: &str, result: &str, date: &str) -> String {
    format!(
        "[Event \"Echo practice game\"]\n[Site \"Local\"]\n[Date \"{date}\"]\n[Round \"?\"]\n[White \"{white}\"]\n[Black \"{black}\"]\n[Result \"{result}\"]\n\n{}\n\n",
        movetext(sans, result)
    )
}

/// Write the game into `dir` and return the file path.
pub fn save_game(sans: &[String], white: &str, black: &str, result: &str, dir: &str) -> Result<String, String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = civil_from_unix(now);
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {dir}: {e}"))?;
    let path = Path::new(dir).join(format!("echo_{y:04}{mo:02}{d:02}_{h:02}{mi:02}{s:02}.pgn"));
    let text = pgn_text(sans, white, black, result, &format!("{y:04}.{mo:02}.{d:02}"));
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_correct() {
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(civil_from_unix(951_868_800), (2000, 3, 1, 0, 0, 0));
        assert_eq!(civil_from_unix(1_794_000_000).0, 2026);
    }

    #[test]
    fn movetext_numbers_moves() {
        let sans: Vec<String> = ["e4", "e5", "Nf3"].iter().map(|s| s.to_string()).collect();
        assert_eq!(movetext(&sans, "*"), "1. e4 e5 2. Nf3 *");
    }
}
