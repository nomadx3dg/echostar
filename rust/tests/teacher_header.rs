//! A model trained with the optional Maia teacher (`echo.py learn --maia-mix ...`) records a `teacher` block in
//! its header. No tensor or feature changed, so the Rust runtime must load such a file unchanged and need
//! neither Maia nor any extra dependency. This test builds that header from the golden model and checks it.

use echo_rs::model::Model;

fn with_header_edit(bytes: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[16..16 + json_len]).unwrap();
    edit(&mut header);
    let json = serde_json::to_vec(&header).unwrap();
    let mut out = bytes[..12].to_vec();
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&bytes[16 + json_len..]);
    out
}

#[test]
fn model_with_a_teacher_block_loads_with_the_same_contents() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
    let plain = std::fs::read(format!("{dir}/golden.echo")).unwrap();
    assert!(
        !String::from_utf8_lossy(&plain[16..400.min(plain.len())]).contains("\"teacher\""),
        "the golden model should not already carry a teacher block"
    );

    let taught = with_header_edit(&plain, |h| {
        h["teacher"] = serde_json::json!({"version": 1, "name": "lc0:maia-1500.pb.gz", "elo": 1500, "mix": 0.25, "games": 12});
        h["teacher_from_the_future"] = serde_json::json!({"anything": [1, 2, 3]});
    });
    assert_ne!(plain, taught);

    let a = Model::from_bytes(&plain).expect("golden model loads");
    let b = Model::from_bytes(&taught).expect("a model with a teacher block loads");
    assert_eq!((a.player.as_str(), a.games, a.positions), (b.player.as_str(), b.games, b.positions));
    assert_eq!(a.migrated_from, b.migrated_from);
    assert_eq!(a.book.len(), b.book.len());
}
