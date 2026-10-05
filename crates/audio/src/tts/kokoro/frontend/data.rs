// Adapted from MicheleYin/misaki-rs at 38bf1a534cce5f6864fb2df595447ca6f6891e74.
// Copyright (c) 2026 MicheleYin, MIT; see licenses/MISAKI-RS-MIT.txt.
use super::lexicon::PhonemeEntry;
use std::collections::HashMap;
pub fn load_us_gold() -> HashMap<String, PhonemeEntry> {
    serde_json::from_str(include_str!("resources/us_gold.json"))
        .expect("bundled canonical US gold dictionary")
}
pub fn load_us_silver() -> HashMap<String, PhonemeEntry> {
    serde_json::from_str(include_str!("resources/us_silver.json"))
        .expect("bundled canonical US silver dictionary")
}
