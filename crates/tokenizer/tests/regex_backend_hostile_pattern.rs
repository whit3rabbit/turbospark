//! A checkpoint's `tokenizer.json` regex runs on Oniguruma through the
//! `tokenizers` `onig` feature. Oniguruma caps retries per start position
//! (10M by default) but not across a search, so a checkpoint can choose:
//!
//! - a pattern that hits the cap: the `onig` crate turns that error into a
//!   panic inside `Regex::search_with_encoding`, or
//! - a pattern that stays under it at every start: encode is then quadratic
//!   or worse in the text, with nothing to cancel it.
//!
//! Both tests fail on the current backend and are ignored for that reason:
//! closing them means changing the regex engine, which moves tokens. The risk
//! is accepted for now and the engine swap is deferred research (see the
//! regex rule in crates/tokenizer/AGENTS.md). Run with `-- --ignored`, and
//! drop the `ignore` once a backend change lands.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use turbospark_tokenizer::MfTokenizer;

/// Loads through the production path, with a table the dialect probe accepts,
/// and a pattern placed where a crafted checkpoint would put it.
fn tokenizer_with_split(name: &str, pattern: &str) -> MfTokenizer {
    let pattern = serde_json::to_string(pattern).unwrap();
    let added: String = [(0, "<unk>"), (1, "<s>"), (2, "</s>")]
        .iter()
        .map(|(id, tok)| {
            format!(
                r#"{{"id":{id},"content":"{tok}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}"#
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{added}],"normalizer":null,"pre_tokenizer":{{"type":"Split","pattern":{{"Regex":{pattern}}},"behavior":"Isolated","invert":false}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{"<unk>":0,"<s>":1,"</s>":2}},"unk_token":"<unk>"}}}}"#
    );
    let dir: PathBuf =
        std::env::temp_dir().join(format!("ts-regex-hostile-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), json).unwrap();
    let loaded = MfTokenizer::load_from_dir(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    loaded.unwrap_or_else(|e| panic!("the crafted tokenizer must load: {e:?}"))
}

#[test]
#[ignore = "accepted risk: fails on the onig backend; acceptance bar for the deferred regex engine research"]
fn a_hostile_pattern_cannot_stall_encode() {
    // Lookaheads leave the search optimizer no required literal to reject on,
    // and every start stays far under the per-start retry cap. A linear or
    // limit-bounded engine answers this in milliseconds.
    let tokenizer = tokenizer_with_split("stall", r"[a-z]*a(?!a)(?!$)");
    let text = "a".repeat(40_000);
    let (done, finished) = mpsc::channel();
    // The stuck encode cannot be cancelled, which is the defect. It is left
    // to die with the test process.
    std::thread::spawn(move || {
        let _ = tokenizer.encode(&text, false);
        let _ = done.send(());
    });
    assert!(
        finished.recv_timeout(Duration::from_secs(3)).is_ok(),
        "encoding 40 KB through a crafted regex took over 3 s"
    );
}

#[test]
#[ignore = "accepted risk: fails on the onig backend; acceptance bar for the deferred regex engine research"]
fn a_hostile_pattern_cannot_panic_encode() {
    let tokenizer = tokenizer_with_split("panic", r"(a+)+$");
    let text = format!("{}b", "a".repeat(40));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tokenizer.encode(&text, false)
    }));
    assert!(
        outcome.is_ok(),
        "a crafted regex panicked encode at the retry limit"
    );
}
