//! Encode equality between a tokenizer REBUILT FROM A GGUF's metadata
//! (`turbospark_repack::tokenizer_files_from_gguf`) and the checkpoint's own
//! HF `tokenizer.json`, through this crate's real loader. This is the gate
//! that matters for that rebuild: matching vocabulary and regex (checked in
//! `crates/repack/tests/gguf_tokenizer_network.rs`) is necessary and not
//! sufficient, since a pre-tokenizer difference only shows in how text splits.
//!
//! `#[ignore]`d and env-gated: `TURBOSPARK_GGUF_TOK_DIR` (a directory written
//! by `crates/repack/examples/gguf_tokenizer_dump.rs`) and
//! `TURBOSPARK_HF_TOK_DIR` (the HF `tokenizer.json` and `tokenizer_config.json`).
//!
//! ```sh
//! cargo run -p turbospark-repack --example gguf_tokenizer_dump -- <gguf url> /tmp/tok/gguf
//! # fetch the HF tokenizer.json + tokenizer_config.json into /tmp/tok/hf
//! TURBOSPARK_GGUF_TOK_DIR=/tmp/tok/gguf TURBOSPARK_HF_TOK_DIR=/tmp/tok/hf \
//!   cargo test -p turbospark-tokenizer --test gguf_tokenizer_encode_equality -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use turbospark_tokenizer::MfTokenizer;

const CORPUS: &[&str] = &[
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing   spaces  ",
    "tabs\tand\nnewlines\r\nand\n\n\nblank lines",
    "Numbers: 0 1 12 123 1234 12345 3.14159 1,000,000 2024-11-20",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "fn main() { println!(\"{}\", 1 + 2); } // comment",
    "SELECT * FROM users WHERE id = 42 AND name LIKE '%a%';",
    "don't can't won't I'm you're they've she'll he'd it's",
    "UPPER lower MiXeD camelCase snake_case kebab-case PascalCase",
    "Emoji \u{1F600} \u{1F680} and symbols \u{00A9} \u{2122} \u{20AC} \u{221E}",
    "Caf\u{00E9} na\u{00EF}ve fa\u{00E7}ade \u{00FC}ber stra\u{00DF}e",
    "\u{4F60}\u{597D}\u{FF0C}\u{4E16}\u{754C}\u{3002}\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}\u{4E16}\u{754C}",
    "\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}, \u{043C}\u{0438}\u{0440}! \u{0645}\u{0631}\u{062D}\u{0628}\u{0627}",
    "URL https://example.com/a/b?c=d&e=f#frag and email a.b@c.org",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "...!!!???---===***+++///\\\\\\|||",
    "a",
    "",
    "    indented\n        more\n            deeper",
    "Mixed 123abc abc123 a1b2c3 x_1 y-2 z.3",
];

fn load(var: &str) -> Option<MfTokenizer> {
    let dir = PathBuf::from(std::env::var_os(var)?);
    Some(MfTokenizer::load_from_dir(&dir).unwrap_or_else(|e| panic!("{var}: {e}")))
}

#[test]
#[ignore = "needs TURBOSPARK_GGUF_TOK_DIR and TURBOSPARK_HF_TOK_DIR"]
fn a_gguf_rebuilt_tokenizer_encodes_exactly_like_the_hf_one() {
    let (Some(gguf), Some(hf)) = (
        load("TURBOSPARK_GGUF_TOK_DIR"),
        load("TURBOSPARK_HF_TOK_DIR"),
    ) else {
        eprintln!("skipped: set TURBOSPARK_GGUF_TOK_DIR and TURBOSPARK_HF_TOK_DIR");
        return;
    };
    // Not vacuous: both loaders really tokenize, and the corpus is long enough
    // that an empty comparison would be noticed.
    assert!(hf.encode("Hello, world!", false).len() >= 3);
    assert_eq!(
        gguf.encode("Hello, world!", false),
        hf.encode("Hello, world!", false)
    );
    let mut diffs = Vec::new();
    for text in CORPUS {
        let (a, b) = (gguf.encode(text, false), hf.encode(text, false));
        if a != b {
            diffs.push((*text, a, b));
        }
    }
    for (text, a, b) in &diffs {
        eprintln!("DIFF {text:?}\n  gguf {a:?}\n  hf   {b:?}");
    }
    assert!(
        diffs.is_empty(),
        "{} of {} corpus strings encode differently",
        diffs.len(),
        CORPUS.len()
    );
}
