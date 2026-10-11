// THROWAWAY probe (deleted after use): how much of a code + prose corpus a
// draft vocabulary of the first N token ids covers, against the best
// possible N-token list (the corpus's own top N by frequency).
use std::path::Path;
use turbospark_tokenizer::MfTokenizer;

#[test]
#[ignore]
fn coverage() {
    let dir = std::env::var("PROBE_TOKENIZER_DIR").unwrap();
    let tok = MfTokenizer::load_from_dir(Path::new(&dir)).unwrap();
    let files = std::env::var("PROBE_FILES").unwrap();
    let mut counts = vec![0u64; 300_000];
    let mut total = 0u64;
    let mut max_id = 0i32;
    for f in files.lines() {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        for id in tok.encode(&text, false) {
            counts[id as usize] += 1;
            total += 1;
            max_id = max_id.max(id);
        }
    }
    let mut ranked: Vec<u64> = counts.clone();
    ranked.sort_unstable_by(|a, b| b.cmp(a));
    println!("tokens {total}, max id {max_id}");
    for n in [8192usize, 16384, 32768, 49152, 65536, 98304, 131072] {
        let prefix: u64 = counts[..n].iter().sum();
        let best: u64 = ranked[..n].iter().sum();
        println!(
            "N {n:>6}: id-prefix coverage {:.4}  best-list coverage {:.4}",
            prefix as f64 / total as f64,
            best as f64 / total as f64
        );
    }
}
