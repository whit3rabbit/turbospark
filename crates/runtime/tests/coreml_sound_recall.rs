//! Recall gate for the Audio library's sound lane. Ignored by default.
//!
//! The lane (`AudioSemanticSearch` in
//! `swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioSearch.swift`)
//! stores 10 s audio windows at 256-d, embeds the query with the
//! `search_query` task at 768-d, cuts it to 256 and renormalizes, and keeps
//! windows scoring at least `minSoundScore`. This gate does the same through
//! the runtime the app calls (`embed_audio_file` on 16 kHz mono WAV,
//! `embed_texts`), scores each clip by its best window as the lane's
//! item-level fusion does, and reports recall@k plus the score distribution
//! the floor has to separate. The floor is read from the Swift source, so
//! the gate measures the value that ships.
//!
//! Model: `TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL` (default
//! `~/models/coreml/embeddinggemma-2-coreml`), FluidInference
//! `embeddinggemma-2-coreml` at `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`,
//! opened with the app's default placement.
//!
//! Clips: `tests/fixtures/sound_recall/clips.json` names each clip's source,
//! SHA-256, license and author. They are real recordings, not synthesized,
//! and are not committed; fetch and convert them first:
//!
//!     python3 -I crates/runtime/tools/fetch_sound_recall_clips.py
//!
//! (writes `~/models/sound-recall-clips`; `TURBOSPARK_SOUND_RECALL_CLIPS`
//! overrides both). Then:
//!
//!     cargo test --release -p turbospark-runtime --test coreml_sound_recall -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use turbospark_runtime::coreml::{EmbeddingGemma2CoreMl, EmbeddingGemma2Options};

/// Same size and task as `AudioSemanticSearch` and the indexer.
/// `TURBOSPARK_SOUND_RECALL_TASK` swaps the query task (`none` for no
/// prefix) to compare prompts; the bounds below hold for the app's task.
const AUDIO_DIM: usize = 256;
const APP_QUERY_TASK: &str = "search_query";
/// Measured on an M4 Max, macOS 26.6 (2026-10-16), default placement,
/// ffmpeg 4.4.6 conversion: hit@1 0.500, hit@3 0.750, hit@5 0.781, MRR
/// 0.649 over 32 queries, identical across two runs. The bounds sit about
/// three queries below. Without a prefix the same set gives MRR 0.590 and
/// with `classification` 0.583, so `search_query` stays the app's task.
const HIT_AT_3_MIN: f64 = 0.65;
const MRR_MIN: f64 = 0.55;

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var_os("TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("models/coreml/embeddinggemma-2-coreml"));
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!("skipping: {} not found", p.display());
        None
    }
}

fn clips_dir() -> Option<PathBuf> {
    let p = std::env::var_os("TURBOSPARK_SOUND_RECALL_CLIPS")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("models/sound-recall-clips"));
    if p.join("converted.json").is_file() {
        Some(p)
    } else {
        eprintln!(
            "skipping: {} has no converted.json; run crates/runtime/tools/fetch_sound_recall_clips.py",
            p.display()
        );
        None
    }
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

fn repo_file(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn json(path: &PathBuf) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// `static let <name>: Float = <value>` from the Swift source. Panics when
/// the declaration moves, so the gate cannot silently measure a stale value.
fn swift_float(rel: &str, name: &str) -> f32 {
    let source = std::fs::read_to_string(repo_file(rel)).unwrap();
    let decl = format!("static let {name}: Float = ");
    let line = source
        .lines()
        .find_map(|l| l.trim().strip_prefix(&decl))
        .unwrap_or_else(|| panic!("{rel}: no `{decl}...`"));
    line.trim().parse().unwrap()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn percentile(sorted: &[f32], p: f64) -> f32 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

struct Clip {
    name: String,
    class: String,
    windows: Vec<Vec<f32>>,
}

#[test]
#[ignore = "needs the Core ML EmbeddingGemma 2 checkpoint and the fetched clip set"]
fn sound_lane_recall() {
    let (Some(model), Some(clips_dir)) = (model_dir(), clips_dir()) else {
        return;
    };
    let manifest = json(&repo_file(
        "crates/runtime/tests/fixtures/sound_recall/clips.json",
    ));
    let converted = json(&clips_dir.join("converted.json"));
    let floor = swift_float(
        "swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioSearch.swift",
        "minSoundScore",
    );

    let task = std::env::var("TURBOSPARK_SOUND_RECALL_TASK")
        .unwrap_or_else(|_| APP_QUERY_TASK.to_string());
    let started = std::time::Instant::now();
    let m = EmbeddingGemma2CoreMl::open(&model, &EmbeddingGemma2Options::default()).unwrap();
    eprintln!("opened in {:?}", started.elapsed());

    let mut clips = Vec::new();
    for c in manifest["clips"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        // The fetch script records what it converted; a clip it did not
        // convert (or converted from different source bytes) is an error,
        // not a smaller clip set.
        let entry = &converted["clips"][name];
        assert_eq!(
            entry["source_sha256"], c["sha256"],
            "{name}: converted from different source bytes; rerun the fetch script"
        );
        let wav = clips_dir.join(entry["wav"].as_str().unwrap());
        let windows = m.embed_audio_file(&wav, Some(AUDIO_DIM)).unwrap();
        assert!(!windows.is_empty(), "{name}: no windows");
        clips.push(Clip {
            name: name.to_string(),
            class: c["class"].as_str().unwrap().to_string(),
            windows: windows.into_iter().map(|w| w.embedding).collect(),
        });
    }
    let windows: usize = clips.iter().map(|c| c.windows.len()).sum();
    eprintln!(
        "{} clips, {} windows, {} classes; query task {task:?}; floor minSoundScore = {floor}",
        clips.len(),
        windows,
        clips
            .iter()
            .map(|c| c.class.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );

    let ks = [1usize, 3, 5];
    let mut hits = [0usize; 3];
    let mut recall = [0f64; 3];
    let mut reciprocal_rank = 0f64;
    let mut relevant_scores = Vec::new();
    let mut irrelevant_scores = Vec::new();
    let mut absent_scores = Vec::new();
    let mut absent_best = Vec::new();
    let mut present_queries = 0usize;
    for q in manifest["queries"].as_array().unwrap() {
        let text = q["text"].as_str().unwrap();
        let class = q["class"].as_str();
        let query = &m
            .embed_texts(&[text], Some(task.as_str()), Some(AUDIO_DIM))
            .unwrap()[0];
        // A clip scores its best window, as the lane's hits fuse per item.
        let mut ranked: Vec<(f32, &Clip)> = clips
            .iter()
            .map(|c| {
                let best = c
                    .windows
                    .iter()
                    .map(|w| dot(query, w))
                    .fold(f32::NEG_INFINITY, f32::max);
                (best, c)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = ranked
            .iter()
            .take(3)
            .map(|(s, c)| format!("{} {s:.3}", c.name))
            .collect();
        let Some(class) = class else {
            // A class the set does not contain: nothing should pass the floor.
            eprintln!("absent  {text:?}: top {}", top.join(", "));
            absent_best.push(ranked[0].0);
            absent_scores.extend(ranked.iter().map(|(s, _)| *s));
            continue;
        };
        present_queries += 1;
        let relevant = clips.iter().filter(|c| c.class == class).count();
        assert!(relevant > 0, "query {text:?}: no clip of class {class:?}");
        let first = ranked.iter().position(|(_, c)| c.class == class).unwrap();
        reciprocal_rank += 1.0 / (first + 1) as f64;
        for (i, &k) in ks.iter().enumerate() {
            let found = ranked[..k.min(ranked.len())]
                .iter()
                .filter(|(_, c)| c.class == class)
                .count();
            hits[i] += usize::from(found > 0);
            recall[i] += found as f64 / relevant.min(k) as f64;
        }
        for (s, c) in &ranked {
            if c.class == class {
                relevant_scores.push(*s);
            } else {
                irrelevant_scores.push(*s);
            }
        }
        eprintln!(
            "rank {:>2} {text:?} ({class}): top {}",
            first + 1,
            top.join(", ")
        );
    }

    let n = present_queries as f64;
    for (i, k) in ks.iter().enumerate() {
        eprintln!(
            "hit@{k} {:.3}  recall@{k} {:.3}",
            hits[i] as f64 / n,
            recall[i] / n
        );
    }
    eprintln!(
        "MRR {:.3} over {present_queries} queries",
        reciprocal_rank / n
    );

    relevant_scores.sort_by(f32::total_cmp);
    irrelevant_scores.sort_by(f32::total_cmp);
    absent_scores.sort_by(f32::total_cmp);
    for (label, v) in [
        ("relevant", &relevant_scores),
        ("irrelevant", &irrelevant_scores),
        ("absent-class", &absent_scores),
    ] {
        eprintln!(
            "{label} pairs {}: min {:.3} p01 {:.3} p10 {:.3} median {:.3} p90 {:.3} max {:.3}",
            v.len(),
            v[0],
            percentile(v, 0.01),
            percentile(v, 0.10),
            percentile(v, 0.5),
            percentile(v, 0.90),
            v[v.len() - 1]
        );
    }
    let at = |v: &[f32], f: f32| v.iter().filter(|&&s| s >= f).count();
    // What a floor would trade: relevant pairs kept against unrelated pairs
    // admitted, and absent-class queries that still return some clip.
    let mut sweep = vec![floor];
    sweep.extend((0..=9).map(|i| 0.58 + 0.02 * i as f32));
    for f in sweep {
        eprintln!(
            "floor {f:.2}: keeps {}/{} relevant, admits {}/{} irrelevant, {}/{} absent queries still match",
            at(&relevant_scores, f),
            relevant_scores.len(),
            at(&irrelevant_scores, f),
            irrelevant_scores.len(),
            absent_best.iter().filter(|&&b| b >= f).count(),
            absent_best.len()
        );
    }

    if task != APP_QUERY_TASK {
        return;
    }
    // The floor's contract is to drop only clearly unrelated windows, so it
    // must keep every relevant pair in this set.
    assert_eq!(
        at(&relevant_scores, floor),
        relevant_scores.len(),
        "minSoundScore {floor} drops relevant clips (lowest relevant score {:.3})",
        relevant_scores[0]
    );
    // Ranking bounds: measured values with about three queries of headroom.
    assert!(
        hits[1] as f64 / n >= HIT_AT_3_MIN,
        "hit@3 {:.3} < {HIT_AT_3_MIN}",
        hits[1] as f64 / n
    );
    assert!(
        reciprocal_rank / n >= MRR_MIN,
        "MRR {:.3} < {MRR_MIN}",
        reciprocal_rank / n
    );
}
