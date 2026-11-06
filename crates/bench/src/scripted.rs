use std::time::Instant;

use foundation::LogitValue;
use runtime::{run_raw_completion, GenerationConfig, RawDecodeProgress, ScriptedLogitProducer};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;

/// Three fixed prompts and a fixed seed, matching the frozen protocol's
/// shape. Content is arbitrary (no real model is being measured); only the
/// count (three) and the fact that they are fixed across runs matters.
pub const FIXED_PROMPTS: [&str; 3] = [
    "Explain the difference between a mutex and a semaphore.",
    "Write a short story about a lighthouse keeper.",
    "Summarize the plot of a three-act play in one paragraph.",
];
/// Fixed RNG seed used across benchmark runs for deterministic sampling.
pub const FIXED_SEED: u64 = 42;
/// Fixed generation budget for each bench prompt in scripted mode.
pub const FIXED_MAX_NEW_TOKENS: u32 = 64;

/// Measurement stats for one bench generation run.
pub struct RunStats {
    /// Number of generated new tokens.
    pub tokens: usize,
    /// Wall-clock decode duration in seconds.
    pub decode_seconds: f64,
}

impl RunStats {
    /// Generated tokens per second of decode time.
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_seconds <= 0.0 {
            0.0
        } else {
            self.tokens as f64 / self.decode_seconds
        }
    }
}

/// Runs the benchmark harness in scripted mode against fixed prompts.
pub fn run_scripted_mode(tok: &MfTokenizer) -> std::process::ExitCode {
    println!(
        "turbospark-bench: {} fixed prompts, seed {FIXED_SEED}, scripted producer (see module docs)",
        FIXED_PROMPTS.len()
    );
    println!(
        "{:<10} {:>12} {:>14} {:>16}",
        "prompt", "tokens", "decode_secs", "tokens_per_second"
    );

    let mut measured: Vec<RunStats> = Vec::with_capacity(FIXED_PROMPTS.len());
    for (index, prompt) in FIXED_PROMPTS.iter().enumerate() {
        // Warmup run: discarded, matching the frozen protocol.
        let _ = run_once(tok, prompt);
        // Measured run.
        match run_once(tok, prompt) {
            Ok(stats) => {
                println!(
                    "{:<10} {:>12} {:>14.4} {:>16.1}",
                    index,
                    stats.tokens,
                    stats.decode_seconds,
                    stats.tokens_per_second()
                );
                measured.push(stats);
            }
            Err(e) => {
                eprintln!("prompt {index} failed: {e}");
                return std::process::ExitCode::from(1);
            }
        }
    }

    let total_tokens: usize = measured.iter().map(|r| r.tokens).sum();
    let total_seconds: f64 = measured.iter().map(|r| r.decode_seconds).sum();
    let aggregate = if total_seconds > 0.0 {
        total_tokens as f64 / total_seconds
    } else {
        0.0
    };
    println!("aggregate: {total_tokens} tokens in {total_seconds:.4}s = {aggregate:.1} tokens/sec");

    std::process::ExitCode::SUCCESS
}

fn run_once(tok: &MfTokenizer, prompt: &str) -> Result<RunStats, String> {
    let vocab_size = tok.vocab_size;
    let h_id = tok.token_to_id("h").unwrap_or(0) as usize;
    let prompt_ids = tok.encode(prompt, true);

    let steps: Vec<Vec<LogitValue>> = (0..prompt_ids.len() + FIXED_MAX_NEW_TOKENS as usize)
        .map(|_| one_hot(vocab_size, h_id))
        .collect();
    let mut producer = ScriptedLogitProducer::new(steps);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, Some(FIXED_SEED))
            .map_err(|e| e.to_string())?,
        max_new_tokens: FIXED_MAX_NEW_TOKENS,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };

    let start = Instant::now();
    let mut tokens = 0usize;
    let result = run_raw_completion(
        &mut producer,
        tok,
        &prompt_ids,
        &config,
        8192,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { .. } = e {
                tokens += 1;
            }
        },
    )
    .map_err(|e| e.to_string())?;
    let decode_seconds = start.elapsed().as_secs_f64();

    Ok(RunStats {
        tokens: result.new_tokens.max(tokens),
        decode_seconds,
    })
}

/// Runs the benchmark harness in real mode using a synthetic model on Metal.
#[cfg(target_os = "macos")]
pub fn run_real_mode(tok: &MfTokenizer) -> std::process::ExitCode {
    let guard = match TempInstallDir::create("turbospark-bench-real") {
        Ok(g) => g,
        Err(e) => {
            eprintln!("failed to create temp install dir: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    let dir = guard.path().to_path_buf();
    let arch = match repack::build_synthetic_gemma4_install(&dir, tok.vocab_size as i64, 2, "bench")
    {
        Ok(a) => a,
        Err(e) => {
            eprintln!("failed to build synthetic install: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    let mut runner = match runtime::RealForwardRunner::open(&dir, arch) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open RealForwardRunner: {e}");
            return std::process::ExitCode::from(1);
        }
    };

    println!(
        "turbospark-bench: {} fixed prompts, seed {FIXED_SEED}, REAL forward pass \
         (synthetic tiny dense model; measures the real GPU dispatch path, \
         not production throughput)",
        FIXED_PROMPTS.len()
    );
    println!(
        "{:<10} {:>12} {:>14} {:>16}",
        "prompt", "tokens", "decode_secs", "tokens_per_sec"
    );

    let mut measured: Vec<RunStats> = Vec::with_capacity(FIXED_PROMPTS.len());
    for (index, prompt) in FIXED_PROMPTS.iter().enumerate() {
        let _ = run_once_real(&mut runner, tok, prompt);
        match run_once_real(&mut runner, tok, prompt) {
            Ok(stats) => {
                println!(
                    "{:<10} {:>12} {:>14.4} {:>16.1}",
                    index,
                    stats.tokens,
                    stats.decode_seconds,
                    stats.tokens_per_second()
                );
                measured.push(stats);
            }
            Err(e) => {
                eprintln!("prompt {index} failed: {e}");
                return std::process::ExitCode::from(1);
            }
        }
    }

    let total_tokens: usize = measured.iter().map(|r| r.tokens).sum();
    let total_seconds: f64 = measured.iter().map(|r| r.decode_seconds).sum();
    let aggregate = if total_seconds > 0.0 {
        total_tokens as f64 / total_seconds
    } else {
        0.0
    };
    println!("aggregate: {total_tokens} tokens in {total_seconds:.4}s = {aggregate:.1} tokens/sec");
    // `runner` drops before `guard` (reverse declaration order), so the
    // install is closed before the guard removes it.
    std::process::ExitCode::SUCCESS
}

#[cfg(target_os = "macos")]
fn run_once_real(
    runner: &mut runtime::RealForwardRunner,
    tok: &MfTokenizer,
    prompt: &str,
) -> Result<RunStats, String> {
    let vocab_size = tok.vocab_size;
    let prompt_ids = tok.encode(prompt, true);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, Some(FIXED_SEED))
            .map_err(|e| e.to_string())?,
        max_new_tokens: FIXED_MAX_NEW_TOKENS,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };

    let start = Instant::now();
    let mut tokens = 0usize;
    let result = run_raw_completion(runner, tok, &prompt_ids, &config, 4096, vocab_size, |e| {
        if let RawDecodeProgress::Token { .. } = e {
            tokens += 1;
        }
    })
    .map_err(|e| e.to_string())?;
    let decode_seconds = start.elapsed().as_secs_f64();

    Ok(RunStats {
        tokens: result.new_tokens.max(tokens),
        decode_seconds,
    })
}

/// Stub for real mode on non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub fn run_real_mode(_tok: &MfTokenizer) -> std::process::ExitCode {
    eprintln!("--real requires macOS (Metal)");
    std::process::ExitCode::from(2)
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<LogitValue> {
    let mut v = vec![LogitValue::from_f32(0.0); vocab_size];
    if index < vocab_size {
        v[index] = LogitValue::from_f32(1.0);
    }
    v
}

/// Scratch install directory for `--real`, removed on drop.
///
/// The name carries the pid plus a time nonce and is created with
/// `create_dir` (not `create_dir_all`), so a stale or planted directory or
/// symlink at the same path is never reused: `AlreadyExists` moves on to the
/// next name. The guard makes every early-return path clean up too.
/// Only `run_real_mode` (macOS) and the tests use it.
#[cfg(any(target_os = "macos", test))]
pub(crate) struct TempInstallDir {
    path: std::path::PathBuf,
}

#[cfg(any(target_os = "macos", test))]
impl TempInstallDir {
    pub(crate) fn create(prefix: &str) -> std::io::Result<Self> {
        let base = std::env::temp_dir();
        let pid = std::process::id();
        for attempt in 0u32..64 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = base.join(format!("{prefix}-{pid}-{nanos:x}-{attempt}"));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create a unique temp install dir",
        ))
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }
}

#[cfg(any(target_os = "macos", test))]
impl Drop for TempInstallDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::TempInstallDir;

    #[test]
    fn temp_install_dir_is_exclusive_per_call() {
        let a = TempInstallDir::create("turbospark-bench-tmptest-excl").unwrap();
        let b = TempInstallDir::create("turbospark-bench-tmptest-excl").unwrap();
        // A colliding name must never hand back a directory that already
        // existed (a stale run's or a planted one).
        assert_ne!(a.path(), b.path());
        assert!(a.path().is_dir() && b.path().is_dir());
    }

    #[test]
    fn temp_install_dir_refuses_preexisting_predictable_path() {
        let planted = std::env::temp_dir().join(format!(
            "turbospark-bench-tmptest-planted-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&planted).unwrap();
        let g = TempInstallDir::create("turbospark-bench-tmptest-planted").unwrap();
        assert_ne!(g.path(), planted.as_path());
        std::fs::remove_dir_all(&planted).unwrap();
    }

    #[test]
    fn temp_install_dir_is_removed_on_drop() {
        let g = TempInstallDir::create("turbospark-bench-tmptest-drop").unwrap();
        let path = g.path().to_path_buf();
        std::fs::write(path.join("weights.bin"), b"x").unwrap();
        drop(g);
        // Early error returns in run_real_mode drop the guard, so the
        // synthetic install must not outlive them.
        assert!(!path.exists());
    }
}
