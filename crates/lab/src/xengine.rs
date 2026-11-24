//! `tslab xengine-kl`: this port against llama.cpp on the SAME GGUF bytes,
//! the two-step flow `.claude/docs/cross-engine-kl.md` documents, as one
//! command. Step 1 runs `bench/tests/logit_dump.rs` on the install (its
//! family's `TURBOSPARK_*_INSTALL_DIR`), step 2 replays the dump's ids through
//! llama.cpp with `scripts/kld_llamacpp.py`, which also runs llama.cpp's own
//! shape floor (batched against cached) and backend floor (CPU against
//! Metal). tslab adds nothing to the numerics: it picks the env var, takes
//! the machine lock, and writes the report beside the experiment's ledger.
//!
//! Needs the checkout (it runs `cargo test` and the script from the source
//! tree tslab was built from), `uv`, and for a family the Homebrew bottle
//! predates (Inkling 2) a llama.cpp source build named by `--llama-cpp`.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::args::Args;

/// The `logit_dump` env var for an install's family, or None when the dump
/// has no arm for it.
pub fn dump_env_for(family: model_io::ModelFamily) -> Option<&'static str> {
    use model_io::ModelFamily as F;
    Some(match family {
        F::Inkling2Loop => "TURBOSPARK_INKLING2_INSTALL_DIR",
        F::Gemma4 => "TURBOSPARK_GEMMA4_INSTALL_DIR",
        F::Deepseek2 => "TURBOSPARK_DSV2_INSTALL_DIR",
        F::GptOss => "TURBOSPARK_GPTOSS_INSTALL_DIR",
        F::Flint => "TURBOSPARK_FLINT_INSTALL_DIR",
        _ => return None,
    })
}

pub fn xengine_kl_cmd(mut args: Args) -> Result<(), String> {
    let model = args.required("--model")?;
    let gguf = PathBuf::from(args.required("--gguf")?);
    let llama = args.take("--llama-cpp")?.map(PathBuf::from);
    let env_override = args.take("--install-env")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "xengine-kl".into());
    args.finish()?;
    let dir = catalog::resolve_model_arg(&model);
    let arch = repack::peek_manifest_arch(&dir)?;
    let install_env = match env_override {
        Some(e) => e,
        None => dump_env_for(arch.family)
            .ok_or_else(|| {
                format!(
                    "logit_dump has no arm for family {}; pass --install-env TURBOSPARK_..._INSTALL_DIR",
                    arch.family.as_str()
                )
            })?
            .to_string(),
    };
    if !gguf.is_file() {
        return Err(format!("{} is not a file", gguf.display()));
    }
    let root = crate::provenance::source_root();
    let lab = crate::paths::lab_dir();
    let _lock = crate::lock::MachineLock::acquire(&lab, &format!("tslab xengine-kl {experiment}"))?;
    let out = crate::paths::experiment_dir(&lab, &experiment)?.join("xengine");
    let port = out.join("port");
    std::fs::create_dir_all(&port).map_err(|e| format!("{}: {e}", port.display()))?;

    crate::say!("tslab xengine-kl: dumping this port's logits ({install_env})");
    run(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "test",
                "-p",
                "turbospark-bench",
                "--test",
                "logit_dump",
                "--release",
                "--",
                "--ignored",
                "--nocapture",
            ])
            .env(&install_env, &dir)
            .env("TURBOSPARK_LOGIT_DUMP_DIR", &port),
        "logit_dump",
    )?;

    crate::say!("tslab xengine-kl: replaying the ids through llama.cpp");
    let mut kl = Command::new("uv");
    kl.current_dir(root)
        .args([
            "run",
            "--python",
            "3.12",
            "--with",
            "numpy",
            "scripts/kld_llamacpp.py",
        ])
        .arg(&gguf)
        .arg(&port)
        .env("TURBOSPARK_LLAMACPP_DIR", out.join("llamacpp"));
    if let Some(l) = &llama {
        kl.env("TURBOSPARK_LLAMACPP_PREFIX", l);
    }
    let report = kl.output().map_err(|e| format!("uv: {e}"))?;
    if !report.status.success() {
        return Err(format!(
            "kld_llamacpp.py failed: {}",
            String::from_utf8_lossy(&report.stderr)
                .lines()
                .last()
                .unwrap_or("")
        ));
    }
    let text = String::from_utf8_lossy(&report.stdout).to_string();
    let path = out.join("report.json");
    std::fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("kld_llamacpp.py output: {e}"))?;
    let line = |label: &str, k: &serde_json::Value| {
        let f = &k["forward"];
        crate::say!(
            "  {label:28} mean KL {:.5}  median {:.5}  p99 {:.4}  top-1 {:.3}",
            f["mean"].as_f64().unwrap_or(f64::NAN),
            f["median"].as_f64().unwrap_or(f64::NAN),
            f["p99"].as_f64().unwrap_or(f64::NAN),
            k["top1_agreement"].as_f64().unwrap_or(f64::NAN)
        );
    };
    crate::say!("tslab xengine-kl: {} rows ({})", v["rows"], path.display());
    if let Some(obj) = v["vs_llamacpp_cached"].as_object() {
        for (name, arm) in obj {
            line(&format!("this port ({name})"), &arm["kl"]);
        }
    }
    line("llama.cpp shape floor", &v["kl_shape_floor"]);
    line("llama.cpp backend floor", &v["kl_backend_floor"]);
    Ok(())
}

fn run(cmd: &mut Command, what: &str) -> Result<(), String> {
    let status = cmd.status().map_err(|e| format!("{what}: {e}"))?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| format!("{what} exited with {status}"))
}

/// Unused outside tests; keeps the path helpers honest about where the
/// script lives.
pub fn script_path(root: &Path) -> PathBuf {
    root.join("scripts").join("kld_llamacpp.py")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dump_arm_tslab_names_exists_in_logit_dump() {
        let src = std::fs::read_to_string(
            crate::provenance::source_root().join("crates/bench/tests/logit_dump.rs"),
        )
        .unwrap();
        for f in model_io::ModelFamily::ALL {
            if let Some(env) = dump_env_for(f) {
                assert!(src.contains(env), "logit_dump has no {env} arm ({f:?})");
            }
        }
        assert!(script_path(crate::provenance::source_root()).is_file());
    }
}
