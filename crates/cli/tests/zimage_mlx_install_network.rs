//! Full-install gate for one pinned published Z-Image MLX variant.
//!
//! The gate is intentionally opt-in and selects one variant per invocation:
//!
//! ```text
//! TURBOSPARK_ZIMAGE_MLX_VARIANT=4bit \
//! TURBOSPARK_ZIMAGE_MLX_INSTALL_DIR=~/models/z-image-turbo-mlx-4bit.image.gturbo \
//!   cargo test -p turbospark-cli --test zimage_mlx_install_network --release \
//!   -- --ignored --nocapture
//! ```
//!
//! `pull-image` stages the source before packing, so the destination volume
//! must hold both the source download and the resulting image install.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use image::{image_quantization_label, ImageManifest};

struct Variant {
    name: &'static str,
    alias: &'static str,
    repo: &'static str,
    revision: &'static str,
    quantization: &'static str,
}

const VARIANTS: [Variant; 4] = [
    Variant {
        name: "2bit",
        alias: "z-image-turbo-mlx-2bit",
        repo: "andrevp/Z-Image-Turbo-MLX-2bit",
        revision: "32b4e9ceb3a813485027b1ea942f199608fb8200",
        quantization: "mlx-affine-linear-weights-group-64-bits-2",
    },
    Variant {
        name: "4bit",
        alias: "z-image-turbo-mlx-4bit",
        repo: "andrevp/Z-Image-Turbo-MLX-4bit",
        revision: "9adc576198c9126874792d35569b53cf2f45a03c",
        quantization: "mlx-affine-linear-weights-group-64-bits-4",
    },
    Variant {
        name: "8bit",
        alias: "z-image-turbo-mlx-8bit",
        repo: "andrevp/Z-Image-Turbo-MLX-8bit",
        revision: "c9f70995562299b1eda9b9145a94dd7a5a1ae0d6",
        quantization: "mlx-affine-linear-weights-group-64-bits-8",
    },
    Variant {
        name: "fp16",
        alias: "z-image-turbo-mlx-fp16",
        repo: "andrevp/Z-Image-Turbo-MLX",
        revision: "e186d7d65d66883270671fcee05324178928ea03",
        quantization: "unquantized",
    },
];

fn binary_path() -> PathBuf {
    let mut path = std::env::current_exe().expect("test binary path");
    path.pop();
    path.pop();
    path.push("turbospark-model");
    path
}

fn directory_bytes(root: &Path) -> u64 {
    fs::read_dir(root)
        .unwrap_or_else(|error| panic!("read installed image {}: {error}", root.display()))
        .map(|entry| {
            let path = entry
                .unwrap_or_else(|error| panic!("read installed image entry: {error}"))
                .path();
            let metadata = fs::symlink_metadata(&path)
                .unwrap_or_else(|error| panic!("stat installed image {}: {error}", path.display()));
            if metadata.is_dir() {
                directory_bytes(&path)
            } else if metadata.is_file() {
                metadata.len()
            } else {
                0
            }
        })
        .sum()
}

fn output_bytes(output: &[u8], prefix: &str) -> u64 {
    String::from_utf8_lossy(output)
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("missing {prefix:?} in command output"))
}

fn peak_rss_bytes(stderr: &[u8]) -> u64 {
    String::from_utf8_lossy(stderr)
        .lines()
        .find_map(|line| {
            line.contains("maximum resident set size")
                .then(|| line.split_whitespace().find_map(|value| value.parse().ok()))
                .flatten()
        })
        .unwrap_or_else(|| panic!("missing macOS peak RSS in /usr/bin/time output"))
}

#[test]
#[ignore = "network and disk: stages one pinned multi-gigabyte Z-Image MLX source"]
fn installs_one_selected_published_zimage_mlx_variant() {
    let Some(variant_name) = std::env::var_os("TURBOSPARK_ZIMAGE_MLX_VARIANT") else {
        eprintln!(
            "set TURBOSPARK_ZIMAGE_MLX_VARIANT to 2bit, 4bit, 8bit, or fp16; skipping full install"
        );
        return;
    };
    let variant_name = variant_name.to_string_lossy();
    let variant = VARIANTS
        .iter()
        .find(|variant| variant.name == variant_name)
        .unwrap_or_else(|| panic!("unsupported Z-Image MLX variant {variant_name:?}"));
    let install_dir = std::env::var_os("TURBOSPARK_ZIMAGE_MLX_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "set TURBOSPARK_ZIMAGE_MLX_INSTALL_DIR to a new destination for the full install"
            )
        });

    let repo = format!("{}@{}", variant.repo, variant.revision);
    let started = Instant::now();
    let mut timed = Command::new("/usr/bin/time");
    timed.arg("-l").arg(binary_path());
    timed.env(
        "TURBOSPARK_HOME",
        std::env::temp_dir().join(format!(
            "turbospark-zimage-mlx-install-{}",
            std::process::id()
        )),
    );
    let output = timed
        .args([
            "pull-image",
            "--repo",
            &repo,
            "--alias",
            variant.alias,
            "--out",
            install_dir.to_str().expect("install path is UTF-8"),
        ])
        .output()
        .expect("turbospark-model runs");
    assert!(
        output.status.success(),
        "{} install failed\nstdout:\n{}\nstderr:\n{}",
        variant.name,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let manifest = ImageManifest::load(&install_dir).expect("load installed image manifest");
    manifest
        .verify_files(&install_dir)
        .expect("verify installed image files");
    assert_eq!(
        image_quantization_label(&manifest).expect("derive installed quantization label"),
        variant.quantization
    );

    let source_bytes = output_bytes(&output.stdout, "downloaded ");
    let install_bytes = directory_bytes(&install_dir);
    let peak_rss_bytes = peak_rss_bytes(&output.stderr);
    eprintln!(
        "zimage_mlx_install_benchmark variant={} source_bytes={} install_bytes={} elapsed_ms={} peak_rss_bytes={}",
        variant.name,
        source_bytes,
        install_bytes,
        started.elapsed().as_millis(),
        peak_rss_bytes
    );
}
