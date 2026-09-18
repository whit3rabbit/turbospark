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

use std::path::PathBuf;
use std::process::Command;

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

fn binary() -> Command {
    let mut path = std::env::current_exe().expect("test binary path");
    path.pop();
    path.pop();
    path.push("turbospark-model");
    let mut command = Command::new(path);
    command.env(
        "TURBOSPARK_HOME",
        std::env::temp_dir().join(format!(
            "turbospark-zimage-mlx-install-{}",
            std::process::id()
        )),
    );
    command
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
    let output = binary()
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
}
