//! `vision_encode` end to end on the synthetic qwen35 vision install: the
//! bench opens the tower, times every stage, and its ledger payload carries
//! one row per page. The tiny tower's numbers mean nothing; that the stages
//! are measured, separated and summed the way the doc says is the claim.
#![cfg(target_os = "macos")]

use serde_json::json;
use turbospark_lab::kernel::vision_encode::{self, VisionEncode, ARMS};
use turbospark_lab::kernel::{BenchOpts, Case, KernelBench, Subject};

#[test]
fn the_tower_bench_separates_stream_reads_from_compute() {
    // `TSLAB_VISION_FIXTURE_DIR` keeps the install for a manual `tslab
    // kernel` run (the ANE vision programs need one with a tower).
    let keep = std::env::var_os("TSLAB_VISION_FIXTURE_DIR").map(std::path::PathBuf::from);
    let dir = keep.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("tslab-vision-encode-{}", std::process::id()))
    });
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    repack::build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &dir,
        256,
        4,
        "tslab-vision-encode",
        1,
    )
    .expect("synthetic vision install");

    let subject = Subject::open(dir.to_str().unwrap()).expect("subject");
    let cases = VisionEncode
        .cases(&subject, &BenchOpts::default())
        .expect("a tower case");
    assert_eq!(cases.len(), 1);
    assert_eq!(
        vision_encode::page_sizes(&cases[0]).unwrap(),
        vision_encode::PAGES.to_vec(),
        "the shipped case carries the oracle corpus"
    );

    // Two small pages through the same path; patch 4 makes the corpus huge.
    let mut params = cases[0].params.clone();
    params["pages"] = json!(["64x64", "32x48"]);
    let case = Case {
        label: cases[0].label.clone(),
        params,
    };
    let opts = BenchOpts {
        reps: 2,
        hash: true,
        ..BenchOpts::default()
    };
    let mut context = gpu::MetalContext::new().expect("metal");
    let result = VisionEncode
        .run(&mut context, &subject, &case, &opts)
        .expect("bench runs");

    assert_eq!(result.arms.len(), ARMS.len());
    let total = result.arm("total").unwrap();
    assert_eq!(total.points.len(), 2);
    assert!(
        total.points[0].x < total.points[1].x,
        "points ordered by patches"
    );
    for p in &total.points {
        assert!(p.ms > 0.0, "total measured");
        assert!(p.out_fnv.is_some(), "--hash hashes the embedding");
    }
    // The pread arm reads every block on the host, apart from compute.
    for arm in ["block_read", "block_gpu", "patch_embed_gpu", "merger_gpu"] {
        let a = result.arm(arm).unwrap();
        assert!(a.points.iter().all(|p| p.ms > 0.0), "{arm} measured");
    }
    let parts: f64 = ["block_read", "block_gpu", "patch_embed_gpu", "merger_gpu"]
        .iter()
        .map(|a| result.arm(a).unwrap().points[1].ms)
        .sum();
    assert!(
        parts <= total.points[1].ms * 1.05,
        "parts {parts} exceed total"
    );

    let payload = vision_encode::ledger_payload(&result).expect("a vision_encode payload");
    let pages = payload["pages"].as_array().unwrap();
    assert_eq!(pages.len(), 2);
    for p in pages {
        assert!(p["peak_footprint_bytes"].as_u64().unwrap() > 0);
        assert!(p["patches_per_s"].as_f64().unwrap() > 0.0);
        let share = p["stream_share"].as_f64().unwrap();
        assert!(share > 0.0 && share < 1.0, "stream share {share}");
        assert_eq!(p["reps"].as_u64(), Some(2), "the first rep is discarded");
        assert!(p["cold"].is_object());
    }
    assert_eq!(payload["tower"]["residency"], "pread");
    assert!(
        payload["tower"].get("vision_pages").is_none(),
        "pages are not repeated"
    );
    if keep.is_none() {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn small_pages(case: &Case) -> Case {
    let mut params = case.params.clone();
    params["pages"] = json!(["64x64", "32x48"]);
    Case {
        label: case.label.clone(),
        params,
    }
}

fn fnvs(result: &turbospark_lab::bench::BenchResult) -> Vec<Option<String>> {
    let total = result.arm("total").unwrap();
    total.points.iter().map(|p| p.out_fnv.clone()).collect()
}

#[test]
fn a_sidecar_on_a_text_only_trunk_times_the_same_tower_as_a_combined_install() {
    let root = std::env::temp_dir().join(format!("tslab-vision-sidecar-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (combined, trunk, sidecar) = (
        root.join("combined"),
        root.join("trunk"),
        root.join("sidecar"),
    );
    for d in [&combined, &trunk, &sidecar] {
        std::fs::create_dir_all(d).unwrap();
    }
    repack::build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &combined,
        256,
        4,
        "tslab-sidecar-combined",
        1,
    )
    .expect("combined install");
    repack::build_synthetic_qwen_gdn_dense_install(&trunk, 256, 4, "tslab-sidecar-trunk")
        .expect("text-only trunk");
    let tower = repack::build_synthetic_vision_sidecar(&sidecar, "tslab-sidecar").expect("sidecar");
    // The fixture's preprocessor_config.json is a `{}` placeholder. Give
    // both installs the same real one, so both runs preprocess alike and
    // the sidecar run reads its own file.
    let config = json!({
        "patch_size": tower.patch_size,
        "temporal_patch_size": tower.temporal_patch_size,
        "merge_size": tower.spatial_merge_size,
        "min_pixels": 1024,
        "max_pixels": 1 << 20,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    })
    .to_string();
    for d in [&combined, &sidecar] {
        std::fs::write(d.join("preprocessor_config.json"), &config).unwrap();
    }

    let opts = BenchOpts {
        reps: 2,
        hash: true,
        ..BenchOpts::default()
    };
    let with_sidecar = BenchOpts {
        vision_sidecar: Some(sidecar.clone()),
        ..opts.clone()
    };
    let mut context = gpu::MetalContext::new().expect("metal");

    let combined_subject = Subject::open(combined.to_str().unwrap()).unwrap();
    let case = small_pages(&VisionEncode.cases(&combined_subject, &opts).unwrap()[0]);
    let reference = VisionEncode
        .run(&mut context, &combined_subject, &case, &opts)
        .expect("combined install runs");

    let trunk_subject = Subject::open(trunk.to_str().unwrap()).unwrap();
    let err = VisionEncode
        .cases(&trunk_subject, &opts)
        .expect_err("a text-only trunk without a sidecar has no tower");
    assert!(err.contains("--vision-sidecar"), "{err}");
    let cases = VisionEncode.cases(&trunk_subject, &with_sidecar).unwrap();
    assert_eq!(
        cases[0].params["sidecar"],
        json!(sidecar.display().to_string())
    );
    let attached = VisionEncode
        .run(
            &mut context,
            &trunk_subject,
            &small_pages(&cases[0]),
            &with_sidecar,
        )
        .expect("trunk plus sidecar runs");
    let payload = vision_encode::ledger_payload(&attached).unwrap();
    assert_eq!(payload["tower"]["params_source"], "preprocessor_config");
    assert_eq!(
        payload["tower"]["sidecar"],
        json!(sidecar.display().to_string())
    );
    assert!(fnvs(&reference).iter().all(|f| f.is_some()));
    assert_eq!(
        fnvs(&attached),
        fnvs(&reference),
        "the same tower bytes embed the same pages"
    );

    let err = VisionEncode
        .cases(&combined_subject, &with_sidecar)
        .expect_err("a trunk with its own tower refuses a sidecar");
    assert!(err.contains("text-only"), "{err}");
    let _ = std::fs::remove_dir_all(&root);
}
