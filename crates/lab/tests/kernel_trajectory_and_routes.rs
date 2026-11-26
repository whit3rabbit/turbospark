//! GPU checks for `gdn_trajectory` and for `moe_routed`'s route sources, on
//! small synthetic shapes: no installs.
#![cfg(target_os = "macos")]

use gpu::{GdnShape, MetalContext};
use serde_json::json;
use turbospark_lab::kernel::gdn_trajectory::{
    fault_self_test, run_pair, Config, Fault, GdnTrajectory, Mismatch, Route,
};
use turbospark_lab::kernel::{moe_routed, BenchOpts, Case, KernelBench, Subject};

fn config(steps: usize, chunk: usize) -> Config {
    Config {
        shape: GdnShape {
            num_k_heads: 2,
            num_v_heads: 4,
            key_head_dim: 32,
            value_head_dim: 32,
            conv_kernel_size: 4,
        },
        layers: 3,
        steps,
        prefill: 70,
        chunk,
    }
}

#[test]
fn one_flipped_ulp_is_reported_at_its_layer_and_step() {
    let mut context = MetalContext::new().expect("metal");
    for (route, chunk) in [
        (Route::Decode, 1),
        (Route::Prefill, 4),
        (Route::LazyFold, 4),
    ] {
        let cfg = config(12, chunk);
        let clean = run_pair(&mut context, &cfg, [route, route], false, None).unwrap();
        assert_eq!(clean.first, None, "{route:?} against itself");
        assert_eq!(clean.digests[0], clean.digests[1]);
        // A state that never moved would make every comparison pass.
        let shorter = run_pair(&mut context, &config(4, chunk), [route, route], false, None);
        assert_ne!(shorter.unwrap().digests, clean.digests);
        let fault = Fault { layer: 2, step: 5 };
        let hurt = run_pair(&mut context, &cfg, [route, route], false, Some(fault)).unwrap();
        assert_eq!(
            hurt.first,
            Some(Mismatch {
                layer: 2,
                step: 5 / chunk * chunk + chunk - 1,
                what: "state"
            }),
            "{route:?}"
        );
        fault_self_test(&mut context, &cfg, route).expect("self-test");
    }
}

#[test]
fn the_bench_reports_both_pairs_with_the_self_test() {
    let mut context = MetalContext::new().expect("metal");
    let arch = repack::tiny_qwen_gdn_dense_arch(512, 4);
    let subject = Subject {
        dir: std::env::temp_dir().join("tslab-gdn-trajectory-no-install"),
        family: arch.family.as_str().to_string(),
        arch: Some(arch),
        audio_config: None,
        inventory: Err("synthetic architecture".into()),
    };
    let opts = BenchOpts::default();
    let cases = GdnTrajectory.cases(&subject, &opts).unwrap();
    assert_eq!(cases.len(), 2);
    for mut case in cases {
        case.params["steps"] = json!("20");
        case.params["prefill"] = json!("33");
        let r = GdnTrajectory
            .run(&mut context, &subject, &case, &opts)
            .expect("runs");
        println!("{}: {}", case.label, r.notes[0]);
        assert_eq!(r.subject["steps"], 20);
        assert_eq!(r.subject["rows"], 53);
        assert_eq!(r.subject["fault_self_test"]["detected"], true);
        assert_eq!(r.subject["fault_self_test"]["step"], 19);
        assert_eq!(r.subject["order_stable"], true);
        assert_eq!(r.arms.len(), 2);
        assert_eq!(
            r.subject["bit_exact"] == true,
            r.subject["first_mismatch"].is_null()
        );
    }
}

fn moe_case(routes: Option<String>) -> (Subject, Case) {
    let subject = Subject {
        dir: std::env::temp_dir().join("tslab-moe-routes-no-install"),
        family: "synthetic".into(),
        arch: None,
        audio_config: None,
        inventory: Err("synthetic".into()),
    };
    let mut params = json!({"d": 256, "f": 128, "top_k": 4, "experts": 16, "silu": true});
    if let Some(r) = routes {
        params["routes"] = json!(r);
    }
    (
        subject,
        Case {
            label: "d256_f128_k4".into(),
            params,
        },
    )
}

#[test]
fn route_sources_run_and_the_sorted_grid_changes_no_bit() {
    let mut context = MetalContext::new().expect("metal");
    let capture = std::env::temp_dir().join(format!("tslab-capture-{}.json", std::process::id()));
    // Two layers of eight passes, overlapping experts, rank order mixed.
    let layer: Vec<u32> = (0..8u32)
        .flat_map(|p| [(p * 3) % 16, (p * 3 + 7) % 16, 5 + (p % 2), 15 - (p % 4)])
        .collect();
    std::fs::write(
        &capture,
        json!({"num_layers": 2, "num_experts": 16, "top_k": 4, "trace": [layer, layer]})
            .to_string(),
    )
    .unwrap();
    let opts = BenchOpts {
        rows: vec![1, 2, 4, 8],
        reps: 2,
        hash: true,
        ..BenchOpts::default()
    };
    for routes in [
        None,
        Some("zipf:1.1".to_string()),
        Some(format!("trace:{}", capture.display())),
    ] {
        let (subject, case) = moe_case(routes.clone());
        let r = moe_routed::MoeRouted
            .run(&mut context, &subject, &case, &opts)
            .unwrap_or_else(|e| panic!("{routes:?}: {e}"));
        let labels: Vec<&str> = r.arms.iter().map(|a| a.label.as_str()).collect();
        assert_eq!(labels, moe_routed::ARMS);
        for m in 0..opts.rows.len() {
            let hash = |a: usize| r.arms[a].points[m].out_fnv.clone().expect("hashed");
            assert_eq!(hash(1), hash(2), "{routes:?}: sorted grid at point {m}");
            assert_eq!(hash(0), hash(1), "{routes:?}: batched against sequential");
        }
        assert_eq!(r.break_even.len(), 2);
        assert!(r.subject["routing"][3]["rows_per_expert"].is_object());
        assert!(r.subject["tile_pin"]["buckets"].is_array());
        assert_eq!(r.case == "d256_f128_k4", routes.is_none(), "{}", r.case);
    }
    // A capture of another shape is refused, not reshaped.
    let (subject, mut case) = moe_case(Some(format!("trace:{}", capture.display())));
    case.params["experts"] = json!(32);
    let err = moe_routed::MoeRouted
        .run(&mut context, &subject, &case, &opts)
        .unwrap_err();
    assert!(err.contains("the install routes"), "{err}");
    let _ = std::fs::remove_file(&capture);
}
