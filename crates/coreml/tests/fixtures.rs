//! Real Core ML round trips on the tiny packages from
//! `tools/gen_coreml_fixtures.py`. Each test compiles into its own scratch
//! cache, so the committed fixtures are never written to.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use turbospark_coreml::{
    compile_if_needed, compute_plan, supports, Capability, ComputeDevice, ComputeUnits,
    CoreMlError, CoreMlModel, DType, FeatureKind, LoadOptions, Tensor,
};

fn fixture(name: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name);
    if p.extension().and_then(|e| e.to_str()) == Some("mlpackage") {
        let weights_dir = p.join("Data").join("com.apple.CoreML").join("weights");
        let _ = std::fs::create_dir_all(&weights_dir);
    }
    p
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let d = std::env::temp_dir().join(format!(
            "ts-coreml-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        Self(d)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn open(name: &str, scratch: &Scratch, opts: &LoadOptions) -> CoreMlModel {
    CoreMlModel::open(&fixture(name), &scratch.0, opts).unwrap()
}

fn cpu() -> LoadOptions {
    LoadOptions::new(ComputeUnits::CpuOnly)
}

#[test]
fn add_f32_runs_and_reports_description() {
    let s = Scratch::new("add");
    let m = open("add_f32.mlpackage", &s, &cpu());
    let names: Vec<&str> = m.inputs().iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(
        m.inputs()[0].kind,
        FeatureKind::MultiArray {
            dtype: Some(DType::F32),
            shape: Some(vec![2, 3])
        }
    );
    let a = Tensor::f32(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let b = Tensor::f32(vec![2, 3], vec![10., 20., 30., 40., 50., 60.]).unwrap();
    let out = m.predict(&[("a", &a), ("b", &b)]).unwrap();
    assert_eq!(out["sum"].shape(), &[2, 3]);
    assert_eq!(
        out["sum"].as_f32().unwrap(),
        &[11., 22., 33., 44., 55., 66.]
    );
    assert_eq!(
        out["diff"].as_f32().unwrap(),
        &[-9., -18., -27., -36., -45., -54.]
    );
}

#[test]
fn add_f32_refuses_bad_inputs_before_core_ml() {
    let s = Scratch::new("add-bad");
    let m = open("add_f32.mlpackage", &s, &cpu());
    let a = Tensor::f32(vec![2, 3], vec![0.; 6]).unwrap();
    let wrong = Tensor::f32(vec![3, 2], vec![0.; 6]).unwrap();
    assert!(matches!(
        m.predict(&[("a", &a), ("b", &wrong)]),
        Err(CoreMlError::Shape { .. })
    ));
    assert!(matches!(
        m.predict(&[("a", &a)]),
        Err(CoreMlError::MissingFeature(_))
    ));
}

#[test]
fn f16_round_trip() {
    let s = Scratch::new("f16");
    let m = open("f16_scale.mlpackage", &s, &cpu());
    let x = Tensor::f16_from_f32("x", vec![1, 4], &[0.5, -1.25, 3.0, 1000.0]).unwrap();
    let out = m.predict(&[("x", &x)]).unwrap();
    assert_eq!(out["y"].dtype(), DType::F16);
    assert_eq!(out["y"].to_f32_vec().unwrap(), vec![1.0, -2.5, 6.0, 2000.0]);
}

#[test]
fn int32_output() {
    let s = Scratch::new("i32");
    let m = open("i32_argmax.mlpackage", &s, &cpu());
    let x = Tensor::f32(vec![1, 8], vec![0., 3., 1., 9., 2., -1., 4., 8.]).unwrap();
    let out = m.predict(&[("x", &x)]).unwrap();
    assert_eq!(out["idx"].as_i32().unwrap(), &[3]);
}

#[test]
fn compile_cache_is_reused() {
    let s = Scratch::new("cache");
    let pkg = fixture("add_f32.mlpackage");
    let first = compile_if_needed(&pkg, &s.0).unwrap();
    let mtime = std::fs::metadata(&first).unwrap().modified().unwrap();
    let second = compile_if_needed(&pkg, &s.0).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        std::fs::metadata(&second).unwrap().modified().unwrap(),
        mtime
    );
    // A compiled model passes straight through.
    assert_eq!(compile_if_needed(&first, &s.0).unwrap(), first);
    // Nothing but the one cache entry is left behind (no staging dirs).
    let entries: Vec<_> = std::fs::read_dir(&s.0).unwrap().flatten().collect();
    assert_eq!(entries.len(), 1);
}

#[test]
fn load_refuses_uncompiled_and_missing_paths() {
    assert!(matches!(
        CoreMlModel::load(&fixture("add_f32.mlpackage"), &cpu()),
        Err(CoreMlError::Load(_))
    ));
    assert!(matches!(
        CoreMlModel::load(Path::new("/nonexistent/x.mlmodelc"), &cpu()),
        Err(CoreMlError::Load(_))
    ));
}

#[test]
fn multifunction_selects_function() {
    let s = Scratch::new("multi");
    let x = Tensor::f32(vec![1, 4], vec![1., 2., 3., 4.]).unwrap();
    if !supports(Capability::MultiFunction) {
        let e = CoreMlModel::open(
            &fixture("multifunction.mlpackage"),
            &s.0,
            &cpu().with_function("triple"),
        )
        .unwrap_err();
        assert!(matches!(e, CoreMlError::OsTooOld { .. }));
        return;
    }
    let double = open("multifunction.mlpackage", &s, &cpu());
    let triple = open(
        "multifunction.mlpackage",
        &s,
        &cpu().with_function("triple"),
    );
    assert_eq!(
        double.predict(&[("x", &x)]).unwrap()["y"].as_f32().unwrap(),
        &[2., 4., 6., 8.]
    );
    assert_eq!(
        triple.predict(&[("x", &x)]).unwrap()["y"].as_f32().unwrap(),
        &[3., 6., 9., 12.]
    );
    assert!(CoreMlModel::open(
        &fixture("multifunction.mlpackage"),
        &s.0,
        &cpu().with_function("quadruple"),
    )
    .is_err());
}

#[test]
fn stateful_accumulator_carries_state() {
    let s = Scratch::new("state");
    if !supports(Capability::State) {
        return;
    }
    let m = open("accumulator.mlpackage", &s, &cpu());
    assert_eq!(m.states().len(), 1);
    let state_name = m.states()[0].name.clone();
    let x = Tensor::f16_from_f32("x", vec![1, 4], &[1., 2., 3., 4.]).unwrap();

    let mut st = m.new_state().unwrap();
    let y1 = m.predict_with_state(&mut st, &[("x", &x)]).unwrap();
    let y2 = m.predict_with_state(&mut st, &[("x", &x)]).unwrap();
    assert_eq!(y1["y"].to_f32_vec().unwrap(), vec![1., 2., 3., 4.]);
    assert_eq!(y2["y"].to_f32_vec().unwrap(), vec![2., 4., 6., 8.]);
    assert_eq!(
        st.read(&state_name).unwrap().to_f32_vec().unwrap(),
        vec![2., 4., 6., 8.]
    );

    // A fresh state starts from zero again.
    let mut st2 = m.new_state().unwrap();
    let y = m.predict_with_state(&mut st2, &[("x", &x)]).unwrap();
    assert_eq!(y["y"].to_f32_vec().unwrap(), vec![1., 2., 3., 4.]);

    // A state cannot be used with another model instance.
    let other = open("accumulator.mlpackage", &s, &cpu());
    assert!(other.predict_with_state(&mut st, &[("x", &x)]).is_err());

    // But its buffers can be copied into a state of that instance, which
    // then continues from them (the prefill-to-decode hand-off).
    let mut st3 = other.new_state().unwrap();
    st3.copy_from(&st).unwrap();
    let y3 = other.predict_with_state(&mut st3, &[("x", &x)]).unwrap();
    assert_eq!(y3["y"].to_f32_vec().unwrap(), vec![3., 6., 9., 12.]);
    // The source is untouched by the copy.
    assert_eq!(
        st.read(&state_name).unwrap().to_f32_vec().unwrap(),
        vec![2., 4., 6., 8.]
    );
}

#[test]
fn stateless_model_refuses_new_state() {
    let s = Scratch::new("nostate");
    let m = open("add_f32.mlpackage", &s, &cpu());
    assert!(m.new_state().is_err());
}

#[test]
fn model_is_shareable_across_threads() {
    let s = Scratch::new("threads");
    let m = std::sync::Arc::new(open("add_f32.mlpackage", &s, &cpu()));
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let m = m.clone();
            std::thread::spawn(move || {
                let a = Tensor::f32(vec![2, 3], vec![i as f32; 6]).unwrap();
                let b = Tensor::f32(vec![2, 3], vec![1.0; 6]).unwrap();
                let out = m.predict(&[("a", &a), ("b", &b)]).unwrap();
                assert_eq!(out["sum"].as_f32().unwrap(), &[i as f32 + 1.0; 6]);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn compute_plan_places_every_op_of_the_loaded_function() {
    let s = Scratch::new("plan");
    let compiled = compile_if_needed(&fixture("add_f32.mlpackage"), &s.0).unwrap();
    if !supports(Capability::ComputePlan) {
        assert!(matches!(
            compute_plan(&compiled, &cpu()),
            Err(CoreMlError::OsTooOld { .. })
        ));
        return;
    }
    let plan = compute_plan(&compiled, &cpu()).unwrap();
    let ops: Vec<(&str, &str)> = plan
        .ops
        .iter()
        .map(|o| (o.operator.as_str(), o.output.as_deref().unwrap_or("")))
        .collect();
    assert_eq!(ops, [("ios16.add", "sum"), ("ios16.sub", "diff")]);
    for op in &plan.ops {
        assert_eq!(op.function, "main");
        assert_eq!(op.preferred, Some(ComputeDevice::Cpu));
        assert_eq!(op.supported, [ComputeDevice::Cpu]);
    }
    let total: f64 = plan.ops.iter().filter_map(|o| o.cost).sum();
    assert!((total - 1.0).abs() < 1e-6, "cost weights sum to {total}");

    // The plan covers only the function the configuration binds; the
    // package's other functions get no placement.
    if supports(Capability::MultiFunction) {
        let multi = compile_if_needed(&fixture("multifunction.mlpackage"), &s.0).unwrap();
        let plan = compute_plan(&multi, &cpu().with_function("triple")).unwrap();
        let placed = |f: &str| {
            plan.ops
                .iter()
                .any(|o| o.function == f && o.preferred.is_some())
        };
        assert!(placed("triple"));
        assert!(!placed("double"));
    }
    assert!(matches!(
        compute_plan(&fixture("add_f32.mlpackage"), &cpu()),
        Err(CoreMlError::Load(_))
    ));
}
