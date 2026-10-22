#![cfg(all(target_os = "macos", feature = "ane-prefill"))]
//! Real shared-plane predictions on a tiny generated fixture, no download.
use half::f16;
use std::path::Path;
use turbospark_coreml::{
    supports, Capability, ComputeUnits, CoreMlModel, LoadOptions, SharedF16Plane,
    SharedPredictionBuffers,
};

fn model() -> CoreMlModel {
    assert!(
        supports(Capability::AsyncPrediction),
        "test requires macOS 14"
    );
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/shared_double.mlpackage");
    let cache = std::env::temp_dir().join(format!("ane-async-fixture-{}", std::process::id()));
    CoreMlModel::open(&fixture, &cache, &LoadOptions::new(ComputeUnits::CpuOnly)).unwrap()
}

fn buffers(value: f32) -> SharedPredictionBuffers {
    let mut input = SharedF16Plane::new(2, 32).unwrap();
    input.write_f16(&[f16::from_f32(value); 64]).unwrap();
    SharedPredictionBuffers {
        inputs: vec![("x".into(), input)],
        outputs: vec![("y".into(), SharedF16Plane::new(2, 32).unwrap())],
    }
}

#[test]
fn fixture_bytes_match_the_recorded_generator_digest() {
    use sha2::{Digest, Sha256};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/shared_double.mlpackage");
    fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                files(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    files(&root, &mut paths);
    paths.sort();
    let mut digest = Sha256::new();
    for path in paths {
        digest.update(
            path.strip_prefix(&root)
                .unwrap()
                .to_str()
                .unwrap()
                .as_bytes(),
        );
        digest.update(std::fs::read(path).unwrap());
    }
    let manifest = std::fs::read_to_string(root.with_extension("manifest.json")).unwrap();
    assert!(manifest.contains(&format!("{:x}", digest.finalize())));
}

#[test]
fn async_matches_sync_and_recovers_owned_planes_repeatedly() {
    let m = model();
    let mut b = buffers(0.0);
    for value in [1.0, -2.0, 3.5, 0.25] {
        b.inputs[0]
            .1
            .write_f16(&[f16::from_f32(value); 64])
            .unwrap();
        m.predict_shared(&[("x", &b.inputs[0].1)], &[("y", &b.outputs[0].1)])
            .unwrap();
        let reference = b.outputs[0].1.read_f16().unwrap();
        b.outputs[0].1.write_f16(&[f16::NAN; 64]).unwrap();
        let (result, returned) = m.predict_shared_async(b).unwrap().wait();
        result.unwrap();
        b = returned;
        assert_eq!(b.outputs[0].1.read_f16().unwrap(), reference);
        assert_eq!(reference, vec![f16::from_f32(2.0 * value); 64]);
    }
}

#[test]
fn dropping_outstanding_ticket_drains_and_releases_model() {
    let m = model();
    drop(m.predict_shared_async(buffers(1.0)).unwrap());
    let (result, b) = m.predict_shared_async(buffers(2.0)).unwrap().wait();
    result.unwrap();
    assert_eq!(
        b.outputs[0].1.read_f16().unwrap(),
        vec![f16::from_f32(4.0); 64]
    );
}

#[test]
fn invalid_names_shapes_and_duplicate_bindings_refused_before_submission() {
    let m = model();
    let mut b = buffers(1.0);
    b.inputs[0].0 = "missing".into();
    assert!(m.predict_shared_async(b).is_err());
    let mut b = buffers(1.0);
    b.outputs[0].1 = SharedF16Plane::new(1, 32).unwrap();
    assert!(m.predict_shared_async(b).is_err());
    let mut b = buffers(1.0);
    b.inputs
        .push(("x".into(), SharedF16Plane::new(2, 32).unwrap()));
    assert!(m.predict_shared_async(b).is_err());
}

#[test]
fn independent_sessions_cannot_complete_each_others_predictions() {
    let first = model();
    let second = model();
    let a = first.predict_shared_async(buffers(1.0)).unwrap();
    let b = second.predict_shared_async(buffers(7.0)).unwrap();
    let (result, b) = b.wait();
    result.unwrap();
    let (result, a) = a.wait();
    result.unwrap();
    assert_eq!(
        a.outputs[0].1.read_f16().unwrap(),
        vec![f16::from_f32(2.0); 64]
    );
    assert_eq!(
        b.outputs[0].1.read_f16().unwrap(),
        vec![f16::from_f32(14.0); 64]
    );
}

#[test]
fn concurrent_cold_loads_publish_complete_models() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/shared_double.mlpackage");
    let cache = std::env::temp_dir().join(format!(
        "ane-cold-load-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!cache.exists());
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    let model = CoreMlModel::open(
                        &fixture,
                        &cache,
                        &LoadOptions::new(ComputeUnits::CpuOnly),
                    )
                    .unwrap();
                    let (result, output) = model.predict_shared_async(buffers(3.0)).unwrap().wait();
                    result.unwrap();
                    assert_eq!(
                        output.outputs[0].1.read_f16().unwrap(),
                        vec![f16::from_f32(6.0); 64]
                    );
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    });
    std::fs::remove_dir_all(cache).unwrap();
}

#[test]
fn plane_host_copies_honor_padded_rows_and_reject_wrong_lengths() {
    let mut plane = SharedF16Plane::new(2, 33).unwrap();
    assert!(plane.stride() > plane.cols());
    let values: Vec<_> = (0..66).map(|i| f16::from_f32(i as f32 / 8.0)).collect();
    plane.write_f16(&values).unwrap();
    assert_eq!(plane.read_f16().unwrap(), values);
    assert!(plane.write_f16(&values[..65]).is_err());
}

#[test]
fn stateful_models_are_refused_before_stateless_submission() {
    if !supports(Capability::State) {
        return;
    }
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/accumulator.mlpackage");
    let cache = std::env::temp_dir().join(format!("ane-async-stateful-{}", std::process::id()));
    let model =
        CoreMlModel::open(&fixture, &cache, &LoadOptions::new(ComputeUnits::CpuOnly)).unwrap();
    assert!(!model.states().is_empty());
    let result = model.predict_shared_async(buffers(1.0));
    assert!(
        matches!(result, Err(turbospark_coreml::CoreMlError::Prediction(message)) if message.contains("stateless"))
    );
}
