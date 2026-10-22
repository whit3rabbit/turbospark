//! Isolated public API handoff probe. No inference scheduler changes.
//! cargo run -p turbospark-runtime --features ane-prefill --release \
//!   --example ane_async_probe -- PACKAGE.mlpackage 12
#[cfg(not(all(target_os = "macos", feature = "ane-prefill")))]
fn main() {
    panic!("requires macOS and --features ane-prefill");
}

#[cfg(all(target_os = "macos", feature = "ane-prefill"))]
fn main() {
    use half::f16;
    use std::path::Path;
    use std::time::Instant;
    use turbospark_coreml::{
        compute_plan, ComputeUnits, CoreMlModel, DType, FeatureKind, LoadOptions, SharedF16Plane,
        SharedPredictionBuffers,
    };
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(args.get(1).expect("PACKAGE.mlpackage"));
    let reps: usize = args.get(2).map_or(12, |s| s.parse().expect("reps"));
    let executor = args.get(3).map_or("paired", String::as_str);
    assert!(["paired", "sync", "async"].contains(&executor));
    let units = std::env::var("ANE_PROBE_UNITS").unwrap_or_else(|_| "ane".into());
    let opts = LoadOptions::new(ComputeUnits::parse(&units).expect("ANE_PROBE_UNITS"));
    let root = std::env::temp_dir().join("turbospark-ane-async-probe");
    let model = CoreMlModel::open(path, &root, &opts).expect("Core ML program");
    eprintln!("compute_plan={:?}", compute_plan(model.path(), &opts));
    let make = |spec: &turbospark_coreml::FeatureSpec| {
        let FeatureKind::MultiArray {
            dtype: Some(DType::F16),
            shape: Some(shape),
        } = &spec.kind
        else {
            panic!("fixed F16 shared plane required");
        };
        assert_eq!(&shape[..2], &[1, 1]);
        assert_eq!(shape.len(), 4);
        (
            spec.name.clone(),
            SharedF16Plane::new(shape[2], shape[3]).expect("plane"),
        )
    };
    let mut buffers = SharedPredictionBuffers {
        inputs: model.inputs().iter().map(make).collect(),
        outputs: model.outputs().iter().map(make).collect(),
    };
    let mut gpu = gpu::MetalContext::new().expect("Metal");
    for round in 0..=reps {
        let mut reference: Option<Vec<Vec<f16>>> = None;
        // Alternate paired order; discard the first pair. Not a real-model
        // or power benchmark. Compilation and allocation are outside timing.
        for is_async in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            if (executor == "sync" && is_async) || (executor == "async" && !is_async) {
                continue;
            }
            gpu::autorelease_pool(|| {
                for (input, (_, p)) in buffers.inputs.iter_mut().enumerate() {
                    let values: Vec<_> = (0..p.rows() * p.cols())
                        .map(|i| {
                            f16::from_f32(
                                0.01 + ((i * 37 + input * 7) % 29) as f32 / 4096.0
                                    + (round % 7) as f32 / 8192.0,
                            )
                        })
                        .collect();
                    p.write_f16(&values).unwrap();
                }
                let started = Instant::now();
                let pass = gpu.begin_pass_labeled("ANE probe producer");
                let wrappers: Vec<_> = buffers
                    .inputs
                    .iter()
                    .map(|(_, p)| {
                        gpu::wrap_page_aligned_no_copy(gpu.device(), p.base_ptr(), p.alloc_size())
                            .unwrap()
                    })
                    .collect();
                for ((_, p), b) in buffers.inputs.iter().zip(&wrappers) {
                    gpu::encode_scalar_mul(
                        &mut gpu,
                        &pass,
                        (b, 0),
                        2.0,
                        (p.rows() * p.stride()) as u32,
                    )
                    .unwrap();
                }
                let (producer, ready) = pass.commit_notifying();
                ready
                    .recv()
                    .expect("producer callback")
                    .expect("producer success");
                producer.wait();
                drop(wrappers);
                let ready_ms = started.elapsed().as_secs_f64() * 1000.0;
                let predict_start = Instant::now();
                if is_async {
                    let (result, returned) = model
                        .predict_shared_async(std::mem::take(&mut buffers))
                        .unwrap()
                        .wait();
                    buffers = returned;
                    result.expect("async output backings");
                } else {
                    let inputs: Vec<_> = buffers
                        .inputs
                        .iter()
                        .map(|(n, p)| (n.as_str(), p))
                        .collect();
                    let outputs: Vec<_> = buffers
                        .outputs
                        .iter()
                        .map(|(n, p)| (n.as_str(), p))
                        .collect();
                    model
                        .predict_shared(&inputs, &outputs)
                        .expect("sync output backings");
                }
                let prediction_ms = predict_start.elapsed().as_secs_f64() * 1000.0;
                let values: Vec<_> = buffers
                    .outputs
                    .iter()
                    .map(|(_, p)| p.read_f16().unwrap())
                    .collect();
                assert!(values.iter().flatten().all(|v| v.is_finite()));
                if let Some(expected) = &reference {
                    assert_eq!(&values, expected, "sync/async bits");
                } else {
                    reference = Some(values.clone());
                }
                // A real dependent GPU consumer, with checked completion.
                let pass = gpu.begin_pass_labeled("ANE probe consumer");
                let wrappers: Vec<_> = buffers
                    .outputs
                    .iter()
                    .map(|(_, p)| {
                        gpu::wrap_page_aligned_no_copy(gpu.device(), p.base_ptr(), p.alloc_size())
                            .unwrap()
                    })
                    .collect();
                for ((_, p), b) in buffers.outputs.iter().zip(&wrappers) {
                    gpu::encode_scalar_mul(
                        &mut gpu,
                        &pass,
                        (b, 0),
                        0.5,
                        (p.rows() * p.stride()) as u32,
                    )
                    .unwrap();
                }
                pass.commit_and_wait_checked().unwrap();
                drop(wrappers);
                for ((_, plane), before) in buffers.outputs.iter().zip(&values) {
                    let expected: Vec<_> = before
                        .iter()
                        .map(|v| f16::from_f32(v.to_f32() * 0.5))
                        .collect();
                    assert_eq!(
                        plane.read_f16().unwrap(),
                        expected,
                        "dependent GPU consumer"
                    );
                }
                let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
                if round > 0 {
                    println!("{{\"round\":{round},\"executor\":\"{}\",\"ready_ms\":{ready_ms},\"prediction_ms\":{prediction_ms},\"wall_ms\":{wall_ms}}}", if is_async {"async"} else {"sync"});
                }
            });
        }
    }
}
