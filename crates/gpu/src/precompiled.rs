//! Optional embedded Metal IR, keyed by the exact composed MSL bytes.
//! Function constants stay unspecialized until the normal pipeline path runs.

struct BundledLibrary {
    hash: &'static str,
    source_len: usize,
    compiler_identity: &'static str,
    bytes: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/precompiled_registry.rs"));

fn enabled(opt_in: Option<&str>, precise_math: bool) -> bool {
    opt_in == Some("1") && !precise_math
}

fn lookup(source: &str) -> Option<&'static BundledLibrary> {
    let hash = model_io::hash_data(source.as_bytes());
    LIBRARIES
        .iter()
        .find(|entry| entry.source_len == source.len() && entry.hash == hash)
}

pub(crate) fn compiler_identity(source: &str) -> Option<&'static str> {
    lookup(source).map(|entry| entry.compiler_identity)
}

pub(crate) fn load(device: &metal::DeviceRef, source: &'static str) -> Option<metal::Library> {
    if !enabled(
        std::env::var("TURBOSPARK_METAL_PRECOMPILED")
            .ok()
            .as_deref(),
        std::env::var_os("TURBOSPARK_METAL_PRECISE_MATH").is_some(),
    ) {
        return None;
    }
    let entry = lookup(source)?;
    // Toolchain/device incompatibility is an optimization miss, not a model error.
    device.new_library_with_data(entry.bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precise_math_and_explicit_opt_in_select_the_source_fallback() {
        assert!(enabled(Some("1"), false));
        assert!(!enabled(Some("1"), true));
        assert!(!enabled(None, false));
        assert!(!enabled(Some("true"), false));
        assert!(!enabled(Some("0"), false));
    }

    #[test]
    fn inventory_preserves_composite_separators_and_iq_codebooks() {
        let composites = [
            concat!(
                include_str!("shaders/dequant_int4.metal"),
                "\n",
                include_str!("shaders/gdn.metal")
            ),
            concat!(
                include_str!("shaders/dequant_iq_lowbit_tables.metal"),
                "\n",
                include_str!("shaders/dequant_iq.metal")
            ),
            crate::moe_prefill_batch_gguf::SOURCE,
            crate::attention_decode::SOURCE,
        ];
        for source in composites {
            assert!(
                INVENTORY.iter().any(|(_, item)| *item == source),
                "runtime composition missing from build inventory"
            );
            if BUILD_COMPLETE {
                assert!(lookup(source).is_some());
            }
        }
    }

    #[test]
    fn changing_source_content_cannot_select_an_old_library() {
        if !BUILD_COMPLETE {
            return;
        }
        let source = include_str!("shaders/utility.metal");
        assert!(lookup(source).is_some());
        // Same length and pointer-independent content identity both matter here.
        let mut changed = source.to_string();
        changed.replace_range(..1, " ");
        assert!(lookup(&changed).is_none());
        assert!(compiler_identity(&changed).is_none());
        assert!(lookup(&format!("{source}\n")).is_none());
        assert!(lookup("").is_none());
    }

    #[test]
    fn packaged_identity_covers_the_actual_embedded_metallib() {
        for (label, source) in INVENTORY {
            let Some(entry) = lookup(source) else {
                continue;
            };
            let identity = compiler_identity(source).unwrap();
            let fields: Vec<_> = identity.split('|').collect();
            assert_eq!(fields.len(), 3, "{label}");
            assert_eq!(fields[0], "packaged-v2");
            let toolchain = fields[1].strip_prefix("toolchain=").unwrap();
            assert_eq!(toolchain.len(), 64);
            assert!(toolchain.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_eq!(
                fields[2],
                format!("metallib={}", model_io::hash_data(entry.bytes)),
                "{label}"
            );
        }
    }

    #[test]
    fn every_bundled_composition_loads_and_exports_the_source_kernel_set() {
        if !BUILD_COMPLETE {
            eprintln!("precompiled library parity skipped: Metal toolchain inventory incomplete");
            return;
        }
        crate::autorelease_pool(|| {
            let device = metal::Device::system_default().expect("Metal device");
            assert_eq!(LIBRARIES.len(), INVENTORY.len());
            for (label, source) in INVENTORY {
                crate::autorelease_pool(|| {
                    let entry = lookup(source).unwrap();
                    let bundled = device
                        .new_library_with_data(entry.bytes)
                        .unwrap_or_else(|err| panic!("{label}: {err}"));
                    let runtime = device
                        .new_library_with_source(source, &metal::CompileOptions::new())
                        .unwrap_or_else(|err| panic!("{label}: {err}"));
                    let mut actual = bundled.function_names();
                    let mut expected = runtime.function_names();
                    actual.sort();
                    expected.sort();
                    assert_eq!(actual, expected, "{label}");
                });
            }
        });
    }

    #[test]
    fn bundled_nonlinear_kernel_outputs_match_runtime_source_compilation() {
        if !BUILD_COMPLETE {
            return;
        }
        crate::autorelease_pool(|| {
            let context = crate::MetalContext::new().expect("Metal context");
            let source = include_str!("shaders/utility.metal");
            let bundled = context
                .device()
                .new_library_with_data(lookup(source).unwrap().bytes)
                .unwrap();
            let runtime = context
                .device()
                .new_library_with_source(source, &metal::CompileOptions::new())
                .unwrap();
            // An odd count exercises the final partial SIMD/threadgroup too.
            let count = 257u32;
            let gate: Vec<_> = (0..count)
                .map(|index| half::f16::from_f32(index as f32 / 32.0 - 4.0))
                .collect();
            let up: Vec<_> = (0..count)
                .map(|index| half::f16::from_f32((index as f32 * 0.37).sin()))
                .collect();
            let gate = context.new_buffer_with_data(&gate);
            let up = context.new_buffer_with_data(&up);
            for kernel in ["gelu_mul_fp16", "silu_mul_fp16", "gelu_erf_mul_fp16"] {
                crate::autorelease_pool(|| {
                    let dispatch = |library: &metal::Library| {
                        let function = library.get_function(kernel, None).unwrap();
                        let pipeline = context
                            .device()
                            .new_compute_pipeline_state_with_function(&function)
                            .unwrap();
                        let out = context.new_output_buffer(u64::from(count) * 2);
                        let pass = context.begin_pass();
                        pass.encode_threads_3d(
                            &pipeline,
                            &[(&gate, 0, 0), (&up, 1, 0), (&out, 2, 0)],
                            &[(&count.to_le_bytes(), 3)],
                            (u64::from(count), 1, 1),
                            (256, 1, 1),
                        );
                        pass.commit_and_wait();
                        crate::read_buffer_f16(&out, 0, count as usize)
                    };
                    let expected = dispatch(&runtime);
                    let actual = dispatch(&bundled);
                    assert!(expected.iter().all(|value| value.is_finite()));
                    assert!(expected.iter().any(|value| value.to_f32().abs() > 1.0));
                    let bits = |values: Vec<half::f16>| {
                        values
                            .into_iter()
                            .map(half::f16::to_bits)
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(bits(actual), bits(expected), "{kernel}");
                });
            }
        });
    }
}
