#![cfg(target_os = "macos")]
//! `MFERENCE_DISPATCH_PROFILE=1`: per-dispatch GPU timing inside one
//! command buffer.
//!
//! Two things have to hold, and the second is the one that could silently
//! corrupt a real decode: profiling opens a NEW compute encoder per
//! dispatch (this hardware samples counters only at encoder boundaries),
//! so a chained pass whose second dispatch reads the first one's output
//! must still produce identical results. The whole file is one test
//! because `MFERENCE_DISPATCH_PROFILE` is read once into a `OnceLock`;
//! setting it per test would race inside a shared test binary.

use half::f16;
use turbospark_gpu::MetalContext;

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn read_halfs(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b)).collect()
}

/// `hidden += delta` then `hidden *= 2`, encoded as one pass. The second
/// dispatch depends on the first, so a broken re-encode shows up as a
/// wrong value, not just a missing timing row.
fn chained_pass(context: &mut MetalContext, label: &'static str) -> Vec<f16> {
    let n = 512usize;
    let hidden: Vec<f16> = (0..n).map(|i| f16::from_f32(i as f32 * 0.01)).collect();
    let delta: Vec<f16> = (0..n).map(|i| f16::from_f32(i as f32 * 0.02)).collect();
    let hidden_buf = context.new_buffer_with_data(&to_le(&hidden));
    let delta_buf = context.new_buffer_with_data(&to_le(&delta));

    let pass = context.begin_pass_labeled(label);
    turbospark_gpu::encode_residual_add(
        context,
        &pass,
        (&hidden_buf, 0),
        (&delta_buf, 0),
        n as u32,
    )
    .expect("encode residual_add");
    turbospark_gpu::encode_scalar_mul(context, &pass, (&hidden_buf, 0), 2.0, n as u32)
        .expect("encode scalar_mul");
    pass.commit_and_wait();
    read_halfs(&hidden_buf, n)
}

#[test]
fn profiles_each_dispatch_without_changing_results() {
    // A virtualized Metal device (CI's "Apple Paravirtual device") exposes
    // no GPU performance-counter hardware at all, and the `metal` crate's
    // `Device::counter_sets` (called from `new_sample_buffer` the moment
    // profiling is enabled below) hard-panics on that rather than
    // returning an empty set -- a `thread caused non-unwinding panic.
    // aborting`, not a catchable `Result` or `panic!`, so this has to be
    // checked BEFORE enabling `MFERENCE_DISPATCH_PROFILE` at all rather
    // than caught after. `new()` alone never touches the counter API, so
    // creating a context to read the device name first is safe on every
    // device.
    let context = MetalContext::new().expect("Metal device");
    let device_name = context.device().name();
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized and exposes no GPU counter hardware; \
             skipping the dispatch-profile test rather than triggering the metal crate's \
             abort on Device::counter_sets"
        );
        return;
    }
    drop(context);

    // Set before anything touches the GPU: `enabled()` caches the env var
    // in a `OnceLock` on the first pipeline creation, so profiling cannot
    // be turned on mid-process.
    std::env::set_var("MFERENCE_DISPATCH_PROFILE", "1");
    let mut context = MetalContext::new().expect("Metal device");
    let profiled = chained_pass(&mut context, "probe");
    let expected: Vec<f16> = (0..profiled.len())
        .map(|i| f16::from_f32((i as f32 * 0.01 + i as f32 * 0.02) * 2.0))
        .collect();
    for (i, (&got, &want)) in profiled.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got.to_f32() - want.to_f32()).abs() <= 0.05,
            "i={i}: one encoder per dispatch changed the arithmetic: {got} vs {want}"
        );
    }

    let report = turbospark_gpu::dispatch_profile_report(1).expect("report");
    for expected in [
        "probe",
        "residual_add_fp16",
        "scalar_mul_fp16",
        "2.0 dispatches/token",
    ] {
        assert!(
            report.contains(expected),
            "report missing {expected:?}:\n{report}"
        );
    }
    assert!(
        !report.contains("(unregistered pipeline)"),
        "every dispatched pipeline should be named:\n{report}"
    );
    // Timestamps resolved to something: a row of exactly 0.000 ms would
    // mean the counter samples came back as the error sentinel.
    assert!(
        !report.contains("0.000 ms/token"),
        "expected non-zero per-dispatch times:\n{report}"
    );
    std::env::remove_var("MFERENCE_DISPATCH_PROFILE");
}
