// THROWAWAY probe (deleted after use): host cost of the per-round GDN
// snapshot and restore at Qwen3.8-27B's shape.
#![cfg(target_os = "macos")]
use turbospark_gpu::{GdnStateManager, MetalContext};

#[test]
#[ignore]
fn snapshot_cost() {
    let context = MetalContext::new().unwrap();
    let config = model_io::qwen_gdn_dense_27b();
    let mut gdn = GdnStateManager::new(context.device(), &config);
    let mb = (gdn.state_bytes_per_layer + gdn.conv_tail_bytes_per_layer) as f64 * 48.0 / 1048576.0;
    let mut snap_ms = Vec::new();
    let mut restore_ms = Vec::new();
    for _ in 0..21 {
        let t = std::time::Instant::now();
        let s = gdn.snapshot(context.queue());
        snap_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let t = std::time::Instant::now();
        gdn.restore(&s, context.queue());
        restore_ms.push(t.elapsed().as_secs_f64() * 1e3);
        drop(s);
    }
    snap_ms.remove(0);
    restore_ms.remove(0);
    snap_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    restore_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("state {mb:.1} MiB: snapshot median {:.2} ms (min {:.2}, max {:.2}); restore median {:.2} ms",
        snap_ms[10], snap_ms[0], snap_ms[19], restore_ms[10]);
}
