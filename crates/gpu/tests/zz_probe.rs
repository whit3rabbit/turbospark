#![cfg(target_os = "macos")]
#[path = "../examples/music3_linear_bench.rs"]
#[allow(dead_code)]
mod bench;

#[test]
fn zz_probe_grid() {
    let mut out = format!(
        "PROBE {}\n",
        turbospark_gpu::Music3Device::new().unwrap().probe_info()
    );
    for (mode, golden) in [
        ("hashes", include_str!("golden/music3_hashes.txt")),
        ("dit_hashes", include_str!("golden/music3_dit_hashes.txt")),
        ("conv_hashes", include_str!("golden/music3_conv_hashes.txt")),
    ] {
        let mut got = Vec::new();
        bench::run(mode, &mut |line| got.push(line));
        let want: Vec<&str> = golden.lines().collect();
        let bad: Vec<_> = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.as_str() != **b)
            .map(|(a, _)| a.rsplit_once(' ').map_or(a.as_str(), |x| x.0).to_owned())
            .collect();
        out += &format!(
            "== {mode}: {} of {} differ (lens {} {})\n",
            bad.len(),
            want.len(),
            got.len(),
            want.len()
        );
        for b in bad {
            out += &format!("  DIFF {b}\n");
        }
    }
    panic!("{out}");
}
