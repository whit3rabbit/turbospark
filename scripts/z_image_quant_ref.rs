//! Run the existing portable quantizer on raw FP32 probe rows.
//! This prevents Python's approximation from certifying its own packing.

#[allow(dead_code)]
#[path = "../crates/compute/src/quant.rs"]
mod quant;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "input-f32-le output-prefix");
    let bytes = std::fs::read(&args[1]).unwrap();
    assert_eq!(bytes.len() % (64 * 4), 0);
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|x| x.is_finite()));
    let row = quant::quantize_int4_affine(&values);
    std::fs::write(format!("{}.packed", args[2]), row.packed).unwrap();
    for (suffix, values) in [("scales", row.scales), ("biases", row.biases)] {
        let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{}.{suffix}", args[2]), bytes).unwrap();
    }
}
