//! The IOReport meter on this host: either it opens and two readings are
//! monotone with a CPU and an ANE channel, or it says why not. Not ignored:
//! the counters need no sudo and no model.
#![cfg(target_os = "macos")]

use turbospark_lab::energy::{block_of, delta, EnergyMeter, ENERGY_MODEL};

#[test]
fn readings_are_monotone_or_the_meter_says_why() {
    match EnergyMeter::open() {
        Err(reason) => {
            eprintln!("energy meter unavailable here: {reason}");
            assert!(!reason.is_empty());
        }
        Ok(meter) => {
            let a = meter.read().unwrap();
            assert!(a.keys().any(|k| block_of(k) == "cpu"), "{:?}", a.keys());
            assert!(a.keys().any(|k| block_of(k) == "ane"), "{:?}", a.keys());
            // Burn a little CPU so the delta is not all zeros.
            let mut x = 0u64;
            for i in 0..20_000_000u64 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
            }
            assert_ne!(x, 1);
            let b = meter.read().unwrap();
            let d = delta(&a, &b).expect("same channels, counters only grow");
            assert!(d.total_j >= 0.0);
            assert!(d.by_block.contains_key("cpu"));
            assert_eq!(d.energy_model, ENERGY_MODEL);
        }
    }
}
