//! A deterministic drafter (greedy block selector, prompt copy) has no q(x)
//! of its own, but its q is the point mass on what it proposed. Leviathan
//! acceptance against that q must commit tokens with exactly the target law:
//! accept x with probability p(x), else draw from p with x removed.
//!
//! This pins the premise on the shipped `ShapedDistribution::point_mass` and
//! `residual`. The runtime still refuses sampled block drafting.

use turbospark_selection::{residual, ShapedDistribution};

#[test]
fn point_mass_rejection_sampling_keeps_the_target_law() {
    let target = [0.5f64, 0.3, 0.2];
    let p = ShapedDistribution::new(vec![0, 1, 2], target.to_vec());
    const SEEDS: u64 = 60_000;
    // The proposal is fixed per arm, as a greedy drafter's is for a context.
    for proposal in 0..3u32 {
        let q = ShapedDistribution::point_mass(proposal);
        let mut committed = [0u64; 3];
        let mut accepted = 0u64;
        for seed in 0..SEEDS {
            // Acceptance and the correction draw read different positions, so
            // they are independent uniforms.
            let u = turbospark_selection::derive::to_unit_interval(
                turbospark_selection::derive::derive_step_value(Some(seed), 0),
            );
            let token = if u * q.probability(proposal) <= p.probability(proposal) {
                accepted += 1;
                proposal
            } else {
                residual(&p, &q)
                    .expect("mass survives outside the proposal")
                    .draw(Some(seed), 1)
            };
            committed[token as usize] += 1;
        }
        // Binomial standard error at 60000 draws is at most 0.0021.
        for (token, &count) in committed.iter().enumerate() {
            let got = count as f64 / SEEDS as f64;
            assert!(
                (got - target[token]).abs() < 0.008,
                "proposal {proposal}: token {token} at {got}, want {}",
                target[token]
            );
        }
        let rate = accepted as f64 / SEEDS as f64;
        assert!(
            (rate - target[proposal as usize]).abs() < 0.008,
            "acceptance {rate} must equal p(proposal)"
        );
    }
}
