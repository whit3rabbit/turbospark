#![cfg(target_os = "macos")]
//! The DSpark Markov step with probabilities on (the verify-width cut): the
//! picked token and its softmax probability over the softcapped scores,
//! against a CPU reference, on real Metal hardware.

use half::f16;
use turbospark_gpu::{
    encode_dspark_markov_argmax, encode_dspark_markov_logits, MetalContext, DSPARK_LOGITS_PER_GROUP,
};

const SOFTCAP: f32 = 30.0;

fn unit(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

fn le16(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

#[test]
fn markov_step_probability_matches_the_cpu_softmax() {
    // A last group that is partly past the vocabulary: those entries must not
    // enter the denominator. Only visible when every score is near -softcap,
    // since that is what softcap(-inf) would add (the last case).
    let (n_vocab, rank) = (800u32, 64u32);
    let stride = 1024u32;
    assert_ne!(n_vocab % DSPARK_LOGITS_PER_GROUP, 0);
    let mut context = MetalContext::new().expect("metal");
    // Spread, then one case where the leader is past the softcap's knee, so a
    // probability taken from raw scores would differ.
    let cases = [
        (7u32, 2.0f32, 0.0f32, 0.0f32),
        (11, 6.0, 3.0, 0.0),
        (13, 2.0, 60.0, 0.0),
        (17, 0.5, 0.0, -90.0),
    ];
    for (seed, spread, boost, shift) in cases {
        let mut s = seed;
        let mut base: Vec<f16> = (0..stride)
            .map(|_| f16::from_f32(unit(&mut s) * spread + shift))
            .collect();
        base[713] = f16::from_f32(base[713].to_f32() + spread + boost);
        base[40] = f16::from_f32(shift + spread + boost - 1.0);
        let fold_b: Vec<f16> = (0..stride)
            .map(|_| f16::from_f32(unit(&mut s) * 0.1))
            .collect();
        let fold_t: Vec<f16> = (0..rank * stride)
            .map(|_| f16::from_f32(unit(&mut s) * 0.05))
            .collect();
        let latent: Vec<f16> = (0..rank).map(|_| f16::from_f32(unit(&mut s))).collect();

        let scores: Vec<f64> = (0..n_vocab as usize)
            .map(|v| {
                let mut z = f64::from(base[v].to_f32()) + f64::from(fold_b[v].to_f32());
                for j in 0..rank as usize {
                    z += f64::from(fold_t[j * stride as usize + v].to_f32())
                        * f64::from(latent[j].to_f32());
                }
                f64::from(SOFTCAP) * (z / f64::from(SOFTCAP)).tanh()
            })
            .collect();
        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        let sum: f64 = scores.iter().map(|x| (x - scores[best]).exp()).sum();
        let want = 1.0 / sum;

        let buf = |c: &MetalContext, v: &[f16]| c.new_buffer_with_data(&le16(v));
        let (b, fb, ft, la) = (
            buf(&context, &base),
            buf(&context, &fold_b),
            buf(&context, &fold_t),
            buf(&context, &latent),
        );
        let groups = u64::from(n_vocab.div_ceil(DSPARK_LOGITS_PER_GROUP));
        let words = |c: &MetalContext, n: u64| c.new_output_buffer(n * 4);
        let (part_v, part_i, part_p) = (
            words(&context, groups),
            words(&context, groups),
            words(&context, groups),
        );
        let (last, proposals, probs) = (words(&context, 1), words(&context, 2), words(&context, 2));
        let pass = context.begin_pass_labeled("markov prob test");
        let parts = encode_dspark_markov_logits(
            &mut context,
            &pass,
            (&b, 0),
            (&fb, 0),
            (&ft, 0),
            (&la, 0),
            (&part_v, 0),
            (&part_i, 0),
            n_vocab,
            stride,
            rank,
            Some(((&part_p, 0), SOFTCAP)),
        )
        .unwrap();
        // Step 1, so a kernel ignoring the step index fails.
        encode_dspark_markov_argmax(
            &mut context,
            &pass,
            (&part_v, 0),
            (&part_i, 0),
            (&last, 0),
            (&proposals, 0),
            parts,
            1,
            Some(((&part_p, 0), (&probs, 0), SOFTCAP)),
        )
        .unwrap();
        pass.commit_and_wait();
        let picked = turbospark_gpu::read_f32_buffer_at(&proposals, 0, 2)[1].to_bits();
        let got = turbospark_gpu::read_f32_buffer_at(&probs, 0, 2)[1];
        assert_eq!(picked as usize, best, "seed {seed}");
        assert!(
            (f64::from(got) - want).abs() <= 5e-3 * want,
            "seed {seed}: got {got}, want {want}"
        );
    }
}
