use super::*;

/// The checkpoint's own frozen `layer_multipliers`
/// (`docs/QWEN4_PHASE0.md` item 4), for the real `Qwen/Qwen3.8-Flash-Next`
/// config: `vocab_size=248320`, `ple_layer_index=0`, `seed=1234`.
#[test]
fn multiplier_derivation_matches_the_frozen_checkpoint_values() {
    let got = build_layer_multipliers(248_320, 3, 0, 1234);
    assert_eq!(
        got,
        vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    );
}

/// A second PLE layer index moves the derivation (through `base_seed`),
/// which is what says `ple_layer_index` is actually read rather than
/// dropped. `docs/QWEN4_PHASE0.md` only froze index 0 (this checkpoint's
/// one PLE layer); this fixes the OTHER argument the derivation reads.
#[test]
fn a_different_ple_layer_index_changes_the_multipliers() {
    let base = build_layer_multipliers(248_320, 3, 0, 1234);
    let other = build_layer_multipliers(248_320, 3, 1, 1234);
    assert_ne!(base, other);
}

/// The checkpoint's own frozen head vocabulary sizes and offsets
/// (`docs/QWEN4_PHASE0.md` item 4): 8 bigram heads (order 0) then 8
/// trigram heads (order 1), `ngram_vocab_size_base = 20_000_000`.
#[test]
fn head_vocab_and_offsets_match_the_frozen_checkpoint_list() {
    let mut running_total = 0i64;
    let (bigram_sizes, bigram_offsets) =
        derive_head_vocab_and_offsets(20_000_000, 8, 0, &mut running_total);
    let (trigram_sizes, trigram_offsets) =
        derive_head_vocab_and_offsets(20_000_000, 8, 8, &mut running_total);

    let sizes: Vec<i64> = bigram_sizes.into_iter().chain(trigram_sizes).collect();
    let offsets: Vec<i64> = bigram_offsets.into_iter().chain(trigram_offsets).collect();

    assert_eq!(
        sizes,
        vec![
            20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
            20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
            20_000_161, 20_000_171,
        ]
    );
    assert_eq!(
        offsets,
        vec![
            0,
            20_000_003,
            40_000_026,
            60_000_059,
            80_000_106,
            100_000_165,
            120_000_228,
            140_000_297,
            160_000_374,
            180_000_455,
            200_000_548,
            220_000_655,
            240_000_802,
            260_000_955,
            280_001_114,
            300_001_275,
        ]
    );
    assert_eq!(running_total, 320_001_446, "the doc's total_vocab_size");
}

fn frozen_multipliers() -> Vec<i64> {
    vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
}

fn frozen_vocab_and_offsets() -> (Vec<i64>, Vec<i64>) {
    let mut running_total = 0i64;
    let (bs, bo) = derive_head_vocab_and_offsets(20_000_000, 8, 0, &mut running_total);
    let (ts, to) = derive_head_vocab_and_offsets(20_000_000, 8, 8, &mut running_total);
    (
        bs.into_iter().chain(ts).collect(),
        bo.into_iter().chain(to).collect(),
    )
}

/// **DISCRIMINATION, not just a plausible number.** Heads 0-7 are bigram
/// heads and must depend on `ctx[0]` (current) and `ctx[1]` (one back)
/// ONLY -- `ctx[2]` (two back) must not move them at all, because a bigram
/// hash that silently depended on a third token would be indistinguishable
/// from a correct one on any fixture that never varies `ctx[2]` alone. This
/// is also the exact shape of the bug this module's header records fixing:
/// the original (wrong) pseudocode paired `ctx[0]` with `prev_2`, which
/// would make bigram heads move when `ctx[2]` changes.
#[test]
fn bigram_heads_are_unaffected_by_the_two_back_token_and_trigram_heads_are() {
    let mults = frozen_multipliers();
    let (vocab, offsets) = frozen_vocab_and_offsets();

    let ctx_a = [111i64, 222, 333];
    let ctx_b = [111i64, 222, 999]; // only ctx[2] (two back) differs

    let rows_a = ple_ngram_rows(&ctx_a, 8, &mults, &vocab, &offsets);
    let rows_b = ple_ngram_rows(&ctx_b, 8, &mults, &vocab, &offsets);

    assert_eq!(
        rows_a[0..8],
        rows_b[0..8],
        "bigram heads (0..8) must not move when only the two-back token changes"
    );
    assert_ne!(
        rows_a[8..16],
        rows_b[8..16],
        "trigram heads (8..16) must move when the two-back token changes"
    );
}

/// The complementary discrimination: the CURRENT token must move both
/// bigram and trigram heads, since `ctx[0]` (current) participates in
/// every n-gram order's mix.
#[test]
fn the_current_token_moves_both_bigram_and_trigram_heads() {
    let mults = frozen_multipliers();
    let (vocab, offsets) = frozen_vocab_and_offsets();

    let ctx_a = [111i64, 222, 333];
    let ctx_b = [777i64, 222, 333]; // only ctx[0] (current) differs

    let rows_a = ple_ngram_rows(&ctx_a, 8, &mults, &vocab, &offsets);
    let rows_b = ple_ngram_rows(&ctx_b, 8, &mults, &vocab, &offsets);

    assert_ne!(rows_a[0..8], rows_b[0..8], "bigram heads must move");
    assert_ne!(rows_a[8..16], rows_b[8..16], "trigram heads must move");
}

/// Every returned row id must land inside its head's addressed span
/// (`[offset, offset + vocab_size)`), which is what a caller strides
/// `NgramTableLayout::row_offset` by. A modulus or offset bug reads a row
/// that belongs to a NEIGHBOURING head's span -- plausible bytes, wrong
/// head.
#[test]
fn every_row_lands_inside_its_own_heads_addressed_span() {
    let mults = frozen_multipliers();
    let (vocab, offsets) = frozen_vocab_and_offsets();
    let ctx = [12_345i64, 67_890, 11_111];
    let rows = ple_ngram_rows(&ctx, 8, &mults, &vocab, &offsets);
    for (h, &row) in rows.iter().enumerate() {
        let lo = offsets[h] as u64;
        let hi = lo + vocab[h] as u64;
        assert!(
            row >= lo && row < hi,
            "head {h}: row {row} outside [{lo}, {hi})"
        );
    }
}

/// The decode-time state machine's ordinary (no EOS) case: each step's
/// `ctx` is `[current, one_back, two_back]`, verified across several
/// distinct tokens so a wrong shift direction or a dropped slot reads a
/// wrong number rather than a coincidentally-matching one.
#[test]
fn ordinary_steps_shift_the_context_without_touching_eos() {
    const EOS: i64 = 99;
    let mut ctx_state = NgramContext::new(2, EOS);

    assert_eq!(ctx_state.step(5, EOS), vec![5, EOS, EOS]);
    assert_eq!(ctx_state.step(6, EOS), vec![6, 5, EOS]);
    assert_eq!(ctx_state.step(7, EOS), vec![7, 6, 5]);
    assert_eq!(ctx_state.step(8, EOS), vec![8, 7, 6]);
}

/// **THE CASE THE MASKING RULE ACTUALLY DECIDES.** One token after an EOS,
/// the trigram slot (two-back) must read as EOS even though a real,
/// non-EOS token sat there before the EOS arrived -- a plain
/// shift-and-drop register (no reset) would leak that stale token into the
/// hash, which the reference's `_shift_right_ignore_eos` explicitly
/// prevents. `docs/QWEN4_PHASE0.md`'s derivation is what proves this
/// rather than merely asserting it.
#[test]
fn one_step_after_eos_the_two_back_slot_reads_as_eos_not_the_pre_eos_token() {
    const EOS: i64 = 99;
    let mut ctx_state = NgramContext::new(2, EOS);

    ctx_state.step(5, EOS);
    ctx_state.step(6, EOS);
    // Feeding EOS itself: ctx uses the state as it stood BEFORE this call.
    assert_eq!(ctx_state.step(EOS, EOS), vec![EOS, 6, 5]);
    // The token right after EOS: two-back must be EOS, not 6.
    assert_eq!(ctx_state.step(7, EOS), vec![7, EOS, EOS]);
    // Normal shifting resumes from here.
    assert_eq!(ctx_state.step(8, EOS), vec![8, 7, EOS]);
    assert_eq!(ctx_state.step(9, EOS), vec![9, 8, 7]);
}

/// A fresh context starts fully padded with EOS, matching the reference's
/// `previous_context = input_ids.new_full(..., eos_token_id)` when no
/// cache exists.
#[test]
fn a_fresh_context_starts_padded_with_eos() {
    const EOS: i64 = 42;
    let mut ctx_state = NgramContext::new(3, EOS);
    assert_eq!(ctx_state.step(1, EOS), vec![1, EOS, EOS, EOS]);
}

/// `find_nth_prime_after` on its own: the first few primes after a small
/// start, independent of the PLE-specific wrapper above.
#[test]
fn find_nth_prime_after_finds_consecutive_primes() {
    assert_eq!(find_nth_prime_after(10, 1), 11);
    assert_eq!(find_nth_prime_after(10, 2), 13);
    assert_eq!(find_nth_prime_after(10, 3), 17);
    assert_eq!(find_nth_prime_after(1, 1), 2);
}

#[test]
#[should_panic(expected = "one multiplier per shift")]
fn ple_ngram_rows_refuses_a_multiplier_count_mismatch() {
    let ctx = [1i64, 2, 3];
    let mults = [1i64, 2]; // one short
    ple_ngram_rows(&ctx, 8, &mults, &[1; 16], &[0; 16]);
}
