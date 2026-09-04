//! `qwen4_exp`'s PLE n-gram hash: derives the checkpoint's own
//! `layer_multipliers` / prime vocabulary sizes as a CROSS-CHECK (the
//! decode flow reads the checkpoint's shipped buffers, per
//! [`crate::PleConfig`]'s own doc), and computes the per-token row ids a
//! decode step needs from the current and preceding raw token ids.
//!
//! **VERIFIED AGAINST THE REAL REFERENCE, NOT DERIVED FROM THE PROJECT'S
//! OWN SUMMARY.** `docs/QWEN4_PHASE0.md` (`crates/model-io`'s consuming
//! doc) originally carried a `ctx = [prev_2, prev_1, cur]` pseudocode for
//! this hash, which turned out to be WRONG when checked against
//! `transformers`' `modular_qwen4_exp.py` (`fc5c5bde8`) directly: the
//! multiplier at index `s` pairs with the token `s` positions BACK from the
//! current one (index 0 is the current token, unshifted), not with a
//! fixed "oldest-to-newest" position. Every function below is transcribed
//! from that source's `_splitmix64`, `_build_layer_multipliers`,
//! `_find_nth_prime_after` and the `NGramEmbedding.forward` hash loop, with
//! the decode-time shift register (`NgramContext`) derived from -- not
//! copied from, since the reference has no such loop -- its
//! `_shift_right_ignore_eos` masking rule. See the doc for the full
//! derivation; the short version is in this module's own doc comments.

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
/// The reference's `_PRIME_1`: multiplies `ple_layer_index` into the base
/// seed. Not a vocabulary prime; a fixed constant of the hash itself.
const HASH_SEED_PRIME: i64 = 10007;

/// The reference's `_splitmix64` finalizer, operated on as u64 WRAPPING
/// arithmetic throughout -- the source's explicit `& _MASK64` after every
/// add and multiply is exactly what `wrapping_add`/`wrapping_mul` already
/// do, so this is a transcription rather than a reduction.
fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

/// The reference's `_build_layer_multipliers`. `ngram_size` multipliers,
/// one per shift `0..ngram_size` (shift 0 pairs with the CURRENT token).
///
/// A cross-check on the checkpoint's own `layer_multipliers` buffer, never
/// the primary source of it: [`crate::PleConfig`]'s doc says why (a
/// checkpoint that ships them wrong and a port that only derives them are
/// the same silent failure from opposite directions).
pub fn build_layer_multipliers(
    unigram_vocab_size: i64,
    ngram_size: i64,
    ple_layer_index: i64,
    seed: i64,
) -> Vec<i64> {
    let max_long = i64::MAX; // (1 << 63) - 1
    let multiplier_max = max_long / unigram_vocab_size.max(1);
    let half_bound = (multiplier_max / 2).max(1) as u64;
    let base_seed = seed.wrapping_add(HASH_SEED_PRIME.wrapping_mul(ple_layer_index));
    (0..ngram_size)
        .map(|index| {
            let step = SPLITMIX_GAMMA.wrapping_mul((index + 1) as u64);
            let value = (base_seed as u64).wrapping_add(step);
            let h = splitmix64(value) % half_bound;
            2 * (h as i64) + 1
        })
        .collect()
}

/// The reference's `_is_prime`: trial division to `isqrt`.
fn is_prime(value: i64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3i64;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The reference's `_find_nth_prime_after`: the `count`-th prime strictly
/// after `start`.
pub fn find_nth_prime_after(start: i64, count: i64) -> i64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// `heads_per_ngram` head vocabulary sizes and offsets for one n-gram order
/// (bigram, trigram, ...), continuing the running total a PRIOR order left
/// off. The reference's `Qwen4ExpTextNGramEmbedding.__init__` computes all
/// `ngram_heads` heads in one flat loop (`global_head_idx` running across
/// every order); this is that loop's per-order slice, callable
/// independently because the running state it needs (`total_vocab_size`,
/// `global_head_idx`) is exactly what the caller already carries across
/// n-gram orders.
///
/// Cross-checks [`crate::NgramTableLayout`]'s own `head_vocab_sizes` /
/// `head_offsets`, for [`build_layer_multipliers`]'s reason.
pub fn derive_head_vocab_and_offsets(
    ngram_vocab_size_base: i64,
    heads_per_ngram: i64,
    global_head_start: i64,
    running_total: &mut i64,
) -> (Vec<i64>, Vec<i64>) {
    let mut sizes = Vec::with_capacity(heads_per_ngram as usize);
    let mut offsets = Vec::with_capacity(heads_per_ngram as usize);
    for local in 0..heads_per_ngram {
        let global_head_idx = global_head_start + local;
        let size = find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1);
        sizes.push(size);
        offsets.push(*running_total);
        *running_total += size;
    }
    (sizes, offsets)
}

/// The per-token n-gram hash: `ctx[s]` is the token `s` positions back from
/// current (`ctx[0]` is the current token itself), `multipliers[s]` pairs
/// with it. Returns one row id per hash head, bigram heads first
/// (`heads_per_ngram` of them), then trigram, and so on for every n-gram
/// order up to `ctx.len() - 1`.
///
/// **THE MULTIPLIER INDEX PAIRS WITH THE SHIFT, NOT WITH A FIXED POSITION IN
/// A THREE-SLOT WINDOW.** This is the module header's corrected finding:
/// `ctx[0]` (current) always multiplies by `multipliers[0]`, `ctx[1]`
/// (one back) by `multipliers[1]`, and so on, for EVERY n-gram order --
/// the trigram term is the bigram term XORed with one more shift, not a
/// separately-indexed triple.
///
/// Multiplication is NOT wrapping: the reference proves
/// `(vocab_size - 1) * max(multiplier) < 2^63` by construction (see
/// `docs/QWEN4_PHASE0.md` item 4's "APPLICATION" note), so an overflow here
/// means that invariant broke, and `checked_mul` turns that into a loud
/// panic naming the values rather than a silently wrong row id.
pub fn ple_ngram_rows(
    ctx: &[i64],
    heads_per_ngram: i64,
    multipliers: &[i64],
    head_vocab_sizes: &[i64],
    head_offsets: &[i64],
) -> Vec<u64> {
    let ngram_size = ctx.len();
    assert_eq!(
        multipliers.len(),
        ngram_size,
        "one multiplier per shift, 0..ctx.len()"
    );
    let heads_per_ngram = heads_per_ngram as usize;
    let orders = ngram_size.saturating_sub(1);
    let total_heads = orders * heads_per_ngram;
    assert_eq!(
        head_vocab_sizes.len(),
        total_heads,
        "one vocab size per hash head across every n-gram order"
    );
    assert_eq!(
        head_offsets.len(),
        total_heads,
        "one offset per hash head across every n-gram order"
    );

    let term = |pos: usize| -> i64 {
        ctx[pos].checked_mul(multipliers[pos]).unwrap_or_else(|| {
            panic!(
                "ctx[{pos}]={} * multiplier[{pos}]={} overflowed i64; the \
                 checkpoint's multiplier bound (vocab_size - 1) * max(mult) < 2^63 \
                 no longer holds",
                ctx[pos], multipliers[pos]
            )
        })
    };

    let mut rows = vec![0u64; total_heads];
    for order_idx in 0..orders {
        let ngram = order_idx + 2; // 2 (bigram), 3 (trigram), ...
        let mut mixed = term(0);
        for pos in 1..ngram {
            mixed ^= term(pos);
        }
        debug_assert!(
            mixed >= 0,
            "XOR of two non-negative i64s cannot set the sign bit"
        );
        for local in 0..heads_per_ngram {
            let h = order_idx * heads_per_ngram + local;
            rows[h] = (mixed as u64 % head_vocab_sizes[h] as u64) + head_offsets[h] as u64;
        }
    }
    rows
}

/// Decode-time state for the n-gram context (recurrent state slot 2 in
/// `docs/QWEN4_PHASE0.md` item 4's "three recurrent states" table): the
/// last `context_len = ngram_size - 1` raw token ids, EOS-boundary aware.
///
/// **DERIVED FROM `_shift_right_ignore_eos`'S MASKING RULE, NOT GUESSED.**
/// Working through its validity condition (`source_position >
/// previous_eos`) one token at a time collapses to: the token just fed
/// always becomes the new one-back slot (whether it was EOS or not, since
/// the masked value at that slot already equals it either way), and every
/// OTHER slot either carries the previous one-back value forward
/// (ordinary shift) or resets to `eos_token_id` outright, when the fed
/// token IS `eos_token_id` -- proven, not assumed, in
/// `docs/QWEN4_PHASE0.md`'s "A DECODE-TIME SHIFT REGISTER" note. A plain
/// "shift and drop the oldest" register, with no reset logic, would agree
/// with the reference on every step except the one immediately after an
/// EOS token, where it would leak a pre-EOS token into the trigram hash
/// that the reference forces to `eos_token_id`.
#[derive(Debug, Clone)]
pub struct NgramContext {
    /// `history[0]` is one token back, `history[1]` two back, and so on.
    /// Length is fixed at construction (`ngram_size - 1`).
    history: Vec<i64>,
}

impl NgramContext {
    /// A fresh sequence: every slot starts at `eos_token_id`, matching the
    /// reference's `previous_context = input_ids.new_full(..., eos_token_id)`
    /// when no cache exists yet.
    pub fn new(context_len: usize, eos_token_id: i64) -> Self {
        Self {
            history: vec![eos_token_id; context_len],
        }
    }

    /// Feeds one newly-decoded (or newly-prefilled) token, returning the
    /// `ctx` slice [`ple_ngram_rows`] wants for THIS token (current token
    /// first, then the state as it stood BEFORE this call), and advancing
    /// the state for the next call.
    pub fn step(&mut self, token: i64, eos_token_id: i64) -> Vec<i64> {
        let mut ctx = Vec::with_capacity(self.history.len() + 1);
        ctx.push(token);
        ctx.extend_from_slice(&self.history);

        if token == eos_token_id {
            self.history.fill(eos_token_id);
        } else {
            // Shift right: token becomes the new one-back slot, every other
            // slot moves one further back, the oldest is dropped.
            for i in (1..self.history.len()).rev() {
                self.history[i] = self.history[i - 1];
            }
            if !self.history.is_empty() {
                self.history[0] = token;
            }
        }
        ctx
    }
}

#[cfg(test)]
#[path = "ngram_hash_tests.rs"]
mod tests;
