# turbospark-audio module guide

The active entry points are the short [crate guide](../../../crates/audio/AGENTS.md) and
[docs/AUDIO.md](../../../docs/AUDIO.md) for scope and provenance. This page is
the deeper map: directory layout, the algorithm notes that do not fit in
doc comments, and the gotchas the tests exist to catch.

## Directory & File Structure

```
crates/audio/
+-- Cargo.toml            # zero dependencies; see the comment block there
+-- src/
|   +-- lib.rs            # crate docs, module list, re-exports
|   +-- error.rs          # AudioError + sample-rate/channel validators
|   +-- waveform.rs       # Waveform: interleaved f32 + rate + channels
|   +-- wav.rs            # RIFF/WAVE reader and f32/i16 writers
|   +-- conversion.rs     # channel layout moves + to_mono_resampled
|   +-- resample.rs       # linear and sinc-Hann polyphase resamplers
|   +-- fft.rs            # ComplexF32, RealFftPlan, radix-2/Bluestein
|   +-- stft.rs           # stft / istft with reflect center padding
|   +-- mel.rs            # MelScale, filterbank, log-mel frontend
|   +-- whisper.rs        # 80/128-band frontend, frame drop, global clamp
|   \-- dsp.rs            # Hann window, peak, rms, gain, normalize
\-- tests/
    +-- wav_roundtrip.rs  # hand-built containers, field by field
    +-- dsp_pipeline.rs   # end-to-end: wav -> mono resample -> log-mel
    \-- reference/
        \-- generate_whisper.py # independent NumPy/official-asset goldens
```

## Algorithms worth knowing before editing

### Real FFT via half-size complex transform

`RealFftPlan::forward` packs `x[2k] + i x[2k+1]`, runs a complex FFT at half size
(radix-2 or Bluestein), then splits bins with

```
X[k]   = 0.5 (Z[k] + conj(Z[m-k])) - 0.5i W_n^k (Z[k] - conj(Z[m-k]))
X[0]   = Z[0].re + Z[0].im        (DC and Nyquist packed as one real pair)
X[m]   = Z[0].re - Z[0].im
```

The inverse rebuilds `Z[k] = 0.5 (S + i conj(W_n^k) D)` with
`S = X[k] + conj(partner)`, `D = X[k] - conj(partner)`, and partner
`X[m-k]` (mod m, so k=0 pairs with the Nyquist bin). Both directions were
derived once and are pinned by DFT parity and round-trip tests; rederive
rather than nudge signs.

The DIT butterfly loop combines elements `len/2` apart inside each chunk.
A version that pairs adjacent elements (`chunks_mut(2)`) compiles, runs,
and produces plausible-looking garbage: only the DFT parity tests catch it.

### Sinc-Hann polyphase resampler

Rates reduce by GCD (`orig_freq`, `new_freq`), the cutoff is
`min(orig, new) * rolloff`, and kernel taps carry `idx = j - width` spanning
`[-width, width + orig_freq)`. For phase `p`, tap `j` has
`t = ((j - width)/orig - p/new) * base` clamped to `+-lowpass_width`, and
the kernel value is `scale * sinc(pi t) * hann(t/lowpass)` with
`scale = base/orig`. The Hann window closes exactly at the clamp, so taps
beyond each phase's active range are identically zero and the convolution
skips them. Output `(b, p)` samples input around
`(b * new + p) * orig / new`; input outside the buffer is zeros, which is
why DC and tone tests measure the interior only.

The upstream kernel cache (a mutex-guarded global) is replaced by
recomputation per call; the soxr arm (runtime dlopen + linear fallback) is
not ported at all. Bit-parity against torchaudio or upstream is unverified
pending a generated fixture; the behavioral gates are DC gain, tone
frequency and amplitude, and f32/f64 mode agreement.

### WAV 24-bit decode

Three-byte little-endian samples sign-extend from bit 23 before the
`/ 8388608` division. A version that shifts bytes up one slot instead
compiles and decodes full-scale negative as -256.0. The hand-built
container test in `tests/wav_roundtrip.rs` exists because the crate's own
writers never produce 24-bit output, so round-trip tests cannot catch this.

### STFT centering and normalization

Centered framing materializes a reflect-padded buffer (torch
`center=True`: the edge sample is not repeated) and frames it like the
uncentered path, so both paths share one frame loop. The inverse multiplies
the IFFT output (already windowed) by the window once more and divides by
the accumulated window energy; multiplying by the squared window instead
scales a quarter-overlap Hann reconstruction by exactly `sum(w^3)/sum(w^2)`
= 5/6, which is a real bug that once passed a loose tolerance check.

## Testing notes

- Unit tests live inline; integration tests exercise cross-module
  behavior (`tests/dsp_pipeline.rs`) and byte-level container handling
  (`tests/wav_roundtrip.rs`).
- The DFT reference in `fft.rs` tests is f64 direct evaluation, not the
  crate's FFT. The pipeline test also uses a direct DFT, but reuses the
  production HTK filterbank and projection. It validates the FFT/frontend
  wiring, not independent filterbank correctness. Official Slaney asset
  coefficients and NumPy Whisper goldens cover that separate boundary.
- `read_wav_f32` (the path wrapper) is covered by an `#[ignore]`d test so
  the default suite never touches the filesystem; run it explicitly where
  writes are allowed.

## Regression map

Run the commands in [AGENTS.md](../../../crates/audio/AGENTS.md#checks).
These tests cover specific contracts. They do not establish blanket parity
with audio.cpp, torchaudio, or a real transcription model.

| Contract | Regression to retain |
|---|---|
| Slaney area uses Hz width, for both Whisper band counts | `mel::slaney_regression::matches_the_official_whisper_filterbank_asset` and `matches_the_official_whisper_128_filterbank_asset` |
| Whisper frame drop precedes the global peak/clamp | `whisper::tests::discarded_tail_energy_does_not_change_retained_frame_normalization`, plus the 80/128 NumPy golden tests |
| Bluestein sizes and FFT allocation limits | `fft::tests::bluestein_size_matches_direct_dft_and_round_trips`, `bluestein_covers_small_even_and_odd_half_sizes`, and `fft::allocation_regression` |
| Padded short windows, inverse coverage, and window-sum-squares | `stft::short_window_regression`, `stft::allocation_regression`, and `stft::tests::perfect_reconstruction_hann_quarter_overlap` |
| PCM32 scale, complete extensible GUID, and RIFF byte counts | `tests/wav_roundtrip.rs`: `decodes_signed_32_bit_pcm_at_the_documented_scale`, `refuses_extensible_guid_suffix_and_extension_size_corruption`, and `writer_output_has_riff_sizes_that_match_the_bytes` |
| Rate/channel validation and bounded coprime resampler tables | `waveform::allocation_regression`, `resample::kernel_budget_regression`, and the resampler length/options tests |

The short-window regression currently covers uncentered rectangular
windows. It does not prove centered short-Hann parity. Resampling has the
behavioral checks described above, without an independent bit-parity fixture.
Run the ignored filesystem WAV test explicitly; the default suite skips it.

### Independent Whisper fixtures

`tests/reference/generate_whisper.py` uses NumPy and the official
OpenAI Whisper `v20250625` `assets/mel_filters.npz`. It verifies SHA256
`7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c`
before loading the 80/128-band matrices. It does not reconstruct Slaney
coefficients from the Rust formula.

From the repository root, with NumPy available:

```sh
python3 crates/audio/tests/reference/generate_whisper.py --asset /path/to/mel_filters.npz
```

Without `--asset`, the script downloads that pinned asset. Regeneration
prints reference values. It does not rewrite Rust tests and is not part
of normal test execution. The crate remains free of external dependencies.
Keep the input construction, FFT/window dimensions, numeric mode, and
tolerance rationale beside the checked-in values. Do not regenerate a
golden from production output to make a mismatch disappear.

The tail-energy fixture must show that the discarded frame would set a
larger global peak. A silent discarded frame or a comparison through the
same helper can pass with the wrong operation order. When changing these
guards, mutation-check Hz width replaced by mel width and frame dropping
moved after normalization. Confirm each mutation applied and its targeted
test failed, then restore the source before the final checks.
