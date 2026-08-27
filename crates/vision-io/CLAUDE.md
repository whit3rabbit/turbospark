# turbospark-vision-io

Portable vision preprocessing for the `qwen3_5` vision tower: image decode,
PIL-bicubic smart resize, normalize, patchify, and the three position tables
(interpolated position embedding, vision rope frequency rows, mRoPE triples).

Plain arithmetic on `Vec<f32>`. No Metal, no macOS, no model install, no
GPU. It is one of the crates the cross-target `cargo check` in the root
`AGENTS.md` covers, and that is a requirement rather than a happy accident:
milestone M-V1 exists to put this half of the pipeline somewhere a non-macOS
build can reach it.

## Directory & File Structure

```
crates/vision-io/
+-- Cargo.toml
+-- src/
|   +-- lib.rs           # crate docs, module list, re-exports
|   +-- error.rs         # VisionIoError
|   +-- params.rs        # PreprocessParams + preprocessor_config.json reader
|   +-- decode.rs        # bytes -> Rgb8Image
|   +-- rounding.rs      # round_half_to_even (Python's round())
|   +-- smart_resize.rs  # resized_dims
|   +-- resize.rs        # PIL bicubic, fixed-point exact
|   +-- normalize.rs     # rescale + per-channel mean/std
|   +-- patchify.rs      # patch_rows, GridThw
|   +-- preprocess.rs    # the composed pipeline
|   +-- pos_embed.rs     # pos_embed_weights (index/weight table)
|   +-- rope.rs          # vision_rope_freq_rows
|   \-- mrope.rs         # mrope_position_triples
\-- tests/
    +-- generated/       # @generated golden fixtures; see "Oracles" below
    +-- smart_resize_parity.rs
    +-- preprocess_parity.rs
    +-- position_tables_parity.rs
    \-- pipeline_units.rs
```

## Development & Test Commands

```sh
cargo test -p turbospark-vision-io

# The off-macOS gate this crate owes. It builds and tests to nothing
# platform-specific, so a `cfg` or a dependency that breaks it is a bug here
# rather than an accepted limitation.
cargo check --target x86_64-unknown-linux-gnu -p turbospark-vision-io
```

### Regenerating the oracles

Every fixture under `tests/generated/` is PROBED from the vendored mlx-vlm
reference at `../mlx-v/mlx-vlm`, never transcribed, so a fixture cannot
encode this port's own misreading of the reference. Each file's header
carries its own regeneration command; `all` does every mode.

```sh
uv run --python 3.12 --with mlx --with mlx-vlm --with numpy --with pillow \
  scripts/qwen3vl_vision_oracle.py all
```

`uv run --with` builds an ephemeral environment, so nothing is installed
globally and nothing enters this workspace.

## Crate Gotchas

1. **The patch row's inner feature order is `(T, P_h, P_w, C)` and the
   reference's is `(C, T, P_h, P_w)`.** This is deliberate. The tower ships
   `patch_embed.proj.weight` as `[1152, 2, 16, 16, 3]`, i.e.
   `[out, T, P, P, C]`, so emitting rows in that order lets the repack copy
   the weight verbatim and the patch-embed GEMM read both operands row-major
   -- no permutation at repack, none per image at decode
   (`docs/VISION_PHASE0.md` item 4). `preprocess_parity.rs` bridges the two
   orders with explicit index arithmetic and asserts the permutation is a
   bijection AND is not the identity, because a permutation inside a parity
   test is exactly the move that lets a wrong answer through. A mismatch
   here is silent wrong numerics: the GEMM keeps its shape, the tower runs,
   and the model simply reads a different image.

2. **PIL, not torchvision, and that authority is per CHECKPOINT rather than
   per crate.** Both are "antialiased bicubic" and they differ only in the
   fixed-point coefficient precision -- by whole 8-bit levels, which after
   this family's normalization is `2/255` on a signal in `[-1, 1]`, four
   orders of magnitude over the bar the parity tests hold. This family's
   processor calls `PIL.Image.resize` and its `preprocessor_config.json`
   declares `resample: 3` (`Image.BICUBIC`), so PIL is what is ported. A
   future family whose processor uses transformers' `tvF.resize` needs the
   torchvision spelling beside it, not instead of it. sconce carries both
   for this reason; only the PIL arm was taken (see `NOTICE`).

3. **Fixtures may not be sourced from JPEGs.** Pillow and the `image` crate
   disagree on individual samples of the same JPEG (different IDCT and
   chroma upsampling), so a JPEG-sourced golden would compare arithmetic
   PLUS a decoder difference and a failure could not be attributed. Every
   fixture starts from synthetic pixels whose bytes are embedded in the
   fixture itself, which also removes any shared generator that could drift
   between the two languages. `decode.rs` is deliberately outside
   `preprocess` so this stays easy.

4. **Two float formulations here are matched to the reference's SPELLING,
   not to the mathematically obvious one.** `linspace` is
   `(stop - start) * (i / (num - 1))`; the algebraically equal
   `(i * last) / (count - 1)` and `i * step` each disagree in the last f32
   bit, which moves a bilinear weight by ~4e-7 and the summed output row by
   ~1.6e-6 -- above the bar. Three orderings were checked against
   `mx.linspace` at four counts before choosing. When a parity failure is
   in the last ULP or two, look for the reference's spelling before
   loosening a tolerance.

5. **The one place the reference is LESS accurate, and the bar says so.**
   mlx's f32 `**` is not correctly rounded: at theta 10,000 it returns
   `inv_freq[2] = 0.3593813478946686` where the correctly-rounded f32 is
   `0.35938137769699097`, about 2.5 ULP low, and this port lands on the
   latter. `rope_case!` is therefore held to 4 ULP where every other fixture
   here is held to an absolute `1e-6` or to exact equality. The slack is
   named and attributed rather than nudged until green.

6. **The pixel budget is read from the checkpoint and has no default.**
   These checkpoints declare `size: {shortest_edge: 65536, longest_edge:
   16777216}` -- pixel COUNTS despite the "edge" naming -- against the
   generic Qwen2-VL library default of 3,136 .. 1,003,520, a factor of 16 on
   the ceiling. Falling back to the generic pair would resize every image to
   a fraction of its intended resolution and produce a correct-looking,
   lower-quality result, so `from_preprocessor_config_json` REFUSES when
   neither spelling is present. `in_channels` is the one value not read: the
   file never carries it, the processor sets `do_convert_rgb` and converts
   unconditionally, so three is what the format's silence means.

7. **`resize_bicubic_pil`'s equal-size early-out is a SPEED path, and a
   mutation deleting it survives the parity suite by design.** Checked
   rather than assumed: at `scale == 1` the half-pixel centre lands every
   sample on an integer offset, the Keys kernel is zero at every nonzero
   integer, and the weights collapse to `[0, 1, 0, 0]`, so the filter path
   reproduces its input exactly. `an_identity_resize_is_the_identity_either_way`
   pins that. The first draft of the module doc claimed the opposite.

8. **The oracle fixture cannot see an `image_mean` / `image_std` swap.**
   Both checkpoints declare `mean == std == [0.5; 3]`, so the two are
   exchangeable and no golden drawn from them discriminates.
   `normalize_applies_mean_before_std_and_per_channel` uses distinct values
   to pin the formula's shape, and
   `the_shipped_mean_and_std_make_the_swap_invisible` states the invariance
   itself so it is not rediscovered later as a finding.

9. **mRoPE counts images one way and places them another, and that is the
   reference's rule rather than a tidy one.** `get_rope_index` sizes its
   loop from `vision_start`-followed-by-`image_pad` pairs and then locates
   each block from the placeholder token's own position. On a malformed
   prompt the two disagree: an unmarked placeholder is not counted, the loop
   does not run, and it falls through as ordinary text. Reproduced rather
   than corrected -- agreeing with the trunk on every prompt it was trained
   on is worth more than being right about one it was not. sconce's own
   mrope walk drives from the markers and was not taken (see `NOTICE`);
   `a_placeholder_with_no_vision_start_is_treated_as_text` pins the choice.

10. **`preprocess_oracle.rs` is 73 KB against the house norm of 5-9 KB for a
    generated fixture, and that is a considered trade rather than an
    oversight.** It is the only fixture here that has to carry PIXELS: the
    other four commit indices, weights or integer triples, while this one
    commits four source images plus the patch matrices they produce. Two
    rounds of shrinking already happened -- the first draft was 245 KB, cut
    by dropping the test geometry from patch 16 to patch 2 (`patch_dim` 1536
    to 24, every loop still exercised because the pipeline is parameterized
    on the geometry) and by fixing a numpy 2.x `repr` that was wrapping every
    float as `np.float32(0.5)`. What remains is real coverage: five cases
    spanning all three resize branches plus the identity one, at full
    per-element comparison. The lever if it ever has to shrink again is
    committing a strided sample of the tiny cases the way `real_geometry`
    already does, which costs per-element coverage on exactly the cases that
    established the `(T, P_h, P_w, C)` order in the first place. Do not reach
    for it without a reason.

11. **Nothing here gathers.** `pos_embed_weights` returns indices and
    weights, and `vision_rope_freq_rows` returns frequency rows; the values
    being gathered live in GPU buffers this crate cannot see. Keep it that
    way -- the split is what lets the index arithmetic, which is where the
    conventions hide, be tested against the reference with no model loaded.
