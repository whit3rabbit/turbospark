# gguf_quantize golden fixtures

Byte-exact output of ggml's own `ggml_quantize_chunk`, used by
`src/gguf_quantize/tests.rs`.

- `in.f32`: 4 rows x 512 little-endian f32 (seeded gaussian with outliers;
  row 3, columns 256..512 are all zero). `im.f32`: 512 f32 importance weights.
- `g_<ggml type id>.bin`: no imatrix. `gi_<id>.bin`: with `im.f32`.
  Ids: 2 Q4_0, 3 Q4_1, 6 Q5_0, 7 Q5_1, 8 Q8_0, 10 Q2_K, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K,
  16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 19 IQ1_S, 20 IQ4_NL, 21 IQ3_S, 22 IQ2_S,
  23 IQ4_XS, 29 IQ1_M.
  Ids 3 and 7 were added with `scripts/ggml_quantize_fixture.c`, which also
  reproduces the committed `g_2`/`gi_2` byte for byte.
  For Q4_0, Q4_1, Q5_0 and Q5_1 the imatrix changes the output (weighted scale search);
  for Q8_0 it has no effect, and `gi_8.bin` equals `g_8.bin`.
- Generated with Homebrew llama.cpp (ggml from that build) through a 20-line C
  program: `ggml_quantize_chunk(type, x, dst, 0, nrows, 512, imatrix_or_NULL)`.

## Second input (`in2.f32`, `im2.f32`, `g2_*.bin`, `gi2_*.bin`)

3 rows x 768, adversarial on purpose: heavy outliers, two zeroed 32-wide
sub-blocks inside a live block, a nearly constant sub-block (distinct values,
because exactly equal values make IQ1's libc `qsort` tie order decide the
result), an all-negative sub-block, a tiny-magnitude block, a large-magnitude
block, and near-zero importance columns. `g2_*` are imatrix-free, `gi2_*` use
`im2.f32`; ids 16, 17 and 19 (IQ2_XXS, IQ2_XS, IQ1_S) have only `gi2_*`
because ggml refuses to run them without an imatrix.

## Decode oracles (`d_<id>.f32`)

`d_2`, `d_3`, `d_6` and `d_7` are the first 2048 floats ggml's own `to_float`
produces from `g_2`, `g_3`, `g_6` and `g_7` (Q4_0, Q4_1, Q5_0, Q5_1), made with
`scripts/ggml_dequant_fixture.c`. `legacy_decoders_match_ggml_to_float` holds
`compute`'s decoders to them exactly, which is what the legacy GPU kernels are
in turn held to.
