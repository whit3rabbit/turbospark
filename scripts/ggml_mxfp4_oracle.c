// Dumps an MXFP4 decode oracle out of ggml itself, so
// `crates/compute/src/quant_gguf_mxfp4.rs` is checked against the format's
// own implementation rather than against a second copy of this port's
// reading of it (ROADMAP M5; gpt-oss ships its routed experts as MXFP4 and
// nothing else this port has seen does).
//
// Sibling of `scripts/ggml_q5_k_oracle.c` and written to the same rule: ggml
// produces BOTH the bytes and the expected floats, so the Rust decoder is
// the only thing under test and the comparison is `==` rather than a
// tolerance. A round trip through this port's own quantizer would pass
// whenever encoder and decoder share a misreading.
//
// IT EMITS TWO TABLES AS WELL AS THE ORACLE, and both are RECOVERED FROM
// ggml BY CONSTRUCTION rather than read out of its source:
//
//   * the 16-entry FP4 codebook, recovered by decoding a block whose shared
//     exponent is the one that makes the scale exactly 1.0 and whose sixteen
//     nibbles are the sixteen indices;
//   * the 256-entry E8M0 shared-exponent table, recovered by sweeping every
//     exponent byte over a block of a known codebook index.
//
// Neither is transcribed and neither can go stale silently: the Rust side
// carries the codebook as a literal and computes the scale by formula, and
// `crates/compute/tests/quant_gguf_mxfp4.rs` asserts both against these.
// That is the same division `iq3xxs_signs` uses -- keep the expression in
// code, keep the check in the generator's output.
//
// Build and run (brew's llama.cpp; same install the two sibling scripts
// compile against):
//
//   cc -O2 -I/opt/homebrew/include scripts/ggml_mxfp4_oracle.c \
//      -L/opt/homebrew/lib -lggml-base -o /tmp/ggml_mxfp4 && /tmp/ggml_mxfp4 \
//      > crates/compute/tests/generated/quant_gguf_mxfp4_oracle.rs
//
// It is NOT wired into the build: this workspace must not gain a libggml
// dependency for one fixture.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"

#define QK_MXFP4 32
// Four blocks, so a decoder that carries block 0's shared exponent forward,
// or that resets a running offset per row instead of per block, has
// somewhere to show up. Q5_K's oracle uses two superblocks for the same
// reason; MXFP4's block is eight times smaller, so it takes more of them.
#define N (4 * QK_MXFP4)

/**
 * Deterministic pseudo-random number generator (LCG) for reproducible oracle patterns.
 */
static uint32_t lcg(uint32_t *s) {
  *s = *s * 1664525u + 1013904223u;
  return *s;
}

/**
 * Builds one MXFP4 block with given shared exponent and uniform codebook index across all 32 elements.
 */
static void build_uniform_block(uint8_t *out, uint8_t e, uint8_t idx) {
  out[0] = e;
  for (int j = 0; j < QK_MXFP4 / 2; ++j) {
    out[1 + j] = (uint8_t)((idx & 0x0F) | ((idx & 0x0F) << 4));
  }
}

/**
 * Main entry point generating MXFP4 oracle tables, test bytes, and expected floats.
 */
int main(void) {
  const enum ggml_type t = GGML_TYPE_MXFP4;
  const size_t block_bytes = ggml_type_size(t);
  const int64_t block_elems = ggml_blck_size(t);
  if (block_elems != QK_MXFP4 || block_bytes != 17) {
    fprintf(stderr, "MXFP4 is %lld elements in %zu bytes, expected %d in 17\n",
            (long long)block_elems, block_bytes, QK_MXFP4);
    return 1;
  }

  const struct ggml_type_traits *tr = ggml_get_type_traits(t);
  if (!tr || !tr->to_float) {
    fprintf(stderr, "no MXFP4 to_float in this libggml\n");
    return 1;
  }

  // --- The shared-exponent table, swept before anything else needs it.
  //
  // Recovered at codebook index 1, whose value is +1.0 in every published
  // description of FP4 E2M1 -- but that is not assumed here either: index 1
  // is only used as a CONSTANT multiplier, and the codebook sweep below
  // divides it back out, so an index-1 value of anything non-zero yields the
  // same table. What would break this is index 1 decoding to zero, which the
  // assertion right after catches.
  float e8m0[256];
  {
    uint8_t blk[17];
    float y[QK_MXFP4];
    build_uniform_block(blk, 128, 1);
    tr->to_float(blk, y, QK_MXFP4);
    const float unit = y[0];
    if (unit == 0.0f) {
      fprintf(stderr, "codebook index 1 decodes to zero; this sweep is void\n");
      return 1;
    }
    for (int e = 0; e < 256; ++e) {
      build_uniform_block(blk, (uint8_t)e, 1);
      tr->to_float(blk, y, QK_MXFP4);
      // y = codebook[1] * scale(e), and scale(128) is what `unit` folds in,
      // so dividing by `unit` and multiplying by scale(128) would be
      // circular. Report the RAW product instead and let the Rust side
      // compare its own `codebook[1] * scale(e)` against it: that keeps both
      // halves of the reconstruction under test rather than just one.
      e8m0[e] = y[0];
    }
  }

  // --- The codebook, at the exponent byte that makes the scale 1.0.
  //
  // WHICH byte that is comes out of the sweep above rather than out of a
  // formula, so a libggml that changed the E8M0 bias would move this with it
  // instead of silently scaling the whole table.
  int unit_e = -1;
  {
    // The scale is 1.0 exactly when index 1 decodes to the same value it
    // does under a unit scale, i.e. when e8m0[e] equals the codebook's own
    // index-1 entry. Find it as the exponent whose neighbours are half and
    // double, which pins it without knowing the entry.
    for (int e = 1; e < 255; ++e) {
      if (e8m0[e] > 0.0f && e8m0[e + 1] == 2.0f * e8m0[e] &&
          e8m0[e] == 2.0f * e8m0[e - 1] && e8m0[e] == 1.0f) {
        unit_e = e;
        break;
      }
    }
    if (unit_e < 0) {
      fprintf(stderr, "no exponent byte yields a unit scale at index 1\n");
      return 1;
    }
  }

  float codebook[16];
  {
    uint8_t blk[17];
    float y[QK_MXFP4];
    for (int idx = 0; idx < 16; ++idx) {
      build_uniform_block(blk, (uint8_t)unit_e, (uint8_t)idx);
      tr->to_float(blk, y, QK_MXFP4);
      codebook[idx] = y[0];
    }
  }

  // --- The oracle proper.
  //
  // Weights spanning both signs, a run of zeros, and a magnitude range wide
  // enough that consecutive blocks land on DIFFERENT shared exponents. The
  // last part is the one that matters: MXFP4's only per-block state is that
  // exponent, so a fixture whose blocks all share one cannot see a decoder
  // that reads block 0's.
  float x[N];
  uint32_t s = 20260811u;
  for (int i = 0; i < N; ++i) {
    const float unit = (float)((int32_t)(lcg(&s) % 2001) - 1000) / 1000.0f;
    // Block b scaled by 2^(2b - 2), so the four blocks are two octaves apart.
    const int b = i / QK_MXFP4;
    x[i] = unit * ldexpf(1.0f, 2 * b - 2);
  }
  for (int i = 40; i < 56; ++i) {
    x[i] = 0.0f;
  }

  uint8_t bytes[N / QK_MXFP4 * 17];
  const size_t total_bytes = (size_t)(N / QK_MXFP4) * block_bytes;
  if (total_bytes > sizeof(bytes)) {
    fprintf(stderr, "buffer too small: %zu\n", total_bytes);
    return 1;
  }
  // The public converter path, the same call `llama-quantize` makes.
  const size_t wrote = ggml_quantize_chunk(t, x, bytes, 0, N / QK_MXFP4,
                                           QK_MXFP4, /* imatrix */ NULL);
  if (wrote != total_bytes) {
    fprintf(stderr, "quantize wrote %zu bytes, expected %zu\n", wrote,
            total_bytes);
    return 1;
  }

  float y[N];
  tr->to_float(bytes, y, N);

  // A fixture whose blocks all landed on one exponent would not test what
  // this one exists to test, so say so rather than hoping.
  {
    int distinct = 0;
    for (size_t b = 0; b < total_bytes / block_bytes; ++b) {
      int seen = 0;
      for (size_t c = 0; c < b; ++c) {
        seen |= bytes[c * block_bytes] == bytes[b * block_bytes];
      }
      distinct += !seen;
    }
    if (distinct < 3) {
      fprintf(stderr, "only %d distinct shared exponents across %zu blocks\n",
              distinct, total_bytes / block_bytes);
      return 1;
    }
  }

  printf("// Generated by scripts/ggml_mxfp4_oracle.c against libggml. Do not "
         "edit.\n");
  printf("//\n// ggml quantized these bytes and decoded these floats, and "
         "both tables below\n// were recovered from it by construction. The "
         "Rust decoder is the only side\n// under test.\n\n");

  printf("/// The 16-entry FP4 codebook, recovered by decoding a block whose "
         "shared\n/// exponent yields a unit scale (byte %d) at each of the "
         "sixteen indices.\n",
         unit_e);
  printf("pub const MXFP4_ORACLE_CODEBOOK: [f32; 16] = [");
  for (int i = 0; i < 16; ++i) {
    printf("%s%#.9g", i ? ", " : "", codebook[i]);
  }
  printf("];\n\n");

  printf("/// `codebook[1] * scale(e)` for every one of the 256 shared "
         "exponent bytes.\n/// Compared against the Rust side's own product, "
         "so BOTH halves of the\n/// reconstruction are under test rather "
         "than the scale alone.\n");
  printf("#[allow(clippy::excessive_precision)]\n");
  printf("pub const MXFP4_ORACLE_E8M0: [f32; 256] = [");
  for (int i = 0; i < 256; ++i) {
    printf("%s%#.9g", i ? ", " : "", e8m0[i]);
  }
  printf("];\n\n");

  printf("/// What `ggml_quantize_chunk` produced for %d elements (%zu bytes "
         "per block,\n/// four blocks two octaves apart so the shared "
         "exponent really varies).\n",
         N, block_bytes);
  printf("pub const MXFP4_ORACLE_BYTES: [u8; %zu] = [", total_bytes);
  for (size_t i = 0; i < total_bytes; ++i) {
    printf("%s%u", i ? ", " : "", bytes[i]);
  }
  printf("];\n\n");

  printf("/// What ggml's `to_float` reads back out of those exact bytes.\n");
  printf("#[allow(clippy::excessive_precision)]\n");
  printf("pub const MXFP4_ORACLE_FLOATS: [f32; %d] = [", N);
  for (int i = 0; i < N; ++i) {
    // `#` forces the decimal point: plain %g prints zero as "0", which is
    // not an f32 literal and reddens the generated file at compile time.
    printf("%s%#.9g", i ? ", " : "", y[i]);
  }
  printf("];\n");
  return 0;
}
