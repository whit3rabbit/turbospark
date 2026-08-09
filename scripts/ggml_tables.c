// Dumps the IQ-quant codebook tables and a set of decode oracle vectors out
// of ggml itself, so `crates/compute/src/quant_gguf_iq.rs` carries DERIVED
// constants rather than transcribed ones (ROADMAP Phase S; the same "ask
// ggml, do not recall" rule as `ggml_type_block`'s doc comment, which is how
// Q3_K = (256, 110) was confirmed).
//
// The tables it recovers are `iq3xxs_grid` (256 entries of 4 bytes) and
// `kvalues_iq4nl` (16 signed values). Neither is exported from libggml, so
// they are recovered BY PROBING rather than by reading a header: build a
// block whose only varying field is the index into the table, dequantize it
// through the public `ggml_get_type_traits(t)->to_float`, and read the table
// entry straight off the output. A transcription error is impossible by
// construction, which matters because both tables are long runs of magic
// numbers that a reviewer cannot check by eye.
//
// It also prints oracle vectors for the hand-packed-block tests: the same
// blocks the Rust tests build, decoded by ggml, so those tests compare
// against ggml and not against a second copy of this port's own reasoning.
//
// Build and run (brew's llama.cpp; b10310 here, same install
// `scripts/llamacpp_logits.c` compiles against):
//
//   cc -O2 -I/opt/homebrew/include scripts/ggml_tables.c \
//      -L/opt/homebrew/lib -lggml-base -o /tmp/ggml_tables && /tmp/ggml_tables
//
// Output is Rust, meant to be pasted. It is NOT wired into the build: this
// workspace must not gain a libggml dependency for three constant arrays.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"

#define QK_K 256

// Block layouts, verbatim from ggml-common.h. Sizes are asserted against
// `ggml_type_size` below rather than trusted, since a silent mismatch here
// would make every probe read the wrong bytes and still print a plausible
// table.
typedef struct {
  uint16_t d;
  uint8_t qs[3 * QK_K / 8]; // 64 grid indices, then 8 sign/scale words
} block_iq3_xxs;

typedef struct {
  uint16_t d;
  uint8_t qs[16];
} block_iq4_nl;

typedef struct {
  uint16_t d;
  uint16_t scales_h;
  uint8_t scales_l[QK_K / 64];
  uint8_t qs[QK_K / 2];
} block_iq4_xs;

static void dequant(enum ggml_type t, const void *blocks, float *out,
                    int64_t n) {
  const struct ggml_type_traits *tr = ggml_get_type_traits(t);
  if (!tr || !tr->to_float) {
    fprintf(stderr, "no to_float for type %d\n", (int)t);
    exit(1);
  }
  tr->to_float(blocks, out, n);
}

static void check_size(enum ggml_type t, size_t expect, const char *name) {
  size_t got = ggml_type_size(t);
  if (got != expect) {
    fprintf(stderr, "%s: ggml says %zu bytes per block, this file assumes %zu\n",
            name, got, expect);
    exit(1);
  }
  if ((size_t)ggml_blck_size(t) == 0) {
    fprintf(stderr, "%s: zero block size\n", name);
    exit(1);
  }
}

// FP16 1.0. Every probe uses d = 1.0 so the dequantized value IS the table
// entry, with no scale to divide back out and no rounding to argue about.
#define F16_ONE 0x3C00

// ---------------------------------------------------------------- IQ4_NL

// One block, all 32 nibbles set to the same index, recovers one table entry.
// Sweeping the index 0..15 recovers the table. The low nibbles land in
// elements 0..15 and the high nibbles in 16..31, which the probe also
// CHECKS rather than assumes: with both nibbles equal, every one of the 32
// outputs must agree, so a wrong split shows up as a spread.
static void dump_kvalues_iq4nl(void) {
  printf("/// The 16 non-linear reconstruction levels IQ4_NL and IQ4_XS share,\n");
  printf("/// recovered from ggml by `scripts/ggml_tables.c` rather than\n");
  printf("/// transcribed. Not evenly spaced: the steps widen toward the tails,\n");
  printf("/// which is the whole point of the type and the reason an affine\n");
  printf("/// habit (`(q - 8) * d`) decodes to something finite and wrong.\n");
  printf("pub const IQ4NL_VALUES: [i8; 16] = [\n   ");
  for (int idx = 0; idx < 16; ++idx) {
    block_iq4_nl b;
    memset(&b, 0, sizeof b);
    b.d = F16_ONE;
    for (int j = 0; j < 16; ++j) {
      b.qs[j] = (uint8_t)(idx | (idx << 4));
    }
    float y[32];
    dequant(GGML_TYPE_IQ4_NL, &b, y, 32);
    for (int j = 1; j < 32; ++j) {
      if (y[j] != y[0]) {
        fprintf(stderr, "IQ4_NL element %d disagrees with 0 at index %d\n", j,
                idx);
        exit(1);
      }
    }
    printf(" %d,", (int)y[0]);
  }
  printf("\n];\n\n");
}

// The nibble split, stated as data rather than as a comment: one block whose
// low nibbles are all 0 and high nibbles all 15. Prints the index of the
// first element that takes the high nibble.
static void check_iq4nl_nibble_split(void) {
  block_iq4_nl b;
  memset(&b, 0, sizeof b);
  b.d = F16_ONE;
  for (int j = 0; j < 16; ++j) {
    b.qs[j] = 0xF0;
  }
  float y[32];
  dequant(GGML_TYPE_IQ4_NL, &b, y, 32);
  int first_high = -1;
  for (int j = 0; j < 32; ++j) {
    if (y[j] != y[0]) {
      first_high = j;
      break;
    }
  }
  printf("// IQ4_NL nibble split: elements 0..%d take the LOW nibble, %d..32\n",
         first_high, first_high);
  printf("// take the HIGH nibble of the same byte.\n\n");
}

// --------------------------------------------------------------- IQ3_XXS

// The grid. One superblock's first 32-element block reads 8 index bytes; set
// them all to the same index and every one of those 32 outputs is a grid
// entry, repeated. Sign bits are left at zero (all positive).
//
// `db` MUST come out exactly 1.0 or the dumped table is silently scaled, and
// the arithmetic is easy to get wrong by one factor: the scale is
// `db = d * (0.5 + (aux >> 28)) * 0.5`, so BOTH halves apply. Scale nibble 0
// with d = 4.0 gives 4.0 * 0.5 * 0.5 = 1.0. An earlier version of this probe
// used nibble 1 with d = 2.0, read that as 1.0, and dumped a table 1.5x too
// large -- which decoded to plausible, correctly-signed, uniformly-wrong
// weights. The `expect_unit_scale` check below is what would have caught it,
// so it stays.
static void dump_iq3xxs_grid(void) {
  printf("/// The IQ3_XXS codebook: 256 entries of four 8-bit magnitudes.\n");
  printf("/// Recovered from ggml by `scripts/ggml_tables.c`. An IQ3_XXS block\n");
  printf("/// stores an INDEX into this table per four elements plus a sign\n");
  printf("/// bit each, which is what makes it a codebook type rather than an\n");
  printf("/// affine or K-quant one: there is no arithmetic that reproduces\n");
  printf("/// these numbers.\n");
  printf("pub const IQ3XXS_GRID: [[u8; 4]; 256] = [\n");
  // Scale nibble 0 under d = 4.0 is db = 1.0. Established once, against a
  // known grid entry, before 256 rows are printed on the strength of it.
  //
  // The `0.5 +` is the part worth pinning, because it is the difference
  // between a linear scale and an offset one, and it is checked by RATIO so
  // the unknown grid entry cancels: nibble n over nibble 0 must read
  // (0.5 + n) / 0.5, i.e. 3x at n = 1 and 5x at n = 2. A missing `0.5 +`
  // would give 0 and undefined; a missing trailing `* 0.5` would leave both
  // ratios unchanged, which is why the absolute value is asserted too.
  {
    block_iq3_xxs b;
    float y[QK_K];
    memset(&b, 0, sizeof b);
    b.d = 0x4400; // FP16 4.0
    dequant(GGML_TYPE_IQ3_XXS, &b, y, QK_K);
    const float unit = y[0];
    if (unit != 4.0f) {
      fprintf(stderr,
              "IQ3_XXS scale probe: grid entry 0 at db = 1.0 reads %f, not the "
              "4.0 every known build of this table gives\n",
              unit);
      exit(1);
    }
    for (int n = 1; n <= 2; ++n) {
      uint32_t aux = (uint32_t)n << 28;
      memcpy(b.qs + QK_K / 4, &aux, 4);
      dequant(GGML_TYPE_IQ3_XXS, &b, y, QK_K);
      const float want = unit * (0.5f + (float)n) / 0.5f;
      if (y[0] != want) {
        fprintf(stderr,
                "IQ3_XXS scale probe: nibble %d reads %f, the expression "
                "d * (0.5 + nibble) * 0.5 predicts %f\n",
                n, y[0], want);
        exit(1);
      }
    }
  }
  for (int idx = 0; idx < 256; ++idx) {
    block_iq3_xxs b;
    memset(&b, 0, sizeof b);
    b.d = 0x4400; // FP16 4.0; with scale nibble 0 this makes db exactly 1.0
    for (int j = 0; j < 8; ++j) {
      b.qs[j] = (uint8_t)idx;
    }
    float y[QK_K];
    dequant(GGML_TYPE_IQ3_XXS, &b, y, QK_K);
    // With all eight index bytes equal, the first 32 outputs are the same
    // 4-entry pattern eight times over. Verified, not assumed.
    for (int j = 4; j < 32; ++j) {
      if (y[j] != y[j % 4]) {
        fprintf(stderr, "IQ3_XXS grid probe: element %d breaks the period at "
                        "index %d\n",
                j, idx);
        exit(1);
      }
    }
    for (int j = 0; j < 4; ++j) {
      if (y[j] != (float)(int)y[j] || y[j] < 0.0f || y[j] > 255.0f) {
        fprintf(stderr, "IQ3_XXS grid entry %d.%d is %f, not a u8\n", idx, j,
                y[j]);
        exit(1);
      }
    }
    printf("    [%3d, %3d, %3d, %3d],\n", (int)y[0], (int)y[1], (int)y[2],
           (int)y[3]);
  }
  printf("];\n\n");
}

// The sign table needs no dumping: `ksigns_iq2xs[i]` is `i` with bit 7 set
// iff `i` has an odd number of set bits, so the eighth sign is a parity bit
// over the seven stored ones. Printed as a CHECK rather than a table, since
// a computed byte cannot go stale but a claim about one can.
static void check_iq3xxs_sign_parity(void) {
  for (int idx = 0; idx < 128; ++idx) {
    block_iq3_xxs b;
    memset(&b, 0, sizeof b);
    b.d = 0x4000;
    for (int j = 0; j < 8; ++j) {
      b.qs[j] = 1; // grid entry 1, whatever it is, as long as it is nonzero
    }
    uint32_t aux = (1u << 28) | (uint32_t)idx; // sign index for l == 0
    memcpy(b.qs + QK_K / 4, &aux, 4);
    float y[QK_K];
    dequant(GGML_TYPE_IQ3_XXS, &b, y, QK_K);
    int parity = __builtin_popcount(idx) & 1;
    int expect_bits = idx | (parity << 7);
    for (int j = 0; j < 8; ++j) {
      int neg = y[j] < 0.0f;
      int want = (expect_bits >> j) & 1;
      if (y[j] == 0.0f) {
        continue; // a zero magnitude carries no sign
      }
      if (neg != want) {
        fprintf(stderr,
                "IQ3_XXS sign parity: index %d element %d is %s, parity rule "
                "says %s\n",
                idx, j, neg ? "negative" : "positive",
                want ? "negative" : "positive");
        exit(1);
      }
    }
  }
  printf("// IQ3_XXS signs: ksigns[i] == i | (popcount(i) & 1) << 7, checked\n");
  printf("// for all 128 indices. No table needed; the eighth sign is parity.\n\n");
}

// ---------------------------------------------------------------- oracles

// Prints one decoded block per type from a deterministic pseudo-random fill,
// as the expected values for the Rust hand-packed tests. The generator is a
// plain LCG so the Rust side can reproduce the same bytes without this file.
static uint32_t lcg(uint32_t *s) {
  *s = *s * 1664525u + 1013904223u;
  return *s;
}

static void dump_oracles(void) {
  printf("/// Decoded by ggml (`scripts/ggml_tables.c`) from the byte pattern\n");
  printf("/// `oracle_bytes()` builds. The expected values come from ggml and\n");
  printf("/// not from this port, which is the whole point: a decoder and a\n");
  printf("/// fixture written from one mental model agree while both are wrong.\n");

  {
    uint32_t s = 12345;
    block_iq4_nl b;
    memset(&b, 0, sizeof b);
    b.d = F16_ONE;
    for (int j = 0; j < 16; ++j) {
      b.qs[j] = (uint8_t)(lcg(&s) & 0xFF);
    }
    float y[32];
    dequant(GGML_TYPE_IQ4_NL, &b, y, 32);
    printf("pub const IQ4NL_ORACLE: [f32; 32] = [");
    for (int j = 0; j < 32; ++j) {
      printf("%s%.1f", j ? ", " : "", y[j]);
    }
    printf("];\n");
  }

  {
    uint32_t s = 12345;
    block_iq4_xs b;
    memset(&b, 0, sizeof b);
    b.d = F16_ONE;
    b.scales_h = (uint16_t)(lcg(&s) & 0xFFFF);
    for (int j = 0; j < 4; ++j) {
      b.scales_l[j] = (uint8_t)(lcg(&s) & 0xFF);
    }
    for (int j = 0; j < 128; ++j) {
      b.qs[j] = (uint8_t)(lcg(&s) & 0xFF);
    }
    float y[QK_K];
    dequant(GGML_TYPE_IQ4_XS, &b, y, QK_K);
    printf("pub const IQ4XS_ORACLE: [f32; 256] = [");
    for (int j = 0; j < QK_K; ++j) {
      printf("%s%.1f", j ? ", " : "", y[j]);
    }
    printf("];\n");
  }

  {
    uint32_t s = 12345;
    block_iq3_xxs b;
    memset(&b, 0, sizeof b);
    b.d = F16_ONE;
    for (int j = 0; j < 3 * QK_K / 8; ++j) {
      b.qs[j] = (uint8_t)(lcg(&s) & 0xFF);
    }
    float y[QK_K];
    dequant(GGML_TYPE_IQ3_XXS, &b, y, QK_K);
    printf("pub const IQ3XXS_ORACLE: [f32; 256] = [");
    for (int j = 0; j < QK_K; ++j) {
      printf("%s%.4f", j ? ", " : "", y[j]);
    }
    printf("];\n");
  }
}

int main(void) {
  check_size(GGML_TYPE_IQ3_XXS, sizeof(block_iq3_xxs), "IQ3_XXS");
  check_size(GGML_TYPE_IQ4_NL, sizeof(block_iq4_nl), "IQ4_NL");
  check_size(GGML_TYPE_IQ4_XS, sizeof(block_iq4_xs), "IQ4_XS");

  printf("// Generated by scripts/ggml_tables.c against libggml. Do not edit.\n\n");
  check_iq4nl_nibble_split();
  dump_kvalues_iq4nl();
  check_iq3xxs_sign_parity();
  dump_iq3xxs_grid();
  dump_oracles();
  return 0;
}
