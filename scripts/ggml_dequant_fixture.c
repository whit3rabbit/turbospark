// Decodes quantized bytes with ggml's own `to_float`, so a CPU dequantizer is
// held to ggml's output rather than to a round trip through its own encoder.
// Not wired into the build (no libggml dependency).
//
//   cc -O2 -I/opt/homebrew/include scripts/ggml_dequant_fixture.c \
//      -L/opt/homebrew/lib -lggml-base -Wl,-rpath,/opt/homebrew/lib -o /tmp/ggml_dq
//   /tmp/ggml_dq <ggml type id> <quantized.bin> <elements> <out.f32>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s type in.bin n out.f32\n", argv[0]); return 2; }
    enum ggml_type type = (enum ggml_type)atoi(argv[1]);
    int64_t n = atoll(argv[3]);
    size_t bytes = ggml_row_size(type, n);
    void *src = malloc(bytes);
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(src, 1, bytes, f) != bytes) { fprintf(stderr, "%s: bad input\n", argv[2]); return 1; }
    fclose(f);
    const struct ggml_type_traits *tr = ggml_get_type_traits(type);
    if (!tr || !tr->to_float) { fprintf(stderr, "no to_float for %d\n", (int)type); return 1; }
    float *dst = malloc(n * sizeof(float));
    tr->to_float(src, dst, n);
    FILE *o = fopen(argv[4], "wb");
    fwrite(dst, sizeof(float), n, o);
    fclose(o);
    return 0;
}
