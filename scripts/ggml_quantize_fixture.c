// Writes ggml's own quantizer output for one type, so the golden fixtures in
// crates/repack/tests/fixtures/gguf_quantize are ggml's bytes rather than this
// port's reading of them. Not wired into the build (no libggml dependency).
//
//   cc -O2 -I/opt/homebrew/include scripts/ggml_quantize_fixture.c \
//      -L/opt/homebrew/lib -lggml-base -Wl,-rpath,/opt/homebrew/lib -o /tmp/ggml_qfix
//   /tmp/ggml_qfix <ggml type id> <in.f32> <rows> <cols> <out.bin> [im.f32]
//
// With im.f32 the output is the imatrix variant (`gi_*`), without it the plain
// one (`g_*`). Ids: 3 Q4_1, 7 Q5_1 (see the fixture README for the rest).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"

static float *load(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    float *v = malloc(n * sizeof(float));
    if (fread(v, sizeof(float), n, f) != n) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s type in rows cols out [im]\n", argv[0]); return 2; }
    enum ggml_type type = (enum ggml_type)atoi(argv[1]);
    int64_t rows = atoll(argv[3]), cols = atoll(argv[4]);
    float *x = load(argv[2], (size_t)(rows * cols));
    float *im = argc > 6 ? load(argv[6], (size_t)cols) : NULL;
    ggml_quantize_init(type);
    size_t cap = ggml_row_size(type, cols) * rows;
    void *dst = malloc(cap);
    size_t n = ggml_quantize_chunk(type, x, dst, 0, rows, cols, im);
    if (n != cap) { fprintf(stderr, "wrote %zu, expected %zu\n", n, cap); return 1; }
    FILE *o = fopen(argv[5], "wb");
    fwrite(dst, 1, n, o);
    fclose(o);
    return 0;
}
