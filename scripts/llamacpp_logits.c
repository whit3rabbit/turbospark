// Teacher-force a fixed token-id sequence through llama.cpp and write every
// position's full-vocabulary logits, so this port can be compared against a
// second engine reading THE SAME QUANTIZED BYTES (ROADMAP Phase G's last
// open gate clause, and the same-precision reference Phase S needs).
//
// Built and driven by `scripts/kld_llamacpp.py`; see that file for what the
// numbers mean and `scripts/kld.py` for the mlx-lm sibling this mirrors.
//
// WHY A HARNESS AND NOT A STOCK BINARY. No shipped llama.cpp tool
// teacher-forces a caller-supplied id list and returns full-vocab logits.
// `llama-perplexity --save-all-logits` comes closest but tokenizes its own
// text file and chunks by n_ctx, which would fight the assistant-slot
// alignment the whole measurement rests on.
//
// IDS GO IN AS IDS. llama.cpp's tokenizer is never asked to encode
// anything, so a tokenizer or chat-template difference cannot surface as a
// numerics gap (`scripts/kld.py`'s rule, and the reason `meta.json` carries
// the id list at all).
//
// LOGITS COME OUT AS f32, unlike this port's f16 dump. The port dumps the
// width it computes in; narrowing llama.cpp's f32 to match would add a
// rounding error this file would then have to defend. Widening is free.
//
//   cc -O2 -o /tmp/llamacpp_logits scripts/llamacpp_logits.c \
//      -I$(brew --prefix)/include -L$(brew --prefix)/lib -lllama
//   /tmp/llamacpp_logits model.gguf ids.i32 out.f32 cached|batched [n_gpu_layers]

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

// Reads token ID list from binary int32 file into heap-allocated array.
static int32_t *read_ids(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open ids file %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 4 != 0) {
        fprintf(stderr, "ids file %s holds %ld bytes, not a whole number of int32\n", path, bytes);
        exit(1);
    }
    int32_t *ids = malloc((size_t)bytes);
    if (fread(ids, 1, (size_t)bytes, f) != (size_t)bytes) {
        fprintf(stderr, "short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    *count = (size_t)bytes / 4;
    return ids;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <model.gguf> <ids.i32> <out.f32> <cached|batched> [n_gpu_layers]\n",
                argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *ids_path = argv[2];
    const char *out_path = argv[3];
    const int cached = strcmp(argv[4], "cached") == 0;
    if (!cached && strcmp(argv[4], "batched") != 0) {
        fprintf(stderr, "mode must be 'cached' or 'batched', got %s\n", argv[4]);
        return 1;
    }
    const int n_gpu_layers = argc > 5 ? atoi(argv[5]) : 0;

    size_t n_ids = 0;
    int32_t *ids = read_ids(ids_path, &n_ids);
    // Row i is the next-token logits after consuming ids[i], so the last id
    // is fed to nobody. Exactly `logit_dump.rs`'s layout.
    const int32_t n_rows = (int32_t)n_ids - 1;
    if (n_rows < 1) {
        fprintf(stderr, "need at least 2 ids, got %zu\n", n_ids);
        return 1;
    }

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)n_ids + 8;
    // One decode call per token in `cached`, one call over the whole
    // sequence in `batched`. n_batch also sizes llama.cpp's own output
    // buffer (n_outputs_max defaults to it), which is n_batch * n_vocab
    // floats -- 577 MiB at 551 rows, so do not round this up.
    cparams.n_batch = cached ? 1 : (uint32_t)n_rows;
    cparams.n_ubatch = cparams.n_batch;
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    // Printed for the driver to guard on, and for a human to eyeball against
    // the prompt the ids came from: a vocabulary that does not line up
    // between the two engines would otherwise read as a numerics gap.
    printf("n_vocab %d\n", n_vocab);
    printf("n_rows %d\n", n_rows);
    printf("mode %s\n", cached ? "cached" : "batched");
    printf("n_gpu_layers %d\n", n_gpu_layers);
    printf("first_pieces ");
    for (int i = 0; i < 8 && i < (int)n_ids; ++i) {
        char piece[64];
        int n = llama_token_to_piece(vocab, ids[i], piece, sizeof(piece) - 1, 0, true);
        if (n < 0) n = 0;
        piece[n] = 0;
        for (int c = 0; c < n; ++c) {
            unsigned char ch = (unsigned char)piece[c];
            if (ch < 0x20 || ch == 0x7f) printf("\\x%02x", ch);
            else putchar(ch);
        }
        printf("|");
    }
    printf("\n");
    fflush(stdout);

    FILE *out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "cannot create %s\n", out_path);
        return 1;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    float max_abs = 0.0f;

    if (cached) {
        // The shape THIS PORT runs in: one token at a time through the KV
        // cache. The headline comparison uses this arm.
        struct llama_batch batch = llama_batch_init(1, 0, 1);
        for (int32_t i = 0; i < n_rows; ++i) {
            batch.n_tokens = 1;
            batch.token[0] = ids[i];
            batch.pos[0] = i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "llama_decode failed at position %d\n", i);
                return 1;
            }
            const float *row = llama_get_logits_ith(ctx, 0);
            if (!row) {
                fprintf(stderr, "no logits at position %d\n", i);
                return 1;
            }
            for (int32_t v = 0; v < n_vocab; ++v) {
                float a = row[v] < 0 ? -row[v] : row[v];
                if (a > max_abs) max_abs = a;
            }
            fwrite(row, sizeof(float), (size_t)n_vocab, out);
            if ((i + 1) % 50 == 0) {
                fprintf(stderr, "cached: %d/%d\n", i + 1, n_rows);
            }
        }
        llama_batch_free(batch);
    } else {
        // One pass over the whole sequence. Same weights, same kernels, same
        // engine as the arm above -- only the forward shape differs, which
        // is what makes the pair a FLOOR for any cross-engine number.
        struct llama_batch batch = llama_batch_init(n_rows, 0, 1);
        batch.n_tokens = n_rows;
        for (int32_t i = 0; i < n_rows; ++i) {
            batch.token[i] = ids[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed on the batched pass\n");
            return 1;
        }
        for (int32_t i = 0; i < n_rows; ++i) {
            const float *row = llama_get_logits_ith(ctx, i);
            if (!row) {
                fprintf(stderr, "no logits for batch index %d\n", i);
                return 1;
            }
            for (int32_t v = 0; v < n_vocab; ++v) {
                float a = row[v] < 0 ? -row[v] : row[v];
                if (a > max_abs) max_abs = a;
            }
            fwrite(row, sizeof(float), (size_t)n_vocab, out);
        }
        llama_batch_free(batch);
    }

    fclose(out);
    // The softcap guard. Gemma's GGUF carries final_logit_softcapping = 30,
    // so a max above it means llama.cpp is NOT applying the cap this port
    // applies, and every divergence downstream would be dominated by that
    // rather than by the weights.
    printf("max_abs_logit %.6f\n", max_abs);
    printf("wrote %s\n", out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(ids);
    return 0;
}
