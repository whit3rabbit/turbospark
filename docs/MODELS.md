# Getting a model: the catalog, the probe, and `pull`

`turbospark-model` finds, inspects and installs models. It has two halves that
answer two different questions, and knowing which one you are asking saves a
lot of time:

- **The catalog** says what has been RUN here. Thirteen rows, each naming a
  repository and a revision that were streamed and generated on real hardware,
  with the gate targets that assert it.
- **The probe** says what COULD be run here. It reads headers, costs KB and
  seconds, and decides: architecture, block types or affine width, expert
  granularity, tokenizer sidecars. This is the half that scales past the
  table.

```sh
turbospark-model list                    # the catalog
turbospark-model info gemma4             # one row in full
turbospark-model pull gemma4             # install it
turbospark-check --model gemma4 --messages-file /tmp/p.json
```

```sh
turbospark-model probe TheBloke/SomeModel-GGUF --file some-model.Q4_K_M.gguf
```

---

## Before anything else: a pull cannot resume

`crates/repack`'s ranged downloader retries a chunk eight times and then gives
up, and giving up costs the whole walk. A `pull` that dies 19 GB into a 26 GB
stream starts again from zero. This is stated in the usage text and again by
`pull` itself before it begins, because it is the property nobody discovers
until it costs them twenty minutes.

Adding resume is a change to the walks in `crates/repack`, not to
`crates/catalog`.

---

## The store

| | |
|---|---|
| root | `$TURBOSPARK_HOME`, default `~/.turbospark` |
| installs | `$ROOT/models/<alias>.gturbo`, or wherever `--out` says |
| record | `$ROOT/installed.json` |

`turbospark-check --model <name>` takes a path OR an alias. **An existing
directory always wins.** A bare name that silently preferred an alias would
run a different model than the one on the command line, fluently and with no
error, so the fallback only applies to a string that is not a directory.

`turbospark-model path <alias>` prints the directory and FAILS if the model is
not installed, so `--model $(turbospark-model path x)` cannot quietly expand
to `--model ''`.

---

## Status tiers

A row's status is evidence, not intent.

| status | means |
|---|---|
| `verified` | has a frozen quality-gate and/or memory-oracle row in [`BENCHMARKS.md`](BENCHMARKS.md). Some number about it is asserted by a test that can go red. |
| `runs` | installed and generated coherent text here, with no frozen row. |
| `caveat` | installs and runs, and the notes disqualify it. Kept BECAUSE the disqualification is invisible otherwise. |

The one `caveat` row is `mixtral`, and it is worth reading as an example of
what the probe is for: Mixtral 8x7B installs correctly, decodes correctly, and
wants **54.5 GiB of pinned expert-slot cache** at the default 16 slots. It is
the smaller model by parameter count than several rows that run fine. What
decides whether a model fits this engine is how finely it splits its experts,
not how big it is, and that number is one multiplication off the header
(AGENTS.md Gotcha 36). `probe` prints it.

---

## Reading a probe

```
RUNNABLE  mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF@main
  declares    llama
  family      llama
  shape       32 layers, 4096 hidden, 32000 vocab, 8 experts top-2
  block types
    Q4_K       113 tensors  20.0 GiB (75.6%)     has kernels
    F32         97 tensors  5.0 MiB (0.0%)       transcoded at repack
  experts     one expert is 108.9 MiB, so the pinned slot cache would be
                 8 slots: 27.2 GiB  <-- will not fit
                16 slots: 54.5 GiB  <-- will not fit
  tokenizer   tokenizer.json, tokenizer_config.json, generation_config.json
  template    tokenizer_config.json:chat_template
```

Four things to read, in this order:

1. **The verdict.** `RUNNABLE` or `REFUSED` with a reason. A refusal names the
   FIRST gate that failed, which is the most fundamental one -- a later gate
   overwriting it produces messages that are true and useless.
2. **The block types.** `transcoded at repack` is not the same claim as `has
   kernels`: F32/F16/BF16 are narrowed at install time and reach no dispatch
   at all. A type shown as `UNSIZED` is one this port cannot even measure, and
   it is reported rather than rendered as zero bytes, because a zero sorts to
   the bottom of a share column and that is the exact inverse of its real rank.
3. **The expert arithmetic.** Gates nothing. Decides everything about fit.
4. **The tokenizer line.** `template NONE FOUND` on an instruction-tuned model
   means it will be framed by its dialect's fallback, which produces fluent
   output that is not an answer (AGENTS.md Gotcha 41).

`probe` exits 0 only on `RUNNABLE`, so `probe X && pull --repo X ...` works.

---

## Installing something not in the catalog

```sh
turbospark-model pull --repo owner/name --alias myname \
  --sidecar-repo owner/original-checkpoint
```

`pull` probes first and refuses on a red verdict; `--force` overrides.

**A GGUF pull almost always needs `--sidecar-repo`.** A GGUF carries its
tokenizer as llama.cpp metadata and this port loads an HF `tokenizer.json`, so
the sidecars have to come from the checkpoint the GGUF was converted from.
Nothing about the weights repo says which one that is, which is why every GGUF
row in the catalog names it explicitly.

### What `pull` does, in order

The order is the design, and it inverts how the `crates/repack/tests/*_network.rs`
tests do it:

1. Probe, and refuse on a red verdict.
2. **Fetch the tokenizer sidecars and verify them** -- load them with
   `MfTokenizer`, render a one-turn conversation, encode it. A few MB and
   milliseconds.
3. Stream the weights through the repack walk.
4. Read the install back through `model_io`'s real loaders.
5. Record it in `installed.json`.

Step 2 comes before step 3 because `Qwen3.8-27B`'s bring-up failed on a 404
for `merges.txt` **after** a 20-minute stream had written a perfectly good
install (AGENTS.md Gotcha 47). By the time a byte of weight data moves, the
install is known to have a tokenizer that loads and a template that renders.

---

## Adding a catalog row

Same admission rule `crates/repack/src/arch_registry.rs` states for its
architecture strings: **a row exists only if that exact repository and
revision were streamed and run here.** A model somebody expects to work is not
a row; a model somebody ran is.

1. Install it with `pull --repo ... --alias ...` and generate with it.
2. Add the row to `crates/catalog/src/models.json`. Pin a commit sha where the
   publisher offers one. Take `download_bytes` from the network guard rather
   than from a listing page -- see below.
3. Run the offline tests: `cargo test -p turbospark-catalog`.
4. Run the guard: `cargo test -p turbospark-catalog --test catalog_network
   --release -- --ignored --nocapture`, and paste its published byte figure
   back into the row.
5. Set `status` honestly. `verified` needs a frozen row in `BENCHMARKS.md`.

### Local rows without a rebuild

`$TURBOSPARK_HOME/models.json` merges over the curated table by alias, using
the same schema and the same validation. `list` marks the rows it introduced.

---

## The rot guard

```sh
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture
```

A file list and a HEAD per row, ~26 seconds for the whole table, downloads
nothing. It checks that every sidecar named in a row exists in the repository
that row names for it, that the weights file exists at the size recorded, and
that every GGUF's `general.architecture` still resolves to the family the row
claims.

**Why the size matters more than it looks.** Every GGUF row is pinned at
`main`, because those publishers offer nothing else. A floating row's frozen
numbers stop meaning anything the moment the file is re-uploaded, and
`download_bytes` is the only fingerprint that would notice. That is why the
tolerance is 2% and why every figure in the table was read off this test's own
HEAD requests: an earlier draft carried round numbers at 10%, under which
`mixtral`'s recorded "26 GB" sat 9.4% from its real 28,448,468,384 and would
have absorbed an entire re-quantization.

---

## Gated repositories

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`). Without one a gated repository
answers 401, and the error says so by name rather than leaving it to look like
a bug here.

---

## See also

- [`MODEL_FAMILY.md`](MODEL_FAMILY.md) -- which architectures run, which are
  recognized and refused, and the parity matrix
- [`NEW_MODEL.md`](NEW_MODEL.md) -- the bring-up checklist for an architecture
  the probe refuses
- [`GTURBO.md`](GTURBO.md) -- the install format `pull` writes
- [`BENCHMARKS.md`](BENCHMARKS.md) -- the frozen rows a `verified` status
  refers to
