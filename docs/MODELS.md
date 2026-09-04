# Getting a model: the catalog, the probe, and `pull`

`turbospark-model` finds, inspects and installs models. It has two halves that
answer two different questions, and knowing which one you are asking saves a
lot of time:

- **The catalog** says what has been run here. Fourteen rows, each naming a
  repository and a revision that were streamed and generated on real hardware,
  with the gate targets that assert it.
- **The probe** says what could be run here. It reads headers, costs KB and
  seconds, and decides: architecture, block types or affine width, expert
  granularity, tokenizer sidecars. This is the half that scales past the
  table.
- **`recommend`** puts the two together and asks the question you probably
  came with: what should *this* machine run? See
  [below](#what-should-this-machine-run).

```sh
turbospark-model list                    # the catalog
turbospark-model recommend               # ...ranked for this machine
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

`--model <name>` takes a path or an alias, on **both** `turbospark-check` and
`turbospark-server`; the two call one `resolve_model_arg`, so an install has
one name whichever binary opens it. **An existing directory always wins.** A
bare name that silently preferred an alias would run a different model than
the one on the command line, fluently and with no error, so the fallback only
applies to a string that is not a directory. The server prints the resolved
directory beside the alias at startup, because it is the one of the two that
runs unattended.

`turbospark-model path <alias>` prints the directory and fails if the model is
not installed. `--model <alias>` is the shorter form; reach for `path` in a
script that would rather fail before starting than serve the wrong model, and
note `--model $(turbospark-model path x)` cannot quietly expand to
`--model ''`.

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

## What should *this* machine run?

```
turbospark-model recommend
turbospark-model recommend --context 8192
turbospark-model recommend --probe          # reads every row's header, ~60 s
turbospark-model recommend --discover 20    # and Hugging Face at large
```

```
machine: 36.0 GiB of memory on Apple M4 Max, 28.1 GiB of Metal working set
fitting against a 4096-token context

  MODEL        EVIDENCE       ALLOCS    ON DISK      TOK/S  VERDICT
  gemma4       verified     ~3.3 GiB   12.1 GiB      33-46  fits, fully resident
  qwen36       verified     ~2.2 GiB   16.8 GiB      33-38  fits, fully resident
  ...
  mixtral      caveat      ~55.0 GiB   27.0 GiB          -  does not fit
```

**The two size columns answer different questions and that is the point.**

- **ALLOCS** is what the engine allocates and what `phys_footprint` charges
  for: the expert-cache slot cache plus the KV cache. Exceeding memory here
  is a failed `open()`.
- **ON DISK** is the whole install, weights included. Exceeding memory here is
  not an error at all -- it is the streaming this engine is built around, and
  it costs throughput rather than correctness.

A single "size" column would have to be wrong for one of the two: a 13 GB
install on a 16 GB machine *runs* (it streams), and a 27 GB one that "fits" by
file size can still be refused on its slot cache. `mixtral` above is refused
at 55 GiB of ALLOCS against 27 GiB on disk, which is AGENTS.md Gotcha 36 in
one line.

The marker on ALLOCS says where the number came from: `*` measured on this
chip, `~` estimated from the checkpoint's shape, `?` nothing has read the
header yet. **TOK/S is only ever quoted, never estimated** -- decode rate does
not track weight bytes on this engine (one architecture reads 18.3 / 14.2 /
19.0 tok/s at 1 / 2 / 4 bits), so a row nobody has measured shows a dash
rather than a guess.

Ordering is: it fits, then how much is known about it, then its measured rate,
then its size. **A discovered repository never outranks a curated row that
fits**, however big or popular, because nothing here has run it.

### What the default arm cannot tell you

Offline, a row that has been through a memory oracle carries its measured
peak and gets an exact answer; a row that has not reports `unknown`. The gap
is not laziness -- the slot cache and the KV are functions of the
checkpoint's **shape**, and nothing offline knows it. Filling it from the
family's baseline would be wrong: one architecture string covers several
checkpoints, `llama` alone covers Mixtral 8x7B, Mistral 7B and TinyLlama
1.1B, and reading a baseline as a checkpoint's own shape is what once shipped
a `head_dim` of 128 to a model with 64. `--probe` reads the real header.

A measured peak is also a peak at **one context and one slot count**, so it
is reported rather than applied when either differs, with a line saying which
one. Gemma 4 is 2,175 MiB at 4,096/16 and 3,654 MiB at 4,096/32; quoting the
first for the second is not an approximation.

### Discovery

`--discover N` pulls the N most-downloaded GGUF text-generation repositories
and puts each through the **same probe** the command above uses, so a
discovered row is refused in the same words and for the same reasons. Where a
repository publishes ten quantizations it picks the largest whose block types
have kernels here; the name is only a pre-filter, and the header decides. A
sharded GGUF is skipped rather than partially installed.

The usual reason a plausible repository still cannot be installed is the
tokenizer: a GGUF carries llama.cpp's representation and this port loads an HF
`tokenizer.json`, so discovery reads the card's `base_model` and, failing
that, tells you to pass `--sidecar-repo`.

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
   first gate that failed, which is the most fundamental one; a later gate
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
2. **Fetch the tokenizer sidecars and verify them**: load them with
   `MfTokenizer`, render a one-turn conversation, encode it. A few MB and
   milliseconds.
3. Stream the weights through the repack walk.
4. Read the install back through `model_io`'s real loaders.
5. Record it in `installed.json`.

Step 2 comes before step 3 because `Qwen3.8-27B`'s bring-up failed on a 404
for `merges.txt` **after** a 20-minute stream had written a perfectly good
install (AGENTS.md Gotcha 47). By the time a byte of weight data moves, the
install is known to have a tokenizer that loads and a template that renders.

### A row can pull a second repository for a drafter head

`CatalogEntry.mtp` (`repo`, `revision`) is optional and names a repository
SEPARATE from `source.repo`, for a checkpoint whose own conversion drops a
multi-token-prediction head (`docs/MTP.md`). Step 3 above reads that
repository's own shard index, keeps only the shard(s) whose tensors are
`mtp.`-prefixed, and merges them into the same multi-shard registry the trunk
streams through -- one install, one resident index, no second artifact.
`qwen38-27b-mtp` is the one row that carries it; see `docs/MTP.md`'s
"Installing a headed checkpoint" for what was verified against real bytes.
Leave the field out entirely for every other row -- it exists only where the
weights repository and the head repository genuinely differ.

---

## The `measured` block

Eleven rows carry one, and it is what `recommend` quotes:

```json
"measured": [{
  "chip": "Apple M4 Max", "context": 4096, "expert_cache_slots": 16,
  "peak_footprint_mib": 2175,
  "decode_tok_s_min": 33.039, "decode_tok_s_max": 45.648,
  "measured_on": "2026-08-16", "source": "this port, four readings, ..."
}]
```

**These are observations; the ceilings and floors that guard them stay in the
oracle targets**, with the paragraph of provenance that justifies each margin
(JSON has nowhere to put a paragraph, and the margins differ per row on
purpose). The two are tied together by
`oracle_common::assert_agrees_with_catalog`, which is NOT `#[ignore]`d and
needs no install: edit either side into disagreement and `cargo test
--workspace` fails on the edit.

The convention is **worst-observed**: the slowest reading of the slowest
protocol case, the fastest of the fastest, and the highest peak. So the pair
brackets what to expect rather than advertising a best case, and
`decode_tok_s_min` is comparable with the oracle's floor by construction.
Pasting a favourable number in silently loosens that check.

Two fields look redundant and are not. `context` is there because every
footprint is a footprint at one window -- on a dense install the window is
most of it. `expert_cache_slots` is there because on a streamed MoE the slot
cache is the dominant term, and `--expert-cache-slots auto` may well resolve
to a different one on your machine than the protocol pinned.

---

## Adding a catalog row

Same admission rule `crates/repack/src/arch_registry.rs` states for its
architecture strings: **a row exists only if that exact repository and
revision were streamed and run here.** A model somebody expects to work is not
a row; a model somebody ran is.

If the model is a new architecture rather than another checkpoint of one that
already runs, the row is the last step rather than the first: see
[`NEW_MODEL.md`](NEW_MODEL.md) Phase 7, which lists the probe and install-driver
match sites a new family has to be wired into before `pull` can reach it at
all. A new GGUF-source family needs none of them -- both halves of that path
are family-agnostic and read `arch_registry.rs`.

1. Install it with `pull --repo ... --alias ...` and generate with it.
2. Add the row to `crates/catalog/src/models.json`. Pin a commit sha where the
   publisher offers one. Take `download_bytes` from the network guard rather
   than from a listing page; see below.
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

- [`MODEL_FAMILY.md`](MODEL_FAMILY.md): which architectures run, which are
  recognized and refused, and the parity matrix
- [`NEW_MODEL.md`](NEW_MODEL.md): the bring-up checklist for an architecture
  the probe refuses
- [`GTURBO.md`](GTURBO.md): the install format `pull` writes
- [`BENCHMARKS.md`](BENCHMARKS.md): the frozen rows a `verified` status
  refers to
