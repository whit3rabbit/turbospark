# turbospark-catalog

The curated model catalog, the header-only Hugging Face probe, and the install
driver that turns either into a `.gturbo` directory. Backs the
`turbospark-model` binary in `crates/cli`.

User-facing documentation: [`docs/MODELS.md`](../../docs/MODELS.md).

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/catalog/
+-- Cargo.toml
+-- src/
|   +-- lib.rs                  # Library root and re-exports
|   +-- models.json             # The curated table, embedded with include_str!
|   +-- entry.rs                # CatalogEntry, Source, Sidecars, SourceKind, Status
|   +-- catalog.rs              # Load embedded + merge a user override, lookup
|   +-- hf.rs                   # HF API file list, resolve URLs, small-file GET, HF_TOKEN
|   +-- probe/
|   |   +-- mod.rs          # Types, the dispatcher, the sidecar check
|   |   +-- gguf.rs         # The GGUF gates: architecture, block types, expert stride
|   |   \-- safetensors.rs  # The MLX gates: model_type, affine width, expert stride
|   +-- install.rs              # The walk driver: plan -> .gturbo install
|   \-- store.rs                # ~/.turbospark layout, installed.json, alias resolution
\-- tests/
    +-- catalog.rs              # The embedded table's structural invariants
    +-- store.rs                # Store layout, the record, and resolution ORDER
    +-- probe.rs                # Every probe gate, offline, both directions
    \-- catalog_network.rs      # The rot guard: does the table still describe reality (ignored)
```

## Key Modules

- `entry.rs`: one catalog row. **Every field exists because it cannot be
  derived from the repo name**, which is the admission rule for the struct as
  much as for the table. `CatalogEntry::validate` enforces the structural ones
  at load, so a malformed user override fails at the point of reading rather
  than a quarter of an hour into a walk.
- `catalog.rs`: the embedded table plus an optional `$TURBOSPARK_HOME/models.json`
  merged over it by alias. A schema version it does not recognize is REFUSED
  rather than best-effort parsed -- the failure mode of guessing is a silently
  dropped field, and a dropped `sidecars.repo` is a 404 after the stream.
- `hf.rs`: KB-scale endpoints only. **Deliberately does not download weights**:
  those go through `repack::HttpRangeSource`, which owns the chunking, the
  retry ladder and the `http1_only()` client setting that the Xet bridge's
  per-edge rate cap makes load-bearing (AGENTS.md Gotcha 46). A second client
  here fetching a multi-GB body would quietly lose all three.
  `Client::file_list` is what makes probing an unknown repository possible at
  all; `content_length` reads `x-linked-size` before `content-length`, because
  a Xet-backed file reports the LFS object's real length in the former.
- `probe/`: four gates, cheapest first. `mod.rs` owns the report types, the
  GGUF-or-safetensors dispatch and the sidecar check; `gguf.rs` and
  `safetensors.rs` each own one format's gates. In both halves the
  `evaluate_*` entry point is split from its fetching wrapper so every gate is
  testable with no network -- which matters because a live probe of a curated
  row takes the accepting path every time, and the refusal paths are where the
  decisions and the wording are.
- `install.rs`: the shape all thirteen `crates/repack/tests/*_network.rs` files
  repeat, written once, with the step order inverted (see Gotcha 1).
- `store.rs`: `Store::resolve`'s ORDER is the load-bearing part; see Gotcha 2.

## Development & Test Commands

```sh
# Offline tests: the table, the store, and every probe gate.
cargo test -p turbospark-catalog

# The rot guard. A file list and a HEAD per row, ~26 s for the whole table,
# downloads nothing. Run it after adding or editing a row, and paste its
# published byte figure back into the row.
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture

# The binary lives in crates/cli.
cargo run -p turbospark-cli --bin turbospark-model -- list
cargo run -p turbospark-cli --bin turbospark-model -- probe owner/name
cargo run --release -p turbospark-cli --bin turbospark-model -- pull tinyllama
```

## Crate Gotchas

1. **SIDECARS ARE FETCHED AND VERIFIED BEFORE ANY WEIGHT BYTE MOVES, and that
   ordering is the whole reason this crate exists rather than a helper
   function.** Every `crates/repack/tests/*_network.rs` file streams multi-GB
   weights first and fetches tokenizer sidecars afterwards. That is how
   `Qwen3.8-27B`'s bring-up failed on a 404 for `merges.txt` AFTER a 20-minute
   stream had written a perfectly good install (AGENTS.md Gotcha 47): the
   artifact was fine and the run was wasted. `install()` fetches the sidecars
   into the destination, loads them with `MfTokenizer`, renders a one-turn
   conversation and encodes it -- a few MB and milliseconds -- and only then
   streams. **Loading is not enough on its own**, which is why it also renders:
   `load_from_dir` succeeds on any valid `tokenizer.json`, template or no
   template, and a template-less instruction-tuned model is framed by its
   dialect's fallback, which produces fluent output that is not an answer
   (Gotcha 41). Do not "tidy" this by moving the sidecars next to the other
   file writes.

2. **`Store::resolve` prefers an existing DIRECTORY over an alias, and that is
   not a tie-break.** `turbospark-check --model` has always taken a path,
   every gate env var in this repo points at one, and a string that silently
   resolved to an alias of the same name would run a DIFFERENT model than the
   one on the command line -- fluently, with no error, and with a footer
   reporting perfectly plausible tok/s. `resolve_model_arg` additionally
   returns the raw string unchanged when there is no store at all, so a
   machine with no `HOME` reports the same "no such install" it always did
   rather than a new failure. `tests/store.rs` pins both.

3. **A PROBE MUST NOT INHERIT A PARSER'S DEFAULT, and one of them is exactly
   inverted here.** `repack::parse_gemma4_quantization` answers
   `Gemma4Quant::default()` -- 4-bit, group 64 -- when `config.json` carries no
   `quantization` block. That is right for its own callers, which have already
   established the checkpoint is MLX-quantized, and catastrophic for a probe
   whose entire question is whether it is: inheriting it makes every BF16
   checkpoint on Hugging Face report as a runnable INT4 one. `probe/safetensors.rs` checks
   the key's PRESENCE separately, before parsing. This is AGENTS.md Gotcha 39's
   shape one layer out -- a default is a claim about what silence means, and
   this module's question makes silence mean something else. Audit any other
   parser default this crate reads for the same inversion.

4. **A later gate must not overwrite an earlier refusal.** `ProbeReport::refuse`
   keeps the FIRST one, because the gates run cheapest-first which is also
   most-fundamental-first. Found live: probing
   `bartowski/Phi-3.5-mini-instruct-GGUF` refuses at the architecture (`phi3`
   is recognized, has no decode flow, and the registry says what it would
   need) and then refuses AGAIN at the sidecars, because a GGUF repo carries
   no `tokenizer.json`. The second message is true and useless -- it sends the
   reader off to find a sidecar repo for a model that would not run with one.

5. **F32/F16/BF16 must not be checked against `EXECUTABLE_GGUF_TYPES`.** They
   are transcoded at repack time and reach no dispatch, so checking them marks
   every real candidate blocked -- which is exactly what
   `scopes_the_dense_llama_candidates`' first run did (`crates/repack/CLAUDE.md`
   Gotcha 5). `TypeShare` carries `transcoded` beside `executable` so the
   output can say "transcoded at repack" rather than "has kernels", because
   the second states something false about this port and is the version that
   sends somebody looking for the F32 GEMV that does not exist.

6. **`download_bytes` is a FINGERPRINT, not just a progress-bar input.** Every
   GGUF row is pinned at `main` because those publishers offer nothing else,
   and a floating row's frozen numbers stop meaning anything the moment the
   file is re-uploaded. The size is the only thing that would notice, which is
   why `catalog_network.rs`'s tolerance is 2% and why every figure in
   `models.json` was read off that test's own HEAD requests. An earlier draft
   carried round numbers at 10%: `mixtral`'s recorded "26 GB" sat 9.4% from
   its real 28,448,468,384 and would have absorbed an entire re-quantization.
   Do not widen the tolerance to accommodate a hand-typed figure; run the
   guard and paste its number back.

7. **The sidecar list is per REPOSITORY, never per family, and
   `tests/catalog.rs` asserts that it VARIES.** `ternary27b` and `qwen38-27b`
   are one architecture and their lists are near-inverses -- the first ships
   `merges.txt` and no `generation_config.json`, the second the reverse. If
   `two_checkpoints_of_one_architecture_have_different_sidecar_lists` ever goes
   green because the two lists became identical, somebody has copied one to
   the other and a 20-minute stream is about to fail at the end.
