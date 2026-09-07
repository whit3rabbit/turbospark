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
|   +-- auth.rs                 # HF_TOKEN resolution (explicit/env/store/cache), whoami-v2 validation
|   +-- vision.rs               # Resolve an installed vision-tower sidecar by (family, hidden_size)
|   +-- probe/
|   |   +-- mod.rs          # Dispatcher and sidecar check
|   |   +-- types.rs        # Probe verdict and result types
|   |   +-- gguf.rs         # The GGUF gates: architecture, block types, expert stride
|   |   \-- safetensors.rs  # The MLX gates: model_type, affine width, expert stride
|   +-- recommend/
|   |   +-- mod.rs          # Machine, Recommendation, the catalog arm
|   |   +-- tests.rs        # Unit tests for recommendation formatting and filtering
|   |   +-- fit.rs          # counted vs mapped: does it fit, and how much context
|   |   +-- fit_tests.rs    # Unit tests for fit and memory sizing
|   |   +-- rank.rs         # the ordering (vendored shape; see NOTICE)
|   |   \-- discover.rs     # popular HF repos, filtered through the probe
|   +-- install.rs              # The walk driver: plan -> .gturbo install
|   +-- stream.rs               # GGUF and MLX streaming and shard writing helpers
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
- `auth.rs`: resolves an `HF_TOKEN` in priority order (explicit override, then
  `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`, then the store's own `hf_token` file,
  then the standard `hf`-CLI cache files) and validates one against
  `whoami-v2`. Read together with Gotcha 12: this is what a caller reaches
  for before trusting a probe's chat-template verdict on a gated repo.
- `probe/`: four gates, cheapest first. `mod.rs` owns the report types, the
  GGUF-or-safetensors dispatch and the sidecar check; `gguf.rs` and
  `safetensors.rs` each own one format's gates. In both halves the
  `evaluate_*` entry point is split from its fetching wrapper so every gate is
  testable with no network -- which matters because a live probe of a curated
  row takes the accepting path every time, and the refusal paths are where the
  decisions and the wording are.
- `recommend/`: what this machine should run. `fit.rs` is the arithmetic and
  reuses `model_io`'s two sizing policies rather than restating them -- the
  same `ExpertCacheSlots::resolve` and `kv_bytes_for_context` `open()` calls,
  which is what stops a recommendation promising a configuration the engine
  then declines. `rank.rs` and the tiering shape in it are adapted from
  shoehorn (`NOTICE`); `discover.rs` is the network arm and gates every
  candidate through `probe/` unchanged. See Gotchas 8-10.
- `install.rs`: the shape every install-writing `crates/repack/tests/*_network.rs` file
  repeat, written once, with the step order inverted (see Gotcha 1).
- `store.rs`: `Store::resolve`'s ORDER is the load-bearing part; see Gotcha 2.
- `vision.rs`: `resolve_vision_sidecar` finds the ONE installed vision-tower
  row pairing with a `(family, hidden_size)`, reading back through
  `model_io::load_vision_sidecar` for every candidate rather than trusting
  the store's own `family` string. Zero matches and more than one match are
  both refused rather than picked between: a revision pin is load-bearing,
  so nothing here silently disambiguates two installed towers of the same
  shape.

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

8. **`counted` AND `mapped` ARE DIFFERENT QUESTIONS AND A FIT MODEL NEEDS
   BOTH.** `counted` is what the engine ALLOCATES -- the expert-cache slot
   cache plus the KV -- so exceeding memory there is a failed `open()`.
   `mapped` is the install on disk, and exceeding memory there is the
   STREAMING this engine is built around: it costs throughput, not
   correctness. Collapsing them into one boolean gets a 13 GB install on a
   16 GB machine wrong in one direction (it runs, and the slot policy's floor
   exists for exactly that machine) or a 27 GB one wrong in the other.

   **The resident core is in `mapped`, which contradicts `crates/bench/CLAUDE.md` Gotcha 1
   and matches every frozen peak.** That gotcha says
   `newBufferWithBytesNoCopy` pins the mapped range into `phys_footprint`;
   Gemma 4 reads 2,175 MiB against a 1.26 GiB core plus 1.5 GiB of slot cache
   plus 320 MiB of KV, and the core is absent. `gptoss_memory_oracle.rs` says
   the same in its own words, and Gotcha 40 says it outright for the dense
   case. Putting the core in `counted` would make the estimate incomparable
   with the rows it is checked against, and would read museGlimmer's 15 GB of
   dense weights as 15 GB of allocation against a measured 536 MiB.

9. **A MEASURED PEAK APPLIES AT ONE CONTEXT AND ONE SLOT COUNT AND NOWHERE
   ELSE.** Both terms it is made of move with those: KV is a pure function of
   the window, and the slot cache is `slots x layers x expert_stride`. Gemma 4
   is 2,175 MiB at 4,096/16 and 3,654 at 4,096/32. Quoting the first against
   an 8,192/32 request is not an approximation, it is a different
   measurement, so `from_entry` reports it instead of applying it and says
   which term disagreed.

   **The trap underneath is that an UNPROBED row resolves 16 slots by
   ignorance**, not by arithmetic: with no `ArchConfig` there is no expert
   stride, so `Auto` divides by nothing and returns `DEFAULT_CACHE_SLOTS`,
   which happens to be the 16 the protocol pins. So the two look like
   agreement. An unprobed row therefore takes the measurement AT ITS OWN
   STATED CONFIGURATION and says so, rather than claiming to describe the one
   `open()` would choose. That distinction is also why `fit()` takes the slot
   policy as a PARAMETER: the accuracy gate has to pin what the protocol
   pinned, and passing `Auto` there reads as a 55% overestimate.

    **THE LOAD GUARD IS A THIRD SUCH PARAMETER AND CARRIES A SHARPER VERSION
    OF THE RULE.** `fit()` takes a `model_io::LoadGuard` and `Machine` carries
    one, because this ranking and `resolve_max_context`'s refusal share a
    memory budget BY CONSTRUCTION -- that is what makes a recommendation worth
    showing. A hub ranking under `relaxed` while its sessions open under
    `strict` promises a fit the loader then refuses, in the one place a user
    has no way to see the two disagree. Measured rather than argued: `strict`
    on a 10 GiB machine drops `gptoss-20b` off the top of a table that
    `relaxed` reports as fitting. `Fit` therefore CARRIES the tier that
    produced it, so `verdict_for_counted` cannot re-derive a verdict under a
    different one after a caller substitutes a measured peak. The default is
    `Relaxed`, the arithmetic every `measured` block in `models.json` was
    taken under (`model-io` Gotcha 3), and `render.rs` names the tier in the
    header only when it is not the default. See `docs/LOAD_GUARD.md`.

10. **THE FAMILY BASELINE IS NOT A SUBSTITUTE FOR A CHECKPOINT'S OWN SHAPE.**
    `known_architecture(family)` is right there and would turn every offline
    `unknown` into a number, and it is wrong for the same reason AGENTS.md
    Gotcha 39 records: one architecture string covers several checkpoints.
    `llama` alone is Mixtral 8x7B, Mistral 7B and TinyLlama 1.1B, whose head
    dimensions and layer counts differ -- reading a baseline as a checkpoint's
    own shape is what shipped a `head_dim` of 128 to a model with 64.
    `recommend --probe` reads the real header instead, and an unread row
    reports `unknown` rather than a plausible wrong number.

11. **`models.json` round-trips byte-identically through Python's
    `json.dumps(obj, indent=2, ensure_ascii=False) + "\n"`**, so a bulk row
    edit can be scripted without reformatting the file. Verify the round trip
    before writing (`json.dumps(json.loads(raw), ...) == raw`), because the
    day it stops being true the diff is the whole table.

12. **A GATED SIDECAR REPO WITHOUT `HF_TOKEN` USED TO READ AS "NO CHAT
    TEMPLATE" RATHER THAN AS AN ERROR; `probe/mod.rs`'s `resolve_chat_template`
    now tells the two apart.** `check_sidecars`'s `tokenizer_config.json`
    branch fetches through `client.get_optional(...)` and hands the `Result`
    to `resolve_chat_template`, which matches `Some(Err(e))` separately from
    `Some(Ok(None)) | None` -- `get_optional` correctly returns
    `Err("GET url: HTTP 401")` for a gated repo with no token, and that arm
    gets its own warning naming `HF_TOKEN` rather than falling into the
    generic "no chat template found in either place" line
    (`an_http_error_gets_its_own_warning_naming_hf_token` in `probe/mod.rs`).
    Measured on `meta-llama/Meta-Llama-3-8B-Instruct` (gated): unauthenticated
    probe reports `template NONE FOUND`; with `HF_TOKEN` set,
    `tokenizer_config.json:chat_template`. Export
    `HF_TOKEN="$(cat ~/.cache/huggingface/token)"` (or wherever `hf auth
    login` wrote it) before trusting a probe's chat-template verdict on a
    gated repo.

13. **THE SLOT POLICY IS A CALLER'S PARAMETER NOW, ALL THE WAY OUT TO THE C
    ABI, AND THAT IS GOTCHA 9 BECOMING REACHABLE RATHER THAN A NEW RULE.**
    `from_entry` and `recommend_catalog` hardcoded
    `ExpertCacheSlots::Auto` while `fit()` already took the policy as an
    argument, so the one caller that could disagree with `open()` about a
    slot count -- a GUI whose inspector is set to 32 -- had no way to say so.
    Both take it now, `DiscoverOptions` carries it beside `context`, and
    `ts_recommend_json` and `ts_probe_json` take it in their options bags
    under `expertCacheSlots`, spelled exactly as `ts_session_open` spells it.
    `the_requested_slot_count_reaches_the_fit` is the guard, and it reads at
    8,192 deliberately: at 4,096 `gemma4`'s frozen row is APPLIED and
    `Fit::slots` comes from the measurement rather than from the policy, so
    the obvious version of that test passes with the parameter ignored.

    **THE VALIDATION HAS TO TRAVEL WITH IT.** `ExpertCacheSlots::Fixed` is
    built from whatever it is handed and the setters panic outside
    `ALLOWED_CACHE_SLOTS`; `crates/ffi` is linked INTO its host, so an
    unvalidated count aborts the app rather than raising an error a GUI can
    show. `ts_session_open` had learned that already and the check lived in
    its own body, which is why the two new entry points would each have
    needed their own copy. It is `wire::expert_cache_slots` now, one
    validator with three callers -- and the mutation that proves it reddens
    the open's pre-existing case as well as the two new ones.

14. **A PROBE REPORTS A FIT, AND ITS `mapped` TERM IS THE PUBLISHED
    CHECKPOINT RATHER THAN THE INSTALL THIS PORT WOULD WRITE.**
    `ProbeReport` carried the expert stride and the slot-cache table and no
    verdict, so an arbitrary repository could be sized and not judged.
    `models::probe_fit` closes that with the same `fit()` the curated rows
    take, at the context and slot count the caller names.

    The one honest gap is the size: `install_bytes` and `download_bytes` are
    separate `CatalogEntry` fields because they differ, and a probe has only
    the second. `recommend::discover` already relied on that approximation
    and states why (a GGUF's expert blobs are written verbatim and only the
    small resident core is transcoded); it is looser for an MLX repository.
    So the JSON says `mappedSource: "download"` and the GUI labels it "as
    published" rather than presenting it as an install size. `counted` is
    unaffected -- it is built from the arch and the stride, which are
    properties of the checkpoint rather than of the container.

15. **`gguf_variants` IS A LISTING AND `choose_quantization` IS A CHOICE, AND
    THE TWO FILTER DIFFERENTLY ON PURPOSE.** The scan picks the largest
    runnable file, so it drops anything naming a type with no kernels. A
    PICKER that dropped those would show three of a repository's eight files
    and read as the repository having three, so the listing keeps them with
    `executable: false` and lets the probe refuse them in the probe's own
    words. Sharded files are the one real exclusion, for the scan's reason
    (installing one shard installs a fraction of a model that then fails to
    open) -- and they are COUNTED rather than merely dropped, because a
    repository publishing nothing else would otherwise present an empty
    picker, which reads as "no GGUF here" instead of "this port cannot walk a
    shard set".
