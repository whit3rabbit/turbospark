---
title: "Rust API: Engine and Frontends"
description: "Public API of the generation engine and the user-facing binaries (turbospark-runtime, turbospark-server, turbospark-cli, turbospark-catalog, turbospark-bench, turbospark-ffi), generated from cargo doc and source doc comments."
diataxisType: "reference"
---

<!-- generated: rust lane, signal: Cargo.toml -->

This page covers the decode engine and everything a user or host app
touches: the prefill+decode loop, the HTTP server, the CLI binaries, the
model catalog and install driver, the benchmark harness, and the C ABI.
Signatures are read from source; doc text is quoted or condensed from
`///` comments. The full rustdoc inventory is under `target/doc/` after
`cargo doc --no-deps`.

## turbospark-runtime (`crates/runtime`)

`#![forbid(unsafe_code)]`. The raw-completion prefill+decode loop, wiring
the `selection` sampler, the `tokenizer` crate's streaming detokenizer and
stop matcher, and a pluggable `LogitProducer` into one token generation
loop. Ported from `Runtime/Generation/RawCompletion.swift` and
`LogitProducer.swift`. 139/149 public items documented.

The crate root re-exports: `config`, `model_io`, `error`, `families`,
`power`, `producer`, `raw_completion`, `real_forward`,
`speculation_policy`, `speculative`, `steering`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `LogitProducer` | trait | Produces next-token logits for the generation loop. |
| `ChunkedPrefillRunner` | trait | A `LogitProducer` that can also process a whole prefill chunk in one call, writing the logits state for the position immediately after the chunk. |
| `SpeculativeProducer` | trait | A `LogitProducer` that can also draft tokens ahead of itself and verify a block of them in one pass; measured 1.44x at block 2 on the qwen dense install. |
| `ScriptedLogitProducer` | struct | A fixed sequence of pre-scripted logit vectors, replayed in order; errors once exhausted. Mirrors the Swift validation fixture. |
| `RealForwardRunner` | struct | The real GPU-forward-pass runner (`real_forward.rs`, macOS only); opens a `.gturbo` install and drives the per-family decode flows. Undocumented struct in source. |
| `run_raw_completion` | fn | `pub fn run_raw_completion(producer: &mut dyn LogitProducer, tokenizer: &MfTokenizer, prompt_ids: &[TokenId], config: &GenerationConfig, max_context: u32, vocab_size: usize, on_progress: impl FnMut(RawDecodeProgress)) -> Result<RawDecodeResult, RuntimeError>`. Feeds every prompt token to `producer` one at a time, then decodes. |
| `StopReason` | enum | Reason why generation stopped. |
| `RawDecodeProgress` | enum | Progress event emitted during prefill and decoding. |
| `is_pure_greedy` | fn | A pure-greedy config always agrees with the single highest-scoring candidate, independent of the seed. |
| `resolve_speculation` | fn | `pub fn resolve_speculation(requested: Speculation, drafter: SpeculativeDrafter, engine_blocker: Option<String>, deterministic: bool) -> Result<SpeculationPlan, String>`. Decides speculation from the request and what the install turned out to be; pure, so the hard-fail/warn split is testable without a 14 GB model. |
| `dflash_speculation_blocker` | fn | Why DFlash2 speculation cannot run on this runner, or `None` if it can; the CLI's hard-fail message for `--speculative-drafter dflash`. |
| `dflash_draft_block` | fn | One DFlash2 round: context-write the committed prefix, run the `block + 1` query rows through the drafter, walk the selector, and return `block` proposals. |

## turbospark-server (`crates/server`)

OpenAI- and Anthropic-compatible generation server on loopback:
request/response envelopes, SSE streaming framing, and the axum router,
wired to `turbospark-runtime`'s raw-completion loop. Two backends
implement the `ChatModel` interface: `ScriptedChatModel` (portable, fixed
logit sequence) and `RealChatModel` (macOS only, a real `RealForwardRunner`
forward pass against a `.gturbo` install). Model dialect auto-selection is
the tokenizer's job here, not the server's. 58/72 public items documented.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `build_router` | fn | `pub fn build_router(state: impl Into<ServerState>) -> Router`. Build the axum router with no options: no auth; every existing caller keeps the exact behavior it had before. |
| `build_router_with_options` | fn | Build the router with `RouterOptions` beyond the shared `ServerState`. |
| `RouterOptions` | struct | Options `build_router_with_options` takes beyond the shared `ServerState`; a struct so a future option has somewhere to land without an ABI-style break. |
| `ServerState` | struct | What axum's `State` carries: which models are attached, and who is watching. |
| `ScriptedChatModel` | struct | Always replays the same scripted logit sequence, regardless of the prompt; `steps` must include one entry per prefill token plus one per decode step. |
| `RealChatModel` | struct | The macOS real-install backend (`real_model.rs`). Undocumented struct in source. |
| `ModelArgs` | struct | Parsed command line arguments for running `turbospark-server` in `--model` mode. |
| `parse_model_args` | fn | Parses the `--model` mode's flags; returns `Ok(None)` when the first argument is not an option, leaving the caller on the legacy positional path. |
| `short_circuit` | fn | Text a `--help` or `--version` token short-circuits to, whichever is reached first in a left-to-right scan. |
| `BindMode` | enum | Interface binding and Tailscale host resolution; resolution fails rather than widening. |
| `tailnet_host` | fn | Accepts exactly one Tailscale IPv4 address; empty, ambiguous, IPv6-only, malformed, and off-range output all fail, none fall back. |
| `completions` | fn | `POST /v1/completions`. The legacy raw-prompt endpoint: no chat template, `tokenizer.encode` is the one call site in this crate. |

## turbospark-cli (`crates/cli`)

`turbospark-check`: the process entry point for the deterministic front
half of the port. Reads `argv`, hands the tokens to
`turbospark-invocation`, applies its pure exit-status/stream-routing
decisions. `turbospark-model` is the second binary: find, inspect, probe,
install and remove models. 15/17 public items documented.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `Error` | enum | The two failure kinds for `turbospark-model`, which get different exit codes: a malformed invocation is the caller's mistake, an operational failure is not. |
| `Options` | struct | Parsed options, flat: every command reads the ones it needs and `reject_unused` refuses the rest, so a flag on the wrong command is an error. |
| `list` | fn | Curated rows, marking which are installed. |
| `info` | fn | One row in full. |
| `probe` | fn | Header-only verdict for an arbitrary repository. |
| `path` | fn | Print an install directory, or fail so `$(...)` does not expand to nothing and silently produce a `--model ''`. |
| `remove` | fn | Delete an install directory. |
| `pull` | fn | `pub fn pull(catalog: &Catalog, store: &Store, client: &Client, positionals: &[String], options: &Options) -> Result<(), Error>`. Install a curated row, or any repository the probe accepts. |
| `recommend` | fn | `pub fn recommend(catalog: &Catalog, client: &Client, options: &Options) -> Result<(), Error>`. What this machine should run, ranked; three sources of shape, cheapest first. |

## turbospark-catalog (`crates/catalog`)

The model catalog, the Hugging Face probe, and the install driver behind
`turbospark-model`. Three layers answering three questions: `Catalog`
(what has been run here, a curated table whose rows name repositories
streamed and generated on real hardware), `probe` (what could be run here,
reading a header and deciding architecture, block types, expert
granularity, tokenizer sidecars; costs KB and seconds), and `install` (the
streaming install driver). 98/99 public items documented.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `CatalogEntry` | struct | One curated model. Not `Eq`: `Measured` carries `f64` throughput readings. |
| `Catalog` | struct | The resolved table: curated rows with any user rows merged over them. |
| `load` | fn | The curated table with `$TURBOSPARK_HOME/models.json` merged over it, if that file exists; a missing override is not an error. |
| `get` / `entries` / `len` / `is_empty` | fn | Row lookup and iteration over the resolved table. |
| `is_user_row` | fn | Whether `alias` came from the user's override rather than the embedded table. |
| `validate` | fn | Structural checks that hold for every row, applied at load so a malformed user override fails at the point of reading. |
| `parse` | fn | `pub fn parse(text: &str) -> Result<Self, String>` (in `hf.rs`). Parse `owner/name` or `owner/name@revision`; an omitted revision is `main`. |
| `probe` | fn | `pub fn probe(client: &Client, repo: &RepoRef, want_file: Option<&str>, sidecar_repo: Option<&RepoRef>) -> Result<ProbeReport, String>`. Probe `repo`, optionally forcing which `.gguf` file to look at. |
| `InstallPlan` | struct | Everything needed to install one model, whether it came from the catalog or from a `--repo` probe. |
| `install` | fn | `pub fn install(plan: &InstallPlan, dir: &Path, client: &Client, progress: impl FnMut(&str)) -> Result<Installed, String>`. Install `plan` into `dir`; `progress` receives both this driver's stage lines and the repack walk's own. |
| `Recommendation` | struct | One ranked candidate from the `recommend` layer. |
| `Store` | struct | The `~/.turbospark` tree. |

## turbospark-bench (`crates/bench`)

Library surface of the bench harness, so the memory-oracle integration
test can drive the exact same real-install flow the `turbospark-bench`
binary runs (`--model` mode): frozen community-protocol prompts, the
Swift-parity `phys_footprint` sampler, and per-case results. 45/46 public
items documented.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `AppMemorySampler` | struct | Current-process physical footprint sampler with peak tracking, the Swift `AppMemorySampler` contract. |
| `sample` | fn | One `phys_footprint` sample; updates the peak. `None` when `task_info` fails. |
| `peak_bytes` | fn | The peak memory footprint in bytes recorded since creation or last reset. |
| `reset_peak` | fn | Resets the recorded peak memory footprint to `None`. |
| `chip_brand_string` | fn | `sysctl machdep.cpu.brand_string`, e.g. `"Apple M5 Pro"`; used by the memory oracle to pick the matching Swift baseline row. |
| `ProtocolCase` | struct | Test case specification for the community benchmark protocol. |
| `run_model_mode` | fn | The real-install protocol run; one shared footprint sampler across warmups and measured runs. |

## turbospark-ffi (`crates/ffi`)

C ABI over the inference engine, for a native GUI host. The contract in
four sentences, quoted from the crate docs: every fallible call returns 0
on success and a non-zero `abi` code otherwise, with a message retrievable
through `ts_last_error` on the same thread; a `const char *` argument is
borrowed for the duration of the call while a `char **` out-parameter is
an allocation the caller returns through `ts_string_free`; options and
results are JSON, so adding a knob is never an ABI break; and a session is
single-threaded except for `ts_session_cancel`, which is safe from any
thread and never blocks. 87/88 public items documented. The hand-written
`turbospark.h` side of this boundary is covered by the cpp lane.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `TsSession` | type | `pub type TsSession = Session`. The opaque handle a caller holds; `TsSession *` in C. |
| `TsServer` | type | The server-side opaque handle type. |
| `TsEventCallback` / `TsInstallCallback` | type | Callback function-pointer types handed across the boundary. |
| `ts_session_open` | fn | `pub unsafe extern "C" fn ts_session_open(model_dir: *const c_char, options_json: *const c_char, out: *mut *mut TsSession) -> c_int`. Opens `model_dir` (a path, or a `turbospark-model` alias) and writes a session handle to `out`; `options_json` may be null or `{}`. |
| `ts_session_close` | fn | Closes a session and frees it; null is a no-op; must not be called while a generation is in flight on another thread. |
| `ts_session_cancel` | fn | Asks the in-flight generation to stop. Safe from any thread and never blocks: the flag it raises lives outside the session's mutex. |
| `ts_server_start` | fn | `pub unsafe extern "C" fn ts_server_start(session: *const TsSession, options_json: *const c_char, out: *mut *mut TsServer) -> c_int`. Starts an in-process HTTP server and writes a handle to `out`; `session` may be null, meaning start with no model attached. |
| `ts_last_error` | fn | Copies this thread's last error message into `buf`, returning the message's own length in bytes excluding the NUL; pass a null `buf` to ask for the length. |
| `ts_string_free` | fn | Frees a string this library handed out through a `char **`. |
| `guard` | fn | Runs `body` with panics caught, mapping one to the `TS_ERR_PANIC` code; every `extern "C"` function in this crate is a call to this and nothing else. |

## Coverage for this tier

Per-crate public-item doc coverage (excluding `pub use` re-exports, parsed
from `crates/*/src`): runtime 139/149, server 58/72, cli 15/17, catalog
98/99, bench 45/46, ffi 87/88. The server crate carries the largest
undocumented share of the tier, mostly handler internals.
