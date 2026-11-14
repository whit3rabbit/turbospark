# OC-12: ACP adapter for editor embedding

Finding: Section 8, "ACP (Agent Client Protocol) support - Missing in
Swift" in [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[ACP docs](https://github.com/anomalyco/opencode/blob/dev/packages/web/src/content/docs/acp.mdx)
and
[session adapter](https://github.com/anomalyco/opencode/blob/dev/packages/cli/src/acp/sessions.ts).

* Author: whit3rabbit
* Status: Implemented, with no editor run (protocol core, local engine with file tools, real-model streaming and tool calls verified with a scripted client; Zed or any real editor is NOT run, see Implementation Record and Remaining)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Expose the local engine to editors over the Agent Client Protocol: a
JSON-RPC-over-stdio adapter so editors like Zed can start and drive
TurboSpark sessions against locally installed checkpoints. Everything
stays on device: the adapter hosts the local engine directly, and the
permission story is explicit rather than inherited.

## The Problem

No ACP adapter exists in the Swift tree or the FFI. The outward-facing
surfaces are the OpenAI-compatible HTTP server
([ServerConnectionView.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerConnectionView.swift)),
which carries no sessions, tools, or permissions, and inbound reads of
other agents' MCP configs
([ExternalAgentMcpReader.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Tools/MCP/ExternalAgentMcpReader.swift)).
An editor cannot drive a TurboSpark agent session today. OpenCode can
be embedded in third-party editors because its agent core is
frontend-agnostic; the same protocol surface is the portable way for
TurboSpark to be drivable from an editor without any network service.

## Goals

* An ACP adapter process: JSON-RPC over stdio implementing the core
  session surface: initialize, session/new, session/prompt, session/
  update (streaming deltas and tool-call events), fs read/write
  bridged to the project tool surface, session/request_permission, and
  session/request elicitation for questions.
* Fully local: stdio transport only, launched by the editor as a
  child process of the user's editor, no listening socket, no pairing,
  no cloud anything.
* Sessions persist under the app support directory as JSON records in
  the established `AppJSONStore` quarantine style, separate from the
  app's own chats in v1, with the separation documented in the
  surface's docs.
* Permissions: tool calls inside ACP sessions run through the same
  risk classification as app sessions. Permission requests forward to
  the editor through the ACP permission request method when the client
  supports it; when it does not, the default is deny-with-message so a
  headless client cannot silently approve tool use.
* Streaming: token deltas and tool events flow as ACP session updates;
  abort propagates both directions.

## Non-Goals

* No sharing of the app's own chat archives with ACP sessions in v1
  (the adapter's sessions are its own records; a later draft can unify
  them).
* No remote transport, no websocket, no daemon-pairing variant.
* No SSH or remote workspace driving (excluded category).
* No GUI: the adapter is headless; the app's UI is not involved.

## Open Questions

* **Where the adapter lives**: (a) a new Rust crate beside
  `crates/server`, hosting the engine directly over the existing
  runtime interfaces, the OpenCode shape and the lean for v1; (b) a
  Swift helper mode of the app. Rust keeps the app unlaunched, matches
  the runtime ownership of sessions, and reuses the server crate's
  hardening; Swift would reuse the permission engine but forces the
  full app process. Lean (a).
* **Permission default without client support**: deny-with-message
  (lean) versus a read-only allowlist (file reads allowed, everything
  else denied).
* **Tool surface in v1**: file tools only (read/write/edit/list), or
  also shell. Lean: file tools only; shell through an editor channel
  needs its own threat review.

## Critical User Journeys

### CUJ 1: editor drives a local model

* **Given**: an editor configured with the adapter command.
* **When**: the user prompts from the editor.
* **Then**: a session starts on a locally installed checkpoint, streams
  into the editor, and persists as an adapter session record.

### CUJ 2: permission forwarding

* **Given**: a session where the agent attempts a file write.
* **When**: the editor supports ACP permission requests.
* **Then**: the request surfaces in the editor; the editor's decision
  is recorded and reused for the session like an app Always-allow.

### CUJ 3: headless client

* **Given**: a client that never answers permission requests.
* **When**: the agent attempts a write.
* **Then**: the call is denied with a message naming the permission;
  nothing executes silently.

## Technical Plan (Architecture)

```text
Editor (ACP client)
  | stdio (JSON-RPC)
  v
ACP adapter (crates/acp or crates/server/src/acp/, per Open Question)
  session registry, update fan-out, permission forwarding
        |
        v
crates/runtime engine (local checkpoints, Metal/MLX/ANE)
  file tools against the declared workspace root
```

The workspace root is declared at session/new and every path tool
resolves inside it with the existing path-safety rules (real-path
resolution, refusal on escape). Sessions are separate from the app's
chats by design in v1; the docs state this so no user expects the
editor transcript in the app.

## Non-Functional Requirements

* **Performance**: first token latency inherits the engine's; the
  adapter adds serialization only.
* **Safety**: deny-by-default on permission forwarding gaps; workspace
  root confinement; no listening sockets.
* **Localization**: adapter strings are log and error lines only (not
  localized by policy); editor-facing protocol strings follow the ACP
  spec's English shapes.
* **Accessibility**: not applicable (headless surface); the docs note
  that UI is the editor's responsibility.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `crates/acp/` (Create crate)
   * **Rationale**: the adapter: JSON-RPC stdio loop, session registry,
     typed method handlers, permission forwarding, abort propagation.
2. `Cargo.toml` (Modify, workspace members)
   * **Rationale**: add the crate.
3. `crates/acp/src/protocol.rs` (Create)
   * **Rationale**: the ACP method and type subset, with serde types
     and round-trip tests against captured fixtures.
4. `crates/acp/src/sessions.rs` (Create)
   * **Rationale**: session records, persistence, workspace-root
     validation.
5. `crates/ffi/` or `crates/runtime/` (Modify, interface only)
   * **Rationale**: whatever session-entry seam the adapter needs that
     the runtime does not already expose; interface-only changes, no
     engine changes.
6. `crates/cli/src/main.rs` (Modify)
   * **Rationale**: the launch subcommand (`turbospark acp`) so editors
     have a stable entry point.
7. `docs/ACP.md` (Create)
   * **Rationale**: how to point an editor at the adapter, session
     separation from app chats, permission defaults, limitations.
8. `CHANGELOG.md` (Modify)
   * **Rationale**: entry under Added.
9. `crates/acp/tests/` (Create)
   * **Rationale**: protocol fixtures, permission-forwarding matrix,
     deny-by-default proof, workspace escape refusal. Mutation check:
     remove the deny default and assert the headless test fails.

### Verification Plan

* **Build**: `cargo build --workspace`.
* **Tests**: `cargo test -p turbospark-acp` plus the workspace suite.
* **Cross-checks**: `cargo fmt --check`, `cargo clippy --workspace
  --tests`.
* **Manual**: drive the adapter from a scripted stdio client first;
  then from Zed with a small project, confirming streaming, permission
  forwarding, and abort. Per the repo's verification rules, the manual
  ACP run is a real-engine check on this Mac, not a release gate.

--------------------------------------------------------------------------------

## Implementation Record

Implemented 2026-11-14 as the new crate `crates/acp` (package
`turbospark-acp`, lib `turbospark_acp`, binary `turbospark-acp`). **Status
Implemented, no editor run**: the stdio JSON-RPC core, session registry, file
tools, permission gate and cancellation (first pass), the local engine in
`crates/cli` (second pass), a scripted independent client (third pass) and
tool wiring for local models (fourth pass) are done. A real model has streamed,
been cancelled and called tools through the adapter, driven by a scripted
client. No editor (Zed or any other ACP client) has been run against the
adapter. Nothing here is an editor interop claim.

### Open Question resolutions

* **Where the adapter lives**: (a), a new Rust crate beside `crates/server`,
  per the draft's lean. It depends only on `serde` and `serde_json`; it does
  not link `turbospark-server`, `runtime` or any GPU crate, so it builds and
  tests off-device. The EGM review's B1 (the Rust crate cannot reuse
  `AppToolPermissionEngine`) stands and is accepted: this is a second, narrower
  agent surface, not the Swift loop. "Same risk classification" therefore means
  this ladder: reads (`read_file`, `list_dir`) run unprompted inside the
  workspace root; writes (`write_file`, `edit_file`) require editor approval;
  nothing else exists. It is not the app's full ladder and does not read the
  app's stored Always-allow decisions.
* **Permission default without client support**: deny-with-message (lean).
  ACP has no capability flag for `session/request_permission` (every client is
  expected to answer it), so "unsupported" is detected by outcome, not at
  `initialize`: an error reply (for example method not found), silence past
  the timeout (60 s, `AcpConfig::permission_timeout`), a `cancelled` outcome,
  an option id the adapter never offered, a malformed reply, a cancelled turn,
  or a closed connection are all refusals. There is no allowlist and no
  auto-approve flag.
* **Tool surface in v1**: file tools only (read, list, write, edit), per the
  lean. No shell.

### Decisions on the review's findings

* **Protocol subset (M4)**: ACP protocol version 1. Methods: `initialize`,
  `session/new`, `session/prompt`, `session/cancel`, `session/update`,
  `session/request_permission`, `fs/read_text_file`, `fs/write_text_file`
  (table in `protocol.rs`). "session/request elicitation" from the Goals is
  not an ACP method and is not implemented. `authenticate` is not offered
  (no login). Prompt content: `text` and `resource_link` only; the
  `initialize` result advertises no image, audio or embedded-context support.
  The fixtures are hand-written from the spec, not captured from an editor.
* **File access path (M4)**: when the client advertises
  `fs.readTextFile` / `fs.writeTextFile`, reads and writes go through the
  editor (unsaved buffers win); otherwise the disk. An editor failure fails the
  tool, it never falls back to disk. `list_dir` is always disk, inside the root.
* **Workspace root (M2)**: must be an existing absolute directory, not `/`,
  not the home directory itself, and inside `--allow-root` when given. Paths
  are normalized lexically, then the longest existing prefix is canonicalized
  (symlinks, `/var` vs `/private/var`) and the result must stay under the
  root. Escapes are refused before any permission prompt. A symlink swap
  between check and use is not closed (documented, same-user threat model).
* **Persistence (M5)**: records are written (JSON, atomic tmp+rename, 0600 in a
  0700 directory, `$TURBOSPARK_STATE_DIR/acp-sessions`, else
  `$TURBOSPARK_HOME/acp-sessions`, else `~/.turbospark/acp-sessions`) but not
  read back: `session/load` is not advertised, so the `AppJSONStore`
  quarantine rule has nothing to apply to yet. `--no-persist` keeps sessions in
  memory only. Separate from the app's chats by design; `docs/ACP.md` says so.
* **Concurrency and stdio (M3, M9)**: one engine per adapter process; two
  editor windows are two processes. stdout is protocol only, diagnostics go to
  stderr and never include bodies. stdin EOF cancels running turns and the
  process exits. Frames are capped at 1 MiB (oversized lines are discarded and
  answered with an error, the stream stays aligned); `read_file` output is
  capped at 256 KiB, edits at 1 MiB files, listings at 1000 entries, 64 tool
  calls per turn, 32 sessions. The "adapter adds serialization only" latency
  claim was not measured (no engine is wired).
* **Cargo**: publishable (`publish.workspace = true`) and listed in
  `publish-crates.yml` ahead of `turbospark-cli`; MSRV 1.82 and edition 2021
  inherited from the workspace.

### What changed

* `crates/acp/` (new): `framing.rs` (bounded line reader), `protocol.rs`
  (JSON-RPC envelope and ACP subset), `paths.rs` (root validation and
  containment), `tools.rs` (file tools and risk classification), `engine.rs`
  (the `Engine` / `TurnHost` seam and `UnwiredEngine`), `sessions.rs`
  (registry state and records), `server.rs` (dispatch, permission forwarding,
  cancellation, editor fs bridge), `main.rs` (binary), tests under
  `crates/acp/tests/`.
* Root `Cargo.toml`: member line; `Cargo.lock`: the `turbospark-acp` entry only.
* `docs/ACP.md` (new).

### Verification run

* `cargo test -p turbospark-acp`: 28 unit, 2 process (real child over pipes:
  protocol-only stdout, clean EOF exit, bad flag exits 2), 15 permission/path,
  13 protocol/session tests, repeated three times without flakes.
  `cargo clippy -p turbospark-acp --tests` clean. Built with a private
  `CARGO_TARGET_DIR` because other sessions held the shared target lock.
* Mutation checks (each applied, test failure observed, then restored):
  timeout treated as approval fails the headless-client test; an unrecognized
  option id treated as approval fails the non-approval matrix; containment
  check removed fails the escape tests; frame bound removed fails the
  oversized-frame tests; `cancelled` mapping removed fails the cancel test;
  shutdown not cancelling or closing fails the EOF test. One mutation
  survived and is equivalent: removing only the per-session cancel in
  `begin_shutdown` changes nothing because the `closed` flag already stops
  turns and pending requests.
* A test-harness race found and fixed along the way: the cancel flag is reset
  on the reader thread at prompt dispatch, not in the worker, so a cancel
  arriving right behind a prompt is not erased.

### Local engine wiring (second pass, 2026-11-14)

Decision taken (reversible): the real engine lives in `crates/cli`, not in
`crates/acp`, so `crates/acp` stays `serde` + `serde_json` only.

* `turbospark acp --model M [--allow-root D]... [--state-dir D | --no-persist]`
  runs the new peer binary `turbospark-acp-agent` (`crates/cli/src/bin/acp.rs`;
  not named `turbospark-acp`, which is `crates/acp`'s protocol-only binary).
  New files: `acp_engine.rs` (portable worker-thread engine over a
  `ChatBackend` trait), `acp_real.rs` (macOS backend over
  `generate::Session`), `acp_lock.rs` (refusal policy). Shared-file edits:
  `Cargo.toml` (bin plus the `turbospark-acp` path dep), `bin/turbospark.rs`
  (the `Acp` passthrough and one help row), `Cargo.lock`.
* The model session is opened with `generate::open_session` and every turn is
  rendered through the checkpoint's chat template and window-fit, as in
  `--chat`. Decode uses the runtime's cancellable loops (plain and chunked), so
  cancel is honored between tokens. Speculation is off. Reasoning is dropped.
  `stream_turn` is not reused because it writes to stdout, the protocol channel.
* One worker thread owns the model (no `unsafe impl Send`); it loads lazily on
  the first prompt and is joined and dropped at stdin EOF.
* Refusal policy for a held model: advisory `flock` per model directory plus a
  live-daemon check; the app is not covered (see `docs/ACP.md`).
* **Tools: wired in the fourth pass (below).** This pass was text-only.
* Tests: 5 fake-backend engine tests (stream order, load once, cancel between
  tokens, release on drop, load failure retried, panic isolated), 4 lock/daemon
  tests (real `flock` conflicts), process tests (flag errors, an unloadable
  model fails the turn and EOF exits 0, the front door lists `acp`), and one
  `#[ignore]`d real-model smoke test. Mutation check: dropping the cancel
  propagation fails the cancel test.
* **Real-model smoke test: run (2026-11-14).** Release build, Apple M4 Max,
  installed `dsv4pro-q35-9b-heretic-mlx4` (the only complete installed chat
  model). `cargo test --release -p turbospark-cli --test acp_agent -- --ignored`
  passed in 3.3 s: the model loaded inside the first prompt, the reply
  streamed as `agent_message_chunk`, a mid-answer `session/cancel` returned
  `cancelled`, the session answered again, stdin EOF exited 0. A second,
  independent-client run (item B) measured the cancel landing 145 ms after the
  `session/cancel` and the whole first turn (load plus first chunk) at 0.78 s
  with a warm page cache. **The host was not quiet** (1-minute load 10.7 to
  11.3, other sessions' test jobs; a quiet host never came within the polling
  window), so those numbers are functional observations, not benchmarks, and
  no first-token latency is claimed. Text only at that point: no tool call
  reached the model.

### Scripted-client interop (third pass, 2026-11-14)

`crates/acp/tests/scripted/client.rs` is a hand-written ACP client that does
not use `turbospark_acp` types. `crates/acp/tests/scripted_client.rs` (9 tests)
drives the real `serve` loop as a child process with a scripted fake engine
(`crates/acp/examples/scripted_agent.rs`): ordered streaming before the
response, cancel mid-stream and a usable session afterwards, a permission
request of the spec shape, deny on silence, reject, `cancelled` and
method-not-found, allow-once as the only way a write lands (and not sticky),
reads without a prompt, JSON-RPC errors with an aligned stream, refused cwds,
EOF mid-turn exits 0. `crates/cli/tests/acp_scripted_real.rs` (`#[ignore]`)
runs the same client against the real agent and the installed 9B: passed
(streamed, cancel landed in 35 to 145 ms across two runs, EOF exit 53 ms).
Mutation checks, each applied, failure observed, then restored: timeout
approves (fails the silent-client test), any non-explicit option approves,
`cancelled` outcome approves, an editor error approves (each fails the
reject/cancel/error test). A first version of the suite shared one workspace
across parallel tests and flaked; each test now gets its own directory.
**Scripted client only: no editor was run.**

### Tool wiring (fourth pass, 2026-11-14)

* `acp_engine.rs`: `LocalEngine::run_turn` is now a bounded round loop.
  `ChatBackend::generate` takes the working conversation (`ChatMsg`: user,
  assistant with calls, tool result) and reports `ToolReq`s. Each call goes
  through `TurnHost::call_tool` (the adapter's gate, unchanged), the result is
  clipped to 16 KiB characters for the model and appended, and the model runs
  again. Caps: `MAX_TOOL_ROUNDS = 8`, `MAX_TOOL_CALLS = 64` (the adapter's own
  `max_tool_calls_per_turn` still applies if lower); either ends the turn as
  `max_turn_requests`. Ids are engine-assigned. Cancel is checked between
  tokens, between calls and between rounds. `tool_schemas()` holds the four
  schemas with the argument names `crates/acp/src/tools.rs` validates
  (`old_text`/`new_text` for `edit_file`).
* `acp_real.rs`: when tools are offered the prompt is rendered with
  `encode_generic_tool_chat`, exactly as `crates/server/src/handler/plan.rs`
  does, and the `TurnSplitter` allowlist is the four names, so parsed calls
  arrive as `TurnEvent::ToolCall`. If the template cannot render tools the
  backend drops to the plain template for the process and says so on stderr.
  Window-fit trimming that leaves a leading tool result removes it (a template
  rejects an orphan). Nothing in the backend executes a tool.
* Not changed: the permission gate, containment, size bounds and the 60 s
  silence-means-deny rule. No flag approves anything.
* Tests: 8 new fake-backend engine tests (call runs and result reaches the
  model, round cap, per-reply call cap and unique ids, cancel between calls,
  clipping, stored tool history not replayed, schema names) and 3 through the
  real `serve` loop in-process (silent client denies the model's write and the
  model is told, only an explicit allow writes, reject leaves the disk alone;
  an escaping `../` read is refused with no prompt). Real model, `#[ignore]`d
  (`scripted_client_real_model_tool_calls`), installed 9B: the model called
  `read_file`, the result came back and it answered with the file's word; asked
  to write, it reached the gate once, the scripted client stayed silent for
  the 60 s timeout, nothing was written and the model told the user it could
  not get approval. One run; the host load was 9 to 10.
* Mutation checks (applied, failure observed, restored; the first three re-run
  after this change): timeout approves, reject approves, containment check
  removed, per-reply call cap removed, round cap raised, cancel between calls
  removed. All six fail a test.
* Verified by: `cargo test -p turbospark-acp` (all green),
  `cargo test --release -p turbospark-cli --bin turbospark-acp-agent`
  (20 passed), clippy on both crates clean of new warnings.

### Remaining (why this is not "editor-verified")

1. **Real-model run: done functionally** (see the bullet above); a timing
   run on a quiet host (load under 10) is still open.
2. **Tools for a local model: done** (fourth pass). Open: only one model
   (the 9B) has been seen to emit a parseable call; other families are
   unmeasured, and tool rounds are not persisted across turns.
3. **Editor interop**: a scripted client written independently from the spec
   has been run (below); a real editor (Zed) has not (streaming, permission
   forwarding, abort). Not run.
4. **Latency and memory**: first-token overhead is unmeasured.
5. **App as a model holder**: the app takes no lock, so the refusal policy does
   not cover it.
6. **CI and release: done.** `-p turbospark-acp` is in the Linux portable
   check in `ci.yml` (build and test already use `--workspace`), `publish.workspace
   = true` replaces `publish = false`, and `publish-crates.yml` lists
   `turbospark-acp` (input choices and `ALL_CRATES`) ahead of `turbospark-cli`.
   `docs/RELEASE.md` publish order updated. Not verified: a real publish
   (`turbospark-cli` cannot dry-run until acp is on crates.io). The release and
   app-bundle scripts enumerate bins explicitly (`bins=(...)` in `release.yml`,
   `scripts/make-app-bundle.sh`, `scripts/make-dmg.sh`) and do NOT ship
   `turbospark-acp-agent`: omission breaks nothing today (no editor has been run
   against it), so it is left out until the real-model and editor-interop items
   above pass. Bundling it is a deliberate follow-up in those three lists.
7. **CHANGELOG entry** (draft item 8): not written per the task's instruction.
8. **`session/load`** and a record loader with quarantine, if resumption is
   wanted. Also `--max-context`, `--max-new` and sampling flags on `acp`.

--------------------------------------------------------------------------------

*Drafted with EGM*
