# OC-12: ACP adapter for editor embedding

Finding: Section 8, "ACP (Agent Client Protocol) support - Missing in
Swift" in [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[ACP docs](https://github.com/anomalyco/opencode/blob/dev/packages/web/src/content/docs/acp.mdx)
and
[session adapter](https://github.com/anomalyco/opencode/blob/dev/packages/cli/src/acp/sessions.ts).

* Author: whit3rabbit
* Status: Draft
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

*Drafted with EGM*
