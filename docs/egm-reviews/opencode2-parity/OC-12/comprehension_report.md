# Stage 6: Comprehension report (OC-12)

Target: docs/designs/opencode2-parity/OC-12-acp-adapter.md (whole file). Reviewer: egm-comprehension (CO). Document read alone, no extra context.

## What this proposes

Add a new headless program, launched as `turbospark acp`, that speaks the Agent Client Protocol (ACP, a JSON-RPC-over-stdio protocol) so an editor such as Zed can start it as a child process. The editor can then create sessions, send prompts, receive streamed tokens and tool-call events, and have file tools run against a workspace folder, all using local model checkpoints with no network. Tool permission requests are forwarded to the editor. If the editor does not answer or does not support the request, the call is denied with a message. Sessions are saved as JSON records, kept separate from the app's own chats in v1. The plan is a new Rust crate (`crates/acp`), a CLI subcommand, docs, and tests.

## How the existing system works, per this document

There is no ACP support today. The only outward-facing surfaces are an OpenAI-compatible HTTP server, which has no sessions, tools or permissions, and a reader of other agents' MCP configs. A Rust runtime crate (`crates/runtime`) runs local checkpoints on Metal/MLX/ANE. A `crates/server` crate exists and has "hardening". The Swift app has a permission engine with "risk classification", an "Always-allow" option, path-safety rules, and a JSON store called `AppJSONStore` with a quarantine style. How these pieces connect to the new adapter is only partly explained.

## Findings

CO-1 Blocker (lines 18-21, 51-55, 74-76, 113-119). The permission and risk-classification engine is said to be reused ("same risk classification as app sessions") but is described as Swift, and the recommended design is a Rust crate that never launches the app. The document never says whether the engine is reimplemented in Rust, exposed over FFI, or ported. "Reused like an app Always-allow" (CUJ 2) is likewise unexplained.

CO-2 Major (lines 70-76, 113, 143, 174). Where the adapter lives is an open question, yet the plan commits to `crates/acp/` and package `turbospark-acp`. The diagram says "crates/acp or crates/server/src/acp/". Line 76 repeats "Lean (a)" from lines 71-72.

CO-3 Major (lines 47-49, 121-125). Sessions persist with `AppJSONStore` (Swift) from a Rust crate. No Rust equivalent, definition of "quarantine style", or location of the app support directory is given.

CO-4 Major (lines 39-43, 148). Method list unclear: stray line break in "session/ update"; "session/request elicitation" is not a clearly named ACP method; "fs read/write bridged to the project tool surface" does not say whether editor fs or local disk is used; no ACP spec version is pinned.

CO-5 Major (lines 39-43, 56-57, CUJ 1). Persisted sessions are mentioned but no list, load or resume method is in scope, so persistence has no stated purpose. How the editor selects the checkpoint is not explained.

CO-6 Major (lines 54-55, 100-105). A client that does not support permission requests and a client that never answers are treated as one case. A supporting client that never answers would hang; no timeout is stated. "Message naming the permission" is vague.

CO-7 Major (lines 80-82, 51, 119, 123). v1 tool surface is open, but CUJ 2 and the architecture assume file writes. "File tools (read/write/edit/list)" are not defined as existing tools; "existing path-safety rules" are not described or linked.

CO-8 Major (lines 154-157). "Whatever session-entry seam the adapter needs" is hand-waving. Unclear whether the runtime has a session concept (line 76 says it owns sessions) or whether `crates/ffi` or `crates/runtime` changes.

CO-9 Major (lines 18-22, 32-36, 3-4). "Stays on device" and "permission story is explicit rather than inherited" are vague; the OpenCode "frontend-agnostic" point is rationale, not a requirement. "Missing in Swift" in the finding reference is unexplained for a Rust proposal.

CO-10 Nit (lines 44-46, 64). "Launched by the editor as a child process of the user's editor" is redundant. "Pairing" and "daemon-pairing" are undefined.

CO-11 Nit (lines 39-43, 56-57). "Abort propagates both directions" does not name the messages or the reverse-direction behavior.

CO-12 Nit (lines 3-8, 18, 28-31, 65, 117, 133-137, 168, 179-180). Undefined or unlinked terms: "excluded category", Metal/MLX/ANE, "the repo's verification rules", "mutation check" procedure, "localized by policy". Relative link style to be checked.

CO-13 Nit (lines 150, 171-180). Fixture source for "captured fixtures" not named (real Zed run or OpenCode). Manual Zed check lacks editor configuration (deferred to docs/ACP.md).
