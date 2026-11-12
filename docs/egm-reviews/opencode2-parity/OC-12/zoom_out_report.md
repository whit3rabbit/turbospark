# Stage 5: Initial zoom out report (OC-12)

Target: docs/designs/opencode2-parity/OC-12-acp-adapter.md (full file, 184 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out reviewer (ZO)

Verdict: the root problem is real, but the architecture rests on reuse claims the code does not support. 2 Blocker, 3 Major, 3 Nit.

ZO-B1 (lines 51-55, 70-76, 107-125, 143-157). The adapter is a Rust crate, but the permission ladder (`AppToolPermissionEngine.swift`) and `AppJSONStore` are Swift. No Rust agent session or tool loop exists in `crates/`. "Same risk classification", file tools, session registry and path-safety rules would all be new Rust reimplementations, forking the main safety invariant. Open Question (a) treats permission-engine reuse as a cost of Swift; for a safety path it is the stronger argument. Choose Swift, or name which Rust components already own tools and permissions. If none, the "large" sizing is understated.

ZO-B2 (lines 154-157, 147-150). Item 5 ("whatever session-entry seam ... interface-only changes") hides the main work: sessions, prompt templating, tool-call parsing, generation loop. Name the existing entry point (for example the `crates/ffi` generate path) and what handles tool calls.

ZO-M1 (lines 68-82, 113). Open Questions decide the architecture but later sections assume answer (a). Commit with reuse evidence or stop prescribing crate paths.

ZO-M2 (lines 39-46, 80, 121-123). Scope wider than need: fs bridging, elicitation, persistence, two-way abort. Two file-access paths (editor fs methods and local file tools) with no stated authority. Persistence in AppJSONStore style is a Swift convention; in-memory sessions would be cheaper for v1.

ZO-M3 (lines 3-8, 26-31). Swift links are relative to repo root, not this folder. ServerConnectionView.swift is only the IP and port settings form; cite `crates/server`. ExternalAgentMcpReader.swift is unrelated to ACP.

ZO-N1 (lines 139-169). File list is more detailed than the plan needs while the key seam is vague; check `crates/cli` layout (`daemon.rs` exists).

ZO-N2 (lines 61-62, 64). Speculative cross-references ("a later draft can unify them", OC-13 for an excluded daemon transport).

ZO-N3 (lines 177-180). The mutation check is the strongest verification item; lead with it.

## Inventive catalyst (IC)

4 Major, 3 Nit, no Blocker. Advisory.

IC-M1 (lines 44-46, 70-76, 109-119, 129). One engine per editor-spawned process: multiple multi-GB checkpoint loads, memory pressure, GPU contention, cold-start time are unaddressed; "adapter adds serialization only" ignores model load. Options: thin stdio shim attaching to a resident engine over a user-only Unix socket, or a documented single-engine lock / refusal when the app holds the model.

IC-M2 (lines 51-55, 70-76). "Same risk classification as app sessions" conflicts with a Rust crate. Ladder position required by INDEX.md (lines 112-114) is not stated. Share a declarative policy table with a cross-language conformance test, or state a narrower Rust v1 ladder and name its position.

IC-M3 (lines 51-55, 77-79, 100-105). Deny-by-default offered as the only alternative to silent approval. Add a user-controlled launch-time policy (flag or file read only from the app support directory, never the workspace) and a timeout that turns an unanswered permission request into a denial.

IC-M4 (lines 47-50, 61-63, 121-125, 151-153). AppJSONStore is Swift; the Rust crate needs a second store with its own quarantine. Two writers to one directory with no locking. Fix a versioned language-neutral record schema, dedicated subdirectory, single-writer lock.

IC-N1 (lines 56-57, 143-145). Stdio failure handling unspecified: stdin EOF mid-generation (orphan process holding GPU memory), stdout pollution corrupting JSON-RPC, half-written records on crash, backpressure from a stalled client.

IC-N2 (lines 41-42, 118-122). Unsaved editor buffers vs disk: no statement of which is authoritative; edits could overwrite dirty buffers. Use client fs capability when advertised, confined disk access otherwise.

IC-N3 (lines 28, 31, 42, 150). Link paths do not resolve from this folder; "session/request elicitation" not confirmed against the ACP method set; fixture source and ACP spec revision not named; MCP-reader citation irrelevant.
