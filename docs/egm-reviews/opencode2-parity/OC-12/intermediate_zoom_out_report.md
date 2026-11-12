# Stage 8: Intermediate zoom out report (OC-12)

Target: docs/designs/opencode2-parity/OC-12-acp-adapter.md (full file). Reviewer: egm-zoom-out (S8). Inputs: Stage 5 report, Stage 6 and 7 reports. Completed.

Verdict: fails root-problem focus and leverage-existing. Stage 5 blockers are still open. Applying Stage 6 and 7 findings as written would roughly double the document and add scope unrelated to the root problem. The reviewer recommends cutting and deciding, not adding.

Code check by the reviewer: `crates/` has no agent session or tool-execution loop. `crates/server` and `crates/tokenizer` parse and validate tool calls but nothing executes them. The permission ladder and `AppJSONStore` are Swift only. No ACP crate or SDK dependency exists.

## Findings on the document as it stands

S8-1 Blocker (lines 51-55, 70-76, 107-119, 143-157). Swift permission engine reused from a Rust crate that never launches the app. "Same risk classification" is false for option (a). Item 5 hides the real work: agent loop, tool dispatch, session registry, path-safety rules, all new Rust. Decide first: Swift helper mode (reuses ladder and store, meets INDEX ladder-position requirement), or a narrower Rust-only ladder with its position named and the sizing corrected. Same finding as ZO-B1, CO-1, DC-1.

S8-2 Blocker (lines 68-82 vs 143-157). Architecture question left open while the file plan commits to `crates/acp`. The diagram shows two paths. Resolve, then delete the unchosen path.

S8-3 Major (lines 24-35). Weak problem statement: `ServerConnectionView.swift` is a settings form, `ExternalAgentMcpReader.swift` is unrelated. Cite `crates/server`, drop the MCP reader, state who needs editor embedding.

S8-4 Major (lines 39-50, 56-57). v1 scope wider than the need. Cut to `initialize`, `session/new`, `session/prompt`, `session/update`, `session/cancel`, `session/request_permission`. In-memory sessions for v1 (removes the AppJSONStore dependency, second-writer and quarantine questions). One file-access path, stated.

S8-5 Major (lines 121-125). Workspace root is client-supplied; a client can declare `/` or `~`. One sentence: root must be an existing directory the user allowed, or confined by a user-only allow list.

S8-6 Major (lines 44-46, 129-130). Engine load cost unaddressed. State a minimal policy: one engine per process, refuse when the app holds the model. Do not design a resident-engine socket shim.

S8-7 Major (lines 100-105). Capability detected at `initialize`; an unanswered request times out into a denial. A one-line requirement.

S8-8 Nit (lines 3-8, 28, 31). Swift links are repo-root-relative; OpenCode links point at `dev` instead of audited commit 7ce35879b8.

S8-9 Nit (lines 61-62, 64). Speculative forward references ("a later draft can unify them", daemon-pairing). Delete.

S8-10 Nit (lines 139-169). File list too granular while item 5 (the seam that matters) is vague. Collapse and name the seam.

S8-11 Nit (lines 177-180). Lead verification with the mutation check (lines 168-169). Waiving the real-engine run as "not a release gate" is acceptable.

## Stage 6 and 7 mitigations to reject or defer

- ST-1 and ST-2: keep Alternatives and Risks sections short. One line for "extend the HTTP server" (no sessions or tools). Do not grow Risks into a full threat model.
- IC-M1 resident engine with Unix socket shim, IC-M4 language-neutral record schema with single-writer lock: both vanish if persistence is cut or Swift is chosen.
- IC-M3 launch-time policy file: a second permission channel is speculative. Keep deny-by-default plus the timeout.
- DC-4 per-path Always-allow, DC-5 symlink/TOCTOU/APFS rules, DC-9 size limits and metrics, DC-10 fake-engine harness and fuzz tests: implementation-level, belong in tasks. Keep only: stdout carries protocol only, logs to stderr, stdin EOF cancels generation.
- DC-15 signing and Sparkle: premature unless Swift helper chosen.
- DC-14, SY-4 localization: one sentence ("the adapter has no UI strings").
- DC-6, CO-4: pin the ACP spec revision in one line; prefer a maintained Rust ACP SDK over hand-written `protocol.rs` after checking it.
- Style-only items (SY-7 to SY-18, ST-6 to ST-8): one cleanup pass.
