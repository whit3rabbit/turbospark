# Stage 7: Critic report (OC-12)

Target: docs/designs/opencode2-parity/OC-12-acp-adapter.md (FeatureRange 1..184). Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). Standards: feature-spec-standard for design and structural. One round only, no remediation, so convergence was not assessed. All three reviewers completed. Reviewers noted they did not read the ACP section (OPENCODE2_COMPARISON.md line 430 onward).

## Design (DC)

DC-1 Blocker (lines 51-55, 76, 116-119, 154-157). The permission design depends on `AppToolPermissionEngine`, which is Swift only. A headless Rust crate cannot call it, so "same risk classification as app sessions" is false for the leaned option. The agent loop and tool dispatch also appear to be Swift. Item 5's "interface-only changes, no engine changes" assumes a Rust agent loop with tool calling that is unverified. Ladder position, required by INDEX (lines 112-114), is not stated. Either choose the Swift helper or cost the Rust port explicitly.

DC-2 Blocker (lines 70-76, 143-157). The central architecture question is open while the file plan commits to a Rust crate. The Swift option is rejected in one sentence. Missing alternatives: extend the existing server crate, a thin shim to the app over a local socket, an existing ACP Rust SDK versus hand-written `protocol.rs`.

DC-3 Major (lines 20-22, 44-46, 87-91, 130). No model-loading or concurrency story: cold start, memory contention with the app or server holding weights, multiple editor windows, single-instance rule, model selection mechanism.

DC-4 Major (lines 51-55, 97-98, 100-105, 131). Permission model under-specified: no timeout for a pending request; stalled request may block the stdio loop; capability detection in `initialize` unstated; session-wide Always-allow for writes is a broad grant (scope should be per path); open allowlist question while NFR asserts deny-by-default; prompt injection from file contents not covered.

DC-5 Major (lines 121-125, 131). Workspace root is client-supplied at `session/new`; a client can declare `/` or `~`. Path-safety rules are not located (Swift or Rust). Symlinks, TOCTOU and case-insensitive APFS paths unaddressed. Constrain roots or require a user allow list.

DC-6 Major (lines 39-42, 56-57, 148-150). Protocol scope vague: "session/request elicitation" does not match known ACP names (`session/request_permission`, `fs/read_text_file`, `fs/write_text_file`, `session/cancel`, `session/update`). No protocol version pin, negotiation or unknown-method handling. ACP fs methods are client-side (editor buffers); the draft describes local disk access. Fixture source unnamed.

DC-7 Major (lines 47-50, 59-63, 151-153). Persistence thin: `AppJSONStore` is Swift; no location, schema, versioning, resume support, retention, concurrent-writer safety; whole-archive rewrite scaling; vault unlock on a locked profile in a headless process.

DC-8 Major (lines 68-82 vs 51-52). Tool surface inconsistent: Goals imply the full app tool set, the Open Question says file tools only; streaming of tool events for unexposed tools unspecified.

DC-9 Major (lines 127-137). NFRs mostly empty: no message size limit, stdout backpressure, framing errors; stdout must carry protocol only and logs go to stderr; EOF and crash behavior (kill in-flight generation, free GPU); no metrics, effort estimate or milestones.

DC-10 Major (lines 177-180). Manual real-engine run waived as "not a release gate". No automated end-to-end test with a fake engine, no stalled-permission test, no malformed or oversized frame test, no cross-version fixture test.

DC-11 Major (lines 26-35). Problem evidence is weak: `ServerConnectionView.swift` is only an IP and port settings form, and `ExternalAgentMcpReader.swift` is unrelated to ACP. No user need, no Zed usage evidence, no success metric.

DC-12 Nit (lines 3-8). Section 8 heading unverified; OpenCode links point to `dev` branch, pin to commit 7ce35879b8.

DC-13 Nit (lines 75, 109-119). "Hardening" of `crates/server` named but unspecified.

DC-14 Nit (lines 133-135). "Not localized by policy" conflicts with the INDEX localization constraint for user-visible permission messages.

DC-15 Nit (line 160). Binary discovery by an editor when the app is a .app bundle with Sparkle updates, helper signing and notarization are unaddressed.

DC-16 Nit (lines 146-148). Workspace member addition should note MSRV 1.82 and edition 2021.

## Structural (ST)

ST-1 Blocker (whole document). No Alternatives Considered section; the Rust versus Swift pair is an unresolved lean. Candidates: extend the HTTP server, third-party ACP shim, do nothing. No rejection may be attributed to a maintainer without a link.

ST-2 Blocker (whole document). No Risks and Mitigations section; safety content is scattered across the Safety NFR, permission Goals and an Open Question aside. Gaps: prompt-injection writes, root escape, hostile JSON-RPC frames, stdout contamination, unbounded session growth, concurrent writers.

ST-3 Major (lines 28, 31). Repo-root-relative Swift links resolve wrongly from this folder; use `../../../swift/...`.

ST-4 Major (lines 111-113 vs 143-146, 70-76, 154). Diagram says "crates/acp or crates/server/src/acp/" while the plan commits to `crates/acp/` and `turbospark-acp`; item 5 is non-committal.

ST-5 Major (lines 68-82 vs 51-55, 168). Two of three Open Questions are already answered in Goals and the Verification Plan; tool-surface lean not reflected in Goals.

ST-6 Major (lines 39-42). Method list malformed: "session/ update" split across a line, "session/request elicitation" not a method token, names not in code font.

ST-7 Nit (lines 39-42, 159, 177-180). Inconsistent code font for protocol and command names.

ST-8 Nit (lines 175-176). Inline code span split across a line break.

ST-9 Nit (lines 24-35). OpenCode's frontend-agnostic core claim has no inline citation.

ST-10 Nit (lines 47-50, 93-98). "AppJSONStore quarantine style" is a Swift type; state what the Rust crate implements (atomic write, quarantine of corrupt files). Also cite `crates/server` instead of `ServerConnectionView.swift`.

## Style (SY)

SY-1 Major (lines 51-55, 127-132). "Same risk classification" is an uncalibrated claim and contradicts Open Question (b); state the ladder position and mechanism or narrow the claim.

SY-2 Major (lines 96-98). Session reuse of the editor decision stated as settled; ACP allow-once and allow-always options are not named.

SY-3 Major (lines 129-130). "Adapter adds serialization only" is unmeasured; also adds the stdio loop, registry and fan-out.

SY-4 Major (lines 66-67, 133-137). Localization sentence conflicts with the INDEX gates; say the adapter has no UI strings. "By policy" has no pointer.

SY-5 Major (lines 18-23). "Permission story is explicit rather than inherited" is vague and conflicts with line 51-52, which inherits the app classification.

SY-6 Nit (lines 32-36). Causal clause about OpenCode design has no evidence; drop or cite.

SY-7 Nit (lines 19-20, 44-46, 64, 132). Same on-device and no-socket fact stated five times; "no cloud anything" is informal.

SY-8 Nit (lines 44-46). "Editor" repeated in "child process of the user's editor".

SY-9 Nit (lines 39-43, 57). Method names unverified; "core session surface" undefined while "abort both directions" implies `session/cancel`.

SY-10 Nit (lines 70-76). Lean stated twice; "Rust keeps the app unlaunched" is awkward.

SY-11 Nit (lines 77-83). "(lean)" labels with no owner; deny default is both a Goal and an Open Question.

SY-12 Nit (lines 100-105). A never-answering client is indistinguishable from an unsupporting one in Goals; state the timeout.

SY-13 Nit (lines 143-145, 154-157). "Whatever session-entry seam" is vague; name it or mark as unknown.

SY-14 Nit (lines 146-147). "Add the crate" is a bare rationale; fold into item 1.

SY-15 Nit (lines 179-180). Verification rules reference unlinked; "this Mac" ambiguous.

SY-16 Nit (lines 6-8). Parity links point to `dev` with no commit pin.

SY-17 Nit (lines 143, 174). Package name `turbospark-acp` and directory `crates/acp/` should agree in the crate item.

SY-18 Nit (lines 47-50). Name what the Rust crate implements, not a Swift type.
