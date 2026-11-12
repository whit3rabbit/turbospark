# Consolidated review report: OC-12 ACP adapter

Target: docs/designs/opencode2-parity/OC-12-acp-adapter.md (full file, 184 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. Reviewers did not read the ACP section of OPENCODE2_COMPARISON.md (line 430 onward), so the draft's quoted Section 8 finding is unverified.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 4, Major 11, Nit 11.

## Blocker

B1. Lines 51-55, 70-76, 116-119, 154-157. Permission and tool logic is Swift (`AppToolPermissionEngine`, agent loop, `AppJSONStore`), but the plan is a Rust crate that never launches the app; "same risk classification" is false for that option, ladder position (INDEX 112-114) is not stated, and the work is understated. Fix: choose the Swift helper mode, or state a narrower Rust v1 ladder with its position and re-size as a second agent runtime. [ZO-B1, IC-M2, CO-1, DC-1, SY-1, S8-1]

B2. Lines 68-82, 113, 143-157. Where the adapter lives is an open question while the plan commits to `crates/acp`; the session-entry seam (item 5) is unnamed and hides the main work. Fix: resolve the question, delete the unchosen path, name the existing entry point and what already handles tool calls. [ZO-B2, ZO-M1, CO-2, CO-8, DC-2, ST-4, SY-13, S8-2]

B3. Whole document (no section). No Alternatives Considered section; Swift helper is dismissed in one sentence. Fix: add a short section: Swift helper, extend `crates/server`, existing ACP Rust SDK versus hand-written protocol, do nothing, each with a technical rejection reason. [ST-1, DC-2, S8 mitigation note]

B4. Whole document (no section). No Risks and Mitigations section; safety content is scattered. Fix: add a brief section: prompt-injection writes, root escape, hostile frames, stdout pollution, concurrent processes, unanswered permission requests. [ST-2, DC-9]

## Major

M1. Lines 51-55, 77-79, 96-105, 131. Permission model under-specified: no timeout for unanswered requests, no capability detection at `initialize`, "never answers" conflated with "unsupported", session-wide Always-allow scope and allow-once options unstated, allowlist question open while the NFR asserts deny-by-default. Fix: detect capability at `initialize`, time out into denial, name honored options, close the allowlist question. [DC-4, CO-6, IC-M3, SY-2, SY-12, S8-7]

M2. Lines 121-125, 131. Workspace root is client-supplied (`/` or `~` possible); path-safety rules are not located. Fix: require an existing user-allowed directory or an allow list, and cite the rules' implementation. [DC-5, S8-5]

M3. Lines 20-22, 44-46, 87-91, 129-130. No model-load or concurrency story (cold start, memory contention with the app, multiple editor windows); "adapter adds serialization only" is unmeasured. Fix: one-engine-per-process policy with refusal when the app holds the model; drop or measure the latency claim. [DC-3, IC-M1, SY-3, S8-6]

M4. Lines 39-43, 56-57, 118-122, 148-150. Protocol scope vague: "session/request elicitation" is not an ACP method name, "session/ update" is broken, no spec revision pinned, `session/cancel` unmapped, fs authority (editor buffers vs disk) unstated, fixture source unnamed. Fix: method table with pinned spec revision; cut v1 to initialize, new, prompt, update, cancel, request_permission; one stated file-access path. [DC-6, CO-4, CO-13, ST-6, SY-9, IC-N2, ZO-M2, S8-4]

M5. Lines 47-50, 59-63, 121-125, 151-153. Persistence uses Swift `AppJSONStore` from Rust; no location, schema, resume method, writer locking, or vault behavior. Fix: keep sessions in memory for v1, or specify what the Rust crate implements (atomic write, quarantine) and the storage location. [DC-7, CO-3, CO-5, IC-M4, ST-10, SY-18, ZO-M2, S8-4]

M6. Lines 51-55, 68-82, 93-98, 168. Open Questions already answered inside Goals and the Verification Plan; v1 tool surface inconsistent between Goals (full app set) and the question (file tools only). Fix: mark decisions, state the v1 tool set in Goals. [ZO-M1, CO-7, DC-8, ST-5, SY-11]

M7. Lines 24-35. Problem evidence weak: `ServerConnectionView.swift` is only an IP and port form and `ExternalAgentMcpReader.swift` is unrelated; no stated user need. Fix: cite `crates/server`, drop the MCP reader, name the need. [ZO-M3, DC-11, S8-3]

M8. Lines 28, 31. Repo-root-relative Swift links do not resolve from this folder. Fix: use `../../../swift/...`. [ST-3, ZO-M3, IC-N3, S8-8]

M9. Lines 56-57, 127-137, 143-145. NFRs and stdio failure handling thin: stdout must carry protocol only (logs to stderr), stdin EOF behavior, orphan process holding GPU memory, backpressure. Fix: state stdout/stderr split and EOF cancels generation; leave size limits and metrics to tasks. [DC-9, IC-N1, S8 mitigations]

M10. Lines 168-169, 171-180. Verification waives the real-engine run and lacks a stalled-permission test and malformed-frame test. Fix: lead with the mutation check, add those two tests, keep the real-engine run noted. [DC-10, ZO-N3, S8-11]

M11. Lines 18-23, 129-130, 133-137. Uncalibrated claims: "permission story is explicit rather than inherited" contradicts the inherited classification; "not localized by policy" has no policy pointer. Fix: plain statement of the deny default; one sentence that the adapter has no UI strings (deny messages shown in the editor noted). [SY-5, SY-4, DC-14, S8 mitigations]

## Nit

N1. Lines 3-8. OpenCode links point to `dev`, not audited commit 7ce35879b8; Section 8 quote unverified. Fix: pin the commit. [DC-12, SY-16, S8-8]

N2. Lines 75, 160. "Hardening" of `crates/server` unspecified; CLI subcommand placement unchecked against `crates/cli` layout (`daemon.rs` exists); binary discovery, signing and notarization unaddressed. Fix: name the hardening and check the CLI layout; defer signing unless Swift helper chosen. [DC-13, DC-15, ZO-N1]

N3. Lines 146-148. Cargo member change should note MSRV 1.82 and edition 2021. Fix: add a clause. [DC-16]

N4. Lines 39-42, 159, 175-176. Inconsistent code font for method and command names; split inline code span. Fix: normalize. [ST-7, ST-8]

N5. Lines 143, 146-147, 154-157, 174. Vague or bare file-list items and package name mismatch (`crates/acp` vs `turbospark-acp`). Fix: fold item 2 into item 1, name the package, name the seam. [SY-13, SY-14, SY-17]

N6. Lines 19-20, 32-36, 44-46, 70-76. Redundant or unsupported wording: on-device repeated five times, "no cloud anything", causal OpenCode clause uncited, "lean" stated twice, "Rust keeps the app unlaunched". Fix: state once, cite or drop. [SY-6, SY-7, SY-8, SY-10, CO-10, ST-9]

N7. Lines 61-62, 64. Speculative references ("a later draft can unify them", daemon-pairing). Fix: delete. [ZO-N2, S8-9]

N8. Lines 139-169. File list more granular than a design needs. Fix: collapse rows. [ZO-N1, S8-10]

N9. Lines 179-180. Verification rules unlinked; "this Mac" ambiguous. Fix: link `.claude/docs/verification.md`, say "a developer Mac". [SY-15]

N10. Lines 39-43, 56-57, 150. "Abort both directions" does not name the messages; "core session surface" undefined. Fix: map to `session/cancel`. [CO-11, SY-9]

N11. Lines 3-8, 18, 65, 117, 168. Undefined terms: "excluded category", Metal/MLX/ANE, "mutation check" procedure, "Missing in Swift" for a Rust proposal. Fix: one-line gloss or link to INDEX. [CO-9, CO-12]

## Traceability

- zoom_out_report.md: ZO-B1, ZO-B2, ZO-M1 to ZO-M3, ZO-N1 to ZO-N3 (zoom-out); IC-M1 to IC-M4, IC-N1 to IC-N3 (inventive catalyst). Feeds B1, B2, M1 to M8, N1, N2, N7, N8.
- comprehension_report.md: CO-1 to CO-13. Feeds B1, B2, M1, M4 to M6, N6, N10, N11.
- critic_report.md: DC-1 to DC-16, ST-1 to ST-10, SY-1 to SY-18. Feeds all Blockers, Majors and Nits.
- intermediate_zoom_out_report.md: S8-1 to S8-11 plus the reject/defer list. Confirms all Stage 5 blockers still open. Recommends rejecting scope-adding asks (IC-M1 resident engine and socket, IC-M4 record schema, IC-M3 policy file, DC-4 per-path scoping, DC-5 symlink and TOCTOU rules, DC-9 limits and metrics, DC-10 fuzz harness, DC-15 signing) and keeping ambiguity-removal and decision fixes; recommends cutting v1 to the minimal session loop with in-memory sessions.
- Merge notes: IC-M2, DC-1, ZO-B1, CO-1, SY-1 merged into B1; ST-4/ZO-M1/CO-2 merged into B2. The reviewers disagree on size of fixes: Stage 7 asks for added sections and tests, Stage 8 asks for a smaller design. Both agree B1 and B2 must be decided first because they change half the other findings.
