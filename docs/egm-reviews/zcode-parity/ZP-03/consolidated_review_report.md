# Consolidated review report: ZP-03 model trajectory viewer

Target: docs/designs/zcode-parity/ZP-03-model-trajectory-viewer.md (full file, 185 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 9, Nit 10.

## Blocker

B1. Lines 108-109, 118-119, 77-83 (CUJ 3). Crash-loss claim is false for a crash: the quit flush does not run, "absent or marked truncated" is two behaviors, the schema has no truncated marker, and per-call appends make the bound the in-flight call. Fix: state clean-quit and crash cases separately; with a DB table, "completed calls present, in-flight call lost"; drop flush and "truncated". [SY-1, IC-1, CO-2, DC-3, ZO-6, S8-M2]
B2. Missing section (after line 136). No Risks and Mitigations section, required by the standard. Fix: add 4 to 6 lines: at-rest payload copy, growth, locked vault, redaction false negatives, ZP-23 contention. [ST-1, S8-B1]

## Major

M1. Lines 51-53, 94-98, 111-127, 142-146. Bespoke encrypted JSONL store duplicates ProfileDatabase; vault vs sidecar open yet asserted; Alternatives weak, "Rejected by author" attribution, DB table and transcript-derived options missing. Fix: decide "table in ProfileDatabase", rewrite Alternative B as table vs file, drop "by author". [ZO-1, ZO-2, IC-2, CO-1, CO-7, DC-5, ST-2, ST-3, SY-5, SY-7, SY-8, S8-M1]
M2. Lines 131-132, 39-40. Privacy: "inherit vault posture" has no mechanism for a JSONL file; pasted secrets persist unredacted. Fix: follows from M1; cap payload size at write time. [DC-2, IC-3, ZO-4, S8-M4]
M3. Lines 33-34, 131-132. Ghost chats not excluded; AppChatUsage.swift says ghost rows carry no conversation content. Fix: add Goal "no rollout for ghost chats". [ZO-3, S8-M3]
M4. Lines 33-36, 142-146. Lifecycle unspecified: chat delete, fork, regenerate, rewind, archive, profile export. Fix: key by chat id with cascade delete; state fork/rewind behavior; turn id link. [ZO-4, CO-4, S8-M4]
M5. Lines 99-104, 149-150, 174, INDEX l.31. "Engine callbacks already carry usage, latency, finish reason" unverified; INDEX says Rust-side logging, draft is Swift-only. Fix: name the callback seam or size the Rust work; fix INDEX. [ZO-7, IC-7, DC-1, ST-4, SY-2, SY-3, S8-M5]
M6. Lines 106-109. Schema too thin: no turn id, single `tokens`, `role` undefined for toolExec, no types or error shape, "model call" undefined. Fix: minimal set (turn id, prompt/completion/cached tokens, explicit kind), foreground calls only; defer sampling params and prompt hash. [ZO-5, IC-5, IC-6, CO-3, DC-7, ST-7, S8-M6]
M7. Lines 7-12, 138-152. ZP-23 dependency unstated; hooks must key by chat id. Fix: one dependency line. [ZO-12, IC-8, DC-6, S8-M7]
M8. Lines 38, 72-75. Tail bound is by count, not bytes; no payload or file cap; no byte bound on pane memory. Fix: per-payload truncation cap and per-chat cap with clear control; follows from M1. [CO-5, DC-4, S8-N1]
M9. Lines 39-40, 64-66, 132, 145-146. Redaction undefined ("user-writable paths", "secret-shaped strings"), no owning component, "shareable" overstated. Fix: drop export from v1 or name an existing redaction helper and call it best-effort. [ZO-9, CO-6, DC-11, ST-8, SY-6, S8-N2]

## Nit

N1. Lines 49-56. Open Questions have no owner or date; close them. [CO-7, DC-14, SY-7]
N2. Lines 57-83, 89-104, 142-161. Placement described three ways (context menu, inspector, dock); diagram omits AppModel+Trajectory; store under Diagnostics/. Fix: choose Inspector section or one placement; reconcile. [ZO-11, ST-6, CO-11, S8-N3]
N3. Lines 69-75, 133-134. Search-all and 10,000-call scale speculative. Fix: tail plus search of loaded entries. [ZO-8, IC-4, S8-N1]
N4. Lines 162-170. Mutation recipes mis-placed and tied to file store; hook owner unnamed. Fix: drop or move. [ZO-10, ST-10, S8-N4]
N5. Lines 3-6, 21, 28. Uncited: TURBOSPARK_DEBUG_PROMPT_IDS, Inspector claim, ZCode timeline file and docs/model-trajectory-viewer.md, UsageDashboardSection.swift. Fix: cite or verify. [DC-9, DC-12, ST-5, ST-12, SY-4, S8-N5]
N6. Lines 177-178. "Ten parity gates", "21 languages" unanchored. Fix: link INDEX. [ST-10, SY-11, CO-9]
N7. Lines 118-119, 126-127, 109, 61. "Rejected by author:" redundant; "quit flush" vs "shutdown flush", per-call vs per-turn inconsistent. Fix: unify wording. [SY-8, SY-10]
N8. Lines 38, 62-64, 129-136. "Usable on long chats" unmeasurable; NFRs lack numbers; "no model gates apply" asserted. Fix: tie to 500/10,000 or drop. [SY-9, ST-9, SY-12, CO-10]
N9. Lines 13-21, 59-75. Default view (whole chat or selected turn) unstated; model-io prompt not logged. Fix: state both. [CO-12, DC-8]
N10. Lines 138-182, 172-181. No localization file listed, no vault-locked or scale tests. Fix: optional; S8 recommends declining. [DC-13, DC-10, IC-9]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-12, IC-1 to IC-9. Feeds B1, M1-M9, N2-N3, N4.
- comprehension_report.md: CO-1 to CO-12. Feeds B1, M1, M4, M6, M8, M9, N1-N2, N5-N9.
- critic_report.md: DC-1 to DC-14, ST-1 to ST-12, SY-1 to SY-12. Feeds all Blockers and Majors.
- intermediate_zoom_out_report.md: S8-M1 to S8-M7, S8-B1, S8-N1 to S8-N5. Confirms all earlier findings open; recommends declining IC-1 queue, IC-2 segments, IC-4 sidecar index, schema expansion beyond the minimal set, scale/concurrency tests, catalog listing, numeric budgets.
- Merge notes: the three reviewers all raised the INDEX Rust mismatch (M5) and the vault contradiction (M1); crash-safety merged into B1.
