# Stage 7: Critic report, ZP-03 (one round)

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All completed. Standard: feature-spec-standard for DC and ST. FeatureRange = lines 1-185. Convergence not assessed (one round).

## Design (DC)

DC-1 Major, INDEX.md l.31 vs lines 87-104, 142-152. INDEX says "Rust-side rollout logging"; draft is Swift-only. Name callbacks or add Rust/FFI work.
DC-2 Major, lines 39-40, 51-53, 106-109, 131-132. Privacy under-specified and contradictory: "inherit vault posture" has no mechanism for a per-chat JSONL file; chat delete cascade, pasted secrets, decrypted-cache purge unaddressed.
DC-3 Major, lines 74, 98, 108-109. Crash safety hand-waved: no buffer described, "flush" meaningless for direct append; "absent or marked truncated" undecided; partial final line repair on next append, fsync, O_APPEND not specified.
DC-4 Major, lines 38, 72-75. Bound is by entry count not bytes; no per-payload cap, no file cap; backward scan with variable line length; disk growth unmitigated.
DC-5 Major, lines 111-127. Alternatives thin; "Rejected by author" invalid; missing: SQLCipher ProfileDatabase rows, deriving from transcript, Rust-side emission; vault vs sidecar not an alternative.
DC-6 Major, lines 7-10, 138-152. ZP-23 dependency unstated; hook must key by chat id; lock behavior under concurrent chats.
DC-7 Nit/Major, lines 106-108. Schema lacks turn/call id, sampling params, error stack, unknown-`v` policy; `role` undefined for toolExec.
DC-8 Nit, lines 18-19, 33-34. Prompt/message list not logged though "model-io" implies it; retries/reasoning not discussed.
DC-9 Nit, line 28. TURBOSPARK_DEBUG_PROMPT_IDS not cited; verify it exists.
DC-10 Nit, lines 172-181. No scale test for 10,000 entries, no concurrent-append or chat-delete tests.
DC-11 Nit, lines 64-66. "Shareable" overstated; redaction is heuristic; best-effort plus preview.
DC-12 Nit, lines 4-6. ZCode docs/model-trajectory-viewer.md and timeline file not cited.
DC-13 Nit, lines 138-182. No localization catalog file listed.
DC-14 Nit, line 8, 11. Open Questions must close before implementation.

## Structural (ST)

ST-1 Blocker, missing section. No Risks & Mitigations: at-rest unredacted payloads, unbounded growth, locked-vault unreadable log, concurrent append contention, schema evolution, redaction false negatives.
ST-2 Major, lines 49-56 vs 31-40, 85-109, 131. Open Questions unresolved while text asserts vault; encrypted tail-read tradeoff unanalyzed.
ST-3 Major, lines 111-127. Alternatives weak, "by author" attribution forbidden without a link; Alternative B near strawman.
ST-4 Major, lines 1-6, 19-21, 98-104. INDEX/draft Rust mismatch; line 103 claim has no source.
ST-5 Major, lines 3-10. Provenance incomplete: ZCode timeline file and design doc omitted; Swift reference (UsageDashboardSection.swift) not linked.
ST-6 Major, lines 57-83, 31-40. Placement described three ways (context menu, inspector, dock); 500-entry page unjustified; CUJ 3 untestable; no flush mechanism or AppShutdownCoordinator hook defined.
ST-7 Major, lines 106-109. Schema is prose with "?" notation: no types, sizes, truncation policy, error shape, token split.
ST-8 Major, lines 138-181 vs 31-40. No component owns search-all; no redactor component or rules; redaction test coverage thin.
ST-9 Nit, lines 129-136. NFRs lack numeric budgets; "no model gates" is a claim.
ST-10 Nit, lines 172-181. Bare "ten gates" and "21 languages"; mutation checks mis-placed; no vault-locked test or perf test.
ST-11 Nit, lines 12, 183, 8-10, 49. Section order (Open Questions early) and metadata placement; confirm template.
ST-12 Nit, line 28. Env var uncited.

## Style (SY)

SY-1 Blocker, lines 108-109, 118-119, 77-83. Crash-loss claim outruns the design: "quit/shutdown flush" does not run on a crash; CUJ 3 promises "absent or marked truncated" but schema has no marker; per-call append makes loss bound the in-flight call, not turn.
SY-2 Major, INDEX l.31. Rust-side scope mismatch.
SY-3 Major, lines 103-104, 149-150. Unverified claim: "engine callbacks already carry usage, latency, finish reason" cites no callback.
SY-4 Major, lines 20-22, 28. "Nothing equivalent; nearest is Inspector" and the env-var workaround cite no source.
SY-5 Major, lines 51-53 vs 131-132. Vault open question conflicts with settled requirement.
SY-6 Major, lines 39-40, 132-133. Redaction scope undefined.
SY-7 Major, lines 51-56. Open questions hedged ("lean vault; author decides"); retention cap absent from plan.
SY-8 Nit, lines 118-119, 126-127. "Reason for Rejection: Rejected by author:" redundant.
SY-9 Nit, lines 38, 62-64. "Usable on very long chats" unmeasurable; tie to 500/10,000.
SY-10 Nit, lines 109, 119, 61. "quit flush" vs "shutdown flush", per-call vs per-turn used loosely.
SY-11 Nit, lines 177-178. Unanchored "ten gates", "21 languages".
SY-12 Nit, lines 46-47, 134. Dense phrasing; name the gate document.

## Comprehension carryover
Findings CO-1 to CO-12 (comprehension_report.md) overlap DC-2, DC-3, DC-7, ST-6, SY-5, SY-6.
