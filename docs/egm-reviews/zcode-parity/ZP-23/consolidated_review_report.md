# Consolidated review report: ZP-23 concurrent background chats

Target: docs/designs/zcode-parity/ZP-23-concurrent-chats.md (full file, 224 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (two spawns hit "Concurrent subagent limit reached" and succeeded on retry).
Persona files actually read: zoom-out agents/zoom_out.md (stages 5 and 8); inventive-catalyst agents/inventive_catalyst.md; comprehension agents/comprehension.md; design agents/design_critic.md; structural agents/structural.md; style agents/style.md. Standards: feature-spec-standard (design, structural), zoom-out-standard (zoom-out).
Source IDs: ZO, IC = zoom_out_report; CO = comprehension_report; DC, ST, SY = critic_report; S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 4, Major 11, Nit 11.

## Blocker

B1. Lines 66-74 vs 111-139. Central mechanism (Swift engine lease, alternating steps) is stated as settled while feasibility is an open question; no fallback or abort criterion. Fix: add a phase-0 spike with pass/fail criterion, mark the coordinator contingent, name the turn-level lease as fallback. [ZO-2, IC-2, CO-1, DC-B2, ST-3, SY-5, S8-B1]
B2. Whole document (lines 159-171 nearest). No Risks and Mitigations section; "no new memory risk" and "2 is safe on 16 GB" are unsupported. Fix: add a short section (KV exhaustion, foreground starvation, approvals for a non-visible chat, partial flush) and mark claims as hypotheses. [DC-B3, ST-1, SY-1, SY-4, S8-N6]
B3. Lines 149-150, 155-157. Alternatives rejected "by author" with no link; B is circular; A is self-contradictory. Fix: concrete technical reasons; link any decision record. [DC-B1, ST-2, SY-6, SY-7, S8-M5]
B4. Lines 141-157, 66-74. Alternatives omit the real fork: Rust-side scheduler vs Swift scheduler vs turn-level lease (plus one line each for engine per chat and KV swap). Fix: add them with rejection reasons. [ST-3, DC-B1, S8-M5]

## Major

M1. Lines 111-139, 175-193. Root cause is per-turn singletons on AppModel (about 25 guard sites in 10+ files); plan lists five files. Fix: name a per-chat turn-state holder as the structural change with a blast-radius note; list SplitChatPanes, AppModel+SplitView and projects. [ZO-1, ZO-4, DC-M3, S8-M1]
M2. Lines 74, 135-137, 167-169. "Round-robin with foreground priority" is contradictory and the performance NFR is unfalsifiable. Fix: one policy (strict foreground priority, background when idle), rewrite as design intent with a measurable criterion. [IC-1, DC-M1, SY-2, ST-6, S8-M4]
M3. Lines 48-50, 75-77. Behavior at the cap and non-generating consumers (title, memory, cron, goal timers) undefined; fixed count stands in for KV. Fix: constant cap of 2 for v1, state refuse or queue at the cap, no settings field yet. [IC-3, DC-M2, CO-4, ZO-7, S8-M2]
M4. Lines 89-109, 118. Tool execution in background chats (shared workspace, permission engine, approval spoofing, auto-approve) and awaiting-state slot accounting unaddressed; no ladder-position statement. Fix: background chats do not auto-run tools, awaiting chats park and release lease and KV, state ladder position. [ZO-5, IC-4, IC-5, DC-M4, ST-1, S8-M3]
M5. Lines 104-109, 165-166. "Atomic per chat", ordered flush, crash recovery with multiple chats undefined. Fix: reuse the existing interrupted-row flow, state crash reconciliation scope. [IC-7, DC-M5, CO-6, S8-M6]
M6. Lines 49-50, 138-139, 161-166. ResidencyPolicy (no Swift type exists), hardware-fit logic, expert caps undefined; "subagents already serialize" uncited. Fix: name real types and files or drop. [ZO-3, CO-3, ST-5, SY-1, SY-8]
M7. Lines 55-59, 96-102. Model-switch refusal is always true while any chat generates, conflates chat-switch and model-switch guards, offers no recourse. Fix: reframe, name chats, mark string as placeholder, one line listing stop-then-switch as follow-up. [IC-6, CO-5, DC-N3, ST-9, S8-N1]
M8. Lines 3-9, 18-24, 85-86, 170. ZP-02 and ZP-21 undefined and dependency direction confusing. Fix: one line each plus links. [CO-2, SY-11, S8-M7]
M9. Lines 66-77. Open Questions misplaced, no owner or exit criterion; cap default reads as decided elsewhere. Fix: move, add exit criteria. [ST-4, SY-4]
M10. Lines 37-53, 219-220. Goals lack measurable acceptance (foreground tok/s ratio, switch latency, memory ceiling). Fix: add them as observations to verify. [ST-6, DC-M1, DC-N5]
M11. Lines 6-8, 135-139. Idle-chat KV lifecycle and subagent slot accounting undefined; cited idle-deactivation parity unaddressed. Fix: one slot per top-level chat with subagents inside it; state idle KV retention or defer explicitly. [DC-M6, IC-8, CO-8, S8-N4]

## Nit

N1. Lines 175-220. Test matrix and mutation checks over-detailed; mutation bullet nested inconsistently (205-208). Fix: collapse. [ZO-6, ST-8, SY-9, S8-N2]
N2. Lines 31-35. Cron dispatch cited as motivation without goal or CUJ; queue draining vs cap not covered. Fix: one sentence routing through the cap. [IC-9, CO-7, S8-N3]
N3. Lines 18-24, 28-31, 51. Rhetoric and figurative phrases ("not by parity inertia", "parks the whole app", "Quit generalizes"). Fix: plain wording. [SY-3, SY-13, SY-14]
N4. Lines 147, 220. "No-unmeasured-gates rule", "bench gates" unlinked. Fix: link. [CO-9, DC-N2]
N5. Line 166. "Sampling owns normalization" unexplained. Fix: tie to concurrency or drop. [CO-10, DC-N1]
N6. Lines 63-64. "Workspace tab" undefined. Fix: link ZP-21. [CO-11]
N7. Lines 118, 177-209. MaxConcurrentGenerations key and UI location unspecified; "settings field" vs "settings-visible". Fix: name key or drop the field per M3. [CO-12]
N8. Lines 170-171, 215. "21 languages" vs INDEX "ten localization parity gates". Fix: one reference. [CO-13, SY-12]
N9. Lines 26, 111. Headings differ from standard names. Fix: "Problem Statement", "Proposed Architecture". [ST-7]
N10. INDEX.md line 4. "Missing" framing vs this "Different" finding. Fix: one-line carve-out. [DC-N4, S8-N7]
N11. INDEX.md. Drafts were written against feat/classlru-replay-arm; guard line numbers confirmed on main only. Fix: re-check on the implementation branch. [ZO-8]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-8, IC-1 to IC-10.
- comprehension_report.md: CO-1 to CO-13 (no Blockers).
- critic_report.md: DC-B1 to DC-N5, ST-1 to ST-10, SY-1 to SY-14.
- intermediate_zoom_out_report.md: S8-B1 to S8-N7. Document unchanged since Stage 5, all findings open. It recommends rejecting scope growth: KV-byte admission and queue, weighted scheduler, per-workspace write lease, model pinning/deferred switch, new in-flight marker, idle KV LRU, metrics apparatus, hardware-aware settings bound. Keep fixes that remove ambiguity.
