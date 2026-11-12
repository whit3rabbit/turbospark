# Stage 5 report: zoom-out and inventive catalyst (INDEX.md, full file)

## Zoom-out reviewer (ZO)

ZO-1 Blocker. Lines 83-84, 95, 99. Step 3 schedules OC-11 "after OC-10's seam" but OC-10 is scheduled in step 7, and step 7 lists OC-11 again. Neither sibling defines a shared seam: OC-10 plans its own EditToolHook in AppModel+Tools.swift, OC-11 plans its own post-write hook there and never mentions OC-10. Fix: delete OC-11 from step 3; add a sentence to both drafts naming the shared seam, or rewrite lines 83-84 to say a small shared hook is extracted first.

ZO-2 Major. Lines 30, 49, 64. 14 drafts (16 findings) + 8 covered rows + 11 excluded = 35, not 31. Verified: the comparison has exactly 31 "Missing in Swift" headings. Overlaps: crash recovery (OC-14 and OC-90), multi-window (covered row and OC-90), provider OAuth (covered row and OC-90), unread/inbox (OC-03 and ZP-02 row), tab strip (OC-04 and ZP-21 row); deep link is folded into the multi-window row. Fix: add a reconciliation note under line 28; fix OC-90 "eleven non-drafted findings".

ZO-3 Major. OC-05 lines 19, 38 vs TurboSparkApp.swift lines 445 and 453: Cmd+[ and Cmd+] are bound to Previous Chat and Next Chat. Index does not flag it. Fix: add a dependency bullet; OC-05 picks other chords or re-homes chat cycling.

ZO-4 Major. OC-04 line 56 claims Cmd+W (and Option+arrow); OC-01 lines 57-58 refuse Cmd+W as system-reserved. Index lines 79-80 say only "build OC-01 first". Fix: state the policy (first-party built-ins exempt, or OC-04 uses another close chord).

ZO-5 Major. OC-03 lines 44-46 claim ZP-02's priority order but ZP-02 line 102 is pending > unread > running > default, so Finished outranks Working; OC-03 puts Working before Finished. Index repeats the pairing (lines 58, 76-78). Fix: reorder OC-03 or state section order is display order only.

ZO-6 Major. Step 2 vs lines 76-78. "Sequence ZP-23 first" for OC-03, yet step 2 places OC-03 early with no gate, and OC-03 is opt-in until ZP-23. Steps never mention ZP-02, ZP-21, ZP-23. Fix: add the ZP prerequisites to the steps or mark OC-03 usable before ZP-23.

ZO-7 Major. Line 54. "1 diagnostics export" is comparison section 7 (heading line 387), not section 1. Fix: change to 7.

ZO-8 Nit. Line 41 and step 1 (line 93). OC-13 "small, self-contained" touches a new crates/server version.rs, a route handler, FFI generator and header, TurboSparkServer.swift, ServerPaneView and tests. Fix: label small-medium and note the server handler.

ZO-9 Nit. Lines 85-87, 42. OC-12 does not reference OC-13; "new Rust crate" pre-empts OC-12's open question (crates/acp vs crates/server/src/acp). Fix: "new crate or server submodule (open)"; add the reference to OC-12 or drop the claim.

ZO-10 Nit. Lines 34-47. Size labels are the index's own; no sibling states a size. Fix: call them index estimates.

ZO-11 Nit. No note on hot-file overlap: AppModel+Tools.swift is edited by OC-02, OC-09, OC-10, OC-11; TurboSparkApp.swift, KeyboardShortcutCatalog, RootView by OC-01, OC-04, OC-05, OC-06. OC-05 history vs OC-04 tab selection interaction is unstated. Fix: one merge-order line.

ZO-12 Nit. Sequencing is by size, not value; no go/no-go gate; OC-12 and OC-08 are the weakest-justified. Fix: say drafts are proposals and name first cuts.

ZO-13 Nit. Lines 129-131. Branch name feat/classlru-replay-arm is not in the comparison; commit 7ce35879b8 and date 2026-10-31 are confirmed. Fix: drop or cite the branch.

Verified OK: titles and numbering match sibling H1s; in-workspace links resolve; constraints match the ZP index; OC-90's eleven items match its table; OC-07 import-step handling matches. Standard clauses on root problem, YAGNI and codebase reuse mostly do not apply to an index. The .kiro links are unverifiable.

## Inventive catalyst (IC)

IC-1 Major. Lines 83-84, 95, 99. Same sequencing contradiction as ZO-1. Plus: the formatter rewrites the file in place, so diagnostics computed before formatting carry stale ranges; format should run before diagnose, which makes "OC-10 first" probably wrong at runtime. Fix: a tiny shared ordered post-edit stage list (format, then diagnose) as an early step, or rewrite the sentence so OC-11 lands its stage first.

IC-2 Major. Lines 73-78, 93-95. The index treats OC-02 and OC-03 as gated on ZP drafts, but the drafts say soft. OC-03 line 47 says Needs you is derivable today, no ZP-02 dependency, and reads ZP-02 fields through a protocol; OC-02 fires from AppModel lifecycle hooks and never reads ZP-02 state ("ZP-02, when built"). The claim that OC-02 "consumes the same state" (line 73) is unsupported. Also step 2 labels OC-03 "app-local" while line 76 says sequence ZP-23 first. Fix: say "soft dependency, no gating"; mark ZP-02 and ZP-23 as enhancements.

IC-3 Major. Lines 93-100. OC-04 (step 5) needs ZP-21 and ZP-23, all ZP drafts are still Draft, and the sequence has no slot for them. Fix: add an "External prerequisites" line listing ZP-02, ZP-21, ZP-23 and the shell-hardening spec items, stating which OC steps are blocked versus enriched.

IC-4 Nit. Lines 30, 49, 62-64. Same tally mismatch as ZO-2 (three overlaps by this reviewer's count). Fix: one reconciliation line.

IC-5 Nit. Dependency bullets omit couplings the drafts state: OC-02 depends on the dock-badge state in the shell-hardening spec (OC-02 lines 69-71); OC-14 on the spec's diagnostics bundle (OC-14 line 24). Fix: list them.

IC-6 Nit. Lines 85-87. OC-12 contains no OC-13 reference (same as ZO-9).

IC-7 Nit. Lines 34, 95. OC-14 is "small-medium" in the table, "medium" in step 3. OC-08 (large) is ordered before OC-10 (large) without rationale. Fix: align labels or add rationale.

IC-8 Nit. Line 34. The row says "hang detection" while the draft title is "Engine stall watchdog". Fix: one name.

Simplifying transform: replace dependency prose with two explicit small interfaces (post-edit stage list, attention-state protocol) so steps 2, 3, 5, 7 are independent of unscheduled outside drafts.
