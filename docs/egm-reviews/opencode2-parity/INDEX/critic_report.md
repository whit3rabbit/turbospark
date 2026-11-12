# Stage 7 report: critic review (INDEX.md, one round, no remediation)

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All three completed. Standard: feature-spec-standard for DC and ST; template sections (Problem Statement, Alternatives, Proposed Architecture, CUJs, per-feature Risks) do not apply to an index and were not flagged. Convergence was not assessed (one round).

## Design critic (DC)

Checked sound: the comparison has exactly 31 "Missing in Swift" headings (summary table line 515); each maps to a draft, a covered row or an OC-90 row; 14 drafts cover 16 findings.

DC-1 Major. Lines 30, 49-69. Counts reach 35 (16 + 8 + 11) with no arithmetic. Overlaps: multi-window (53, 64), provider OAuth and API key (59, OC-90 line 23), crash finding split OC-14 / OC-90, unread and tab strip (58, 60 vs 44-45). Fix: one reconciliation table, each of 31 findings in one primary row.

DC-2 Major. Lines 95, 99. OC-11 is in step 3 and step 7, before OC-10's seam. Fix: remove from step 3.

DC-3 Major. Lines 83-84. The OC-10 to OC-11 seam is not in the siblings (OC-11 never mentions OC-10 or a seam; OC-10 never mentions OC-11). Fix: add a shared-seam sentence to both drafts, or soften the index to "intended".

DC-4 Major. Lines 85-87. OC-12 does not reference OC-13. Fix: add the reference in OC-12 or drop the requirement.

DC-5 Major. Lines 76-78 vs step 2 (94). "Sequence ZP-23 first" for OC-03 contradicts OC-03 lines 69-71 and 131-132 (opt-in, works before and after ZP-23). Fix: "OC-03 can ship first (opt-in); ZP-23 makes Working rows common".

DC-6 Major. Lines 89-100. External gates (ZP-02 for OC-02, ZP-21 and ZP-23 for OC-04, shell-hardening import wizard for OC-07) are not on the timeline. Fix: add a "requires" column.

DC-7 Major. Lines 79-80. OC-04 line 56 claims Cmd+W; OC-01 lines 57-58 refuse it as system-reserved, so "chords join OC-01's override store" cannot hold. Fix: state Cmd+W is a fixed binding owned by OC-04 and align OC-01's reserved list.

DC-8 Major. Lines 47, 93. OC-05 (lines 19, 38) treats Cmd+[ and Cmd+] as free; they are bound to Previous/Next Chat at TurboSparkApp.swift lines 445 and 453 (confirmed by the orchestrator). OC-05 is step 1, before OC-01. Fix: add a program-level dependency and have OC-05 resolve it first.

DC-9 Major. Lines 58, 76-78. OC-03 orders Working before Finished; ZP-02 line 102 is pending > unread > running. OC-03 line 46 claims the same order. Fix: record one canonical order and correct OC-03.

DC-10 Nit. Line 30 and 34-47. Finding labels are comparison section numbers, never stated. Fix: legend sentence.

DC-11 Nit. Lines 49-60. 9a rows are "covered elsewhere" and also drafted (44-45). Fix: mark "shared".

DC-12 Nit. Line 53. Deep link finding is folded into the multi-window row, unnamed. Fix: name it.

DC-13 Nit. Lines 53, 55, 109. .kiro links unverifiable.

DC-14 Nit. Lines 116-120. Re-verify "none drafted" for LAN against OC-12 and the daemon mention at line 86.

DC-15 Nit. Lines 9-11, 127. Audit 2026-10-31 on a feature branch vs draft 2026-11-11; not stated whether re-run against main.

## Structural (ST)

ST-1 Major. Lines 93-99 vs 83-84. Same sequencing contradiction as DC-2.
ST-2 Major. Lines 3-5, 28-69. Counts do not reconcile (35 vs 31). Same as DC-1.
ST-3 Major. Lines 34-47. Finding column uses section numbers that repeat (two "3", four "8"); the 16-finding count is checkable only from Size notes. Fix: name the section, add a findings-count column.
ST-4 Major. Lines 53, 55 (and 81-82). The .kiro/specs links resolve to nothing (only audio-workspace exists); the import wizard wording suggests the spec may not exist yet. Fix: confirm on another branch or mark "planned, not yet in tree".
ST-5 Nit. Line 95. Step 3 labelled "medium" but holds OC-14 and OC-11 (small-medium) and OC-02 (medium); step 1 holds only "small". Fix: align labels.
ST-6 Nit. Lines 76, 94. OC-03 in step 2 with no ZP-23 note. Same as DC-5.
ST-7 Nit. Line 104. The zcode-parity link text wraps mid-link. Fix: keep on one line.
ST-8 Nit. Lines 11, 131. Draft date stated twice.
ST-9 Nit. Line 21. No explicit range statement "OC-01 to OC-14 plus OC-90". All files exist.
Verdict: heading hierarchy and tables are clean; the other in-workspace links resolve.

## Style (SY)

SY-1 Major. Lines 3-4, 30, 64. 31-finding accounting does not add up. Same as DC-1.
SY-2 Major. Lines 95, 99, 76-78. Sequencing contradicts dependency list. Same as DC-2 and DC-5.
SY-3 Major. Lines 33-47, 93-100. Size notes are undefined and inconsistent across table and steps (OC-03, OC-11, OC-14, "medium-large"). Fix: define the scale once, use identical labels, separate Size and Note columns.
SY-4 Major. Lines 6-7, 98. "Fit a local-first, macOS-only, native-Swift app" is not testable. Fix: point to the offline and program constraints as the test.
SY-5 Nit. Line 38. "pure integration" is unprovable.
SY-6 Nit. Line 100. "largest surface" is subjective; OC-08 and OC-10 are also large.
SY-7 Nit. Lines 33-47, 51-60. Finding labels undefined; "9a" unexplained; table order follows no rule.
SY-8 Nit. Lines 34-110. Undefined acronyms: LSP, ACP, PTY, MSRV, EGM, YAGNI; "ten localization gates" has no pointer.
SY-9 Nit. Lines 85-87. "once a daemon-based transport ever exists" is speculative hedging. Fix: delete or state a rule.
SY-10 Nit. Lines 116-120. "User directive 2026-11-11" cites no record; "LAN-facing choices (none drafted)" is unclear. Fix: "No drafted feature uses a LAN or hosted service."
SY-11 Nit. Lines 64-69. "YAGNI, reopenable" and "docs task with a concrete recommendation" are vague. Fix: name the trigger or link the OC-90 section.
SY-12 Nit. Lines 112-114. "Local green is not release evidence" is a slogan.
SY-13 Nit. Lines 127-131. Provenance redundant; "per repo rules" names no rule; "re-check a link" is ambiguous. Fix: rewrite as suggested.
SY-14 Nit. Line 90. "Small, independent wins first" conflicts with steps 2-3. Fix: "Self-contained items first".
No em dashes found.
