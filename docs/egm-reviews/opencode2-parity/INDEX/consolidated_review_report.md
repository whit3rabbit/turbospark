# Consolidated review report: OpenCode2 parity INDEX

Target: docs/designs/opencode2-parity/INDEX.md (full file, 133 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. Every reviewer completed on the first spawn; none failed.
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.
Standard applicability: this is an index, so Problem Statement, Alternatives, Proposed Architecture, CUJs and per-feature Risks of feature-spec-standard do not apply; review concentrated on accuracy against siblings, sequencing, consistency, completeness. The .kiro links (lines 53, 55, 81-82) do not resolve in this worktree (only .kiro/specs/audio-workspace exists) and were not verified.

Counts after deduplication: Blocker 1, Major 10, Nit 11.

Verified by me against the files: the comparison has exactly 31 "Missing in Swift" headings; OC-90 has 11 rows; titles and numbering match sibling H1s; Cmd+[ and Cmd+] are bound to Previous/Next Chat at swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift lines 445 and 453; OC-04 line 56 claims Cmd+W while OC-01 lines 57-58 refuse it; OC-03 lines 44-46 order Working before Finished while ZP-02 line 102 is pending > unread > running.

## Blocker

B1. Lines 83-84, 95, 99. Step 3 schedules OC-11 "after OC-10's seam" but OC-10 is step 7, and step 7 lists OC-11 again. Neither OC-10 nor OC-11 names a shared seam or mentions the other. Fix: remove OC-11 from step 3, keep both in step 7, and soften "land first" to "intended" (note format should run before diagnose). [ZO-1, IC-1, CO-4, DC-2, DC-3, ST-1, SY-2, S8-1]

## Major

M1. Lines 3-5, 28-69. Coverage does not reconcile: 16 + 8 + 11 = 35, not 31. Split across tables: crash recovery, multi-window, provider OAuth, unread, tab strip; deep link is hidden in the multi-window row; OC-90's "eleven non-drafted" is partly drafted. Fix: one sentence under line 28 listing the overlaps and the arithmetic. [ZO-2, IC-4, CO-1, CO-3, CO-12, DC-1, DC-11, DC-12, ST-2, SY-1, S8-2]

M2. Lines 47, 93 vs OC-05 lines 19, 38. OC-05 calls Cmd+[ and Cmd+] free; they are bound to Previous/Next Chat (TurboSparkApp.swift 445, 453), and OC-05 runs in step 1. Fix: add a dependency bullet; OC-05 picks other chords or re-homes chat cycling before shipping. [ZO-3, DC-8, S8-4]

M3. Lines 79-80 vs OC-04 line 56, OC-01 lines 57-58. OC-04 claims Cmd+W (and Option+arrow) that OC-01 refuses as reserved. Fix: bullet stating the policy; resolve in the drafts. [ZO-4, DC-7, S8-4]

M4. Lines 58, 76-78 vs OC-03 lines 44-46, ZP-02 line 102. OC-03 claims ZP-02's priority order but puts Working before Finished. Fix: correct OC-03 (or say section order is display order); index must not assert a shared order. [ZO-5, DC-9, S8 move-to-drafts]

M5. Lines 73, 76-78, 94. OC-02 and OC-03 are soft-dependent on ZP-02 and ZP-23 (OC-03 line 47, 69-71, 131-132; OC-02 never reads ZP-02 state), yet the index says "sequence first" and "consumes the same state", and puts OC-03 in step 2 as "app-local". Fix: "soft dependency, no gating"; drop "consumes the same state". [ZO-6, IC-2, CO-5, DC-5, ST-6, SY-2, S8-3]

M6. Lines 89-100. External prerequisites (ZP-02, ZP-21, ZP-23, shell-hardening import wizard) are not on the timeline and none of the ZP drafts is built. Fix: one line stating they block only OC-04's tab strip and OC-07's import step. [IC-3, CO-6, DC-6, S8-3]

M7. Lines 42, 85-87. "OC-12's docs must reference OC-13" has no counterpart in OC-12, rests on a speculative daemon transport, and "new Rust crate" pre-empts OC-12's open crate-vs-server-submodule question. Fix: delete the bullet; "new crate or server submodule (open)". [ZO-9, IC-6, CO-7, DC-4, S8-5]

M8. Lines 53-55, 81-82. The desktop-shell-hardening and inspector-terminal specs are not in this tree, so "covered elsewhere" and the import-step handoff are unverified; line 54 "1 diagnostics export" is comparison section 7. Fix: mark "spec not in this tree; confirm"; correct the number. [ZO-7, ST-4, DC-13, S8-6]

M9. Lines 30, 33-60. Finding labels (1, 3, 4, 7, 8, 9a) are comparison section numbers, never stated, and repeat across rows. Fix: one legend sentence. [CO-2, CO-13, DC-10, ST-3, SY-7, S8-7]

M10. Lines 33-47, 93-100. Size labels are undefined index estimates and inconsistent between table and steps (OC-14, OC-11, OC-03, OC-13 "small" but spans Rust and Swift). Fix: mark them "index estimates" and align OC-14 and OC-13 labels; do not add a scale. [ZO-8, ZO-10, IC-7, ST-5, SY-3]

(Also raised, advised for the draft, not the index: OC-10 vs the "no Python or Node runtime" constraint; optional clause "language servers are user-installed, not bundled". [CO-8])

## Nit

N1. Lines 126-131. Branch feat/classlru-replay-arm is not in the comparison; "per repo rules" names no rule; "re-check a link" is ambiguous; dates repeated. Fix: drop the branch, cite verification.md, one date. [ZO-13, CO-11, DC-15, ST-8, SY-13, S8-8]

N2. Lines 6-7, 98. "Fit a local-first, macOS-only app" is not testable. Fix: point to the offline and program constraints as the test. [SY-4]

N3. Lines 116-120. "LAN-facing choices (none drafted)" is unclear and the directive has no cited record. Fix: "No drafted feature uses a LAN or hosted service." [CO-10, SY-10, DC-14]

N4. Lines 1-12, 21, 108-114. Undefined terms (EGM, ZCode, LSP, ACP, PTY, MSRV, YAGNI, dock badge spec, "ten gates"). Stage 8 advises link at first use, no glossary. [CO-9, SY-8]

N5. Lines 34, 53-55. OC-14 is "hang detection" in the row but "Engine stall watchdog" in the draft; OC-02 and OC-14 depend on the shell-hardening spec (dock badge, diagnostics bundle) but only OC-07's import step is listed. Fix: one name; optional clause each. [IC-5, IC-8]

N6. Lines 38, 100, 112-114, 64-69. "pure integration", "largest surface", "Local green is not release evidence", "YAGNI, reopenable" are subjective or slogans. Stage 8 calls these style churn. [SY-5, SY-6, SY-11, SY-12]

N7. Line 90. "Small, independent wins first" conflicts with steps 2 and 3. Fix: "Self-contained items first". [SY-14]

N8. Hot-file overlap (AppModel+Tools.swift, TurboSparkApp.swift, KeyboardShortcutCatalog) and no go/no-go. Stage 8 advises dropping. [ZO-11, ZO-12]

N9. Line 104 link text wraps mid-link; line 131 repeats the line 11 date; no range statement for OC-01 to OC-14 plus OC-90. [ST-7, ST-8, ST-9]

N10. Line 21 and 4-5. Add one sentence: all drafts are proposals; ZP and .kiro items may be unbuilt. [CO-6, DC-15]

N11. Line 59 and OC-90. Provider OAuth and API-key path appear in both tables (covered in M1). Line 60 and 58: 9a rows appear as both covered and drafted; mark "shared". [DC-11]

## Dropped or deferred per Stage 8 (scope-adding, not carried)

Glossary; size scale and Size/Note columns; per-finding reconciliation table; status or requires columns and slip rules; shared post-edit stage list and attention-state protocol (IC simplifying transform); hot-file merge order; first-cuts naming; style rewrites in N6.

## Traceability

- zoom_out_report.md: ZO-1 to ZO-13, IC-1 to IC-8. Feeds B1, M1-M10, N1, N5, N8.
- comprehension_report.md: CO-1 to CO-13. Feeds B1, M1, M5-M7, M9, N1, N3, N4, N10.
- critic_report.md: DC-1 to DC-15, ST-1 to ST-9, SY-1 to SY-14. Feeds all.
- intermediate_zoom_out_report.md: S8-1 to S8-8 confirm B1, M1-M9 and set the drop list above.
- Merge notes: the OC-11 sequencing finding from all stages merged into B1; count findings merged into M1.
