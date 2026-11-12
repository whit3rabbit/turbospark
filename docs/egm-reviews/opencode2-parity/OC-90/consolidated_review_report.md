# Consolidated review report: OC-90 excluded and decision findings

Target: docs/designs/opencode2-parity/OC-90-excluded-and-decision-findings.md (172 lines)
Preset: Feature spec only (Stages 5, 6, 7, 8; Stage 9 and semantic split skipped). FeatureRange = whole file, DetailedRange = N/A.
Stages: all ran, no reviewer failed. Stage 5 zoom-out and inventive-catalyst; Stage 6 comprehension; Stage 7 design, structural, style; Stage 8 zoom-out.
Reports in this folder: zoom_out_report.md, comprehension_report.md, critic_report.md, intermediate_zoom_out_report.md.

Applicability note: this is a decision log, not a feature design. Problem, Alternatives and Risks were applied in reduced form. Architecture, implementation plan, test plan and open questions do not apply. The .kiro/ links (lines 43, 140) and the claims resting on them are unverifiable in this checkout.

Verdict: no Blockers. Every exclusion verdict is defensible against the stated policy. The defects are in the framing: a false closing claim, a policy baseline and several premises that ZP-22 made stale, two reopen triggers that cannot fire, and one broken link. Table vs body is consistent on all 11 rows and "eleven" matches INDEX.md.

Counts: Blocker 0, Major 14, Nit 11.

## Source ID key

ZO = Stage 5 zoom-out, IC = Stage 5 inventive-catalyst, CO = Stage 6 comprehension, DE / ST / SY = Stage 7 design / structural / style, S8 = Stage 8 zoom-out. IDs refer to the numbered findings in each stage report.

## Blocker

None.

## Major

1. **Decision log claims are false or unclassified (166-168).** "Reopen conditions are recorded per finding above" holds only for multi-window, pairing, motion tuning and PTY daemon. "None permanent except OAuth" conflicts with WSL (N/A) and with policy-bound exclusions. Fix: add a Class/Reopen column (permanent by decision, N/A, policy-reopenable, docs task) and rewrite the log from it. Sources: ZO-1, ZO-2, IC-9, CO-13, DE-4, ST-2, SY-1, S8-M2.
2. **Broken link (113).** TurboSparkMotion.swift is repo-root-relative and does not resolve from this folder. Fix: `../../../swift/TurboSparkApp/Sources/TurboSparkApp/Theme/TurboSparkMotion.swift`. Sources: ZO-3, IC-10, CO-9, DE-1, ST-1, SY-4, S8-N1.
3. **Policy baseline contradicts adopted ZP-22 (8-9, 164-168).** "No cloud services or accounts" conflicts with hosted-API providers. Fix: state the narrow rule (no app-operated hosted infrastructure or identity; user-directed API calls with user keys allowed) once in the preamble. Sources: IC-6, CO-14, S8-M1.
4. **Cassette premise is stale (151-160).** "No provider HTTP traffic to record" ignores ZP-22's Swift SSE clients and their fixture needs. Fix: reword the reason and route replay fixtures to ZP-22. Sources: IC-1, S8-M4.
5. **Cloud usage premise is stale (99-106).** ZP-22 introduces third-party cost; ZP-90 line 131 already treats quota views as ZP-22 follow-ups. "Analog already exists" overstates (source rates Partial). Fix: cross-reference ZP-90 and ZP-22 follow-up and cite the actual surface. Sources: IC-4, S8-M4, ZO-5, SY-6.
6. **Crash reporting overclaim (49-59).** "No honest local analog short of a service" is false; local post-mortem capture needs no network. Fix: narrow to "no transmission" and say local capture is covered by OC-14 and the shell-hardening export. Sources: ZO-9, IC-2, SY-3, S8-M5.
7. **Multi-window reopen condition is unsatisfiable (21, 45-47).** Tabs and split panes never serve a second display. Fix: restate (for example, evaluate a pop-out tab host after ZP-21's context store lands) and label the row "excluded, partly covered". Sources: IC-3, S8-M3, ST-10.
8. **Pairing rationale omits the existing bind rule (87-97, 143).** "Loopback-only" ignores ZP-90's loopback-or-single-verified-Tailscale-IPv4 rule. Fix: cite it in the reopen note. Sources: IC-5, S8-M7.
9. **Translation governance is untracked and unenforced (121-134).** No owner, no acceptance criteria, gate-or-convention unstated, effort claim unsupported, CLDR is not a terminology authority. Fix: state the requirement as its own item with an owner, say gate or convention, drop "recommended, small" and the CLDR terminology claim. Sources: IC-7, SY-2, ZO-6, DE-note, ST-10, S8-M8.
10. **PTY daemon framing and trigger mismatch (30, 136-149).** Table says "after inspector-terminal", body says "if users ask"; persistence-versus-no-orphans is not binary (respawn with restored state). Fix: unify the trigger and note the respawn alternative. Sources: IC-8, S8-M6.
11. **Motion rationale is factually loose (108-119).** "Six-ish knobs" versus nine named animations; "dozen animated sites" counts a different quantity. Fix: state the real count and one measurable trigger. Sources: DE-2, ZO-4, IC-11, SY-5, ST-9, S8-N2.
12. **Document is not self-contained (3-9, 40, 78, 93-94).** Finding and Section numbers refer to the external report, table rows have no topic line, ZP-90 section numbers carry the SSH and pairing reasoning, and ZP-18, ZP-23 and "ZCode program" are unlinked or undefined. Fix: add a topic per row, note once that numbers refer to the comparison report, restate one reason sentence per ZP-90 citation, link ZP-18 and ZP-23. Sources: CO-1, CO-2, CO-3, CO-4, ST-6, ST-7, SY-7, S8-N4, S8-N5.
13. **OAuth permanence has no provenance link (63-72).** Rests on "author, 2026-11-11" and "recorded in ZP-90" with no link. Also clarify that multiple named API-key providers are not excluded and ZP-22 coverage of per-provider accounts is unbacked. Fix: link ZP-90 section 7.2 and state the account point. Sources: ST-3, S8-N6, S8-N8, DE-5, IC-12.
14. **Counts do not reconcile (3, 17-31).** "Eleven" counts sub-items with no stated rule; INDEX arithmetic is 16 drafted + 8 covered + 11 excluded = 35 against a goal of 31. Fix: add one line mapping the 31 findings to dispositions and noting overlaps. Sources: CO-7, DE-3, S8-N3.

## Nit

1. Crash paragraph "respectively" pairs three residuals with two owners (52-56). Fix: name OC-14 for hang detection and guarded relaunch, the spec for diagnostics export. Sources: CO-6, DE-7, ST-8, S8-N7.
2. Verdict labels inconsistent ("remote category", "remote control axis") (19-31). Fix: use a controlled set. Sources: SY-10, S8-N11.
3. Uncited claim that stubs and fixtures exist and Rust owns provider fixtures (155-157). Fix: add a path or drop. Sources: SY-13, S8-N10, ZO-9.
4. "The report itself says" and "legitimately" lack a report section cite (82-83, 105, 153). Fix: cite the section. Sources: SY-11.
5. Subjective phrasing (44-45 "existing strength", 94-95 "feasible engineering but deserves", 104 "honest"). Fix: state the behavior or decision rule. Sources: SY-8, SY-9, SY-6.
6. Intro is a run-on of four categories and omits the "docs task" class (2-9). Fix: split and add the fifth class. Sources: SY-14, ZO-7, ST-5.
7. "Tabs instead" (21) versus "tabs and split panes" (45-47); split panes unexplained. Fix: align wording. Sources: CO-8.
8. "Design-check ratchet" and "permission ladder" undefined; "whenever the app wants it" (66-70) gives no ZP-22 trigger. Fix: link or define. Sources: CO-4, CO-9, CO-11.
9. "The OpenCode package" and "cassettes" undefined (158-160); "21 languages" unlisted, "AGENTS.md" unexplained (127-134). Fix: name and define. Sources: CO-10, CO-12.
10. Status wording: "Draft" document calls ZP-22 "adopted" (12, 60-62); clarify designed versus implemented. Sources: CO-14.
11. No fixed per-finding fields or Risks content (what users lose per exclusion). Fix: use What it is / Why not drafted / Decision / Reopen condition / Cost of exclusion. Sources: ST-4, DE-note.

## Unverifiable

* .kiro/specs/desktop-shell-hardening and inspector-terminal links (43, 140) and the claims resting on them: single-instance lock, redacted bundle, PTY kill-on-quit, hidden pane keeps shell alive (141-145). Sources: CO-5, ZO-9, S8.

## Traceability

Stage 5: ZO-1..9, IC-1..12. Stage 6: CO-1..14. Stage 7: DE-1..7, ST-1..10, SY-1..14. Stage 8: S8-M1..M8, S8-N1..N12. Each consolidated item above lists its sources. Items with no IC or S8 counterpart (Nit 4, 8, 9, 10) come from comprehension or style only. No remediation was applied and the target file was not edited.
