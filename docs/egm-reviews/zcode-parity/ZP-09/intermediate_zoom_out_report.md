# Stage 8: Intermediate zoom out report (ZP-09)

Reviewer: egm-zoom-out (S8). Completed. Target unedited since Stage 5, so every Stage 5 finding is still open. Re-verified: OfficeArchive.swift is a read-only unzip wrapper; no OfficeRenderModel, bundled-skills directory, artifact folder symbol, or zip writer exists in swift/.

## A. Stage 5 findings still open

- S8-B1 Blocker, lines 105, 118-119. Writer half and hardened zip access do not exist; three of four writers depend on it. Fix: one in-process OOXML package writer in the file list with zip strategy and size limits. [ZO-1, IC-1, DC-1, ST-2]
- S8-M1 Major, lines 99-102, 141. Agent-facing interface and ladder position undefined. Fix: one document-write tool, bounded JSON spec per format, stated ladder rung. [ZO-2, IC-2, CO-1, DC-2]
- S8-M2 Major, lines 45, 74, 141, 168. Artifact folder and panel assumed; ZP-07, ZP-08, ZP-19 prerequisites unstated. [ZO-3]
- S8-M3 Major, lines 33-37, 56-63. Four writers in one draft, subset undecided. Fix: explicit v1 cut line, defer charts and shapes, evaluate NSAttributedString docx export. [ZO-4]
- S8-M4 Major, line 82. CUJ 2 VBA example incoherent. [ZO-5]
- S8-M5 Major, lines 41-44, 144-145. Postcheck location (tool-side vs agent-invoked) unstated. [ZO-6]
- Nits open: placement and ZP-07 root (ZO-7), garbled item 12 (ZO-8), "21 languages" and Create-entry UI (ZO-9), shell-out absence test (ZO-10), missing alternatives (ZO-11).

## B. Stage 6 and 7 fixes to accept (clarify or shrink scope)

Name and link dependencies (CO-3, DC-11); define tool and spec (CO-1, CO-2, DC-2); say what exists and what is new (CO-4); replace VBA example (CO-6, ST-7); reword PDFKit-adjacent (CO-11, DC-12, ST-6); fix item 12 (CO-13, DC-10, ST-9, SY-11); change the shell-out test to "writers do not spawn processes" (DC-9); technical reasons in alternatives and fix Alternative B (DC-5, IC-9); mark the assumed option in open questions (ST-5, CO-7); add a short Risks and Mitigations section (DC-8, ST-4).

## C. Proposed fixes that add scope; reject or defer

- IC-5 raw-part passthrough (new attack surface): shrink the subset instead.
- IC-3, DC-7, ST-11 independent validator, cross-checks, golden fixtures in real Office: keep one postcheck with named checks; drop "opens in Word" as a goal.
- IC-4 bounded agent retry: keep temp-then-move and diagnostics returned once.
- IC-6 generating skill limits from the tool schema: defer.
- DC-6 supply-chain analysis: one line, shell-out output is untrusted and no macros are written.
- DC-13 extra Non-Goals: at most "no macros".
- ST-10, ST-11, ST-12, SY-13: one-line mentions only.
- IC-8, DC-3 adding Create-entry UI: drop the UI and "21 languages" claims instead (ZO-9, SY-4).
- IC-7 detailed Core Text or NSPrintOperation path: one sentence plus CJK/RTL stated as a limit.
- Third-party OOXML library alternative: at most one line, rejected for dependency reasons.

## D. Added findings

- S8-M6 Major, lines 33-45, 94-116. Simplest root fix: cut v1 to docx and pdf plus xlsx cells without charts; pptx and charts are heaviest with no reuse path. Otherwise every fix above covers four formats.
- S8-M7 Major, lines 97-116. Writers described as tools, a layer, and pure functions. Fix: one tool dispatching to per-format pure functions returning part dictionaries, with one shared package writer.
- S8-N1 Nit, lines 25-29. No-Python/Node constraint cited without link. Fix: link .kiro/steering/roadmap.md.

## Verdict

Not ready. One Blocker and five open Majors. Fix the Blocker and the tool interface first, then apply Section B and reject Section C.
