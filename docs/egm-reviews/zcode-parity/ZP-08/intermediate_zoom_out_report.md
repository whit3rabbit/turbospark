# Stage 8: Intermediate zoom out report (ZP-08 office document rendering)

Target: full file (205 lines), unedited since Stage 5. Reviewer: egm-zoom-out (S8), completed. Inputs: zoom_out_report.md, comprehension_report.md, critic_report.md, INDEX.md. Spot check by grep confirmed `QuickLookPreviewView` at ArtifactPanelView.swift:294 and FilePreviewView.swift:175, and `AppArtifactRenderKind` at State/AppArtifact.swift:35.

Verdict: not passing. The document is untouched, so every Stage 5 blocker and major is still open. Later-stage fixes would grow a design the standard already calls too big.

## Still open from Stage 5

S8-A1. Blocker. Lines 16-17, 24-26, 173-183. Problem statement stale: `.opaque` already renders QuickLook inline; the "opaque external-opener leaf" does not exist. Root problem (QuickLook fidelity and navigation gaps) unstated and unmeasured. [ZO-1]

S8-A2. Blocker. Lines 129-136. QuickLook rejected "by author" with no data; line 135 fallback appears in no plan item. [ZO-2, DC-3, ST-2, ST-3, CO-7, IC-3]

S8-A3. Major. Lines 34-47, 92-112, 153-194. Three bespoke OOXML engines remain in scope while Non-Goals admit lower fidelity than QuickLook. [ZO-3, ST-8]

S8-A4. Major. Lines 114-116, 169-172. Reuse unverified; `OfficeArchive` per-entry `unzip -p` and cumulative extraction limits will be exhausted; item 6 undefined. [ZO-4, DC-1, CO-1, DC-2]

S8-A5. Major. Lines 181-184. Wrong enum path; Codable fallback and switch sites missing. [ZO-10, DC-4]

S8-A6. Major. Lines 58-64, 173-183. `FilePreviewView.swift` missing from file list; open question already answered. [ZO-9, SY-3, CO-2, DC-9, ST-4, SY-11]

S8-A7. Major. Lines 41-44, 99. PDF "fourth viewer" has no code change. [ZO-8, CO-3, DC-7, ST-5]

S8-A8. Major. Lines 55-56, 74-77. VBA notice cannot trigger for docx/xlsx/pptx. [ZO-7, DC-10, CO-6]

S8-A9. Major. Lines 138-146. 50 MB ceiling and "first N" caps unrelated to existing `DocumentTextExtractor.Limits`. [ZO-6, CO-5, DC-5, SY-6]

S8-A10. Major. Lines 45-46, 110-111, 190-193. Read-only invariant mis-specified; mtime/size test does not prove the claim. [ZO-5, CO-1, DC-2, SY-4]

S8-A11. Nit. Lines 146, 199. "21 languages" versus INDEX "ten localization parity gates". [ZO-13, IC-7, DC-8, ST-6, SY-16]

## Later-stage findings endorsed

S8-B1. Major. Lines 140-141, 114-115. Attacker-controlled `.rels` targets and external references have no rule. Fix: a short constraint that rels targets resolve only inside the archive and external links are never followed. [DC-1]

S8-B2. Major. Whole document. No Risks and Mitigations section. Keep it short, derived from the open findings. [ST-1]

S8-B3. Major. Lines 34-43, 52. Theme exclusion makes most real fills wrong; evidence that a native renderer costs more than assumed, supporting scope reduction over adding a theme map. [IC-4, DC-6, CO-6, SY-7]

## Proposed fixes that add scope the standard would reject

- Reject for v1: XPC helper (IC-2), snapshot plus reload store (IC-1), Fidelity-view toggle (IC-3, wrong answer to a self-inflicted problem), VoiceOver deliverable for custom views (DC-8, disappears if QuickLook stays the renderer), CJK goal (ZCode web-runtime concern).
- Constrain: fidelity matrix and corpus (DC-6) to a short table of QuickLook failures; degradation ladder (IC-5) to one rule: parse failure or cap hit falls back to QuickLook plus Model text; phasing (ST-8) to "one format or none until the QuickLook gap is measured"; extra alternatives (DC-3, ST-3) to two or three short ones (PDF conversion, keep QuickLook).
- Accept: rels rule, short risks section, gate named by name, heading and formatting fixes, INDEX spec links, dropping the CJK claim.

## Simplest path

1. Rewrite the Problem around measured QuickLook gaps on representative agent-written files. If no material gaps, close ZP-08 as already satisfied with a read-only mtime test and soften the INDEX dependency (line 81).
2. If gaps exist, scope v1 to the smallest addition (read-only test, size handoff, at most one native format, likely xlsx), with QuickLook as default and fallback.
3. Fix the file list: add FilePreviewView.swift, correct the AppArtifact.swift path, drop item 6 unless defined, add the Codable fallback to `.opaque`.
4. Cheap mitigations: rels rule, short risks section, gate by name, drop or specify the VBA notice, drop or define PDF, resolve the open question.

Net: remediation should shrink the document, not grow it.
