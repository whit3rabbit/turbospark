# Consolidated review report: ZP-08 office document rendering

Target: docs/designs/zcode-parity/ZP-08-office-document-rendering.md (full file, 205 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. Reviewers read ZCODE_COMPARISON.md lines 1-237 only (zoom-out also read 459-463), so ZCode claims were not checked against finding 5.7.
Source IDs: ZO and IC = zoom_out_report, CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 5, Major 12, Nit 11.

## Blocker

B1. Lines 16-17, 24-26, 173-183. Problem statement is stale: `.opaque` already renders QuickLook inline plus an "Open with default app" button (ArtifactPanelView.swift 291-308; FilePreviewView.swift 168-202). The "opaque external-opener leaf" does not exist. Fix: restate the problem around measured QuickLook gaps on representative agent-written files, or close as already satisfied. [ZO-1, S8-A1]

B2. Lines 129-136, 82-90. QuickLook rejected "by author" with no evidence; the "fallback for unsupported elements" (line 135) has no plan item. Fix: evaluate current QuickLook output first; keep it as default and fallback, or delete line 135. [ZO-2, IC-3, DC-3, ST-2, ST-3, CO-7, SY-8, S8-A2]

B3. Lines 114-115, 140-141, 169-172. Security reuse claim does not match `OfficeArchive` (per-entry `unzip -p`, temp copy, cumulative text-extraction limits that rendering will exhaust); `.rels` targets and external references have no rule. Fix: add render limits with numbers and a rule that rels resolve only inside the archive and external links are never followed. [DC-1, ZO-4, S8-A4, S8-B1]

B4. Lines 45-46, 110-111, 169-172, 193. Read-only invariant mis-specified: no handle API exists, mtime/size test misses other writes and size-preserving writes, extension routing for `.docm/.xlsm/.pptm` and legacy formats undefined. Fix: define it as no writes to the source path and no network or process spawn except the sanctioned unzip; list routed extensions. [DC-2, ZO-5, CO-1, SY-4, SY-5, S8-A10]

B5. Whole document (risk section missing near line 138). No Risks and Mitigations section, required by the standard. Fix: add a short section (corrupt or encrypted files, memory blowup, divergence from Office, read-only failure, external references). [ST-1, S8-B2]

## Major

M1. Lines 34-47, 92-112, 153-194. Three bespoke OOXML engines in one plan, lower fidelity than QuickLook by the draft's own Non-Goals, no phasing. Fix: start with one format or none until the QuickLook gap is measured. [ZO-3, ST-8, S8-A3]

M2. Line 181, item 9. Wrong enum path (real: State/AppArtifact.swift:35); Codable String enum needs unknown-value fallback to `.opaque`; `displayLabel` and ArtifactPanelView.swift:291 switches not listed. Fix: correct path, list switch sites, add fallback. [ZO-10, DC-4, SY-3, S8-A5]

M3. Lines 58-64, 173-183. `FilePreviewView.swift` (real routing site, `hasRenderedAlternative`) missing from file list; Model-text toggle is in CUJ 1 but in no plan item; open question already answered. Fix: add file, name the toggle constraint, resolve the question. [ZO-9, CO-2, DC-9, ST-4, ST-5, SY-11, S8-A6]

M4. Lines 41-44, 99. PDF "fourth viewer" has no file, change or test. Fix: remove or state no code change. [ZO-8, CO-3, DC-7, ST-5, S8-A7]

M5. Lines 55-56, 74-77. VBA notice cannot trigger for `.xlsx`; "disabled-mode" unexplained. Fix: drop it or name macro-enabled types in routing. [ZO-7, DC-10, CO-6, S8-A8]

M6. Lines 138-146. 50 MB ceiling, "first N" and "cap row" have no numbers, no tie to `DocumentTextExtractor.Limits`, no parse timeout or cancellation; 50 MB compressed can expand 10 to 100 times. Fix: concrete caps per format and a timeout. [ZO-6, CO-5, DC-5, SY-6, S8-A9]

M7. Lines 34-43, 52-56. Theme exclusion breaks "cell styles" and colors in the common case; no fallback stated; merged cells, hidden sheets, RTL not covered; the CJK claim is never planned. Fix: state theme-color fallback and a short list of known QuickLook-versus-native gaps; drop or scope CJK. [DC-6, IC-4, CO-6, SY-7, S8-B3]

M8. Lines 126-127, 135-136, 118-136. Alternatives attributed to "the author"; QuickLook and macOS-version claims uncited; ZP-06 unlinked; viable alternatives missing. Fix: concrete trade-offs; add PDF conversion and keep-QuickLook (two or three short ones). [DC-3, ST-2, ST-3, SY-8]

M9. Lines 22-30, 4-6, 123-124. Unsupported claims and missing links: review loop depends on git pane and rewind draft (unnamed, INDEX lines 57-59 give paths), ZCode "CJK font runtime" uncited, ZP-06 and ZP-09 undescribed, circular justification with ZP-09. Fix: link the specs, cite CONTRACT.md sections or drop, state standalone value. [CO-4, DC-8, SY-1, SY-2, ST-7, ZO-14]

M10. Lines 143-145, Verification Plan. Failure handling is truncation and a size handoff only; encrypted, corrupt, legacy, mid-write files uncovered. Fix: one rule, any parse failure or cap hit falls back to QuickLook plus Model text, with failure fixtures. [IC-5, CO-8, DC-8, S8 constrain]

M11. Lines 146, 195-202. "21 languages" versus INDEX "ten localization parity gates"; `make design-check` and verification.md reference absent; no view-level tests. Fix: reference the gates by name, add design-check. [ST-6, DC-8, ZO-13, IC-7, SY-16, S8-A11]

M12. Lines 45-46, 110-112, 189-193, 80. Freshness: ZP-09 rewrites files while viewed; a half-written zip reads as corrupt; the mtime test proves only no viewer write. Fix: cover with the M10 fallback rule; the reviewer's snapshot and reload proposal (IC-1) and XPC helper (IC-2) are scope growth, rejected by Stage 8. [IC-1, IC-2, DC-8, S8-C1, S8-C2]

## Nit

N1. Lines 153-194. Target Files list formatting: items 3-5 lack sub-bullets, item 9 garbled "basename" parenthetical, item 10 indentation, "mutation checks" jargon and mis-nested, items 7-9 no test entry. Fix: normalize, say "negative tests". [ST-9, DC-11, SY-15, CO-10]

N2. Lines 22, 92. Headings differ from standard names. Fix: "Problem Statement", "Proposed Architecture". [ST-10]

N3. Lines 1-10. No link to INDEX.md or Swift sources; CONTRACT.md link not pinned to a commit; ZCODE_COMPARISON line 461 (DocxView.tsx) uncited. Fix: add links, pin commit. [ST-11, DC-12]

N4. Lines 94-112, 100-108 versus 169. Diagram misaligned; `OfficeXMLParsers` "extended" but not in the modify list. Fix: realign, state where new parsing lives. [ST-12, CO-11, SY-13]

N5. Lines 16-17, 29, 126, 148. "preview surfaces" undefined; "weakens", "easier", "where sensible" subjective. Fix: define or remove. [SY-9, SY-10, SY-14]

N6. Lines 12-31. Overview and Problem repeat the same point. Fix: merge. [SY-18]

N7. Lines 77-90. CUJ 2 omits cell comments; CUJ 3 "documented limits list" lacks assertions; frozen-pane mapping unspecified. Fix: add expected visible result. [CO-9, SY-17]

N8. Lines 52-56. "Documented limit" has no home; charts get a placeholder, footnotes nothing. Fix: point to the footer spec. [SY-12]

N9. Line 192. Zip-bomb test coverage claimed by reference to tests not shown to cover the new reads. Fix: name the test file. [SY-5]

N10. Line 51 and INDEX line 81. No forward link to ZP-09; QuickLook already covers the review need, so the INDEX dependency may be overstated. Fix: add a link, soften the INDEX claim. [SY-19, ZO-12]

N11. Lines 146-149. "tied to user font settings where sensible"; DESIGN.md not linked. Fix: name the setting or drop. [CO-10, SY-14]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-14, IC-1 to IC-7. Feeds B1-B4, M1-M4, M5, M6, M9, M11, M12, N6, N10.
- comprehension_report.md: CO-1 to CO-11. Feeds B2, B4, M3-M7, M9, M10, N1, N4, N7, N11.
- critic_report.md: DC-1 to DC-12, ST-1 to ST-12, SY-1 to SY-19. Feeds all Blockers, M1-M12, N1-N11.
- intermediate_zoom_out_report.md: S8-A1 to A11, S8-B1 to B3, S8-C1 to C10. Confirms every Stage 5 blocker and major still open; recommends rejecting XPC helper, snapshot store, Fidelity-view toggle, accessibility deliverable and CJK goal, and constraining the fidelity matrix, degradation ladder and phasing. Overall: remediation should shrink the document.
- Merge notes: IC-3 merged into B2 (fidelity toggle is the same gap as the unplanned fallback); IC-1 and IC-2 recorded as M12 with Stage 8 rejection noted; severity of DC-1 and DC-2 kept at Blocker; ST-1 is the only independent Blocker from the structural review.
