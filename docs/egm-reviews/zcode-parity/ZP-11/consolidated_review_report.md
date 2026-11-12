# Consolidated review report: ZP-11 image preview annotation

Target: docs/designs/zcode-parity/ZP-11-image-annotation.md (full file, 164 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO/IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 10, Nit 10.
Note: S8 rates B1 and B2 as Major and Nit on its own axis; they stay Blocker here because Stage 7 (feature-spec standard) rated them so and the standard's Alternatives Rule is a hard fail.

## Blocker

B1. Lines 38-41, 61-62, 72-93, 114-116, 135-138. Export-to-composer path is undesigned and the "attachment folder" does not exist in the app sources; ImagePreviewView has no composer access. Fix: write the PNG through ManagedAssetStore, attach via AppModel.addPromptAttachment (State/AppModel+Chat.swift:430), pass an onSendToChat(Data) callback, delete the folder claim. [DC-2, ZO-1, CO-6, ST-6, S8-M1]
B2. Lines 103, 110. Both alternatives are "Rejected by author" with no technical reason or link; Alt B is a slogan. Fix: one concrete trade-off each (PencilKit data model mixing; external round trip loses context). [DC-1, ST-2, SY-6, S8-N11]

## Major

M1. Lines 15-19, 31-40, 80-81, 127-129. Tool set inconsistent (pen in Goals not Overview; text "optional" yet needed by CUJ 1). Fix: box, arrow, text as standalone tools; pen to Non-Goals. [ZO-2, IC-4, CO-1, ST-3, SY-7, S8-M3]
M2. Lines 16-18, 38-40, 86-89. "Original-plus-overlay metadata" export has no format, consumer or test. Fix: flattened only; move to Non-Goals. [ZO-3, CO-2, DC-5, ST-4, SY-1, S8-M2]
M3. Lines 31-33, 77, 91-93, 133. Natural-pixel vs normalized coordinates; magnification vs pixel space unreconciled; stroke and font units undefined. Fix: natural-pixel space from source dimensions, magnification as view transform, units in image pixels. [CO-4, ST-5, SY-3, IC-3, S8-M5]
M4. Lines 36-37, 48-52, 64-70. History scope contradicts CUJ 2; persistence open yet assumed session-only; undo granularity undefined. Fix: resolve session-only; history is annotations only; one entry per completed gesture. [IC-1, CO-3, CO-7, DC-7, SY-4, SY-5, S8-M4]
M5. Lines 31-33, 80-83, 91-93. No annotation data model (pen point list, text anchor, font size, wrapping). Fix: define a small annotation enum and history entry. [DC-4, CO-4]
M6. Lines 23-34, 72-79. Current view is SwiftUI ScrollView (FilePreviewLeaves.swift:22); NSScrollView plan is a rewrite with no hosting design. Fix: state NSViewRepresentable around NSScrollView and keep the ImagePreviewView(image:) signature. [DC-3, ZO-4, S8-N9]
M7. Lines 31, 117-119. Per-size-bucket raster and "60 fps" are unverifiable, undefined and contradict "no quality loss"; no image size cap. Fix: drop both claims, say measure first, add one max-pixel or export cap line. [IC-3, DC-6, CO-5, ZO-5, SY-2, ST-13, S8-N6, S8-N12]
M8. Lines 95-165 (missing section). No Risks and Mitigations section (standard requires it). Fix: add 3 to 4 lines (decode failure, failed write, oversize image, Retina scale vs natural pixels). [ST-1, DC-8, S8 scoped down]
M9. Lines 125-149, 139-141. Target files omit the composer handoff, localization catalog, and the second ImagePreviewView caller ArtifactPanelView.swift:281; "Create or Modify" circular with item 4. Fix: list both callers and the reused API; move the view to Files/ImagePreviewView.swift. [ST-6, ZO-6, IC-5, DC-9]
M10. Lines 38-40, 86-93, 144-147. Overlay and exporter are two renderers of one model; hash test proves determinism, not fidelity. Fix: one sentence stating a single shared draw(annotation, in: CGContext, scale:) path; skip golden-image infrastructure. [IC-2, S8 scoped down]

## Nit

N1. Lines 130-131. "Create or Modify" ambiguity. Fix: decide (see M9). [ZO-6, CO-10, DC-9, ST-8, SY-10]
N2. Line 157. "21 languages" conflicts with INDEX "ten localization parity gates". Fix: reference gates by name. [ZO-7, IC-6, DC-10, ST-7, SY-13, CO-11, S8-N8]
N3. Lines 144-150. PNG byte-hash not stable across OS versions; "mutation check" undefined, mis-nested, no immutability mutation. Fix: pixel equality; labeled sub-field; define. [CO-12, DC-10, ST-9, SY-11]
N4. Lines 97-106. ZP-10 unlinked and not listed as related; "Services" undefined. Fix: link ZP-10; say macOS Services menu. [CO-9, ST-11]
N5. Lines 3-5. Cites one ZCode file; finding 5.9 also cites ImageEditorToolbar.tsx and CONTRACT.md; link points at main. Fix: optionally add and pin a60ba6a7. [ST-12, SY-16]
N6. Lines 153-160. Verification lacks pass criteria; CUJ assertions not mapped to tests; design-check qualifier vague. Fix: map assertions to named tests; say "make design-check". [ST-13, SY-12, SY-14, CO-11]
N7. Lines 23, 98-99, 109. Loose or unsourced claims. Fix: cite FilePreviewLeaves.swift; tighten wording. [SY-8, SY-9]
N8. Lines 114, 120-121. No keyboard path to place or move annotations; accessibility has no verification item. Fix: note as v1 limitation. [DC-11, SY-17]
N9. Lines 57-62, 115-116. "Natural size" (pixels vs points) and "read-only" mechanism undefined; read-only point repeated four times. Fix: source pixel dimensions; say "never written", state once. [CO-13, CO-14, SY-15]
N10. Lines 3-5, 64-70, 131. Redo in Goals but never exercised; INDEX size note should stay "medium" after cuts. Fix: drop redo or test it. [S8-N13, S8-N14, CO-8 rejected by S8]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-8 (zoom-out), IC-1 to IC-6 (inventive catalyst). Feeds B1, M1-M3, M6, M7, M9, N1, N2, N5.
- comprehension_report.md: CO-1 to CO-14. Feeds B1, M1-M5, M7, N1-N4, N9.
- critic_report.md: DC-1 to DC-11, ST-1 to ST-13, SY-1 to SY-17. Feeds both Blockers and all Majors.
- intermediate_zoom_out_report.md: S8-M1 to S8-M5, S8-N6 to S8-N14. Recommends adopting ambiguity-removal fixes and rejecting IC-1 persistence, IC-2 golden images, IC-3 tiling, full DC-8 risk list, DC-11 keyboard path, CO-8 behavior summary, and cosmetic SY items.
- Merge notes: IC-1 is kept under M4 only as the Open Question resolution; its autosave subsystem is rejected by S8.
