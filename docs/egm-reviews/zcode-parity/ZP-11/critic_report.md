# Stage 7: Critic report (ZP-11)

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All completed. One round only (no remediation, so convergence was not assessed). Standards: feature-spec-standard for design and structural.

## Design (DC)

- DC-1 Blocker, 103, 110. Both alternatives rejected "by author" with no technical reason or link. Fix: state concrete trade-offs (PencilKit data model mixing, usability on macOS) or link the decision.
- DC-2 Blocker, 38-41, 61-62, 72-93, 115. Flattened PNG to composer path is undesigned. AppPromptAttachment (State/AppChat.swift:114) PreviewKind has pdf, image, text; no named API (AttachmentImporter, AppModel+Files), "attachment folder" undefined, ImagePreviewView lacks composer access, no cleanup lifecycle. Fix: add a data-flow subsection naming the model API, the view-to-composer callback, and the exported file lifecycle.
- DC-3 Major, 23-34, 72-79. Current view is SwiftUI ScrollView with scaledToFit (FilePreviewLeaves.swift:22, called from FilePreviewView.swift:161); NSScrollView plan is a rewrite into NSViewRepresentable; hosting, gestures, coordinate conversion undesigned. Fix: choose NSViewRepresentable plus AppKit annotation view and state magnification-to-image-space mapping.
- DC-4 Major, 31-33, 80-83, 91-93. No annotation data model; pen needs point list, text needs anchor, font size, wrapping; undo granularity of a drag-move ambiguous. Fix: define the annotation enum and what one history entry is.
- DC-5 Major, 38-40. Overlay-metadata export has no format, consumer, or test. Fix: delete from v1 or specify.
- DC-6 Major, 117-119. 60 fps via per-size-bucket raster has no bucket definition or memory bound; no image size cap; huge images and export could exhaust memory or exceed model image limits. Fix: set a max pixel dimension, downsampling and export cap.
- DC-7 Major, 36-37, 48-52, 64-70. History scope inconsistent ("transform and annotation" vs annotations only); persistence "lean" is not a decision. Fix: annotations only; decide session-only with a reason.
- DC-8 Major, 112-121. Thin risks: decode failure, EXIF orientation, wide-gamut color in flatten, Retina scale vs natural pixels, animated GIF source, huge export, failed write. Fix: add Risks and Mitigations.
- DC-9 Nit, 131, 135-137. Create or Modify circular with item 4. Fix: decide move or in place.
- DC-10 Nit, 148-149, 153-157. "21 languages" vs INDEX "ten gates"; mutation check covers mapping only. Fix: verify count; add immutability mutation.
- DC-11 Nit, 114, 120-121. Pointer-drag tools have no keyboard path to place or move annotations. Fix: add one or note as limitation.

## Structural (ST)

Heading hierarchy clean; Goals, Non-Goals, Problem, Architecture, Alternatives present.

- ST-1 Major, 95-165. No Risks and Mitigations section (standard requires it); NFR lacks failure modes. Fix: add the H2.
- ST-2 Major, 103, 110. "Rejected by author" with no link breaks the Alternatives rule; "the loop is the feature" is a product claim. Fix: concrete trade-offs.
- ST-3 Major, 15-19, 31-40, 80-81. Tool list inconsistent (Overview arrow/box/text; Goals add pen; text "optional" but CUJ 1 needs it). Fix: one list.
- ST-4 Major, 38-40, 86-89. Metadata export in Goals, absent from plan and tests. Fix: design or move to Non-Goals.
- ST-5 Major, 33, 91, 133. Natural-pixel vs normalized coordinates; NSScrollView magnification reconciliation missing. Fix: one canonical space and conversion.
- ST-6 Major, 125-149, 157. Target files omit composer handoff, localization catalog, tool palette file. Fix: add or name the reused API.
- ST-7 Major, 157. "21 languages" vs INDEX line 116 "ten localization parity gates". Fix: cite repo count.
- ST-8 Nit, 131. Create or Modify. Fix: decide.
- ST-9 Nit, 148-149. Mutation bullet breaks field pattern. Fix: labeled sub-field.
- ST-10 Nit, 48-52. Open question already assumed in Goals. Fix: mark default assumed.
- ST-11 Nit, 101. ZP-10 unlinked, not listed as related. Fix: link and note boundary.
- ST-12 Nit, 3-5. Finding 5.9 also cites ImageEditorToolbar.tsx and CONTRACT.md. Fix: add links.
- ST-13 Nit, 153-160. Verification has no pass criteria; no check for 60 fps. Fix: add measurable check or drop claim.

## Style (SY)

- SY-1 Major, 38-40, 61 vs 15-19. Export contract inconsistent (metadata mode; "optional flattening" vs default flatten). Fix: drop or define; one phrasing.
- SY-2 Major, 31-32, 117-119, 61. Unverifiable claims (no quality loss, 60 fps, track exactly). Fix: give tolerance, image size, measurement method.
- SY-3 Major, 91-93 vs 31, 77. Coordinate space stated two ways. Fix: pick one.
- SY-4 Major, 20, 36-37, 61-63 vs 147. History scope contradiction. Fix: annotations only.
- SY-5 Major, 50-52. Open question lacks owner and conflicts with Goals. Fix: resolve session-only; sidecar to Non-Goals.
- SY-6 Major, 103, 110. "Rejected by author" is the whole rationale; slogan. Fix: concrete reason.
- SY-7 Nit, 30, 81. Tool list mismatch. Fix: same four everywhere.
- SY-8 Nit, 98-99, 109. Loose claims ("free drawing and erasing", "zero code"). Fix: tighten.
- SY-9 Nit, 23. Current behavior has no source reference. Fix: cite FilePreviewLeaves.swift.
- SY-10 Nit, 131. Create or Modify. Fix: decide.
- SY-11 Nit, 148-149. Mutation check phrased as instruction. Fix: reword as a test description.
- SY-12 Nit, 156. design-check qualifier unexplained. Fix: explain or drop.
- SY-13 Nit, 157. 21 vs ten. Fix: cite.
- SY-14 Nit, 61, 158-160. CUJ assertions not mapped to named tests. Fix: map.
- SY-15 Nit, 114-116. Read-only point repeated (16-17, 62, 146-147). Fix: state once.
- SY-16 Nit, 3, 5. Link to external main can move; INDEX pins a60ba6a7. Fix: pin.
- SY-17 Nit, 120-121. Accessibility has no verification item. Fix: add or call it intent.
