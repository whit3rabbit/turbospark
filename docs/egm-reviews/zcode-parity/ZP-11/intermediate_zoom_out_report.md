# Stage 8: Intermediate Zoom Out report (ZP-11)

Reviewer: egm-zoom-out (completed). Input: full file, plus zoom_out_report.md, comprehension_report.md, critic_report.md. No Blockers on the zoom-out axis; it rates DC-1 and DC-2 as Major and Nit respectively on its axis (wiring choice with an existing answer, wording fix).

## Adopt (remove ambiguity or scope)

- S8-M1 Major, 135-138, 61-62, 114-116. Delete the invented "attachment folder"; route the PNG through ManagedAssetStore and AppModel.addPromptAttachment (State/AppModel+Chat.swift:430, precedent ElementPickerController.attachSelectedCandidate); view takes one onSendToChat(Data) callback. Covers ZO-1, DC-2, CO-6, ST-6.
- S8-M2 Major, 38-40, 16-18, 86-89. Cut the metadata export; flattened only; add to Non-Goals. Covers ZO-3, CO-2, DC-5, ST-4, SY-1.
- S8-M3 Major, 31-33, 80-81, 16-17, 127-129. One tool set: box, arrow, text (text standalone); pen to Non-Goals; drag-after-placement stays. Covers ZO-2, CO-1, ST-3, SY-7, IC-4.
- S8-M4 Major, 48-52, 64-70, 36-37. Resolve to session-only; history is annotations only; one entry is one completed gesture including a drag-move. Covers ZO-8, CO-3, CO-7, DC-7, SY-4, SY-5.
- S8-M5 Major, 91-93, 31, 77. One coordinate authority: natural-pixel space from source dimensions; magnification is a view transform; stroke and font units are image pixels. Covers CO-4, ST-5, SY-3.
- S8-N6 Nit, 117-119. Remove per-size-bucket raster and the 60 fps number; say measure first. Covers ZO-5, CO-5.
- S8-N7 Nit, 130-131, 139-141. Move the leaf to Files/ImagePreviewView.swift, keep ImagePreviewView(image:) with an optional onSendToChat so ArtifactPanelView.swift:281 still compiles; list both callers. Covers ZO-6, IC-5, CO-10, DC-9, ST-8, SY-10.
- S8-N8 Nit, 157. Reference localization gates by name, not "21 languages". Trim "design-check for the tool palette".
- S8-N9 Nit, 72-77. One sentence: view becomes an NSViewRepresentable around NSScrollView.
- S8-N10 Nit, 144-147. Assert pixel dimensions and equality, not PNG byte hash; keep source-file hash check; define natural size as source pixels.
- S8-N11 Nit, 103, 110. Replace "Rejected by author" with one concrete reason each.

## Reject (adds scope)

IC-1 persistence subsystem; IC-2 golden-image infrastructure (a single shared draw function is acceptable as one sentence); IC-3 CATiledLayer; DC-8 full risks list (accept 3 to 4 lines: decode failure, failed write, oversize image); DC-11 keyboard placement path (note as limitation); CO-8 ZCode behavior summary; ST-12, SY-16, SY-14, SY-15, SY-17 cosmetic (pinning a60ba6a7 optional); ST-9, SY-11, SY-12, SY-13 trivial.

## Missed by earlier stages

- S8-N12 Nit, 117-119, 61. No maximum image size; cap export or reuse an existing attachment image limit (not checked in ManagedAssetStore or AttachmentImporter).
- S8-N13 Nit, 3-5, 64-70. Redo is in Goals but never exercised by a CUJ or test; simplest fix is dropping redo.
- S8-N14 Nit, 131 vs INDEX. After cuts, confirm the INDEX size stays "medium".
