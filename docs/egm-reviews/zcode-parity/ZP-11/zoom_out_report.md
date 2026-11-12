# Stage 5: Initial Zoom Out report (ZP-11)

Target: docs/designs/zcode-parity/ZP-11-image-annotation.md (full file, 164 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out (ZO)

Verified in code: ImagePreviewView is a 12-line SwiftUI struct at FilePreviewLeaves.swift:22-34 taking an NSImage. Callers: Files/FilePreviewView.swift:161 and Generation/ArtifactPanelView.swift:281. Attachment plumbing exists: AppPromptAttachment (State/AppChat.swift:114), AppModel.addPromptAttachment (State/AppModel+Chat.swift:430), ManagedAssetStore, AttachmentImporter, ElementPickerController.attachSelectedCandidate (precedent). No "attachment folder" found in app sources.

- ZO-1 Major, lines 135-138, 114-116. Invents an "attachment folder" and composer handoff; existing ManagedAssetStore and addPromptAttachment own both. Fix: write the PNG through ManagedAssetStore, attach via addPromptAttachment, delete the folder claim.
- ZO-2 Major, lines 31-35, 127-129. Scope is a mini drawing app (pen, drag after placement, general history) while CUJs need box, arrow, text. Overview (16-17) lists arrow/box/text, Goals add pen. Fix: cut v1 to box, arrow, text with delete and undo; reconcile Overview, Goals, diagram.
- ZO-3 Major, lines 38-40, 72-89. "Original-plus-overlay metadata" export has no consumer or schema. Fix: ship flattened only; drop from Goals.
- ZO-4 Nit, lines 28, 72-77. NSScrollView magnification in a SwiftUI ScrollView view with no bridging stated. Fix: say it becomes an NSViewRepresentable around NSScrollView and keep the ImagePreviewView(image:) signature.
- ZO-5 Nit, lines 117-119. Per-size-bucket rasterization is premature. Fix: remove; measure first.
- ZO-6 Nit, lines 130-131, 139-141. "Create or Modify" ambiguity; item 4 edits the same file; ArtifactPanelView is an unlisted caller. Fix: pick move or edit in place; list both callers.
- ZO-7 Nit, lines 148-149, 157. "21 languages" conflicts with INDEX "ten localization parity gates"; mutation check over-specific; "make design-check for the tool palette" vague. Fix: reference the gates by name.
- ZO-8 note: persistence Open Question should be resolved to session-only now.

## Inventive catalyst (IC)

- IC-1 Major, lines 31-40, 50-52, 64-70. Persistence is treated as in conflict with "never persisted into original file". Fix: make annotations a Codable value type, history annotation-only, view transform separate and unundoable, optional autosave in app support keyed by image hash (or resolve to session-only explicitly). Liabilities: cleanup policy, privacy retention note.
- IC-2 Major, lines 38-40, 86-93, 117-119, 144-147. Two renderers (overlay at view resolution, Core Graphics flatten at natural size) risk drift; hash-stable test proves determinism, not fidelity. Fix: one shared draw(annotation, in: CGContext, scale:) used by both; add golden-image test at two scales with tolerance.
- IC-3 Major, lines 31, 75-77, 117-119, 133. Bucketed raster conflicts with "no quality loss at any zoom"; bucket count and eviction unstated; "NSScrollView magnification" vs "natural pixel space" coordinate authority unclear. Fix: use tiled rendering (CATiledLayer) or state buckets and cap; state that natural-pixel space derives from image pixel size, not magnification.
- IC-4 Nit, lines 32-33, 81. Text tool vs label bound to a shape is undefined. Fix: pick one.
- IC-5 Nit, lines 131, 139-141, 148-149. Create or Modify deferral; FilePreviewView.swift not listed though it constructs ImagePreviewView. Fix: move to Files/ImagePreviewView.swift and list FilePreviewView.swift and new initializer inputs.
- IC-6 Nit, line 157 vs INDEX. 21 languages vs ten gates; state feature is user-initiated, not an agent tool. Fix: reference gate by name.
