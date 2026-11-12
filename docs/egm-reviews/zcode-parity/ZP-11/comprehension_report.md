# Stage 6: Comprehension report (ZP-11)

Reviewer: egm-comprehension (completed). Read the document alone.

## What this proposes

Turn the app's image preview, which today only shows a picture and scrolls, into a small annotation tool. The user can pan and zoom the image, draw arrows, boxes, freeform pen strokes and text labels on top, select, drag and delete them, and undo and redo. A "Send to chat" action renders the image plus annotations into a new PNG at the image's natural size and attaches it to the chat composer. The original file is never changed. Annotations are stored separately as overlay data and are lost when the preview closes (leaned toward, still an open question). The design copies the behavior of a feature in another product (ZCode).

## How the existing system works, per this document

There is an `ImagePreviewView`, described as a plain, scrollable, aspect-fitted `NSImage` with no interaction beyond scrolling. Its code lives in `FilePreviewLeaves.swift`, which routes image files to it. A "composer" accepts attachments, and an "attachment folder" exists where exported files are written. The document does not describe how the composer or attachment folder work, or what `make swift-lib` and `make design-check` do.

## Findings (no Blockers)

CO-1 Major, 16-19, 38-40, 91-92, 127-129. Tool set inconsistent (Overview: arrow/box/text; Goals: plus pen); text is a tool or a label is unclear. Fix: one canonical tool set; define text's relation to shapes.
CO-2 Major, 16-18, 38-40. Overlay-metadata export has no format, consumer, CUJ or test. Fix: specify or cut; keep flattened only.
CO-3 Major, 36-37, 64-70, 91-92. History semantics unclear: "transform and annotation history" vs undo not touching pan/zoom; redo, operation granularity and depth limit undefined. Fix: annotation operations only; define granularity.
CO-4 Major, 31-33, 72-77, 91-92. Coordinate space stated as natural pixels and as normalized; stroke width and font size scaling undefined. Fix: pick one storage space; define stroke and font units.
CO-5 Major, 72-77, 117-119. "Programmatic zoom" and "size bucket" undefined and in tension with "no quality loss at any zoom". Fix: define bucket and reconcile.
CO-6 Major, 59-62, 114-116. Composer handoff and attachment folder undefined (location, naming, cleanup, no-active-chat case, where the button lives). Fix: name the existing attachment mechanism and lifecycle.
CO-7 Major, 50-53, 59-62, 148-149. Persistence is open yet CUJ and tests assume session-only; unsent-annotation close behavior unstated. Fix: state session-only default and discard behavior.
CO-8 Major, 1-5, 21-27. Parity claim depends on external ZCode with no summary of copied behaviors. Fix: list the matched ZCode behaviors.
CO-9 Nit, 97-106. ZP-10 not linked; "Services" undefined. Fix: link ZP-10; say macOS Services menu.
CO-10 Nit, 131. "Create or Modify" ambiguity. Fix: say extract from FilePreviewLeaves.swift into a new file.
CO-11 Nit, 153-157. Make targets, "app scheme", palette design check, "21 languages" unexplained. Fix: one line each or link.
CO-12 Nit, 140-150. PNG byte-hash stability is not guaranteed across OS versions; "mutation check" undefined. Fix: assert pixel equality or dimensions; define term.
CO-13 Nit, 57-62. "Natural size" on Retina (pixels vs points) undefined. Fix: source pixel dimensions.
CO-14 Nit, 115-116. "Opened read-only" mechanism unnamed. Fix: name it or say "never written".
