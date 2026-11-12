# ZP-11: Image preview annotation and editing

Finding 5.9 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[image-preview-editor contract](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/image-preview-editor/ImagePreviewSurface.tsx).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Upgrade `ImagePreviewView` from a passive image display to an annotation
surface: draggable arrow/box/text annotations over a pan-and-zoom image,
with history owned by the open preview and natural-size export into the
composer. Original bytes stay read-only; annotations live as overlay
data with optional flattening at export.

## The Problem

The current image preview is a plain scrollable, aspect-fitted NSImage
with no interaction beyond scrolling. A user who wants to point at a
region of a screenshot for the model ("what is wrong here?") must edit
in an external app first. ZCode embeds an editor contract in its image
preview with exactly this loop.

## Goals

* Pan and zoom in natural pixel coordinates (no quality loss at any
  zoom).
* An annotation layer with arrow, box, and freeform-pen tools plus
  optional text labels; annotations are draggable, selectable, and
  deletable.
* Transform and annotation history owned solely by the open preview
  (undo/redo), never persisted into the original file.
* Export at natural size into the composer: flattened (annotations
  burned in) or as original-plus-annotation-overlay metadata. Default is
  flattened, since the model consumes plain images.

## Non-Goals

* No crop, resize, filters, or rotation in v1 (ZCode's contract is
  annotation-centric).
* No multi-layer management; one annotation layer per preview session.

## Open Questions

* **Annotation persistence**: ZCode keeps annotations in preview state
  only (lost on close). Options: session-only (simpler, matches ZCode)
  or sidecar persistence per image path. Lean session-only day one.

## Critical User Journeys

### CUJ 1: annotate and ask

* **Given**: a screenshot open in the image preview.
* **When**: the user draws a box around a region, adds an arrow and
  text, and clicks "Send to chat".
* **Then**: a flattened PNG at natural size attaches to the composer;
  the original file bytes are unchanged (verified by hash in tests).

### CUJ 2: pan, zoom, undo

* **Given**: a large image with annotations.
* **When**: the user zooms to 400 percent and pans; then undoes twice.
* **Then**: annotations track the image transform exactly at every zoom
  level; undo removes the last two annotation operations without
  touching pan/zoom state.

## Technical Plan (Architecture)

```text
ImagePreviewView (upgraded)
  pan/zoom: NSScrollView magnification + programmatic zoom
  natural-pixel coordinate space for annotations
       |
       v
AnnotationLayerView (new)
  tools: arrow, box, pen, text
  drag/select/delete; hit testing in image space
  history stack (owned by the preview session)
       |
       v
AnnotationExporter
  flatten: render base image + layer at natural size
  (Core Graphics), hand to composer as attachment
```

Annotations store endpoints in normalized image coordinates so zoom
never resamples them; rendering happens in the overlay layer at view
resolution, flattening re-renders at natural resolution.

## Alternatives Considered

### Alternative A: PencilKit for the annotation layer

* **Pros**: free drawing and erasing.
* **Cons**: PencilKit strokes are its own data model; arrows/boxes/text
  still need custom views, so two toolkits would be mixed. ZP-10's
  whiteboard already owns PencilKit.
* **Reason for Rejection**: Rejected by author: custom lightweight
  layer keeps arrow/box/text first-class and avoids toolkit mixing.

### Alternative B: open in an external editor via Services

* **Pros**: zero code.
* **Cons**: breaks the in-app review loop; the round trip loses context.
* **Reason for Rejection**: Rejected by author: the loop is the feature.

## Non-Functional Requirements

* **Security/privacy**: original files are opened read-only; flattened
  exports are new files in the attachment folder; no writes to source
  bytes.
* **Performance**: the overlay renders at view resolution; flatten
  happens on export only; pan/zoom stays at 60 fps on large images by
  rasterizing the base image once per size bucket.
* **Accessibility**: tools are keyboard-reachable; annotations carry
  accessibility labels with type and position; localized strings.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Files/AnnotationLayerView.swift` (Create)
   * **Type / Scope**: annotation overlay.
   * **Proposed Changes**: tools, hit testing in image space, drag and
     delete, history stack.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Files/ImagePreviewView.swift` (Create or Modify; the current leaf lives in `FilePreviewLeaves.swift`)
   * **Type / Scope**: preview surface.
   * **Proposed Changes**: pan/zoom upgrade to natural pixel space with
     programmatic zoom, tool palette, export action, layer hosting.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Files/AnnotationExporter.swift` (Create)
   * **Type / Scope**: flattening and export.
   * **Proposed Changes**: Core Graphics flatten at natural size, write
     to the attachment folder, composer handoff.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Files/FilePreviewLeaves.swift` (Modify)
   * **Type / Scope**: routing.
   * **Proposed Changes**: route image leaves to the upgraded preview.
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ImageAnnotationTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: coordinate mapping at multiple zoom levels,
     history semantics (undo does not touch pan/zoom), flatten output
     dimensions and hash stability, source-file immutability (hash
     unchanged).
   * Mutation checks: scale annotations with the view instead of the
     image space and assert the zoom mapping test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the annotation tests.
* **Static Analysis and Linting**: `make design-check` for the tool
  palette.
* **Localization**: catalog additions in 21 languages.
* **Manual**: annotate a screenshot, send to chat, verify the model
  receives the flattened image; confirm the original file hash is
  unchanged.

--------------------------------------------------------------------------------

*Drafted with EGM*
