# ZP-10: Whiteboard canvas

Finding 5.8 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[WhiteboardPane](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/WhiteboardPane.tsx)
and its
[whiteboard doc](https://github.com/zai-org/ZCode/blob/main/docs/whiteboard.md).

* Author: whit3rabbit
* Status: Implemented
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Add a whiteboard side-pane tab: per-workspace drawing boards with pen and
eraser, color swatches, stroke width control, undo/redo/clear, persisted
stroke data, and one-click PNG export into the chat composer. Native
PencilKit on macOS provides the drawing surface.

## The Problem

There is no drawing canvas anywhere in the app; searches for PencilKit,
PKCanvasView, whiteboard, and SwiftUI Canvas match only audio waveform
views. Users explaining a geometry problem, sketching an architecture
diagram, or annotating a hand-drawn mock must draw in another app and
attach the file manually. ZCode's whiteboard makes sketches first-class
composer attachments.

## Goals

* A whiteboard tab in the side pane, per workspace.
* Boards auto-name non-colliding titles and persist stroke data across
  tab close and app restart.
* Pen and eraser with six color swatches and adjustable stroke width,
  undo/redo/clear.
* One-click PNG export that attaches the export into the chat composer
  as an image attachment (also reachable from the `@` attachment panel).

## Non-Goals

* No text tool, shapes, or multi-user collaboration in v1 (ZCode's own
  surface is stroke-based).
* No agent-side canvas reading in v1: the exported PNG reaches the
  model as an ordinary image attachment through the existing attachment
  path.

## Open Questions

* **Board scoping**: per workspace (ZCode) vs per chat. Per workspace
  matches ZCode and reuse across chats; per chat keeps context tighter.
  Lean per workspace, author decides.

## Critical User Journeys

### CUJ 1: sketch and send

* **Given**: a workspace with a chat open.
* **When**: the user draws on the whiteboard and clicks "Export to
  chat".
* **Then**: a PNG renders at natural size and attaches to the composer
  as an image attachment; the composer shows the attachment chip; the
  board itself stays unchanged.

### CUJ 2: persistence

* **Given**: a board with strokes.
* **When**: the user closes the pane and reopens it, or restarts the
  app.
* **Then**: the strokes return. Boards persist per workspace with
  auto-named titles that never collide.

## Technical Plan (Architecture)

```text
WhiteboardPaneView (side-pane tab, new)
  PKCanvasView wrapper (NSViewRepresentable)
  toolbar: pen/eraser, six swatches, width slider,
           undo/redo/clear, export
       |
       v
WhiteboardStore (new)
  per-workspace board list, auto-naming
  stroke data as PKDrawing data blobs (PNG for export)
  persisted under the profile state root via AppJSONStore
  (drawing blobs stored as files beside the archive,
   JSON store holds metadata only)
```

PencilKit on macOS supports Apple Pencil and mouse drawing
(`PKCanvasView` with `drawingPolicy`). Undo/redo ride the canvas's
coordinator; clear is a new empty drawing.

## Alternatives Considered

### Alternative A: SwiftUI Canvas with hand-rolled strokes

* **Pros**: no AppKit bridge.
* **Cons**: reimplements palm rejection, smoothing, and pressure; more
  code for worse feel.
* **Reason for Rejection**: Rejected by author: PencilKit is the
  platform drawing primitive.

### Alternative B: store strokes as vector JSON

* **Pros**: editable strokes later.
* **Cons**: v1 scope grows; PKDrawing's own serialization is sufficient
  for persistence and re-render.
* **Reason for Rejection**: Rejected by author: defer vector editing to
  future work.

## Non-Functional Requirements

* **Performance**: strokes render on the canvas layer; export renders
  off the main thread; board files are written on change with a debounce.
* **Design compliance**: toolbar follows DESIGN.md controls and the
  token ratchet; all strings localized (21 languages).
* **Accessibility**: tool controls carry accessibility labels; the
  canvas gets a label describing the board and its content state
  (strokes count).

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Whiteboard/WhiteboardPaneView.swift` (Create)
   * **Type / Scope**: pane UI.
   * **Proposed Changes**: board list, canvas wrapper, toolbar, export
     action.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Whiteboard/WhiteboardStore.swift` (Create)
   * **Type / Scope**: persistence.
   * **Proposed Changes**: per-workspace boards, metadata JSON via
     AppJSONStore, drawing blobs as files, auto-naming, debounced
     writes.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SplitChatPanes.swift` (Modify)
   * **Type / Scope**: side-pane routing.
   * **Proposed Changes**: add the whiteboard tab beside existing pane
     targets.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerView.swift` (Modify)
   * **Type / Scope**: composer intake.
   * **Proposed Changes**: accept a whiteboard export as an image
     attachment (same path as a pasted image).
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/WhiteboardStoreTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: persistence round-trip, auto-naming
     non-collision, export PNG dimensions, quarantine behavior on
     corrupt metadata.
   * Mutation checks: break auto-naming uniqueness and assert the
     collision test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the store tests.
* **Static Analysis and Linting**: `make design-check` for the toolbar.
* **Localization**: catalog additions in 21 languages.
* **Manual**: draw with mouse, restart, confirm persistence; export to
  chat and confirm the attachment reaches the model as an image.

## Implementation notes (2026-11-13)

Implemented in `Whiteboard/WhiteboardStore.swift` (boards and persistence),
`Whiteboard/WhiteboardCanvasNSView.swift` (pointer capture, strokes, eraser),
`Whiteboard/WhiteboardCanvasController.swift` (tools, history, swatches),
`Whiteboard/WhiteboardExporter.swift` (PNG), `Whiteboard/WhiteboardPaneView.swift`
(pane UI), `State/AppModel+Whiteboard.swift` (open/close, composer handoff) and
`Theme/WhiteboardPalette.swift` (pen colors and paper). Tests:
`WhiteboardStoreTests`, `WhiteboardCanvasTests`. Deviations from the draft,
following the EGM review (`docs/egm-reviews/zcode-parity/ZP-10/`):

* **PencilKit has no canvas view on macOS (M4, spike result).** `PKCanvasView`
  and `PKToolPicker` are iOS and Mac Catalyst only: the macOS SDK ships the
  PencilKit data model and renderer (`PKDrawing`, `PKStroke`, `PKInk`,
  `PKDrawing.image(from:scale:)`) and nothing else. The wrapper in the draft
  cannot be built. `PKDrawing` stays the persisted format and its renderer
  draws the committed strokes; `WhiteboardCanvasNSView` supplies mouse and
  trackpad capture, a live polyline preview, and a stroke eraser (a touched
  stroke is removed whole, not partially erased; one gesture is one undo
  step). Pressure is not used. This is the draft's rejected Alternative A in
  its minimal form (own input, PencilKit's storage and rendering).
* **Host is the right column, not `SplitChatPanes` (B1).** The whiteboard is a
  new `AppRightColumnClaimant.whiteboard` opened from a top-bar button.
  Precedence: artifact, html preview, file preview, browser, whiteboard,
  inspector. The browser and whiteboard are exclusive in their setters. The
  inspector shortcut and the settings toggle close it like the browser.
  Width is `AppChromeLayout.whiteboardPanelWidth` (520).
* **Composer seam is `addPromptAttachment` (B2).** `PromptComposerView` is
  untouched. Export renders a PNG, stores it through `ManagedAssetStore` and
  appends it to the focused chat's draft via the same path as ZP-11, by
  extracting `AnnotationExporter.pngData(from:)` and
  `AppModel.attachPNGToPrompt` from ZP-11 rather than duplicating them. The
  `@` attachment panel goal is dropped.
* **Scoping is recorded as a decision (M1):** per workspace. The key is the
  selected project's id in Projects mode, else one shared "chat" bucket.
  Export goes to the focused chat (`selectedChatID`).
* **Persistence (M2):** `whiteboards/boards.json` (AppJSONStore, titles and
  timestamps) plus `whiteboards/<workspace>/<id>.drawing` (PKDrawing data).
  The blob is the source of truth: a blob with no row is adopted on load as
  a new auto-titled board, a row with no blob is an empty board, an
  unparseable blob is renamed `corrupt-<stamp>-<name>` and the board opens
  empty with a warning, and unreadable metadata is quarantined by AppJSONStore.
  Writes debounce 600 ms and also flush on pane close, board switch and
  `shutdown()`. Delete is permanent and confirmed.
* **Toolbar is custom (M3).** `PKToolPicker` does not exist here either. The
  strip uses `AppPillPicker` (pen, eraser), six fixed-sRGB swatches, a width
  slider (1 to 24) and `AppIconButton` undo, redo and clear. Undo and redo
  are whole-drawing snapshots (100 deep), not the window's `NSUndoManager`.
* **Export (M5):** bounds plus a 24 pt margin at 2x on opaque white (the
  canvas paper is fixed white too), pixel count capped at 16 million by
  lowering the scale, empty board refused with a message, rendered on the
  main actor (the off-main claim is dropped). With no vision model active
  the PNG still attaches and the toast gives the vision refusal reason
  instead of claiming success; the turn itself refuses to send an image
  then, as for any image.
* **Naming and board management (M8):** the lowest free "Board N" per
  workspace; rename dedupes ("Plan 2"); new, rename, delete in the pane
  header; the first open in a workspace creates "Board 1". Vector editing is
  a non-goal.
* Limitations: no zoom or pan (strokes past the pane edge are kept but not
  shown), no keyboard drawing path (the controls are keyboard reachable, the
  canvas is pointer only), partial erasing is not supported, and the pane was
  not driven by hand in the running app. The canvas is pointer-tested with
  synthetic events only.

--------------------------------------------------------------------------------

*Drafted with EGM*
