# ZP-08: Office document rendering

Finding 5.7 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[office-runtime contract](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/office-runtime/CONTRACT.md)
(DocxView, XlsxView, PptxView, PdfView with a read-only invariant).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Render office documents natively in the preview surfaces instead of
mapping docx/xlsx/pptx to "opaque + external opener". TurboSpark already
hardens and parses these formats for text extraction
(`OfficeArchive`, `OfficeXMLParsers`, `DocumentTextExtractor`); this
draft adds viewer layers on top of the same hardened archive access.

## The Problem

When the agent writes or the user attaches a docx/xlsx/pptx, the preview
offers only "Open with default app". ZCode's office-runtime renders
DocxView, XlsxView, PptxView, PdfView with a CJK font runtime and a
read-only invariant (viewers never write workspace files). TurboSpark
users must leave the app to see what the agent wrote, which weakens the
per-turn review loop that the git pane and rewind draft (existing specs)
depend on.

## Goals

* `AppArtifactRenderKind` gains real kinds for docx, xlsx, pptx instead
  of `.opaque`.
* A native docx viewer: paragraphs with run-level bold/italic/underline,
  headings, lists, tables, and embedded images.
* A native xlsx viewer: sheet tabs, scrollable grid, cell styles, number
  formats, frozen headers, and cell comments; formulas display cached
  values (no recalculation).
* A native pptx viewer: slide list with page navigation, text frames,
  basic shape rendering (rectangles, ellipses, lines, pictures, text
  boxes).
* PDF already renders via PDFKit; formalize it as the fourth viewer.
* A read-only invariant: viewers never write the source file, enforced
  by the viewer layer taking only a read-only file handle.

## Non-Goals

* No editing or saving of office documents (read-only invariant; ZCode
  holds the same line). Generation is ZP-09's problem.
* No full OOXML fidelity: no themes, SmartArt, charts (xlsx chart parts
  render as a placeholder row listing the chart), footnotes/endnotes in
  v1. Each is listed as a documented limit.
* No macro (VBA) execution ever; xlsx VBA parts are shown as a
  disabled-mode notice only.

## Open Questions

* **Where viewers live**: extending `FilePreviewView` /
  `FilePreviewLeaves` (attachment preview) and `ArtifactPanelView`
  (agent-written artifacts) with office kinds, vs a standalone office
  pane. Extending both keeps one render path per entry point; a
  standalone pane duplicates routing. Lean extending both.

## Critical User Journeys

### CUJ 1: review an agent-written docx

* **Given**: the agent wrote report.docx into the workspace.
* **When**: the user opens the artifact panel entry.
* **Then**: the docx renders with headings, runs, lists, and tables;
  the preview/source toggle shows the extraction text ("Model text")
  the model received; the file's mtime and size are unchanged after
  viewing (read-only invariant).

### CUJ 2: inspect a spreadsheet

* **Given**: an attached xlsx with three sheets.
* **When**: the user opens the attachment preview.
* **Then**: sheet tabs render, the first sheet shows a scrollable grid
  with number formats and frozen headers, and the VBA notice appears
  only if VBA parts exist.

### CUJ 3: presentations

* **Given**: an attached pptx.
* **When**: the user opens the preview.
* **Then**: slides render in a page navigator with text frames and
  basic shapes; a documented limits list appears in the pane footer.

## Technical Plan (Architecture)

```text
FilePreviewView / ArtifactPanelView
        |
        v
OfficeDocumentView (SwiftUI, kind-dispatched)
  DocxView | XlsxView | PptxView | (PdfView = existing PDFKit)
        |
        v
OfficeRenderModel (parse layer, reuses hardened access)
  OfficeArchive (existing hardened unzip)
  OfficeXMLParsers (existing SAX text extraction, extended)
  + new read-only style/layout passes:
     docx: styles.xml -> run props, numbering, tables
     xlsx: sharedStrings, styles, sheet XML -> grid model
     pptx: slide XML -> frames and basic shapes
        |
        v
Read-only file handle; viewers never write.
```

The parse layer stays layered on `OfficeArchive` so the hardening
(zip-bomb bounds, path containment) applies to rendering too. Rendering
is derived from cached formatting in the files only; no recalculation.

## Alternatives Considered

### Alternative A: WKWebView-based office rendering

* **Pros**: HTML/CSS layout is easier.
* **Cons**: contradicts the native posture, drags in a web host for
  attachments, and conflicts with ZP-06's page sandbox philosophy
  (pages are plugin-only).
* **Reason for Rejection**: Rejected by author: native viewers keep the
  trust boundary simple.

### Alternative B: QuickLook-first with custom fallback

* **Pros**: QuickLook renders office formats for free.
* **Cons**: QuickLook quality varies by macOS version, cannot guarantee
  the read-only invariant or consistent presentation, and offers no
  sheet/slide navigation API surface.
* **Reason for Rejection**: Rejected by author: QuickLook is the
  fallback for unsupported elements, not the primary renderer.

## Non-Functional Requirements

* **Security**: all file access through `OfficeArchive` hardening; no
  XML entity expansion; bounded element counts per document (render the
  first N of any unbounded list with a documented cap row).
* **Performance**: parse on a background queue; render lazily per sheet
  and per slide; 50 MB document ceiling for in-app rendering with a
  clean external-opener handoff above it.
* **Localization**: viewer chrome strings in the catalog, 21 languages.
* **Design compliance**: viewer chrome follows DESIGN.md; document
  content itself uses plain typography tied to the user font settings
  where sensible.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/OfficeRenderModel.swift` (Create)
   * **Type / Scope**: parse-to-render model.
   * **Proposed Changes**: docx style/table pass, xlsx grid model
     (sharedStrings, styles, number formats, frozen panes), pptx frame
     model with basic shapes; bounded-element caps; all access through
     OfficeArchive.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/OfficeDocumentView.swift` (Create)
   * **Type / Scope**: viewer container.
   * **Proposed Changes**: kind-dispatched container with sheet tabs
     (xlsx), slide navigator (pptx), heading-outline jump bar (docx),
     limits footer, read-only badge.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/OfficeViews/DocxView.swift` (Create)
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/OfficeViews/XlsxView.swift` (Create)
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/OfficeViews/PptxView.swift` (Create)
6. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/OfficeArchive.swift` (Modify)
   * **Type / Scope**: hardened archive access.
   * **Proposed Changes**: expose a read-only handle API for the render
     layer (viewers never write); no behavior change to extraction.
7. `swift/TurboSparkApp/Sources/TurboSparkApp/Files/FilePreviewLeaves.swift` (Modify)
   * **Type / Scope**: attachment preview routing.
   * **Proposed Changes**: route docx/xlsx/pptx to OfficeDocumentView
     instead of the opaque external-opener leaf.
8. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactPanelView.swift` (Modify)
   * **Type / Scope**: artifact routing.
   * **Proposed Changes**: office kinds route to OfficeDocumentView
     instead of `.opaque`.
9. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactPanelView+Kinds.swift` (Modify; basename, the `AppArtifactRenderKind` enum owner)
   * **Type / Scope**: render kinds enum.
   * **Proposed Changes**: add docx/xlsx/pptx kinds; retire the opaque
     mapping for these three.
10. `swift/TurboSparkApp/Tests/TurboSparkAppTests/OfficeRenderModelTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: fixture documents per kind (minimal docx,
     xlsx with two sheets, pptx with two slides, one VBA-notice case);
     style and number-format mapping; bounded caps; read-only invariant
     (mtime and size unchanged after a render pass); zip-bomb bounds
     inherited from the OfficeArchive tests.
   * Mutation checks: break a bounds check and assert the cap test
     fails; let the viewer write and assert the invariant test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for OfficeRenderModelTests.
* **Localization**: catalog additions in 21 languages.
* **Manual**: open fixture documents from both entry points; confirm
  read-only invariant; confirm external-opener handoff above the size
  ceiling.

--------------------------------------------------------------------------------

*Drafted with EGM*