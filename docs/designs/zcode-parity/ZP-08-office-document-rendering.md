# ZP-08: Office document rendering

Finding 5.7 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[office-runtime contract](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/office-runtime/CONTRACT.md)
(DocxView, XlsxView, PptxView, PdfView with a read-only invariant).

* Author: whit3rabbit
* Status: Implemented (docx, xlsx and pptx viewers with a QuickLook fallback; deviations under Implementation notes)
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
## Implementation notes

Applies the EGM review (`docs/egm-reviews/zcode-parity/ZP-08`) Blockers and
Majors. Swift only, no Rust change, no new dependency, no Package.swift or
resource change.

### What the review changed

* **Problem restated (B1, M1)**: `.opaque` already rendered QuickLook plus an
  "Open with default app" button, so the problem was never "no preview". The
  gap is fidelity and navigation, measured against the QuickLook behavior the
  panel already ships: no sheet or slide model the app can reason about, no
  per-turn control over what is shown, no guarantee the same pane renders the
  same way across macOS versions. All three formats shipped together because
  they share one hardened loader, but the contract is unchanged: **QuickLook
  stays the default fallback** (B2). Any parse failure, cap hit, encrypted or
  mid-write file shows a bounded error card above the QuickLook preview, with
  the external-opener button on the card (M10, ZP-17 CUJ 2).
* **Routed extensions (B4, M5)**: `docx docm xlsx xlsm pptx pptm`
  (`OfficeFormat.extensions`). A macro-enabled file shows the footer notice
  "Macros in this file are never run." when `vbaProject.bin` exists; the part
  is never read. Legacy binary `.doc .xls .ppt` and everything else stay
  `.opaque` (QuickLook).
* **Read-only invariant (B4)**: no write to the source path and no process
  spawn other than the sanctioned `unzip` that `OfficeArchive` already uses.
  There is no handle API: the loader opens the file read-only, and
  `OfficeArchive` parses a private 0700 temp copy it deletes on deinit. The
  test asserts content (not only mtime and size) and that nothing appears
  beside the source.
* **Render limits (B3, M6)**, all in `OfficeRenderLimits`:

| Bound | Value |
| :--- | :--- |
| File on disk | 50 MB (above it: error card, QuickLook, opener) |
| Archive entries | 5,000 |
| Declared uncompressed total | 256 MB |
| Declared over file size | 200x, checked once the claim passes 8 MB |
| One part / all parts read | 16 MB / 96 MB (the entry read is byte-capped regardless of the declared size) |
| Parse deadline | 20 s |
| XML nodes / depth per part | 400,000 / 64 |
| docx blocks / table cells / images | 4,000 / 10,000 / 16 (8 MB each) |
| xlsx sheets / rows / columns / cells per sheet | 32 / 1,000 / 64 / 30,000 |
| pptx slides / shapes per slide / table cells | 200 / 300 / 2,000 |

  A cap cut sets `OfficeRenderNotes.truncated`, shown as a footer line; it is
  never silent.
* **Hostile-input rules (B3)**: table-of-contents names containing `..`, a
  leading `/`, a backslash or NUL refuse the whole archive; a symlink at the
  source path is refused; nothing is extracted to disk (zip-slip and symlink
  entries are inert); entry names with `* ? [ ]` are never requested because
  `unzip` reads them as patterns; relationships resolve only to entries inside
  the archive and `TargetMode="External"` is counted and never followed (no
  network access exists in this layer); XML with a DOCTYPE or ENTITY
  declaration, a UTF-16 BOM or a NUL near the start is refused, and the
  parser runs with external entities off (no XXE, no entity expansion);
  embedded images are raster-only, size-capped, and decoded through a
  thumbnail path that refuses a claimed canvas over 40 megapixels.
* **Theme fallback and known gaps (M7)**: `theme1.xml` is not parsed. A docx
  run uses the literal rgb Word writes beside a theme color; xlsx and pptx
  theme colors use Office's default palette (with tint for xlsx), so a custom
  theme shows default colors. The fidelity matrix versus the QuickLook the
  panel already ships:

| Element | Native viewer | QuickLook |
| :--- | :--- | :--- |
| Headings, runs, lists, tables | rendered | rendered |
| Run font sizes (docx) | not honored (user font setting wins) | honored |
| Headers, footers, footnotes, comments | not rendered | partly |
| Page layout, columns, text boxes | not rendered | rendered |
| Charts, SmartArt, media | placeholder or absent | rendered |
| Merged cells, hidden rows and columns, borders, conditional formats (xlsx) | not rendered | rendered |
| Frozen rows | pinned | n/a |
| Frozen columns | scroll | n/a |
| Cell comments, sheet tabs, formatted cached values | rendered | partly |
| Slide layouts, masters, backgrounds, rotation, animation | not rendered (placeholders without geometry use fixed boxes; white slide) | rendered |
| Sheet and slide navigation, outline | yes | slide list only |
| Right-to-left and complex scripts | shaped by SwiftUI text, no bidi block layout | rendered |

  The QuickLook column is the author's expectation of the system previewer,
  not a measurement on a specific macOS version; the review's request to
  measure it on representative agent-written files is still open.

  CJK: the viewers use the system font stack, which falls back to the system
  CJK fonts. ZCode's "CJK font runtime" is dropped from the goals (it ships
  fonts for a non-macOS renderer; there is nothing to bundle here). No CJK
  fixture test was added.
* **Phasing (M1)**: not phased. The three parsers are small trees over one
  bounded XML builder, and each falls back to QuickLook, so a weak format
  costs no more than the old behavior.
* **PDF (M4)**: no code change. The existing `PDFDocumentView` is the PDF
  viewer; the "fourth viewer" goal is satisfied by routing as it stands.
* **Where it lives (M3, open question closed)**: both entry points route to
  one `OfficeDocumentView`. The attachment pane (`FilePreviewView`) renders it
  for office sources, and its "Model text" toggle still reaches the
  extraction (`hasRenderedAlternative` includes office sources). The artifact
  panel renders it for the new kinds; its Source arm shows the same extraction
  (the text the model receives), not raw bytes.
* **Enum (M2)**: `AppArtifactRenderKind` is at
  `State/AppArtifact.swift`, not `ArtifactPanelView+Kinds.swift`. It gains
  `docx xlsx pptx` and an unknown-raw-value fallback to `.opaque`.
  `formatLabel` keeps the extension text for the new kinds.
* **Freshness (M12)**: the artifact panel passes the artifact revision as a
  reload key, so a rewrite re-parses. A half-written zip is the M10 fallback.
  No snapshot store and no XPC helper were built (scope growth the review's
  Stage 8 rejected).
* **Localization (M11)**: 14 new keys in 21 languages; the parity gates pass.

### Deviations from the draft

* Parsers are split across `OfficeRenderModel.swift` (loader, limits, XML
  tree, colors), `OfficeDocxModel.swift`, `OfficeSpreadsheetModel.swift`
  (with the number formatter) and `OfficeSlidesModel.swift`, and the views
  are `OfficeDocumentView.swift` plus `OfficeViews/{Docx,Xlsx,Pptx}View.swift`.
  `OfficeXMLParsers` (the SAX text extractor) is unchanged; the render layer
  uses its own bounded tree because the extractor streams text only.
* `OfficeArchive` gained only `entrySizes` (declared sizes from the listing).
  No handle API; see the read-only invariant above.
* Parsing is eager per document (all sheets, all slides, within the caps), on
  a detached task; rendering is lazy (rows via `LazyVStack`, one slide at a
  time). Per-sheet lazy parsing was dropped because `OfficeArchive` charges a
  cumulative byte budget across reads.
* Document content draws on fixed white paper (`Theme/OfficePaper.swift`),
  not the themed surface: file colors are chosen for a white page. Only the
  chrome follows the theme.
* Number formats: General, fixed decimals, thousands, percent, literals and
  currency, negative sections, and common date and time tokens, with fixed
  US separators. Scientific, fractions, elapsed time and conditions fall
  back to the cached value as General.
* Not verified by running the app: no visual check of the three views in a
  launched build. The parse layer and routing are unit tested
  (`OfficeRenderModelTests`); the SwiftUI views compile and pass the
  localization and design gates.

### Tests and mutation checks

`OfficeRenderModelTests` builds every fixture in code with a stored-zip
writer (so an entry can be named `../x` or `/etc/passwd`). Mutations applied
and caught: ratio check disabled (bomb test fails; the fixture sits under the
part cap so only the ratio check can refuse it), entry-name containment
disabled (zip-slip and absolute-name tests fail), DOCTYPE and ENTITY refusal
disabled (XXE test fails), wildcard-name guard disabled (wildcard test
fails), symlink refusal disabled (symlink test fails).
