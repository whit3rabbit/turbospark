# ZP-09: Office document generation

Finding 9.1 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's document skills (docx/xlsx/pptx/pdf) from the
[document-skills-plugin doc](https://github.com/zai-org/ZCode/blob/main/docs/document-skills-plugin.md).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Let the agent create professional office files: docx, xlsx, pptx, and
pdf. ZCode does this with four skill plugins that drive JavaScript and
Python libraries. TurboSpark cannot bundle Node or Python in the product
path, so the same outcome ships as a native document-writing layer in
Swift plus skills that instruct the agent to use it, with a
user-installed-tooling fallback path for fidelity cases the native layer
cannot reach.

## The Problem

TurboSpark's only document writing is chat export. An agent that should
produce a report, a deck, or a spreadsheet has nothing to write with.
ZCode's skills solve this with docx-js, python-pptx, and xlsx writers,
but the roadmap forbids bundling Python or Node in the product path, so
those implementations cannot be adopted as-is.

## Goals

* A native Swift document-writing layer produces valid docx, xlsx, pptx,
  and pdf files (PDF via Core Graphics, PDFKit-adjacent APIs).
* Four bundled skills (docx, xlsx, pptx, pdf) teach the agent the
  writing layer's operations and their limits, modeled on ZCode's skill
  structures (cover recipes, per-scene playbooks, postcheck steps)
  without the JS/Python tool references.
* A postcheck step per skill: after writing, the agent re-opens the file
  with the ZP-08 render model and validates structure (openable,
  expected part counts, non-empty text).
* The user-installed-tooling fallback: if the user has Python (or
  scripts) available, skills may offer shell-outs, always through the
  ordinary `Bash` tool and its permission ladder.
* Output lands in the chat's artifact folder and surfaces in the
  artifact panel for immediate review (ZP-08 viewers).

## Non-Goals

* No bundled Python/Node runtime (roadmap constraint).
* No editing of existing office files in v1; the writing layer creates
  new files. Round-trip editing is future work.
* No template marketplace or cover-art assets beyond simple, license-safe
  recipes in skill text.

## Open Questions

* **Native writer scope**: full OOXML feature surface is enormous. The
  proposal is a deliberate subset per format (headings, runs, lists,
  tables, images for docx; sheets, cells, styles, number formats, basic
  charts for xlsx; text frames, basic shapes, images for pptx). Any
  feature outside the subset goes through the shell-out fallback. Does
  the subset match expected use? Decide at implementation.
* **PDF path**: native CG-based writer vs LaTeX-style shell-out only.
  Lean native for text-and-image PDFs, shell-out for LaTeX.

## Critical User Journeys

### CUJ 1: agent writes a report

* **Given**: the user asks for a formatted project report as docx.
* **When**: the agent activates the docx skill and calls the writing
  layer.
* **Then**: report.docx lands in the artifact folder, opens in the ZP-08
  viewer, and the postcheck passes (valid zip, expected parts, non-empty
  text). The user reviews it in place.

### CUJ 2: fallback path

* **Given**: the user needs a feature outside the native subset (for
  example an Excel chart with VBA-created elements).
* **When**: the skill instructs a shell-out to user-installed tooling.
* **Then**: the Bash tool asks per the ladder, the script runs, and the
  postcheck validates the result; if the tooling is missing the skill
  reports the gap instead of failing silently.

### CUJ 3: pdf handout

* **Given**: the user asks for a PDF handout with a chart image.
* **When**: the agent uses the pdf skill.
* **Then**: the native CG writer produces a text-and-image PDF with
  correct page sizes; the postcheck confirms openability via PDFKit.

## Technical Plan (Architecture)

```text
Bundled skills (docx, xlsx, pptx, pdf)
  instruct the agent; carry limits + postcheck steps
        |
        v
Agent calls writer tools (or writes via REPL file bridge)
        |
        v
DocumentWriter layer (Swift)
  DocxWriter: OOXML zip via OfficeArchive's writer half
  XlsxWriter: sheets/styles/number formats/basic charts
  PptxWriter: slides/frames/basic shapes/images
  PdfWriter: Core Graphics text + image pages
        |
        v
Postcheck: re-open with ZP-08 OfficeRenderModel
  validate structure + non-empty text
        |
        v
Artifact panel (ZP-08 viewers) for review
```

Skills are authored under the bundled-skill root from ZP-07. Writers
share the hardened zip access; they never shell out on their own.

## Alternatives Considered

### Alternative A: port ZCode's skills verbatim

* **Pros**: battle-tested content.
* **Cons**: they reference docx-js, python-pptx, and bundled scripts;
  adopting them means bundling a JS/Python runtime, which the roadmap
  forbids.
* **Reason for Rejection**: Rejected by author: runtime constraint.

### Alternative B: generate HTML and convert

* **Pros**: HTML is easy for the model.
* **Cons**: HTML-to-office conversion requires the same runtime tooling
  (pandoc/LibreOffice) with worse fidelity control.
* **Reason for Rejection**: Rejected by author: adds a conversion
  dependency with unpredictable output.

## Non-Functional Requirements

* **Security**: writers write only into the chat artifact folder; the
  shell-out path goes through the ordinary Bash permission ladder; no
  network access for any writer.
* **Quality**: the postcheck is mandatory in every skill; a failed
  postcheck is reported to the user, not retried silently.
* **Localization**: skill content is English (agent-facing, matching
  ZCode), while the Create-entry UI strings are catalog strings in 21
  languages.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/DocumentWriter/DocxWriter.swift` (Create)
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/DocumentWriter/XlsxWriter.swift` (Create)
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/DocumentWriter/PptxWriter.swift` (Create)
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/DocumentWriter/PdfWriter.swift` (Create)
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/DocumentWriter/DocumentPostcheck.swift` (Create)
6. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/docx/SKILL.md` (Create)
7. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/xlsx/SKILL.md` (Create)
8. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/pptx/SKILL.md` (Create)
9. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/pdf/SKILL.md` (Create)
10. `swift/TurboSparkApp/Sources/TurboSparkApp/State/SkillManager.swift` (Modify)
   * **Type / Scope**: discovery of the four new bundled skills.
11. `swift/TurboSparkApp/Tests/TurboSparkAppTests/DocumentWriterTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: per-format fixture generation (write, then
     re-open with OfficeXMLParsers and assert structure), postcheck
     pass and fail paths, artifact-folder confinement (an attempted
     write outside the folder fails), shell-out absence (no bundled
     runtime).
   * Mutation checks: corrupt a written zip's content types part and
     assert the postcheck fails; relax folder confinement and assert
     the confinement test fails.
12. `swift/docs/` docs entry: a document generation section in the
    appropriate swift doc (basename: extend `swift/docs/SWIFT_AUDIO_MEETING_NOTES.md`-style pattern with a new `swift/docs/SWIFT_DOCUMENTS.md`) (Create)

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for DocumentWriterTests.
* **Localization**: catalog additions in 21 languages (UI strings only;
  skills are agent-facing English).
* **Manual**: generate one document per format from chat, review each
  in the ZP-08 viewers, and run each postcheck.

--------------------------------------------------------------------------------

*Drafted with EGM*