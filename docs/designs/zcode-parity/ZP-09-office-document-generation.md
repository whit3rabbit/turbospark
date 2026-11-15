# ZP-09: Office document generation

Finding 9.1 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's document skills (docx/xlsx/pptx/pdf) from the
[document-skills-plugin doc](https://github.com/zai-org/ZCode/blob/main/docs/document-skills-plugin.md).
Constraint: no Python or Node runtime in the product path, from the program
constraints in [INDEX.md](INDEX.md) and the roadmap.

* Author: whit3rabbit
* Status: Implemented (v1 subset; deviations listed under "Implementation record")
* Date: 2026-11-11 (implemented 2026-11-14)

--------------------------------------------------------------------------------

## Overview

Let the agent create new docx, xlsx, pptx and pdf files. ZCode does this with
four skill plugins that drive JavaScript and Python libraries (docx-js,
python-pptx and an xlsx writer). TurboSpark cannot bundle Node or Python in the
product path, so the same outcome ships as ONE native tool, `document_write`,
that dispatches to per-format pure Swift writers over a shared in-process OOXML
package writer, plus four bundled skills that teach the agent the spec and its
limits. Features outside the subset go through a skill-level fallback to
user-installed tooling, always through the ordinary Bash tool and its ladder.

## The Problem

TurboSpark's only document writing is chat export. An agent asked for a report,
a deck, or a spreadsheet has nothing to write with, and ZCode's implementations
need a JS or Python runtime the roadmap forbids bundling.

## Goals

* One agent-facing tool, `document_write(path, spec)`, writing docx, xlsx, pptx
  and pdf from a bounded JSON spec (v1 subset below).
* Four bundled skills (docx, xlsx, pptx, pdf) teaching the spec shape, limits,
  recipes and fallback, without any JS or Python tool references. Skill
  structure follows ZCode's: a recipe per common document, then a postcheck
  section.
* A mandatory tool-side postcheck on a staged file. A failed check writes
  nothing and is reported once.
* Output lands in the project (under `resolveSecurePath`), is registered as a
  chat artifact, and opens in the ZP-08 viewers for review.

## Non-Goals

* No bundled Python or Node runtime.
* No editing of existing office files; the tool creates new files only and
  never overwrites.
* No macro output (docm, xlsm, pptm are refused), no external links, no
  embedded objects.
* No template marketplace or cover-art assets.
* No Create-entry UI. The agent reaches the tool through chat and the skills,
  so no new UI strings beyond one tool label.
* v1 deferrals: xlsx formulas, charts, merges, validation; pptx pictures,
  tables and shapes; docx headers, footers, hyperlinks, fields; pdf links and
  forms.

## Decisions (resolving the earlier open questions)

* **Native versus delegation**: native Swift, no subprocess in any writer. The
  four writers and the shared zip writer are pure functions over bytes and spawn
  nothing. Delegation to user-installed tooling exists only as skill text that
  runs through `Bash`.
* **Writer scope** (v1): docx headings, runs, lists, tables, PNG/JPEG images,
  page breaks. xlsx cells with named number formats, fills, widths, frozen rows.
  pptx title and content slides with bullets. pdf via Core Graphics and Core
  Text with the docx block vocabulary. The subset was chosen as the smallest
  that covers a report, a data sheet, a plain deck and a handout, and everything
  outside it is stated in the skills.
* **PDF path**: native Core Graphics and Core Text. PDFKit only READS (used by
  the postcheck); it cannot write. `NSAttributedString` document export was
  considered and not used: it does not give page-size, table or image control
  and routes through AppKit text views. LaTeX stays out of scope.
* **Artifact folder**: there is no per-chat artifact folder. Files are written
  into the project at the path the agent chooses and registered as artifacts
  through `ArtifactRegistrar`.

## Critical User Journeys

### CUJ 1: agent writes a report

* **Given**: the user asks for a formatted project report as docx, in a project.
* **When**: the agent loads the docx skill and calls `document_write` with
  `path: "report.docx"` and a block spec.
* **Then**: the file is built, staged, postchecked (archive, content-types,
  relationships, render), moved into the project, and registered as an
  artifact. The result names the checks that passed. The user reviews it in the
  ZP-08 viewer.

### CUJ 2: fallback path

* **Given**: the user needs a feature outside the subset, for example a pivot
  table or conditional formatting in an xlsx.
* **When**: the skill says which feature is out of the subset and offers one
  shell-out to tooling the user already has installed, checked first with
  `command -v`.
* **Then**: the Bash tool asks per the ladder and the script runs, writing a NEW
  path. Its output is untrusted: no macro-enabled file, no overwrite, and the
  agent reads it back or asks the user to review. If the tooling is missing the
  skill reports the gap and stops. The native postcheck does not cover this
  path, and the skill says so.

### CUJ 3: pdf handout

* **Given**: the user asks for a PDF handout with a chart image already in the
  project.
* **When**: the agent uses the pdf skill with an `image` block.
* **Then**: Core Graphics and Core Text produce a text-and-image PDF at the
  chosen page size, and the postcheck confirms PDFKit opens it, every page has
  the expected size, and text is extractable.

## Technical Plan (Architecture)

```text
Bundled skills (docx, xlsx, pptx, pdf): spec shape, limits, recipes, fallback
        |
        v
document_write(path, spec)            category .fileWrite, workspace-rooted
  path: resolveSecurePath + validateWritePath + sensitive path + extension
        + must not exist
  spec: DocumentSpec.decode -> DocumentSpecValidation (caps, shapes, colors)
  images: PNG/JPEG by magic bytes, project-confined, regular file, bounded
        |
        v
Per-format pure writers                DocxWriter XlsxWriter PptxWriter PdfWriter
  OOXMLPackage -> ZipArchiveWriter     in-process zip (Compression framework)
        |
        v
Stage in private temp dir -> DocumentPostcheck
  archive, content-types, relationships, render  (pdf: pdf)
        |  pass                          |  fail: nothing written, one report
        v
moveItem into project (fails if it exists) -> ArtifactRegistrar -> ZP-08 viewers
```

The zip writer is new code (`OOXMLPackageWriter.swift`), not a half of
`OfficeArchive`: that type is a read-only `/usr/bin/unzip` wrapper and stays so.
The package reader used by the postcheck is also in-process and separate from
`OfficeArchive`, because `unzip` reads its entry argument as a wildcard and
cannot address `[Content_Types].xml`, and because an independent reader proves
more than the render path that shares parsers with the writer.

## Dependencies and sequencing

* ZP-08 (office rendering, `57324bcf`) is a prerequisite for the review step
  and for the postcheck's `render` check.
* ZP-07 owns the bundled-skills root: skills live in
  `Sources/TurboSparkAppResources/BundledSkills/<name>/SKILL.md` (resources are
  in the `TurboSparkAppResources` target, not `TurboSparkApp`), and
  `SkillManager.discoverBundledSkills()` already discovers them, so no
  SkillManager change is needed.
* ZP-19 (`image_search`, `image_fetch`) supplies web images; the docx skill
  tells the agent to cite the source page.

## Alternatives Considered

### Alternative A: port ZCode's skills verbatim

* **Pros**: battle-tested content.
* **Cons**: they call docx-js, python-pptx and bundled scripts, so adopting
  them means a JS and Python runtime in the product path.
* **Reason for Rejection**: violates the runtime constraint.

### Alternative B: generate HTML and convert

* **Pros**: HTML is easy for the model.
* **Cons**: conversion needs an external converter (pandoc or an office suite)
  that the app cannot assume is installed and cannot version. Fidelity is
  whatever that converter does, and the conversion would be a second, unaudited
  writer.
* **Reason for Rejection**: unpredictable output and an assumed external tool.

### Alternative C: skills only, model writes raw OOXML with `write_file`

* **Pros**: no new tool.
* **Cons**: a zip is binary and `write_file` writes text; the model would need a
  shell zip step per file, with no escaping, size or relationship guarantees and
  no postcheck.
* **Reason for Rejection**: no enforcement point for any safety property.

### Alternative D: a third-party OOXML library

* **Pros**: more features.
* **Cons**: a new dependency, which needs a separate decision, and a large
  surface to audit for hostile input.
* **Reason for Rejection**: not needed for the v1 subset; revisit if the subset
  proves too small.

### Alternative E: system exporters (`NSAttributedString` docx, `NSPrintOperation` pdf)

* **Pros**: no hand-built XML or layout.
* **Cons**: AppKit text-system dependence, no xlsx or pptx equivalent, and little
  control over tables, page size and package contents.
* **Reason for Rejection**: covers one of four formats poorly.

## Non-Functional Requirements

* **Security**: the tool writes only inside the project root through
  `resolveSecurePath` and `validateWritePath`, refuses sensitive paths, never
  overwrites, and the writers open no network connection and spawn no process.
  The shell-out fallback goes through the ordinary Bash ladder and its output is
  untrusted.
* **Ladder position**: category `.fileWrite`, workspace-rooted, so a projectless
  chat cannot call it. It answers to the project File write permission
  (deny, ask, auto), the sensitive-path hard gate and the tool profiles
  (Read-only removes it). Registered in `AppToolCatalog.builtInCategory`,
  `AppToolRegistry.supportedToolNames` and `workspaceRootedToolNames`.
* **Quality**: the postcheck is part of the tool, so no skill can skip it. A
  failed postcheck is reported to the user, not retried silently. Its limit: it
  proves a well-formed package the ZP-08 viewer reads, not that Word, Excel or
  PowerPoint opens it.
* **Localization**: skill content is English (agent-facing, as in ZCode). The
  only UI string is the tool label "Write Document", in all 21 catalog
  languages.

## Risks and Mitigations

* **Size and resource exhaustion**: caps on spec bytes, blocks, table cells,
  rows, columns, slides, pages, images (count, bytes, pixels) and output bytes,
  all below `OfficeRenderLimits`. A spec over a cap is refused whole.
* **Formula injection**: text is always a shared string and no formula is ever
  written, so `=`, `+`, `-`, `@` prefixes are inert. Formulas are a v1 non-goal.
* **XML injection**: all text is escaped and XML-forbidden characters dropped;
  colors are six hex digits; number formats are a closed set; sheet names follow
  Excel's rules. Tests feed hostile strings and re-parse.
* **External content**: no API emits an external relationship, macro part,
  custom XML or hyperlink; the postcheck rejects any.
* **Overwrite and clobbering**: never overwrites; the move into the project
  fails if the path exists, closing the check-then-write race.
* **Fonts and scripts**: docx names Calibri and pdf uses Helvetica with system
  fallback. CJK and right-to-left in pdf rely on Core Text and are not verified
  against a reference rendering; the pdf skill says to ask the user to inspect.
* **Real Office compatibility**: not verified here. Mitigation is a
  schema-ordered, minimal-parts writer and an honest statement in the tool
  result and the docs; a manual open in Word, Excel and PowerPoint is a listed
  follow-up.
* **ZP-08 sequencing**: if the loader is unavailable the `render` check cannot
  run, so ZP-09 must not ship without ZP-08. Both are in the tree.

## Implementation record

Files (under `swift/TurboSparkApp/`):

* `Sources/TurboSparkApp/Presentation/DocumentWriter/`: `DocumentSpec.swift`,
  `DocumentSpecValidation.swift`, `OOXMLPackageWriter.swift`,
  `DocxWriter.swift`, `XlsxWriter.swift`, `PptxWriter.swift`, `PdfWriter.swift`,
  `DocumentImage.swift`, `DocumentPostcheck.swift`.
* `Sources/TurboSparkApp/Tools/File/DocumentWriteTool.swift` (schema and
  executor), wired in `AppToolRegistry.swift`, `AppToolRegistry+Vocabulary.swift`,
  `AppToolCatalog.swift` and `ToolPresentation.swift`.
* `Sources/TurboSparkAppResources/BundledSkills/{docx,xlsx,pptx,pdf}/SKILL.md`.
* `Tests/TurboSparkAppTests/DocumentWriterTests.swift`.
* `Localization/Localizable.xcstrings`: "Write Document" in 21 languages.
* `swift/docs/SWIFT_DOCUMENTS.md` (new), `SWIFT_SKILLS.md`, `Tools/File/EGM.md`.

Deviations from the original draft:

* One tool instead of four writer entry points, and the writers are pure
  functions the tool dispatches to (review B2, M8).
* No SkillManager change: bundled skills are discovered already (review M9).
* No artifact folder: files go into the project (review M2).
* A new in-process zip writer and a separate zip reader for the postcheck; the
  postcheck lives in the tool and checks content types, relationships and spec
  counts, not only a re-parse (review B1, M3).
* v1 subset cut (review M1): no xlsx formulas or charts, no pptx pictures,
  tables or shapes, docx tables have no merges.
* Never overwrites at all, rather than overwriting under permission, because a
  binary file cannot satisfy the read-before-overwrite gate.
* "Shell-out absence" is tested as "the writers spawn no process" (review M6).
* The Create-entry UI and its 21-language strings were dropped (review M2).
* Not verified: opening generated files in real Word, Excel or PowerPoint, and
  CJK or right-to-left PDF text against a reference rendering.

### Verification Plan

* **Build**: `swift build` in `swift/TurboSparkApp`.
* **Unit tests**: `swift test --filter DocumentWriterTests`: per-format
  round trips through `OfficeRenderModel`, `OfficeArchive` and the independent
  zip reader; hostile text; confinement and no-overwrite; limits; ladder
  position; postcheck pass and fail paths with mutation checks (corrupt content
  types, CRC, relationships, counts); the no-process scan with its own mutation
  check; skill discovery and native-only text.
* **Localization**: the parity gates and `make design-check`.
* **Manual follow-up**: generate one document per format from chat, review each
  in the ZP-08 viewers, and open each in the real Office application.

--------------------------------------------------------------------------------

*Drafted with EGM*
