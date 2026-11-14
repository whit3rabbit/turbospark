# Swift document generation (ZP-09)

How the agent creates docx, xlsx, pptx and pdf files. Design record:
`docs/designs/zcode-parity/ZP-09-office-document-generation.md`. Rendering the
result for review is ZP-08 (`Presentation/Office*.swift`).

Keep all code, comments and docs ASCII: no emojis and no em dashes.

All Swift paths are relative to `swift/TurboSparkApp/Sources/TurboSparkApp/`.

## Shape

One tool, `document_write` (`Tools/File/DocumentWriteTool.swift`), takes a
project-relative `path` (the extension picks the format) and a `spec`, a JSON
object passed as a string. The executor validates the path, decodes and bounds
the spec, loads any images, calls one pure writer per format, stages the bytes
in a private 0700 temp folder, runs the postcheck there, and only then moves
the file into the project.

```
document_write(path, spec)
  resolveSecurePath + validateWritePath + sensitive-path + extension + no-overwrite
  DocumentSpec.decode -> DocumentSpecValidation.validate   (limits, shapes, colors)
  load images (PNG/JPEG by magic bytes, project-confined, regular file)
  DocxWriter | XlsxWriter | PptxWriter | PdfWriter          (pure functions)
  OOXMLPackage -> ZipArchiveWriter (in-process, Compression framework)
  stage in temp dir -> DocumentPostcheck -> moveItem into the project
```

Files, all under `Presentation/DocumentWriter/` except the tool:

| File | Role |
|---|---|
| `DocumentSpec.swift`, `DocumentSpecValidation.swift` | The spec types and every limit |
| `OOXMLPackageWriter.swift` | CRC-32, zip writer, zip reader (postcheck only), content types, XML escaping |
| `DocxWriter.swift`, `XlsxWriter.swift`, `PptxWriter.swift` | OOXML parts per format |
| `PdfWriter.swift` | Core Graphics + Core Text layout and flow |
| `DocumentImage.swift` | Bounded PNG/JPEG intake |
| `DocumentPostcheck.swift` | The mandatory check |
| `Tools/File/DocumentWriteTool.swift` | Tool schema and executor |

Bundled skills `docx`, `xlsx`, `pptx`, `pdf` live in
`TurboSparkAppResources/BundledSkills/` and are discovered by
`SkillManager.discoverBundledSkills()` with no code change.

## v1 subset

- docx: headings 1 to 6, styled runs, bullet and numbered lists (each ordered
  list restarts at 1), tables with an optional header row, PNG/JPEG images,
  page breaks, letter or A4.
- xlsx: sheets, string/number/boolean/date cells, bold/italic, text and fill
  colors, eight named number formats, alignment, column widths, frozen header
  rows. No formulas, charts, merges, validation, conditional formats, borders.
- pptx: title and content slides (title, subtitle, bullets to three levels,
  paragraphs), 16:9, one built-in master, layout and theme. No pictures,
  tables, shapes, charts, notes.
- pdf: headings, runs, lists, equal-column tables, images, page breaks, flowing
  over pages, Core Text shaping. No links, forms, LaTeX, headers or footers.

Anything else is a non-goal for v1 and goes through the skills' fallback (a
user-installed tool run through the ordinary Bash tool and its prompt).

## Security invariants

- The writers never spawn a process, open a socket or read the filesystem.
  `DocumentWriterTests.testWritersNeverSpawnAProcessOrTouchTheNetwork` scans the
  directory (with a mutation check on the scanner). The ZP-08 `OfficeArchive`
  `unzip` wrapper stays read-only and is used only by the postcheck's `render`
  step and by tests.
- Spec text is hostile. `OOXMLText.escape` escapes the five XML characters and
  drops characters XML 1.0 forbids. Colors must be six hex digits. Number
  formats are a closed set of names. Sheet names follow Excel's rules.
- No external relationship can be emitted: `OOXMLPackage.addRelationships` takes
  internal targets only and nothing writes `TargetMode`. No macro part, no
  custom XML, no embedded object, no hyperlink.
- A text cell is always a shared string, never a formula, so a value such as
  `=cmd|...` or `@SUM(...)` cannot execute (formula injection). That is also
  why formulas are not in v1.
- Images are PNG or JPEG by magic bytes only (never SVG), at most 8 MiB and 40
  megapixels, a regular file inside the project after symlinks are resolved (a link out of
  the project is refused), never a sensitive path.
- Output: 20 MiB, 2 MiB of spec, caps on blocks, rows, cells, slides and pages
  (`DocumentWriteLimits`) set below `OfficeRenderLimits`, so a file written here
  always opens in the ZP-08 viewers. A spec over a limit is refused whole.

## Permission ladder

`document_write` is category `.fileWrite` and workspace-rooted (a projectless
chat cannot call it). It answers to the project File write permission, the
sensitive-path hard gate and the tool profiles exactly as `write_file` does
(`AppToolCatalog.builtInCategory`, `ToolRiskClassifier`). It NEVER overwrites,
even with permission: a binary file the agent cannot have read cannot satisfy
the read-before-overwrite gate. The agent picks a new path.

A produced document is registered as a chat artifact (`ArtifactRegistrar`,
origin `.sentToUser`), so it appears in the artifact panel and opens in the
ZP-08 viewers.

## Postcheck

Runs inside the tool on the staged file. A failure writes nothing and is
reported once with the failing check's name.

1. `archive`: `ZipArchiveReader` parses the central directory and checks every
   member's CRC and size, plus name safety and uniqueness.
2. `content-types`: `[Content_Types].xml` parses, covers every part, and
   declares the main part with the expected type. The ZP-08 loader never reads
   this part, and `unzip` cannot address it (its name is a wildcard pattern), so
   this check is what catches a package Word would call corrupt.
3. `relationships`: every relationship target exists, none is external, none
   escapes the package.
4. `render`: `OfficeRenderModel.load` opens it; heading, table, sheet and slide
   counts match the spec; there is readable text.

PDF runs `pdf`: `%PDF-` header, PDFKit opens it, every page has the expected
size, extractable text exists.

Mutation checks in `DocumentWriterTests` corrupt content types, CRCs,
relationships and counts and assert the matching check fails.

## Limits and honest gaps

- The postcheck proves a well-formed package that the app's own viewer reads. It
  does NOT prove Word, Excel or PowerPoint opens it: no Office application was
  available to verify against. Treat generated files as unverified in real
  Office until someone opens them there.
- PDF text uses Helvetica with system fallback fonts. CJK and right-to-left
  scripts rely on Core Text and are not checked against a reference rendering.
- Zip members use raw deflate when it shrinks them, otherwise stored. No zip64,
  encryption or data descriptors (the reader refuses them).
- No editing of existing files, no templates, no cover art.
