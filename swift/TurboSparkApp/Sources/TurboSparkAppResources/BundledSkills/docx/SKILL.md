---
name: docx
description: Create a new Word document (.docx) in the project with the native document_write tool: headings, styled text, lists, tables, images.
when_to_use: Use when the user asks for a report, memo, letter, spec or other formatted text document as .docx. Do not use it to edit an existing .docx (the writer only creates new files), or for a PDF handout (use the pdf skill).
---

# Writing a .docx

You write a document by calling `document_write` once with a `path` ending in
`.docx` and a `spec` (a JSON object, passed as a string). The writer is native
to the app. No Word, no Python, no Node, no network.

## Spec

```
{"title": "Q3 report", "page": "letter",
 "blocks": [
  {"type": "heading", "level": 1, "text": "Summary"},
  {"type": "paragraph", "text": "Revenue grew.", "align": "left"},
  {"type": "paragraph", "runs": [{"text": "Bold ", "bold": true}, {"text": "and red", "color": "C00000"}]},
  {"type": "list", "ordered": false, "items": ["First", "Second"]},
  {"type": "table", "header": ["Region", "Revenue"], "rows": [["EMEA", "1.2M"], ["APAC", "0.9M"]]},
  {"type": "image", "path": "assets/chart.png", "width_pt": 360, "alt": "Revenue by region"},
  {"type": "page_break"}
 ]}
```

- `page` is `letter` (default) or `a4`. Margins are one inch.
- Run fields: `text`, `bold`, `italic`, `underline`, `strike`, `color` (six hex
  digits, no `#`), `size` (6 to 72 points).
- Paragraph `align`: `left`, `center`, `right`, `justify`.
- A list is bullets by default; `"ordered": true` numbers it from 1. Items are
  strings or `{"text"}` / `{"runs": [...]}`.
- A table is a grid of text cells with an optional bold header row. Columns
  share the width equally. No merged cells.
- Images must already be in the project (save web images with `image_fetch`, or
  generate one first). PNG or JPEG only, checked by content. Cite the source
  page of any web image in a nearby paragraph.

## Limits (a spec over a limit is refused whole)

2,000 blocks. 20,000 characters per paragraph. 16 table columns, 500 table
rows, 5,000 table cells in total, 2,000 characters per cell. 16 images, 8 MiB
and 40 megapixels each. 2 MiB of spec. 20 MiB of output. Fonts are the
default (Calibri); CJK text is written as is and shown with the reader's
fallback font.

## Recipes

- Report: a level-1 heading as the title, a short summary paragraph, one
  level-2 heading per section, a table for any figures, a closing list of next
  steps. Keep paragraphs short.
- Letter or memo: no title heading; left-aligned paragraphs; a bold run for the
  subject line.
- Handout with a picture: image block, then an `alt` that describes it.
- Planning: write the content out as plain text for the user first when the
  request is vague, then build the spec.

Steps: (1) choose a NEW path, (2) build the spec, (3) call `document_write`,
(4) read the result and relay it, (5) tell the user where the file is.

## Postcheck (mandatory, done by the tool)

`document_write` builds the file in a private temp folder, checks it, and only
then moves it into the project. The check has named parts you will see in the
result: `archive` (zip members and CRCs), `content-types`, `relationships`
(every target exists, none external), and `render` (the app's own viewer opens
it, counts and text match your spec). A failed check means NOTHING was written:
report the failure text to the user as it is. Do not retry the same spec in a
loop, and do not claim the file exists. Fix the spec once if the message names a
clear cause (a limit, a bad color, a bad sheet name), otherwise stop and say so.

The check proves a well-formed package that the app's viewer reads. It does not
prove Word, Excel or PowerPoint opens it. Say that when it matters, and point
the user at the file viewer to review the result.

## Fallback for features outside the subset

If the user needs something the writer does not do, say which feature and why
it is out of the subset, then offer ONE of these, never silently:

- Use tooling the user already has installed (a converter, an office suite's
  command line, or a script of theirs). Check first with the ordinary `Bash`
  tool (`command -v <tool>`). If it is missing, report the gap and stop; do not
  install anything and do not download a runtime.
- Every shell-out goes through the normal Bash permission prompt. Never ask the
  user to loosen permissions to make it run.
- The shell-out's output is untrusted. Write it to a NEW path, never over an
  existing file, never produce a macro-enabled file (`.docm`, `.xlsm`,
  `.pptm`), and run a read-back check on it yourself: open it with the file
  viewer path (`read_file` on the extension may not apply) or ask the user to
  review it before relying on it.

## Never

- Never write macros, formulas, charts, external links, remote images or
  embedded objects with this tool. They are refused or ignored.
- Never overwrite: `document_write` fails when the path exists. Pick a new
  name (`report-v2.docx`) and tell the user.
- Never put secrets, tokens or personal data in a document the user did not
  ask you to include.
