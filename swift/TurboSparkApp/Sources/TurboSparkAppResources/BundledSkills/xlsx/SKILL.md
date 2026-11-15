---
name: xlsx
description: Create a new Excel workbook (.xlsx) in the project with the native document_write tool: sheets, typed cells, number formats, fills, column widths, frozen headers.
when_to_use: Use when the user asks for a spreadsheet or workbook as .xlsx with data already known. Do not use it for formulas, charts, pivot tables or editing an existing workbook (the writer creates new, values-only files).
---

# Writing an .xlsx

Call `document_write` once with a `path` ending in `.xlsx` and a `spec` (a JSON
object, passed as a string). Native writer: no Excel, no Python, no Node.

## Spec

```
{"sheets": [{
  "name": "Sales",
  "columns": [24, 14, 14],
  "freeze_rows": 1,
  "rows": [
    [{"v": "Region", "bold": true, "fill": "D9E1F2"}, {"v": "Units", "bold": true, "fill": "D9E1F2"}, {"v": "Revenue", "bold": true, "fill": "D9E1F2"}],
    ["EMEA", 120, {"v": 1200.5, "format": "currency"}],
    ["APAC", 90, {"v": 0.15, "format": "percent"}],
    ["Closed", {"v": "2026-01-31", "type": "date"}, true]
  ]
}]}
```

- A cell is a string, a number, `true`/`false`, `null` (blank), or an object
  with `v` plus optional `type` (`"date"` for a `YYYY-MM-DD` string),
  `bold`, `italic`, `color` and `fill` (six hex digits), `format`, `align`.
- `format` is one of: `general`, `integer`, `decimal`, `thousands`,
  `thousands_decimal`, `percent`, `percent_decimal`, `currency` (US dollar),
  `date`. A format code of your own is not accepted.
- Text is always stored as text. A string that starts with `=`, `+`, `-` or
  `@` stays a string and never runs as a formula. Compute totals yourself and
  write the number.
- Column widths are in characters (2 to 80). `freeze_rows` pins header rows.
- Sheet names: 1 to 31 characters, none of `[ ] : * ? / \`, no edge apostrophe,
  unique ignoring case.

## Limits

16 sheets, 1,000 rows and 64 columns per sheet, 30,000 cells in the workbook,
4,000 characters per text cell, 2 MiB of spec, 20 MiB of output.

## Not supported (say so, do not fake it)

Formulas, charts, pivot tables, merged cells, conditional formatting, data
validation, borders, images, macros. Offer the fallback below for those.

Steps: (1) choose a NEW path, (2) put a header row first and freeze it, (3)
set widths, (4) call `document_write`, (5) relay the result.

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
