---
name: pdf
description: Create a new PDF handout or report (.pdf) in the project with the native document_write tool: headings, paragraphs, lists, tables and PNG or JPEG images, flowing across pages.
when_to_use: Use when the user asks for a PDF handout, one-pager or printable report. Do not use it for LaTeX or equation typesetting, fillable forms, or editing an existing PDF.
---

# Writing a .pdf

Call `document_write` once with a `path` ending in `.pdf` and the same block
`spec` the docx skill uses (a JSON object, passed as a string). The writer is
native (Core Graphics and Core Text): no LaTeX, no Python, no Node.

## Spec

```
{"title": "Trail guide", "page": "a4",
 "blocks": [
  {"type": "heading", "level": 1, "text": "Hawk Ridge loop"},
  {"type": "paragraph", "text": "6 km, moderate."},
  {"type": "list", "ordered": true, "items": ["Park at the lot", "Take the north fork"]},
  {"type": "table", "header": ["Mile", "Landmark"], "rows": [["1", "Creek"], ["2", "Overlook"]]},
  {"type": "image", "path": "assets/map.png", "width_pt": 400, "alt": "Route map"},
  {"type": "page_break"}
 ]}
```

Block types, run fields, `align`, `page` (`letter` default or `a4`), and limits
are as in the docx skill (2,000 blocks, 16 images). Extra for pdf: at most 200
pages. A table row never splits across pages; a row taller than a page is an
error. Links, bookmarks, forms and headers or footers are not produced.

## Limits of the layout

- Body text is Helvetica 11 pt with system fallback fonts. CJK is drawn with
  the fallback font; right-to-left and complex scripts rely on Core Text and
  have not been checked against a reference rendering. For those, tell the
  user to inspect the result.
- Images are scaled to the page width and moved to the next page if they do not
  fit.

Steps: (1) choose a NEW path, (2) build the spec, (3) call `document_write`,
(4) relay the result: the page count and extractable character count come from
the postcheck.

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
