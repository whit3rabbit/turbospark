---
name: pptx
description: Create a new PowerPoint deck (.pptx) in the project with the native document_write tool: title slides and content slides with bullets and text.
when_to_use: Use when the user asks for a presentation or slide deck as .pptx. Do not use it for pictures, charts, tables or shapes on slides, or for editing an existing deck (v1 writes text slides only).
---

# Writing a .pptx

Call `document_write` once with a `path` ending in `.pptx` and a `spec` (a JSON
object, passed as a string). Native writer: no PowerPoint, no Python, no Node.

## Spec

```
{"slides": [
  {"layout": "title", "title": "Launch plan", "subtitle": "Q4 2026"},
  {"layout": "content", "title": "Goals",
   "bullets": ["Ship the beta", {"text": "Fix top 10 bugs", "level": 1}],
   "paragraphs": ["One plain sentence under the bullets."]},
  {"title": "Risks", "bullets": ["Schedule", "Staffing"]}
]}
```

- `layout` is `title` (centered title, optional subtitle) or `content`
  (default: title, then paragraphs, then bullets).
- Bullets are strings or `{"text", "level"}` with level 0 to 2.
- Slides are 16:9, one built-in layout, default theme. The deck is as plain as
  that: say so instead of promising design.

## Limits

100 slides, 16 bullets and paragraphs per slide, 1,000 characters per text,
2 MiB of spec, 20 MiB of output. Font size drops automatically for dense
slides (more than 7 items); better to split the slide.

## Recipes

- One idea per slide; at most six short bullets; no sentences longer than a
  line.
- Open with a `title` slide, end with a `content` slide of next steps.
- Put anything that is a picture, chart or table in the fallback path below, or
  describe it in a bullet and tell the user.

Steps: (1) choose a NEW path, (2) outline the deck for yourself, (3) call
`document_write`, (4) relay the result.

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
