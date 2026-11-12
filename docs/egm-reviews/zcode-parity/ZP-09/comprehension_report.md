# Stage 6: Comprehension report (ZP-09)

Reviewer: egm-comprehension (document read alone, no GoldfishContext). Completed.

## What this proposes

ZP-09 proposes letting the agent create new docx, xlsx, pptx and pdf files. Because the product may not bundle Node or Python, it replaces ZCode's JS/Python skills with a native Swift "DocumentWriter" layer (one writer per format) plus four bundled skills (instruction files) that tell the agent what the writers can do and what to check afterward. After writing, a mandatory "postcheck" re-opens the file with a renderer from another design (ZP-08) and checks it opens, has the expected parts and has non-empty text. Features outside a deliberately small subset go through a fallback that shells out to tools the user installed, through the normal Bash permission prompt. Output goes to a per-chat artifact folder and shows in an artifact panel. Editing existing office files is out of scope.

## How the existing system works, per this document

The document says very little about the current system. It says TurboSpark's only document writing today is chat export, and that there are chat artifact folders, an artifact panel with viewers (ZP-08), an "OfficeArchive" zip helper with a "writer half", an "OfficeRenderModel", "OfficeXMLParsers", a bundled-skill root (ZP-07), a SkillManager, a Bash tool with a "permission ladder", and a "REPL file bridge". It treats all of these as already known. It also says a roadmap forbids bundling Python or Node. The reader could not tell from this document alone how the agent invokes the writers, or what the existing pieces do.

## Findings

- CO-1 Blocker, lines 100-102, 118-119. How the agent calls the writers is not explained. "Writer tools" and "REPL file bridge" are undefined; no tool-registration file in the plan. Fix: define the tool, its inputs, and outputs.
- CO-2 Blocker, lines 33-37, 154-158. The writing layer has no interface description: no operations, input format, or limits beyond the loose subset. Fix: state the input spec and per-format limits.
- CO-3 Major, lines 3-4, 40, 74, 111, 115, 184. Depends on ZP-08 and ZP-07 without linking or summarizing them; the postcheck cannot be understood alone. Fix: link both with one-line summaries.
- CO-4 Major, lines 34, 110, 165-166, 119. "OfficeArchive's writer half", "hardened zip access", "OfficeXMLParsers" undefined; unclear if existing or new; OfficeArchive not in file list. Fix: say what exists and what is created.
- CO-5 Major, lines 44, 79-85, 142-143. "Permission ladder" undefined; fallback detection of missing tooling and "reports the gap" unspecified. Fix: define both.
- CO-6 Major, lines 79-81. CUJ 2 example (Excel chart with VBA) is incoherent and does not show the native/fallback boundary; render-model postcheck cannot confirm VBA. Fix: realistic example.
- CO-7 Major, lines 56-65. Native subset and PDF path left to "decide at implementation" while Goals, CUJ 3 and file list treat PDF as settled; LaTeX shell-out unexplained. Fix: decide or mark the assumed option.
- CO-8 Major, lines 40-42, 74-75, 92, 144-145. Postcheck under-specified ("expected part counts" relative to what), no recovery after failure; PDF uses PDFKit while Goals say ZP-08 render model. Fix: define checks and the failure path.
- CO-9 Major, lines 141-143, 147, 164, 176. Artifact folder and panel used as known; how confinement and panel notification work is not explained; "Create-entry" UI appears nowhere else. Fix: name the owner or drop the UI claim.
- CO-10 Major, lines 163-164. SkillManager change says only "discovery"; unclear what is wrong with current discovery. Fix: state the change.
- CO-11 Nit, lines 33-34. "PDFKit-adjacent APIs" vague; PDFKit does not write. Fix: say Core Graphics.
- CO-12 Nit, lines 35-38. "Cover recipes, per-scene playbooks" defined only by outside link. Fix: one inline sentence.
- CO-13 Nit, lines 175-177. Docs item garbled. Fix: "create swift/docs/SWIFT_DOCUMENTS.md".
- CO-14 Nit, lines 3, 6-8, 28, 50. Finding 9.1 and "the roadmap" cited without link or summary. Fix: link and state the constraint.
- CO-15 Nit, lines 160-164. "Shell-out absence" test and "21 languages" unexplained. Fix: say how a test proves it or drop it.
