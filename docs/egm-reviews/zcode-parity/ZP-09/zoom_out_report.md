# Stage 5: Initial zoom out report (ZP-09)

Target: docs/designs/zcode-parity/ZP-09-office-document-generation.md (full file, 188 lines)
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out reviewer (ZO)

- ZO-1 Blocker, lines 105, 118-119. "OfficeArchive's writer half" and "hardened zip access" do not exist. OfficeArchive.swift is a read-only wrapper over /usr/bin/unzip (list/extract, timeouts, byte budgets). No zip write path exists; three writers depend on it. A /usr/bin/zip shell-out would contradict "never shell out". Fix: add an explicit OOXML package writer component with a chosen zip strategy; drop the writer-half wording.
- ZO-2 Major, lines 15-20, 94-116. The agent-facing interface is undefined ("writer tools or REPL file bridge"). No tool names, schema, or AppToolPermissionEngine ladder position (INDEX requires one). Confinement has nothing to bind to. Fix: one document-write tool taking a JSON spec per format, state ladder position.
- ZO-3 Major, lines 45, 74, 141, 168. "Chat artifact folder" and "artifact panel" are assumed; no symbol found. ZP-08 and ZP-07 dependencies are stated only mid-paragraph. Fix: name the owner or make it a prerequisite; state ZP-07 and ZP-08 dependencies.
- ZO-4 Major, lines 33-37, 56-63. Scope: four writers including xlsx charts, pptx shapes, in one draft; subset left to "decide at implementation". NSAttributedString officeOpenXML export is not evaluated for docx. Fix: explicit v1 cut line, defer charts, evaluate Cocoa export.
- ZO-5 Major, line 82. CUJ 2 example "Excel chart with VBA-created elements" is incoherent (VBA needs xlsm). Fix: realistic example.
- ZO-6 Major, lines 41-44, 144-145. Postcheck location for shell-out output is unstated; skill-text postcheck can be skipped. Fix: say tool-side versus agent-invoked per path.
- ZO-7 Nit, lines 154-158, 118. Writers sit under Presentation beside readers; bundled-skills root does not exist yet (ZP-07). Fix: placement note, mark ZP-07 prerequisite.
- ZO-8 Nit, lines 175-177. Docs item garbled. Fix: "create swift/docs/SWIFT_DOCUMENTS.md".
- ZO-9 Nit, lines 146-148, 178-180. "Create-entry UI strings" and "21 languages" refer to UI absent from the plan; INDEX names ten parity gates. Fix: add the UI or drop the claim.
- ZO-10 Nit, lines 171-172. "Shell-out absence" test not meaningful. Fix: optional.
- ZO-11 Nit, lines 123-137. Alternatives thin: no system exporters option, no skills-only baseline. Fix: add both.

## Inventive catalyst (IC)

- IC-1 Major, lines 104-107, 119. Same false writer-half premise. Fix: pure in-process OOXMLPackageWriter (stored or deflate, CRC32), format writers as pure functions to part dictionaries; add to file list and test it.
- IC-2 Major, line 101 and security NFR. Same undefined interface. Fix: one create_document tool with a bounded JSON spec; validator is the single gate; confinement in the tool.
- IC-3 Major, lines 111-112, 168. Postcheck shares OfficeXMLParsers with the tests and ZP-08 render, so writer and reader can be wrong together; weak proxy for "opens in Word". Fix: independent structural validator (content types, rels, targets, index bounds), ZP-08 render as second layer, native cross-checks, golden fixtures.
- IC-4 Major, lines 144-145, CUJ 2. Failure is only reported to the user. Fix: return diagnostics to the agent with bounded retry; write to temp and move after postcheck passes.
- IC-5 Major, lines 56-63, CUJ 2-3. Scope is binary native-vs-shell-out; most Macs lack python libs; CUJ 3 chart image has no source (ZP-19 unreferenced). Fix: raw-part passthrough with deny-list (no macros, no external rels), or phase charts; cite ZP-19.
- IC-6 Major, files 6-10, line 118. Skills hand-tuned text can drift from writer limits; ZP-07 and ZP-19 not listed as dependencies. Fix: generate limits from the tool schema; list dependencies.
- IC-7 Major, lines 108, 157. CG has no text flow, pagination, or CJK/RTL fallback; Core Text or NSPrintOperation not named. Fix: name Core Text framesetter or NSAttributedString print path.
- IC-8 Nit, lines 165-174. No size/time caps; no symlink or ".." confinement test. Fix: add caps and test.
- IC-9 Nit, lines 7-9, 130-133, 175-176. Alternative B says pandoc/LibreOffice need the same runtime, but neither is Python or Node; real objection is install and fidelity. Fix: reword.
