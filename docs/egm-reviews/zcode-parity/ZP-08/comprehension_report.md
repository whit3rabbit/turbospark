## What this proposes

Add native (SwiftUI) viewers for docx, xlsx and pptx files in TurboSpark's two preview entry points: attachment preview and the artifact panel for agent-written files. Today these formats are treated as "opaque" and offer only "Open with default app". The viewers would be built on the existing hardened zip/XML access used for text extraction, in a new parse layer (OfficeRenderModel) plus a kind-dispatched container (OfficeDocumentView). PDF, already rendered by PDFKit, is formalized as the fourth viewer. The viewers are strictly read-only, with no editing, no macros, no recalculation, and documented fidelity limits. Documents over 50 MB are handed off to an external opener.

## How the existing system works, per this document

TurboSpark already has `OfficeArchive` (hardened unzip with zip-bomb bounds and path containment), `OfficeXMLParsers` (SAX text extraction) and `DocumentTextExtractor`, which together produce the "Model text" the model receives. An enum `AppArtifactRenderKind` maps docx/xlsx/pptx to `.opaque`. `FilePreviewLeaves` (attachment preview) and `ArtifactPanelView` (agent artifacts) route those kinds to an external-opener leaf. PDF files render with PDFKit. The document also mentions a git pane, a rewind draft and a DESIGN.md as existing specs, but does not describe them.

## Findings (CO)

CO-1. Major. Lines 45-46, 110-111, 171-172. The read-only invariant is claimed to be enforced by "a read-only file handle", but the document never says what that handle is (FileHandle, path, or in-memory Data), how OfficeArchive opens files today, or why a read handle prevents writes. The only check is the mtime/size test, which misses a write that restores size and mtime.

CO-2. Major. Lines 62-64, 7-11. The one open question says "Lean extending both" while the plan and CUJs already commit to extending both. Unclear whether it is decided; no owner or resolution criterion.

CO-3. Major. Line 99, lines 44-45. PDF is listed as the fourth viewer, but the file list creates three views and never says where PDF routes or what "formalize" changes. No implementation or verification item for the PDF goal.

CO-4. Major. Lines 22-30, 3-6, 123-124. "CJK font runtime" is mentioned and not carried into goals; scope unclear. "git pane and rewind draft (existing specs)", "ZP-06", "ZP-09" and "page sandbox philosophy" are referenced with no link or one-line description.

CO-5. Major. Lines 140-142, 52-54, 114-116. "First N of any unbounded list" gives no value for N and no per-kind caps. "Documented cap row" is undefined. OfficeArchive zip-bomb bound values and their compatibility with a 50 MB document are not stated. "No XML entity expansion" has no stated mechanism.

CO-6. Major. Lines 36-43 versus 52-56, 85-90. Goals list cell styles and number formats while Non-Goals exclude themes; theme-derived colors have no stated fallback. The pane-footer "limits list" has no defined content or owner and does not appear in CUJ 1 or 2. "Disabled-mode notice" for VBA is unexplained.

CO-7. Major. Lines 129-136. QuickLook is "the fallback for unsupported elements", but no plan item, file or CUJ describes it, and it is unclear how a per-element fallback works inside a native view. It also appears to conflict with the read-only guarantee for which Alternative B was rejected.

CO-8. Major. Lines 85-90, 96-111, 143-145. The over-50 MB external-opener handoff appears only in NFRs and manual test; no CUJ or UI state. Error behavior is absent: corrupt, encrypted, parse failure, zip-bomb bound hit. Unclear whether the fallback is "Open with default app" or Model text.

CO-9. Nit. Lines 77-83. Cell comments and frozen headers are goals, but only frozen headers appear in CUJ 2. "Model text" is defined only parenthetically; the "preview/source toggle" is not stated as existing or new, and the file list has no change adding it.

CO-10. Nit. Lines 146-149, 181, 192-194. DESIGN.md and the 21-language list are not linked; "where sensible" for font settings is vague; line 181 "(Modify; basename, ...)" is unclear shorthand; "mutation checks" is undefined jargon.

CO-11. Nit. Lines 98-103 versus 155-160. The diagram says OfficeXMLParsers is "extended" but the file list never modifies it; OfficeArchive is modified with "no behavior change to extraction". Unclear where new parsing lives.
