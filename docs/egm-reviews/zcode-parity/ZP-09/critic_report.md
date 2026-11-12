# Stage 7: Critic report (ZP-09)

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). Standard: feature-spec-standard for design and structural. One round, no remediation, convergence not assessed. All three completed.

## Design critic (DC)

- DC-1 Blocker, lines 105, 118-119. "OfficeArchive's writer half" does not exist; OfficeArchive.swift is a read-only /usr/bin/unzip wrapper. No zip writer in the file list; writers "never shell out" conflicts with the only archive code shelling out. Ordering rules (content types first) and size bounds unaddressed. Fix: add a zip-writer component with approach, limits, tests.
- DC-2 Blocker, lines 99-101, 141. Agent-facing tool unspecified: no name, schema, registry change, or ladder position (INDEX requires one). Confinement unenforceable (symlinks, "..", overwrite, collisions). Fix: define the tool, JSON spec, ladder rung, path rules.
- DC-3 Major, lines 45, 146-147, 183. Artifact folder, panel, Create-entry UI undefined and absent from file list. Fix: cite owner or add files.
- DC-4 Major, lines 33-34, 58-63. Document model and subset undefined; xlsx "basic charts" undefined; main scoping decision deferred. Fix: define model, settle or evidence-tie subset.
- DC-5 Major, lines 121-137. Alternatives weak: "Rejected by author", Alt A straw man, Alt B claim wrong (pandoc/LibreOffice are not Node/Python). Missing: third-party Swift OOXML library, NSAttributedString/Core Text PDF, skills-only. Fix: technical reasons, add options.
- DC-6 Major, lines 40-44, 78-85. Shell-out fallback bypasses folder confinement; pip install supply-chain risk; output is untrusted input to the ZP-08 parser; macro output policy unstated. Fix: analyze and state limits.
- DC-7 Major, lines 39-42, 74-75, 144-145, 92. Postcheck overclaimed: shares OfficeXMLParsers with the writer, proves nothing about Word/Excel opening; "mandatory" skill text is advisory. Fix: independent validator, tool-side enforcement, fixtures checked in real apps.
- DC-8 Major, lines 141-148 (missing section). No risks section: size caps, image decode, fonts/CJK/pagination, overwrite, cancellation, XML escaping, xlsx formula injection. Fix: add Risks & Mitigations.
- DC-9 Major, lines 165-174. "Shell-out absence" test contradicts the allowed fallback. Fix: "writers do not spawn processes".
- DC-10 Nit, lines 175-176. Docs item garbled. Fix: "create SWIFT_DOCUMENTS.md".
- DC-11 Nit, lines 118, 163-164. SkillManager has no bundled-skill mechanism; depends on ZP-07; ZP-08 and ZP-19 dependencies not stated. Fix: Dependencies line.
- DC-12 Nit, line 34. "PDFKit-adjacent" wrong. Fix: Core Graphics/Core Text.
- DC-13 Nit, Non-Goals. Missing: no macro output, encryption, tracked changes; no success metric. Fix: add.

## Structural (ST)

- ST-1 Major, lines 121-137. Alternatives: "Rejected by author" attribution without link; only two strawmen. Fix: as DC-5.
- ST-2 Major, lines 94-119, 105. Writer-half claim unsupported; OfficeArchive not in Modify list. Fix: as DC-1.
- ST-3 Major, lines 99-102, 154-177. Agent entry point undefined; no registry change. Fix: as DC-2.
- ST-4 Major, lines 141-148. Missing Risks and Mitigations section. Fix: add one (include ZP-08 sequencing risk).
- ST-5 Major, lines 56-65. Open questions hold decisions Goals and CUJ 3 already assume; no owner; chart-image source unstated. Fix: mark assumed option, name owner.
- ST-6 Nit, lines 33-34, 92. PDFKit wording. Fix: name CGPDFContext/Core Text.
- ST-7 Nit, lines 80-81. CUJ 2 VBA example poor. Fix: pivot table or track changes.
- ST-8 Nit, lines 154-177. File list inconsistent (items 1-9 lack sub-bullets), missing artifact write path, Create-entry UI, catalog file; ZP-07 dependency implicit. Fix: complete the list.
- ST-9 Nit, lines 175-176. Garbled item 12.
- ST-10 Nit, lines 3-9. Swift sources from finding 9.1 (AppChatExport, DocumentTextExtractor, OfficeArchive) not cited. Fix: add.
- ST-11 Nit, lines 178-185. Verification lacks external-validity check, design-check, verification.md pointer. Fix: add.
- ST-12 Nit, lines 144-145. No failure surface, partial-file handling, size or time limits. Fix: state them.

## Style (SY)

- SY-1 Major, lines 101, 130-137, 15-21. Invocation path inconsistent, no ladder position; REPL bridge not found in sources. Fix: name tool or delete the parenthetical.
- SY-2 Major, lines 39-41, 111, 168. Postcheck says ZP-08 OfficeRenderModel, tests say OfficeXMLParsers; "writer half" presented as existing. Fix: one name, state ZP-08 dependency.
- SY-3 Major, lines 154-162. bundled-skills path may be wrong (resources live under TurboSparkAppResources); dir does not exist. Fix: verify program-wide.
- SY-4 Major, lines 148, 183, 147. Create-entry UI and "21 languages" vs INDEX "ten parity gates". Fix: reconcile.
- SY-5 Major, lines 33-34, 41, 80, 91-92, 125, 144-145. Unmeasurable or unsupported claims: "expected part counts", "correct page sizes", "battle-tested", VBA example. Fix: define or drop.
- SY-6 Major, lines 42-44, 82-86, 170-171. Missing-tooling report depends on skill text and has no test. Fix: add verification.
- SY-7 Nit, line 15. "professional". Fix: drop.
- SY-8 Nit, lines 20, 42, 61. Inconsistent fallback term. Fix: pick one.
- SY-9 Nit, lines 58-65. Open question without owner or criterion; "enormous", "Lean native". Fix: reword.
- SY-10 Nit, lines 127-137. Cite roadmap path for the constraint.
- SY-11 Nit, lines 175-176. Garbled item 12.
- SY-12 Nit, lines 25-26. "xlsx writers" is not a named library. Fix: name it.
- SY-13 Nit, lines 90, 172-174. Image source (ZP-19) unstated; mutation checks only for OOXML. Fix: state source and scope.
