# Consolidated review report: ZP-09 office document generation

Target: docs/designs/zcode-parity/ZP-09-office-document-generation.md (full file, 188 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO, IC = zoom_out_report; CO = comprehension_report; DC, ST, SY = critic_report (design, structural, style); S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 9, Nit 10.

## Blocker

B1. Lines 105, 118-119. "OfficeArchive's writer half" and "hardened zip access" do not exist (read-only /usr/bin/unzip wrapper); all OOXML writers depend on it, and "never shell out" conflicts. Fix: add one in-process OOXML package writer to the file list with zip strategy, limits, tests. [ZO-1, IC-1, CO-4, DC-1, ST-2, SY-2, S8-B1]
B2. Lines 99-102, 141. Agent-facing tool undefined ("writer tools or REPL file bridge"): no name, schema, registry change, or ladder position required by INDEX; confinement has nothing to bind to. Fix: one document-write tool taking a bounded JSON spec, state ladder rung and path rules. [ZO-2, IC-2, CO-1, CO-2, DC-2, ST-3, SY-1, S8-M1]

## Major

M1. Lines 33-37, 56-65. Scope: four formats at once, subset and PDF path left to "decide at implementation" while CUJ 3 assumes it; charts undefined. Fix: explicit v1 cut (docx, pdf, xlsx cells; defer charts, pptx shapes), mark the assumed option, evaluate NSAttributedString export. [ZO-4, IC-5, CO-7, DC-4, ST-5, S8-M3, S8-M6]
M2. Lines 45, 74, 141, 146-147, 168. Artifact folder, panel, and ZP-07, ZP-08, ZP-19 prerequisites assumed; Create-entry UI and "21 languages" have no plan entry (INDEX says ten gates). Fix: state dependencies, drop the UI claim. [ZO-3, ZO-9, CO-3, CO-9, DC-3, SY-4, IC-6, S8-M2]
M3. Lines 39-42, 74-75, 144-145. Postcheck shares parsers with the writer, proves little about Word opening, is skill-text advisory, and has no stated location or failure path. Fix: one tool-side postcheck with named checks, temp-then-move on pass, drop "opens in Word". [ZO-6, IC-3, IC-4, CO-8, DC-7, SY-5, S8-M5]
M4. Lines 141-148 (missing section). No Risks and Mitigations: size caps, formula injection, XML escaping, fonts/CJK, overwrite, ZP-08 sequencing. Fix: add a short section. [DC-8, ST-4, IC-8]
M5. Lines 121-137. Alternatives: "Rejected by author", Alt B claim wrong (pandoc/LibreOffice are not Node or Python), missing skills-only, system exporters, third-party library. Fix: technical reasons, add options briefly. [DC-5, ST-1, IC-9, ZO-11]
M6. Lines 40-44, 78-86, 170-171. Shell-out fallback: output untrusted, bypasses folder confinement, macro policy unstated; missing-tooling report untested; "shell-out absence" test contradicts fallback. Fix: one line on untrusted output and no macros, test "writers do not spawn processes". [DC-6, DC-9, SY-6, CO-5, ZO-10]
M7. Line 82 (also 80-81). CUJ 2 VBA example incoherent. Fix: use pivot table or conditional formatting. [ZO-5, CO-6, DC-13, ST-7, SY-5]
M8. Lines 97-116, 108, 157. Writers described inconsistently; PDF text layout (Core Text, CJK/RTL) unaddressed. Fix: one tool dispatching to per-format pure functions plus shared package writer; PDF via Core Text, CJK/RTL as stated limit. [S8-M7, IC-7]
M9. Lines 154-164. bundled-skills root does not exist and the path may differ from the resources layout; SkillManager change unexplained. Fix: verify path, state ZP-07 dependency and what changes. [SY-3, ZO-7, CO-10, DC-11, ST-8]

## Nit

N1. Lines 175-177. Item 12 garbled. Fix: "create swift/docs/SWIFT_DOCUMENTS.md". [ZO-8, CO-13, DC-10, ST-9, SY-11]
N2. Lines 33-34, 92. "PDFKit-adjacent" wrong; PDFKit does not write. Fix: Core Graphics/Core Text. [CO-11, DC-12, ST-6]
N3. Lines 3-9, 25-29. Constraint and finding sources cited without links; Swift sources from 9.1 not cited. Fix: link roadmap, add sources. [CO-14, ST-10, S8-N1, SY-10]
N4. Lines 35-38. Cover recipes and playbooks defined only by outside link. Fix: one inline sentence. [CO-12]
N5. Lines 15, 20, 42, 61. "professional" unmeasurable; fallback term inconsistent. Fix: drop and unify. [SY-7, SY-8]
N6. Lines 58-65. "enormous", "Lean native", no owner or criterion. Fix: reword, name owner. [SY-9]
N7. Lines 25-26. "xlsx writers" is not a named library. Fix: name it. [SY-12]
N8. Lines 154-177. File list inconsistent (items 1-9 lack sub-bullets). Fix: complete. [ST-8]
N9. Lines 178-185. Verification lacks design-check and verification.md pointer; mutation checks only for OOXML. Fix: one-line mentions. [ST-11, SY-13]
N10. Non-Goals. Add "no macro output". [DC-13]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-11, IC-1 to IC-9. Feeds B1, B2, M1-M3, M5-M9, N1.
- comprehension_report.md: CO-1 to CO-15. Feeds B1, B2, M1-M3, M6, M7, M9, N1-N4.
- critic_report.md: DC-1 to DC-13, ST-1 to ST-12, SY-1 to SY-13. Feeds all Blockers and Majors, N1-N10.
- intermediate_zoom_out_report.md: S8-B1, S8-M1 to S8-M7, S8-N1. Confirms all Stage 5 findings open. Recommends rejecting scope-adding fixes: IC-5 raw-part passthrough, IC-3 independent validator and real-Office fixtures, IC-4 retry loop, IC-6 schema-generated skills, DC-6 supply-chain analysis, extra Non-Goals and Create-entry UI. Those are reflected above as the narrower fixes.
- Merge notes: Section C of Stage 8 items are listed as rejected, not as findings.
