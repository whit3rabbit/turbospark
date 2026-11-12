# Consolidated review report: ZP-19 web image search

Target: docs/designs/zcode-parity/ZP-19-web-image-search.md (full file, 168 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; the style reviewer's first spawn hit "Concurrent subagent limit reached" and succeeded on one retry.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 5, Major 9, Nit 10.

## Blocker

B1. Lines 39-42, 68, 91-93, 131-156. `image_fetch` and the "attachment URL intake" do not exist as an agent tool; WebFetchExecutor rejects image MIME types; not in file list, schema, ladder or tests. Fix: extend the existing web fetch path to save an image to a workspace path, state destination, ladder rung and guards, add the files. [ZO-3, IC-3, CO-1, DC-2, ST-4, SY-7, S8-B1]
B2. Lines 36-38, 53-59, 82-85, 95-97. "MCP first, then Brave" does not mirror WebSearchExecutor (hardcoded providers, no user MCP calls); Open Question conflicts with Goals. Fix: day one is Brave images only, drop MCP from Goals and diagram, note MCP tools are callable normally. [ZO-1, IC-2, CO-3, CO-9, DC-1, ST-6, SY-5, S8-B2]
B3. Lines 49, 118-121, 144-148. `SearchSettingsPaneView.swift` does not exist and no Brave key storage or UI exists; "existing Keychain-backed storage" is unsupported. Fix: cite the real store or scope a minimal shared key field and Keychain entry (opt-in, not at launch). [ZO-2, IC-4, CO-5, DC-4, S8-B3]
B4. Lines 22-29, 95-97. "Reuse WebSearchExecutor wholesale" and the description of the existing stack are wrong; sanitize is text-shaped and helpers are private. Fix: rewrite accurately, name shared helpers and visibility changes. [IC-1, ZO-5, CO-2, DC-3, SY-3, S8-M1, S8-M2]
B5. Whole document (near 116-125). No Risks & Mitigations section, required by the standard (prompt injection, licensing, quota, privacy, expired URLs). Fix: add a short section with mitigations. [ST-1, DC-7, IC-6, S8-M7]

(B4 is rated Major by the zoom-out and style reviewers and Blocker by none directly; it is listed here because S8 treats it as the premise of the plan. If a stricter count is wanted, treat B4 as Major.)

## Major

M1. Lines 140-143, 129-148. Ladder positions unstated (INDEX line 121) and registration files missing (AppToolCatalog, ToolPresentation, ToolCallNormalization, AppHookToolNameAliases, AppToolRegistry). Fix: name the rung per tool and add files. [ZO-4, DC-9, ST-5, SY-8, S8-M4]
M2. Lines 122-123, 66. 90 s timeout copied from a hosted service; text default is 25 s; cap 8 conflicts with text defaults and CUJ 1. Fix: inherit 25 s, state cancellation and clamp, one default count, drop "tunable". [ZO-6, IC-4b, IC-8, DC-5, SY-4, S8-M3]
M3. Lines 71-77, 137-138. CUJ 2 is an either/or. Fix: adopt the text tool's availability rule, cite it, one testable Then. [IC-7, CO-4, DC-8, SY-6, S8-M5]
M4. Lines 61-69, 16-18. CUJ 1 depends on ZP-09, sequenced later. Fix: end at image saved with source retained; pptx embed becomes a ZP-09 note; gloss ZP-09. [IC-5, CO-6, S8-M6]
M5. Lines 99-114. Rejections attributed to "the author" without trade-offs; missing alternatives (image mode on web_search, MCP-only, other providers). Fix: concrete reasons, add the web_search image mode alternative. [DC-6, ST-2, ST-3, SY-9, S8-M8]
M6. Lines 14-20, 105. ZCode office-plugin delegation and search/fetch split claims are uncited. Fix: cite or narrow. [SY-1, S8-M8]
M7. Lines 1-10. Missing Sources block required by INDEX 126-129. Fix: add ZCode `.mcp.json` link, Swift search note, `docs/document-skills-plugin.md`. [ST-7, S8-N2]
M8. Lines 33-35, 43, 119-121. Row schema and "provenance tag" undefined; domain filter owner unclear; "no user identifiers" has no mechanism. Fix: define fields and where filtering runs; list image-row sanitization rules. [CO-7, DC-3, ZO-7, S8-N3]
M9. Lines 149-164. Tests miss hostile-title sanitization, non-image and private-target fetch refusal, SSRF mutation check, design-check and localization gates; mutation checks mis-nested. Fix: retarget and move to Verification. [DC-10, ZO-8, ST-9, ST-10, SY-12, S8-N4]

## Nit

N1. Line 125. "21 languages" versus INDEX "ten localization parity gates". Fix: cite a verified count. [DC-11, CO-10, ST-8, SY-7, S8-N1]
N2. Lines 3-6. "Parity target" overstates; argument-name adoption unstated. Fix: "Reference implementation". [DC-12, SY-11, S8-N5]
N3. Lines 8-10. INDEX "small" sizing is understated unless scope is cut. Fix: cut scope as in B2, or re-size. [DC-13, IC-9, S8-N2]
N4. Lines 24-25. "greps match nothing" is unscoped. Fix: state paths and date. [SY-2, S8-M2]
N5. Lines 16-28, 38-39, 112. Informal or undefined wording ("zero-config", "companion", "twin", "low risk"). Fix: tighten. [SY-10, SY-8]
N6. Lines 33-38, 82-85. Inconsistent backend and tool names. Fix: normalize. [ST-11]
N7. Lines 131-148, 154. "Ladder", "basename", "mutation checks" unexplained; file label packs an aside. Fix: gloss and move aside. [CO-8, ST-10, S8-N6]
N8. Lines 51, 118-121. "Attachment cache rules", "per-origin rules", credential rules uncited. Fix: link or delete. [CO-11, SY-13, S8-N6]
N9. Lines 118-120. "Key never logged" has no assertion method. Fix: specify log capture. [SY-12]
N10. Lines 106, 112, 114. Subjective rejection wording. Fix: concrete reasons. [SY-9]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-8, IC-1 to IC-9 (IC-4b merged into M2).
- comprehension_report.md: CO-1 to CO-11.
- critic_report.md: DC-1 to DC-13, ST-1 to ST-11, SY-1 to SY-13.
- intermediate_zoom_out_report.md: S8-B1 to S8-B3, S8-M1 to S8-M8, S8-N1 to S8-N6. Confirms every earlier finding is open. Recommends three decisions: Brave-only day one, fetch extends the existing web fetch path, key and UI scoped as minimal new work. Recommends rejecting scope-adding asks: provider protocol, MCP contract, license and safesearch fields, per-backend settings, tunable timeout, standalone image_fetch with full test matrix.
