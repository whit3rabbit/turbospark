# Consolidated review report: ZP-07 plugin creator

Target: docs/designs/zcode-parity/ZP-07-plugin-creator.md (full file, 162 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (the first inventive-catalyst spawn hit a concurrent-subagent limit and was re-run once, then completed).
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 3, Major 8, Nit 9.
Severity note: Stage 8 judged DC-1 and DC-2 over-severe; here the attribution issue is Major and the containment issue stays Blocker because it is a security claim (CUJ 2 and Security NFR) the design cannot back. Stage 8 also advised against a full Risks section; the standard requires one, so B3 asks for a short one.

## Blocker

B1. Lines 31-33, 51-56, 79-81, 130, 135-139. Delivery mechanism is stated three ways (undecided, `Plugins/BuiltinSkills/...`, `Resources/bundled-skills/...`); the resource dir does not exist, SwiftPM resources live in `TurboSparkAppResources` via `.process`, and SkillManager has no unremovable system scope (SkillScope is `.userGlobal` and `.projectLocal` only). Fix: close the Open Question, use one path via the TurboSparkAppResources target and Bundle.module, define the new scope or flag with same-name precedence, note the divergence from ZCode's builtin plugin. [ZO-1, ZO-2, ZO-6, IC-1, IC-4, CO-1, CO-12, DC-3, DC-4, ST-4, SY-1, SY-2, S8-1]
B2. Lines 36-37, 69-75, 119-121, 144-148. Path containment is skill text only, yet CUJ 2 promises "nothing is written outside the plugin folder" and Security calls it mandatory; tests only assert strings; no ladder position stated (INDEX line 121). Fix: reword as advisory, name AppToolPermissionEngine as the control with its ladder position, add one line on symlinks and overwrite, define the error name. A deterministic validator tool (IC-2, DC-2) is an alternative; Stage 8 recommends against it as scope creep. [ZO-4, IC-2, CO-3, DC-2, SY-3, SY-4, S8-2]
B3. Lines 117-124 (missing section). No Risks and Mitigations section, required by the standard. Fix: add a short section: malformed JSON from small models, existing or unwritable target path, symlink escapes, skill active in unrelated chats, schema drift of the bundled skill, MCP command strings written into manifests. [DC-8, ST-1, ST-8]

## Major

M1. Lines 39-41, 94-95, 140-143. No mechanism for the prefilled chat draft. Fix: name the seam (AppChat `draft` or AppModel+Chat `promptText`), say the draft is editable and not auto-sent, state the no-model case in one line. [ZO-3, IC-3, DC-5, S8-3]
M2. Lines 22-27, 34-38, 41. Problem is mis-stated (an agent can already write files; the gap is manifest schema and install knowledge), and the dev marketplace.json schema and consumer are uncited. Fix: reword the Problem; cite PluginMarketplaceManager.swift and PluginManifestParser.swift or drop the dev marketplace.json goal; drop or define "update" steps. [ZO-5, IC-6, CO-4, CO-6, DC-6, DC-14, S8-4]
M3. Lines 119-121, 35-36. Hook trust cited as existing without a source; hooks are not a listed component; MCP command strings in manifests are an unanalyzed surface. Fix: cite AppHookStore.swift and SWIFT_PLUGINS.md; state hooks are out of scope or trigger the existing prompt on registration. [CO-5, DC-7, SY-3, S8-5]
M4. Lines 79-96. Diagram order opposes CUJ 1 and implies a data flow from agent to pane; "Skill tool" undefined. Fix: reorder to pane, draft, agent, skill; name the tool. [CO-2, ST-5, S8-6]
M5. Lines 103, 106-107, 114-115. "Rejected by author" is attribution without a link, and "cannot author content" is categorical. Fix: drop the attribution, keep the technical reason, add one line naming the seeded-plugin delivery option. [DC-1, ST-2, SY-7, CO-9, S8-7]
M6. Lines 153-159, 147, 157. Verification gaps: "activates through the Skill tool" is not a unit test; manual step has no pass criterion; no coverage of CUJ 2, menu entry or localized draft; "21 languages" versus INDEX "ten localization parity gates"; `make design-check` and verification.md not referenced. Fix: reclassify, add pass criteria, reconcile the count. [DC-9, ST-6, ST-8, SY-6, CO-10, CO-11, S8-9]
M7. Lines 5-6, 24-27, 31-33. Claims about existing Swift behavior ("unremovable, always discovered", "creator surface covers skills only") lack source links; INDEX lines 126-129 require them. Fix: add links, phrase unremovable as a requirement. [SY-2, SY-5, ST-7, S8-10]
M8. Lines 5-6, 51-56. Parity claim diverges: ZCode ships a builtin plugin, this design picks a non-plugin root, and ZP-05, ZP-18, ZP-20 may later need a second delivery path. Fix: state the divergence and a migration path in one sentence. [ZO-6, DC-4, IC-4]

## Nit

N1. Lines 149-151. Item 5 markup malformed (unclosed bold, no Type/Scope and Proposed Changes). Fix: match items 1-4. [ZO-8, IC-5, CO-8, DC-10, ST-3, SY-8, S8-8]
N2. Lines 3-6. SKILL.md link not cited (ZCODE_COMPARISON line 399); Swift links missing. Fix: add. [ST-7, DC-11, S8-10]
N3. Lines 144-148. Test file name `BundledSkillsTests.swift` is generic. Fix: name after behavior. [ZO-7, IC-5]
N4. Lines 22-27, 39-40. "Exact steps" and "only requested components" are unverifiable. Fix: define the output contract once. [SY-9]
N5. Lines 122-124. Parenthetical repeats the clause; "Import" menu hosts a create action. Fix: drop parenthetical. [SY-10]
N6. Lines 46-49. "The skill plus file tools are the editor" is figurative. Fix: "the agent edits files with its existing file tools." [SY-11]
N7. Lines 22-27, INDEX 35, 97. No user need or metrics; "small" sizing is optimistic given the SkillManager scope. Fix: leave metrics out (YAGNI); note sizing in INDEX. [DC-12, DC-13]
N8. Lines 53-56. Open Question has no owner and reads as decided. Fix: mark as decided (see B1). [IC-4, ST-8]
N9. Lines 120-121. Reconcile with M3; cite the trust implementation file. [ZO-9]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-9 (zoom-out), IC-1 to IC-6 (inventive catalyst). Feeds B1, B2, M1, M2, M7, M8, N1-N3, N9.
- comprehension_report.md: CO-1 to CO-12. Feeds B1-B3, M2-M4, M6, N1, N4.
- critic_report.md: DC-1 to DC-14, ST-1 to ST-8, SY-1 to SY-11. Feeds all Blockers and Majors, N1-N8.
- intermediate_zoom_out_report.md: S8-1 to S8-10. Confirms all Stage 5 findings still open (document unedited). Recommends rejecting scope-adding asks: IC-2 and DC-2 validator tool, IC-3 extra machinery, DC-6 load test, DC-5 clobber policy, DC-12 metrics, DC-13 resizing, DC-14 update flow, and a full Risks section (overridden here by B3 because the standard requires one).
- Merge notes: B1 merges DC-3, DC-4, ST-4, SY-1 and the open question; B2 merges ZO-4, CO-3, DC-2, SY-3, SY-4.
