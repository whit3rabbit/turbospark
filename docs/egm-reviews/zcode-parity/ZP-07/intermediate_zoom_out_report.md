# Stage 8: Intermediate zoom out report (ZP-07 plugin creator)

Target: ZP-07-plugin-creator.md (full file, 162 lines, unedited since Stage 5). Reviewer: egm-zoom-out (S8). Inputs: zoom_out_report.md, comprehension_report.md, critic_report.md. Completed.

Verdict: the core shape is sound and small (skill is the editor, UI is only an entry point, no wizard, no auto-install). No simplicity Blockers. Remaining work is ambiguity removal and fixing factual claims, not adding scope. Stage 7 Blockers DC-1 and DC-2 are over-severe for a Draft (Major at most); DC-3 is a real Major.

Code spot-checks (grep): no "bundled-skills" or "BuiltinSkills" resource anywhere under swift/. SkillScope has only `.userGlobal` and `.projectLocal(projectPath:)` (AppSkill.swift lines 4-8). marketplace.json is parsed in PluginMarketplaceManager.swift, PluginManifestParser.swift, SkillMarketplaceManager.swift and McpMarketplaceManager.swift, so a consumer exists but the doc does not cite it.

## Open findings

S8-1. Major. Lines 31-33, 51-56, 79-81, 130, 135-139. Delivery stated three ways; resource dir does not exist; a new unremovable SkillManager scope is the most expensive item in a "small" draft. Ask: close the Open Question, give one path (TurboSparkAppResources target via Bundle.module), say the root is a new scope or flag with a same-name precedence rule, state the divergence from ZCode (a builtin plugin). One sentence on the cheaper option (seeded plugin or existing import). Same as ZO-1, ZO-2, IC-1, IC-4, CO-1, DC-3, DC-4, ST-4, SY-1.

S8-2. Major. Lines 36-37, 69-75, 119-121, 147-148. Containment is skill text only; CUJ 2 and Security overstate it. Ask: reword as advisory, name the file-tool permission ladder (AppToolPermissionEngine) as the control, state the ladder position (INDEX line 121). Same as ZO-4, CO-3, DC-2, SY-3, SY-4.

S8-3. Major. Lines 39-41, 94-95, 140-143. No mechanism for the prefilled draft. Existing seams: AppChat `draft`, AppModel+Chat `promptText`. Ask: name the seam, state the draft is editable and not auto-sent. Same as ZO-3, DC-5, IC-3.

S8-4. Major. Lines 22-27. Problem statement is wrong: an agent with file tools can already write a plugin; the gap is manifest schema and install-flow knowledge. Cite PluginMarketplaceManager.swift and PluginManifestParser.swift for the marketplace.json consumer, or drop the dev marketplace.json goal. Same as ZO-5, IC-6, DC-6.

S8-5. Major. Lines 119-121, 35-36. Hook trust cited as existing but hooks are not in the component list. Ask: cite AppHookStore.swift and SWIFT_PLUGINS.md; say hooks are out of scope or trigger the existing prompt on registration. Same as CO-5, DC-7.

S8-6. Major. Lines 79-96. Diagram runs opposite to CUJ 1; "Skill tool" undefined. Reorder: pane, draft, agent, skill. Same as CO-2, ST-5.

S8-7. Major. Lines 106-107, 114-115. "Rejected by author" is attribution. Ask: delete it, keep the technical reason, add one line naming the seeded-plugin delivery option. Same as DC-1, ST-2, SY-7, CO-9.

S8-8. Nit. Lines 149-151. Item 5 malformed.
S8-9. Nit. Lines 130, 143-148, 157. Test name, "21 languages" versus INDEX "ten localization parity gates", reclassify the Skill-tool activation test as integration or manual, add `make design-check`.
S8-10. Nit. Lines 3-6. Add the SKILL.md link (ZCODE_COMPARISON line 399) and the Swift source links INDEX Provenance requires.

## Recommended rejections (scope-adding)

- IC-2 and the DC-2 option of a deterministic validator tool: adds an agent-facing tool, contradicts the "skill plus file tools" non-goal. Cover symlinks and overwrite with one line of skill-text rules.
- IC-3 extra machinery (disabled menu, skill-resolution check, no-model handling).
- DC-6 load test through PluginManager (a manual line is enough).
- DC-8 and ST-1 as a full Risks section: add 3-4 bullets to the NFR section instead (malformed JSON from small models, existing target path, no write access, skill active in unrelated chats). Note this deviates from the standard's required section; see consolidated report B3.
- DC-5 clobber policy and no-model state; CO-6, CO-7, CO-10 contract and listing expansions; DC-12 metrics; DC-13 resizing; DC-14 and CO-6 update flow (drop "update" from line 41); CO-11, CO-12 build and packaging explanations beyond the one correct path.
- IC-4 ZP-06 note beyond one line: "template covers only currently supported components."

Priority: fix S8-1 to S8-5 first. Demote DC-1 and DC-2 from Blocker.
