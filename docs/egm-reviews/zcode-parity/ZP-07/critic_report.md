# Stage 7: Critic report (ZP-07 plugin creator)

Target: ZP-07-plugin-creator.md (full file, 162 lines). Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). Standard: feature-spec-standard for design and structural. One round only; no remediation, so convergence was not assessed. All three completed.

## Design critic (DC)

Verdict: not ready to implement. Original severities kept.

DC-1. Blocker. Lines 106-107, 114-115. Both alternatives end "Rejected by author", an attribution the standard prohibits without a link. Fix: state the technical reason only; add a viable alternative (Swift-side deterministic scaffolder, or ZP-05 seeding or ZP-20 guide skills).

DC-2. Blocker. Lines 24-25, 69-75, 119-121, 144-148. Containment is prose in SKILL.md; the agent writes with ordinary file tools. "Nothing is written outside the plugin folder" is unverifiable; tests only assert strings. Symlinks, overwrite of an existing folder, and AppToolPermissionEngine ladder position (INDEX line 121) are unaddressed. Fix: add an app-owned deterministic validator, or state that containment is advisory and rely on the write ladder; name the ladder position; add a hostile-path fixture test.

DC-3. Blocker. Lines 53-56, 80-82, 130-139, 5-6. Spec does not match the codebase. Diagram path `Plugins/BuiltinSkills/...` conflicts with `Resources/bundled-skills/...`; that Resources dir is not present in the app target though Package.swift declares `.process("Resources")`; `.process` flattens layouts; SwiftPM resources are not found via Bundle.main (TurboSparkApp.swift line 315). SkillManager has no "unremovable system" scope; adding one touches effective-skills memoization (about lines 296-376), the dedupe key (about line 45) and any delete or edit UI. ZCode's reference is a builtin plugin while the design picks a non-plugin root. Fix: reconcile the path, name the loader API (Bundle.module), list SkillManager consumers affected, state the divergence.

DC-4. Major. Lines 31-33, 51-56. Open question resolved in the body; a parallel bundled root may conflict with ZP-05, ZP-18, ZP-20 (INDEX lines 87-88). Fix: close the question and state a migration path.

DC-5. Major. Lines 39-40, 140-143. No mechanism for the prefilled draft: composer injection API, target chat, behavior if generating or text unsent, no-model-loaded state, no auto-send. Fix: specify the entry API, clobber policy and no-model state.

DC-6. Major. Lines 34-38, 132-134, 158-159. Generated artifacts undefined: accepted plugin.json fields, marketplace.json shape, which components PluginManager supports. Fix: link the parser; add a test that scaffolded output loads through PluginManager.

DC-7. Major. Lines 119-121, 36. Hook trust statement mismatches the component list (no hooks). MCP command strings written into manifests are an unanalyzed supply-chain surface. Fix: say hooks are out of scope or what the user sees; state which approval applies on registration.

DC-8. Major. Whole doc, nearest 117-124. No Risks and Limits section (invalid JSON from small local models, partial scaffolds, existing target path, no write access, skill active in unrelated chats, cleanup). Fix: add Risks and Mitigations.

DC-9. Major. Lines 153-159, 147, 157. Verification is weak: no test for menu entry, localized draft, or activation; "Skill tool" unverified; "21 languages" versus INDEX "ten localization parity gates"; verification.md not referenced. Fix: add tests, reconcile count.

DC-10. Nit. Line 150. Item 5 malformed (unclosed bold, no Type/Scope fields).
DC-11. Nit. Lines 4-6. SKILL.md link not cited (ZCODE_COMPARISON line 399).
DC-12. Nit. Lines 22-27. No user need, authoring frequency or success metrics.
DC-13. Nit. INDEX lines 35, 97. "Small", first-wave sizing conflicts with SkillManager scope and chat-draft plumbing.
DC-14. Nit. Lines 44-49, 41. "Update" steps printed but no update flow in scope.

## Structural reviewer (ST)

Verdict: not ready to pass. Heading hierarchy clean; Problem, Goals/Non-Goals, Architecture, Alternatives present.

ST-1. Major. Lines 117-124. Missing Risks and Mitigations section; Security bullet is a requirement, not a risk with mitigation (unenforced validation, symlink and unicode tricks, overwrite, malformed marketplace.json, schema-drift maintenance of a bundled skill).
ST-2. Major. Lines 106-108, 114-115. "Rejected by author" without link; only two UI-shaped alternatives; the real fork (bundled root vs ZP-05 seeded plugin) is only an Open Question; hand-authored template doc not considered.
ST-3. Major. Lines 149-151. Item 5 malformed.
ST-4. Major. Lines 79-96 vs 130 vs 31-33. Skill path stated two ways; Goals hedge while plan commits.
ST-5. Nit. Lines 79-96. Diagram order inverted versus CUJ 1; implies a data flow from Agent to pane that does not exist.
ST-6. Nit. Line 157 vs INDEX. 21 languages versus ten parity gates.
ST-7. Nit. Lines 3-6. SKILL.md not linked; Swift source links (PluginSettingsPaneView.swift, SWIFT_PLUGINS.md) required by INDEX Provenance are missing.
ST-8. Nit. Lines 51-56, 153-159. Open Question has no owner or resolution criterion; verification does not trace to CUJ 2, the menu entry or the localized draft.

## Style reviewer (SY)

Verdict: fix 1 to 6 before the draft is treated as implementation-ready. No Blockers raised.

SY-1. Major. Lines 31-33, 53-56. Delivery named three ways; two conflicting paths; Open Question vs committed plan. Fix: state once, one path, close the question.
SY-2. Major. Lines 31-33, 130-139. "Unremovable, always discovered" and "ZCode's bundled-skills pack" asserted without evidence. Fix: phrase as requirements and cite the ZCode source.
SY-3. Major. Lines 119-121, 147-148. SHA-256 hook trust claimed as existing with no cited file; "mandatory" containment is not code-enforced. Fix: call it advisory or name the enforcer; cite the trust source.
SY-4. Major. Lines 71-75. CUJ 2 states guaranteed outcomes a skill cannot guarantee; error name undefined. Fix: define the name; state dependence on agent compliance and the permission engine.
SY-5. Major. Lines 5-6, 24-27. Claims about existing Swift code lack source links; INDEX lines 126-129 require them.
SY-6. Major. Lines 148-157. "Activates through the Skill tool" listed as a unit test; manual step has no pass criterion; 21 vs ten unreconciled.
SY-7. Major. Lines 103, 106-107, 114-115. "Rejected by author" is attribution; "cannot author content" is categorical. Fix: state the reason; use "does not".
SY-8. Nit. Lines 150-151. Item 5 malformed.
SY-9. Nit. Lines 22-27, 39-40. "Exact" steps and "only the requested components" unverifiable; define the output contract once.
SY-10. Nit. Lines 122-124. Parenthetical repeats the preceding clause; "Import" menu holds a create action.
SY-11. Nit. Lines 46-49. "The skill plus file tools are the editor" is figurative; state it plainly.
