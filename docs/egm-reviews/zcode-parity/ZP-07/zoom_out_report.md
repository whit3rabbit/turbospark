# Stage 5: Initial zoom out report (ZP-07 plugin creator)

Target: docs/designs/zcode-parity/ZP-07-plugin-creator.md (full file, 162 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out reviewer (ZO)

Overall: small and correctly scoped (skill as editor, UI only an entry point). Findings are mostly stale or wrong repo references and one missing mechanism. No Blockers.

ZO-1. Major. Lines 130, 143, diagram 80-81. The planned path `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/plugin-creator/SKILL.md` does not exist. SwiftPM resources live in the `TurboSparkAppResources` target (Package.swift lines 45-49, `.process("Resources")`) at `swift/TurboSparkApp/Sources/TurboSparkAppResources/Resources/`. `.process` can flatten or rewrite directories, so the plan must say `.copy` or process behavior. The diagram path `Plugins/BuiltinSkills/...` contradicts file 1. The doc never says how the app reads the file (`Theme/ResourceBundle.swift` Bundle.module forwarding is the seam).

ZO-2. Major. Lines 135-139, 31-33. SkillManager has no bundled or system root. Discovery is `discoverUserSkills` and `discoverProjectSkills`, deduped by lowercased name, scopes `.userGlobal` and `.projectLocal`. "Unremovable, always offered, system content" needs a new scope or flag touching SkillsSettingsPaneView, SkillEditorSheet and SkillManager+Files. State the scope, the same-name precedence rule, and that the `skill` tool path also sees the root. Alternative: ship through existing seeding or import.

ZO-3. Major. Lines 39-41, 140-143, 94-95. No mechanism for the Import menu item to open a prefilled chat. The menu is `ExtensionImportMenu { addLocalFolderItem }` in PluginSettingsPaneView (about line 128); drafts exist in AppChat (`draft`) and AppModel+Chat (`promptText`). Name the entry point and the cross-pane navigation, and say how the draft carries editable placeholders (CUJ 1).

ZO-4. Major. Lines 36-37, 69-75, 119-121, 147-148. Containment is enforced only by skill text; the test asserts strings. CUJ 2 and the Security bullet overstate a guarantee. The real control is the file-tool permission ladder (AppToolPermissionEngine, which INDEX says drafts must reference). State the ladder position and reword the skill as advisory.

ZO-5. Major. Lines 22-27. "Agent cannot be asked to scaffold one" is wrong: an agent with file tools can already write a plugin. The real gap is manifest schema and install-flow knowledge. Dev marketplace.json consumer is undefined. No cross-reference to ZP-20, which likely overlaps the skill content.

ZO-6. Nit. Lines 3-6. ZCode's creator is a builtin plugin; the Open Question leans to a bundled root. State the divergence. The Open Question is undecided while the plan already commits.

ZO-7. Nit. Lines 144-148, 156-157. "Activates through the Skill tool" is an integration test, not unit. `BundledSkillsTests.swift` is generic. "21 languages" versus INDEX "ten localization parity gates".

ZO-8. Nit. Lines 149-151. Item 5 markup is malformed (unclosed bold) and lacks Type/Scope and Proposed Changes.

ZO-9. Nit. Lines 120-121. SHA-256 plugin-hook trust approval is cited as existing; cite the implementing file.

## Inventive catalyst (IC)

IC-1. Major. Lines 79-96, 130. False compromise: bundled delivery is framed as ZP-05 seeding versus a new SkillManager root, with inconsistent paths and no packaging mechanism (SwiftPM resource entry, Bundle.module). Fix: pick one path, state packaging, and put the bundled root behind a one-function provider ("list bundled skill directories") so ZP-05, ZP-20 and ZP-18 can register without editing SkillManager. Define name-collision precedence.

IC-2. Major. Lines 36-38, 69-75, 119-121, 144-148. Containment as prose is not validation. Fix: ship a deterministic validator the skill calls (resolves manifest paths, rejects `..`, absolute escapes, symlinks), tested with real fixtures; state the ladder position and scope writes to the target folder. Cost: new code path and tool surface beyond "skill only".

IC-3. Major. Lines 39, 140-143, CUJ 1. Entry point depends on the model activating a skill by name; no handling if the skill is disabled, missing, or no model is loaded. Fix: menu action checks the skill resolves and disables or explains otherwise; define no-model state; do not inline the full skill text.

IC-4. Major. Lines 51-56, 32. Open Question is half decided while the plan commits to the bundled root. Manifest template has a hidden ZP-06 dependency (UI plugin fields). Fix: close the question, add a scope note that the template covers only currently supported components until ZP-06.

IC-5. Nit. Lines 150, 157. Item 5 markup; localization count; verification lacks `make design-check` and the verification.md gates INDEX requires for UI; generic test name.

IC-6. Nit. Lines 3-6, 16, Non-Goals. dev marketplace.json schema and consuming flow not stated; existence of a local marketplace.json install flow unconfirmed. Fix: cite the Swift consumer or drop the goal and keep register-local-folder only.
