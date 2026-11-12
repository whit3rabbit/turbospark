**What this proposes**

The doc proposes a "plugin creator" skill, shipped inside the app, that an AI agent follows to scaffold a plugin. A plugin is a folder with a manifest at `.claude-plugin/plugin.json` plus only the components the user asks for (skills, commands, agents, MCP servers). The skill checks the structure and that no path escapes the plugin folder, writes a dev marketplace.json that lists the plugin, and ends by printing exact manual install steps. The plugins settings pane gets a "Create Plugin" item in its Import menu. It opens a prefilled chat draft that tells the agent to use the skill. The user then installs the result through the existing flows, with no auto-install.

**How the existing system works, per this document**

The plugins pane can register a local folder as a plugin but cannot create one. The only creation UI is SkillEditorSheet, which covers skills only. A SkillManager discovers skills and would need a new "bundled root" to offer a skill the user cannot remove. Plugin hooks run only after an existing SHA-256 trust approval. The app has a catalog of localized strings in 21 languages. A separate design, ZP-05, may seed builtin plugins, and ZCode (another product) already ships this pattern.

## Findings (comprehension reviewer, CO)

No blocker: every section could be paraphrased. CO-1 to CO-3 are most likely to change the implementation.

CO-1. Major. Lines 31-33, 51-56, 79-81, 130. The skill's location is stated three ways: undecided (ZP-05 seeded plugin or bundled root) in Goals and Open Questions, `Plugins/BuiltinSkills/plugin-creator/SKILL.md` in the diagram (line 81), `Resources/bundled-skills/plugin-creator/SKILL.md` in the plan (line 130). The open question leans one way while the rest commits. A reader cannot tell which path is real or whether the decision is closed.

CO-2. Major. Lines 79-96. The diagram reads as a chain from skill to SkillManager to Agent to pane, but the pane entry starts the flow. Arrows run opposite to the CUJ order (lines 60-67). "Skill tool" is never defined, and how the draft names the skill so the agent finds it is not stated.

CO-3. Major. Lines 33-37, 69-75, 119-121, 144-148. Containment is enforced only by skill text, and the doc never says so. CUJ 2 promises refusal with a named error and nothing written outside the folder; the agent writes with general file tools and tests only assert rule strings (line 147). Security calls it "mandatory" without naming an enforcer. The "named error" is undefined and "absolute escapes" does not cover symlinks.

CO-4. Major. Lines 3-6, 16, 32, 52-55. Undefined terms: "dev marketplace.json" (schema, location, consumption), "register-local-folder flow", "ZP-05 seeding", "ZCode bundled-skills pack", "Finding 4.10", and what "unremovable, always discovered" means in SkillManager.

CO-5. Major. Lines 119-121, 35-36. The SHA-256 plugin-hook trust approval is not explained (what is hashed, when requested, whether an agent-created plugin triggers it). The component list omits hooks, so it is unclear whether the creator scaffolds hooks.

CO-6. Major. Lines 34-36, 62-64, 128-133. Skill inputs and outputs are underspecified: how target path and components are collected, what the draft contains, what the manual add/install/update steps look like. "Update" (line 41) is not covered elsewhere.

CO-7. Major. Lines 135-139 vs line 31. "Named in the listing as system content" is a UI behavior absent from Goals and CUJs; the listing and its appearance are unspecified.

CO-8. Nit. Lines 149-151. Item 5 markup is broken (unclosed bold, merged fields).

CO-9. Nit. Lines 106, 114. "Rejected by author" is attribution, not rationale.

CO-10. Nit. Lines 122-125, 157. "21 languages" has no list or catalog reference; strings needing translation beyond menu item and draft prompt are not named.

CO-11. Nit. Lines 153-159. `make swift-lib` and "the app scheme" are not explained; manual verification omits CUJ 2.

CO-12. Nit. Lines 3, 130. Resource packaging and runtime bundle lookup are not explained.
