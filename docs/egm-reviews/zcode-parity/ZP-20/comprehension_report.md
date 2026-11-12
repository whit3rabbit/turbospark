# Stage 6: Comprehension report (ZP-20)

Reviewer: egm-comprehension (completed; persona agents/comprehension.md). It read only the target.

## What this proposes

The document proposes adding two sets of built-in instructions (skills) to the TurboSpark app. The first is a "turbospark-guide" pack: one configuration guide skill plus five diagnosis skills (hooks, commands, skills, MCP, plugins). Each tells the agent how to inspect the relevant stores, list state, and walk the user through verification, citing the app's real paths and settings panes. The second is a "visualize" skill: a strict contract for the agent when it writes an interactive HTML file (single file, 1 MB cap, no external CDN, self-contained SVG or canvas charts) so it renders in the app's artifact panel. Both would be plain SKILL.md files in a bundled resources folder, found by SkillManager, and could be disabled but not removed. A test would check that every path the guide mentions exists in the app sources.

## How the existing system works, per this document

TurboSpark currently ships no content skills, only four built-in agent definitions. The agent has no guided path for diagnosing broken hooks, commands, skills, MCP servers or plugins. An "artifact panel" (ArtifactWebView) already renders HTML, is offline until the user grants access, and has "existing grants". A "bundled-skill root" mechanism, attributed to another design (ZP-07), is said to exist. SkillManager discovers skills, and the agent activates them through a "Skill tool". It reads stores with its existing file tools. The document does not say what ZP-07 provides, so the reader cannot tell whether the bundled root exists or is proposed elsewhere.

## Findings (CO)

CO-1 Major. Lines 45, 103, 143-152. The bundled-skill root (ZP-07) is never explained; unclear whether it exists, how "unremovable but disable-able" is enforced, and what SkillManager item 8 must do if ZP-07 already provides discovery. Needs a one-paragraph ZP-07 summary.
CO-2 Major. Lines 36-40, 65-71, 130-131. The guide must cite "actual paths and settings panes" but none are listed (no store locations, no pane names, no location for MCP spawn env and logs).
CO-3 Major. Lines 130-131, 153-158. Drift guard is vague: how path strings are identified in prose, what "exists in app sources" means, pane names required but not tested, mutation check unclear.
CO-4 Major. Lines 56-61. Open question on content source of truth is unresolved, yet NFRs and tests assume option (a); unclear if the test works under (b).
CO-5 Major. Lines 3-8, 22, 52, 54, 90, 126. Reliance on external references the reader cannot see (finding 9.14, 9.9 "Partial", ZP-06 option (b), ZP-07, ZP-90). ZCode is never introduced; the in-scope third of 9.14 is never named.
CO-6 Major. Lines 41-44, 51, 77-79, 133-135. Visualize contract only sketched: "fragment-free" unclear, no enforcement or behavior above 1 MB, "by default" implies an undescribed exception path, the grant model is not described.
CO-7 Major. Lines 77-79. "Save as" is not shown to exist or to be delivered; no file in the list implements it.
CO-8 Major. Lines 24-32, 106-120. Skill-tool activation mechanics (trigger by description, required SKILL.md frontmatter) are not explained.
CO-9 Nit. Lines 36-38, 143-148. Unclear whether diagnose-* are separately activatable skills or sections of the index; CUJs treat them as separate.
CO-10 Nit. Lines 18, 36, 41, 160-166. Naming inconsistency: zcode-guide vs turbospark-guide; "plugins" in the title vs "skill pack"; "plugins" is also a diagnosis topic.
CO-11 Nit. Lines 150-152, 160-166. Undefined terms: `make swift-lib`, "app scheme", hooks, commands.
CO-12 Nit. Lines 136-137. "Listing strings localized": which listing, where the strings live; no file covers it.
CO-13 Nit. Lines 118, 126. "Rejected by author" is not a reason.
CO-14 Nit. Line 29. The "no in-app documentation surface" problem has no matching goal.
