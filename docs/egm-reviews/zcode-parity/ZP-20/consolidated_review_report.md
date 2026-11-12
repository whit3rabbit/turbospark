# Consolidated review report: ZP-20 bundled content plugins

Target: docs/designs/zcode-parity/ZP-20-bundled-content-plugins.md (full file, 170 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed. Several spawns hit "Concurrent subagent limit reached" and succeeded on retry (Stage 7 design: 2 retries; Stage 7 style: 2 retries; Stage 8 zoom-out: 3 retries); none ended as not run.
Persona files actually read (as reported by the spawn instructions, reviewers were told to read first): zoom_out.md (stages 5 and 8), inventive_catalyst.md, comprehension.md, design_critic.md, structural.md, style.md, all under /Users/whit3rabbit/.claude/plugins/cache/egm/egm/0.1.1/agents/. Reviewer reports confirm reading for zoom-out, inventive-catalyst, comprehension, design, structural and style; the Stage 8 report did not restate it.
Source IDs: ZO/IC = zoom_out_report, CO = comprehension_report, DC/ST/SY = critic_report, S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 0, Major 11, Nit 11.

## Blocker

None.

## Major

M1. Lines 45, 91, 103, 143-154. The bundled-skill root has no confirmed owner or code: ZP-07 is a Draft with two folder names, INDEX names ZP-05, SkillManager has no bundled or unremovable concept, and item 8 is one line. Fix: add a dependency paragraph naming the owner, what it provides, the one SkillManager seam changed, and reconcile folder and test names. [ZO-1, ZO-2, IC-3, CO-1, DC-1, ST-3, SY-1, S8-K1]
M2. Lines 3-8, 16-22, title. Scope mapping broken: ZP-90 has no 9.14 entry, visualize is finding 9.9 (not in the INDEX row), "plugins" in the title but only skills ship, "half" vs "thirds". Fix: record exclusions inline or add to ZP-90, update INDEX, rename to "content skills". [ZO-5, IC-6, CO-5, CO-10, ST-3, ST-6, SY-1, S8-K2]
M3. Lines 56-61, 128-132, 152-158. Drift guard checks path strings only, with no extraction rule and no pane names; open question left unowned. Fix: close the question as (a), state the extraction rule, and limit the NFR claim to path strings. Reject manifest tooling (S8-R2). [ZO-3, IC-1, CO-3, CO-4, DC-3, SY-5, SY-10, S8-N1]
M4. Lines 41-44, 73-79, 133-135. Visualize contract is undefined and unenforced: "fragment-free", "by default", "trigger grants silently" undefined; 1 MB cap not found in code; "save as" unverified; CDN blank-render recovery missing. Fix: define terms, source or drop the cap, state it is advisory, cut or verify "save as". Reject the validator (S8-R1). [ZO-4, IC-2, CO-6, CO-7, DC-4, DC-12, SY-3, S8-K5, S8-K8]
M5. Lines 36-40, 143-149. Nested diagnose-* skills may not be discovered. Fix: state whether discovery recurses, else flatten; consider one guide skill with reference files. [IC-3, CO-9, DC-2, S8-K4, S8-K6]
M6. Lines 111-127. Alternatives: "Rejected by author" with no reason or link; strawman Alternative A. Fix: one-line concrete reasons; keep two alternatives. [DC-5, ST-2, CO-13, IC-4, SY-8, S8-R3]
M7. Whole doc (nearest 128-137). No Risks and Mitigations section; secrets in MCP config and env, permission-ladder position, agent ignoring the contract, disabled skill. Fix: add a short section with a redaction sentence and "no new tools". [ST-1, DC-6, DC-9, IC-5, S8-K7]
M8. Lines 36-40, 65-71, 83-86, 130-131. Guide never names the stores and panes it cites; MCP log location unknown; CUJ 3 names surfaces outside the goals. Fix: list stores and panes per skill; drop "exact" and "real". [CO-2, ZO-8, SY-7, S8-K3]
M9. Lines 25-26, 33, 134. Claims about existing behavior cite no file or symbol. Fix: add paths or mark unverified. [SY-2, ZO-4, S8-K8]
M10. Lines 88-109. Architecture is a bare ASCII flow; no enforcement point or grant boundary shown. Fix: add a short component and data-flow note. [ST-4]
M11. Lines 24-32, 106-120. Skill-tool activation mechanics and required SKILL.md frontmatter are not explained. Fix: one paragraph. [CO-8]

## Nit

N1. Lines 143-151. Item 8 lacks a change description; resource declaration in Package.swift or Xcode project missing; mutation bullet is a fragment (lines 157-158). Fix: add both. [ZO-2, ST-7, SY-10, S8-N2]
N2. Lines 26, 46. "Like other skills" contradicts "zero built-in content skills". Fix: "like user skills". [ST-8, S8-N3]
N3. Lines 136-137. Localization line is a fragment; which listing strings and where is unstated. Fix: full sentence, name the SkillManager listing path. [ZO-9, CO-12, DC-10, SY-9]
N4. Lines 160-166. Verification omits `make design-check`, localization gates, the guide skill, and a no-CDN method (cite ArtifactWebViewSecurityTests). Fix: add. [ST-8, DC-11, SY-11, S8-K9]
N5. Lines 36-37. Six listed skills with no listing-cost sentence (SkillManager+Budget.swift). Fix: one sentence. [ZO-6, DC-9]
N6. Lines 24, 88. Headers differ from standard names. Fix: "Problem Statement", "Proposed Architecture"; move Open Questions after architecture. [ST-5, ST-8]
N7. Lines 36-44. "actual" and "common chart forms" are filler. Fix: list the forms. [SY-4]
N8. Lines 52-53. "ZP-06 option (b)" unexplained. Fix: restate in a few words. [SY-6, CO-5]
N9. Lines 150-166. Undefined terms (`make swift-lib`, app scheme, hooks) and loose verification terms. Fix: define once. [CO-11, SY-11]
N10. Line 29. "No in-app documentation surface" problem has no matching goal. Fix: drop or add goal. [CO-14]
N11. Lines 3-11, 170. ZCode links unverified; "zcode-guide" vs "turbospark-guide" naming; INDEX sizing. Fix: re-check links, one naming note. [ZO-11, CO-10, SY-12]

## Review guidance from Stage 8

Reject as scope growth: artifact validator, manifest with placeholders, extra alternatives (doctor command), provider contract, always-on index, success metrics. Keep the ambiguity-removing fixes (M1-M5, M7-M9).

## Traceability

- zoom_out_report.md: ZO-1 to ZO-11, IC-1 to IC-6.
- comprehension_report.md: CO-1 to CO-14.
- critic_report.md: DC-1 to DC-12, ST-1 to ST-8, SY-1 to SY-12.
- intermediate_zoom_out_report.md: S8-R1 to S8-R8, S8-K1 to S8-K9, S8-N1 to S8-N3.
