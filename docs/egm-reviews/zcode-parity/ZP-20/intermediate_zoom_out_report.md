# Stage 8: Intermediate zoom out report (ZP-20)

Reviewer: egm-zoom-out (completed; persona agents/zoom_out.md, standard zoom-out-standard). Target unchanged since Stage 5, so every Stage 5 finding is still open. Verified in this pass: ZP-90 has no 9.14 entry (so lines 21-22 do not resolve); INDEX lines 87-88 name ZP-05 as the delivery mechanism while the doc names ZP-07; ZP-07 itself uses two folder names (`Plugins/BuiltinSkills/` line 81, `Resources/bundled-skills/` line 130) and ZP-09 uses the latter; no 1 MB cap exists in any Artifact source.

## Open from Stage 5
ZO-1/ZO-2 (root owner unreconciled), ZO-4 (contract not checked against ArtifactWebView), ZO-5 (9.9 not in INDEX), IC-6/ST-3 (broken ZP-90 reference).

## Asks to reject (add scope or complexity)
S8-R1. IC-2 artifact-path validator: new code outside the file list, heuristic. Keep the contract as advisory prose and say so; cite or drop the 1 MB number.
S8-R2. IC-1 manifest with placeholder resolution, and the manifest part of DC-3: more machinery than six markdown files need. Close the open question as (a); add one line on how paths are extracted.
S8-R3. DC-5/ST-2 extra alternatives (doctor command is new surface; generated guide is option (b)). Keep two; replace "Rejected by author" with a one-line reason.
S8-R4. IC-3 provider contract: belongs in ZP-05 or ZP-07.
S8-R5. IC-5 always-on index: a disabled skill degrading to generic help is acceptable; one sentence in Risks.
S8-R6. Success metrics and budget analysis (DC-9): reject; keep the Risks section and a one-sentence listing cost.
S8-R7. SY-12 and ZO-10: cosmetic.
S8-R8. No extra localization file; state which listing strings are localized and where.

## Asks to keep (remove ambiguity)
S8-K1. One dependency paragraph: blocked on the bundled-root owner (pick ZP-05 or ZP-07), what it provides, the single SkillManager seam item 8 changes, and reconcile folder and test file names. Highest value.
S8-K2. Fix the 9.14/9.9 mapping: add 9.14 to ZP-90 or state exclusions inline; add 9.9 to the INDEX row; rename title from "plugins" to "content skills"; explain "half" vs "thirds".
S8-K3. Name the stores and panes per diagnose skill; drop "exact" and "real"; drop shortcuts and profiles from CUJ 3 or add them to goals.
S8-K4. State whether SkillManager discovery recurses; if not, flatten the layout.
S8-K5. Define visualize terms; replace "by default" and "trigger grants silently" with a plain no-network rule; say what the agent does when a CDN reference renders blank; reference ArtifactWebViewSecurityTests.
S8-K6. Consider one guide skill with reference files instead of six listed skills (also settles K4 and listing cost).
S8-K7. One sentence on redacting env values and keys, and "no new tools; existing ladder positions apply" (INDEX requires a ladder statement).
S8-K8. Cite the symbol for offline-until-granted; verify "save as" or cut it; give an MCP log location that exists or soften CUJ 1.
S8-K9. Verification: add `make design-check`, localization gates, and a manual step for the guide skill.

## New issues
S8-N1 Major. Lines 128-132, 152-158. The drift test proves strings exist, not that the text matches the app; scope the NFR claim to "path strings".
S8-N2 Nit. Lines 143-151. Item 8 has no change description; resource declaration in Package.swift or the Xcode project is not mentioned.
S8-N3 Nit. Line 46 vs 26. "Like other skills" contradicts "zero built-in content skills"; say "like user skills".
