# Stage 7: Critic report (ZP-20)

Single round, no remediation. Reviewers: egm-design (DC, persona agents/design_critic.md, standard feature-spec-standard), egm-structural (ST, persona agents/structural.md, standard feature-spec-standard), egm-style (SY, persona agents/style.md). Convergence not assessed.

## Design critic (DC)

DC-1 Major. Lines 45, 103, 143-150. Bundled-skill root is unconfirmed and the citation is circular. SkillManager has no bundled or unremovable concept (only per-scope disable keys). INDEX names ZP-05 as the delivery path. Item 8 does not specify a new scope case, disable-key namespacing, deletion protection, name collisions with user skills, or root resolution under SwiftPM, app bundle and tests. Fix: state the ZP-07/ZP-05 dependency and ordering; specify the SkillManager change.
DC-2 Major. Lines 36-40, 143-149. Nested diagnose-* skills may be invisible if discovery is flat. Fix: state naming and activation, confirm discovery recurses.
DC-3 Major. Lines 128-131, 152-158. Drift guard checks path strings only; no pane names, keys, or extraction rule; mutation check tests the fixture. Fix: machine-readable manifest of paths and pane identifiers, or at least an extraction rule.
DC-4 Major. Lines 36-44, 130-135. Visualize contract has no enforcement; 1 MB cap unjustified and not found in code; "must not trigger grants silently" is vague; a CDN reference renders blank offline with no recovery path. Fix: name the enforcer (tool or panel check), justify the cap, say how the agent detects and recovers.
DC-5 Major. Lines 111-127. Alternatives thin: "Rejected by author" with no link; Alternative A is a strawman; missing real-plugin-via-ZP-05, generated guide, and a built-in doctor command. Fix: concrete trade-offs and add those alternatives.
DC-6 Major. Lines 63-71, 130-131. CUJ 1 not grounded: MCP log location unknown; reading MCP config and env can expose secrets, no redaction guidance; permission-ladder position unstated (INDEX requires it). Fix: add a security note and ladder statement, or "no new tools".
DC-7 Nit. Lines 1-8. Title says plugins but ships skills; ZCode guide is an npm package so parity is functional only.
DC-8 Nit. Lines 56-61. Open question has no criteria or owner, while NFRs assume part of (b).
DC-9 Nit. Lines 128-137. No Risks section, success metrics, or listing and context budget for six extra skills (SkillManager+Budget.swift exists).
DC-10 Nit. Line 137. Which listing strings are localized and where is unspecified.
DC-11 Nit. Lines 163-165. No-CDN check is manual; reference ArtifactWebViewSecurityTests.
DC-12 Nit. Lines 73-79. "Save as" not verified to exist.

## Structural (ST)

ST-1 Major. Whole document (closest 128-137). No Risks & Mitigations section. Unaddressed: drift, agent ignoring the contract, oversized or malformed HTML, secrets in MCP config, user disabling a depended-on skill. Fix: add the section.
ST-2 Major. Lines 111-126. Alternatives Rule: rejection attributed to author without a link; only two alternatives; content-source and delivery choices not treated as alternatives. Fix: as DC-5.
ST-3 Major. Lines 20-22, 45, 144-149. Cross-references broken: ZP-07 vs ZP-05 ownership of the bundled root is unreconciled; ZP-90 has no 9.14 entry, so "recorded as excluded in ZP-90" does not resolve. Fix: add a 9.14 entry to ZP-90 or record the exclusions here; fix the root owner.
ST-4 Major. Lines 88-109. Architecture is an ASCII flow only; no component or data-flow description; enforcement point and grant boundary not shown. Fix: add a short component and data-flow description.
ST-5 Nit. Lines 56-61. Open Questions placed before architecture; pre-selects an answer.
ST-6 Nit. Lines 3-8, 16-22. "Half" vs "thirds"; 9.9 (Partial) place in the program unexplained.
ST-7 Nit. Lines 143-158. File list formatting inconsistent; no localization file; no state file for disable-able.
ST-8 Nit. Lines 31, 46, 160-166. Verification omits localization parity gates and `make design-check`; "like other skills" contradicts "zero built-in content skills". Headers differ from standard names (lines 24, 88).

## Style (SY)

SY-1 Major. Lines 45-46, 18-22, 91. ZP-07 cited as the mechanism without saying what it provides; INDEX says ZP-05; title says "plugins" but body ships skills. Fix: name the owning draft, define or rename "plugins".
SY-2 Major. Lines 25-26, 33, 134. Claims about existing behavior (zero built-in skills, four agent definitions, ArtifactWebView offline-until-granted) cite no file or symbol. Fix: add path or symbol, or mark unverified.
SY-3 Major. Lines 37-44, 133-135. "Fragment-free", "self-contained charts", "by default" (no override or decider) and "trigger grants silently" are undefined; 1 MB cap has no rationale and no stated match to ZCode's contract. Fix: define each term and source the cap.
SY-4 Nit. Lines 36-44. "actual" and "common chart forms" are filler or vague. Fix: list the forms.
SY-5 Nit. Lines 56-61. Open question hedges and ignores the drift-guard test already committed in NFRs; (b) is not tied to a file.
SY-6 Nit. Lines 52-53. "ZP-06's open question (b)" is an unexplained cross-reference.
SY-7 Nit. Lines 66-86. "exact" and "real" are unmeasurable; log location unstated.
SY-8 Nit. Lines 115-126. Thin pros and cons; "Rejected" repeated under "Reason for Rejection".
SY-9 Nit. Lines 136-137. Fragment without a verb; listing strings have no implementation step.
SY-10 Nit. Lines 150-157. SkillManager row lacks Proposed Changes; mutation bullet is a fragment; drift test covers paths not pane names.
SY-11 Nit. Lines 162-166. Verification terms loose ("small dataset", "broken MCP config"); no method for confirming no-CDN; other diagnosis skills and the guide have no manual step.
SY-12 Nit. INDEX line 48. INDEX sizes the draft small-medium while the plan lists nine files.
