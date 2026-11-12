# Stage 7: Critic report (OC-90)

Reviewers: egm-design (DE), egm-structural (ST), egm-style (SY). All completed. Standard: feature-spec-standard for DE and ST (applies only in reduced form to a decision log: Problem, Alternatives rule, Risks; Architecture, plan, tests, open questions do not apply). SY had no standard. No Blockers from any reviewer.

## egm-design

* DE-1 Major, 112-113: broken link to TurboSparkMotion.swift (repo-root-relative). Use `../../../swift/...`.
* DE-2 Major, 108-119: motion rationale "six-ish knobs" is wrong; the file defines nine named animations plus a press-scale style and a helper. Reopen trigger counts call sites, not parameters. State the real count and a measurable trigger.
* DE-3 Nit, 3, 160-168 vs INDEX 28-69: 16 drafted + 8 covered + 11 excluded = 35 against a goal of 31 findings. Overlaps (multi-window, crash recovery, API-key path) are explained in prose but not mapped one-to-one. Add a reconciliation line.
* DE-4 Nit, 160-168: "reopen conditions recorded per finding" is overstated (missing for crash reporting, SSH, cloud usage, HTTP recording, WSL). "None permanent except OAuth" conflicts with policy-bound exclusions. Say which are policy-bound and which judgment-bound.
* DE-5 Nit, 66-70: claim that ZP-22 covers OpenCode's provider page is unbacked; nothing in ZP-22 addresses multiple named accounts per provider.
* DE-6 Nit, 40, 94: ZP-23 and ZP-18 exist but are not linked.
* DE-7 Nit, 54-56: "respectively" pairs three residuals with two owners. OC-14 owns hang detection and guarded relaunch; the shell-hardening spec owns diagnostics export.
* Also: no record of what users lose per exclusion. Cloud usage (99-106) should cross-reference ZP-90 line 131 (quota views as ZP-22 follow-ups). Translation entry has no owner or acceptance criteria.

## egm-structural

* ST-1 Major, 113: broken link (same as DE-1). Other relative links resolve.
* ST-2 Major, 166-168: decision-log claims not backed by the body (reopen conditions present only for multi-window, pairing, motion, process-daemon). "None of these exclusions" is loose because translation governance is not an exclusion.
* ST-3 Major, 63-66, 71-72, 167-168: Alternatives rule gap. Permanence of OAuth rests on "author, 2026-11-11" with no link; "recorded in ZP-90" at line 65 has no link to ZP-90 section 7.2.
* ST-4 Nit, whole doc: no fixed per-finding fields (What it is / Why not drafted / Decision / Reopen condition) and no Risks content.
* ST-5 Nit, 1-13: no labelled Overview, unlike ZP-90.
* ST-6 Nit, headings and table: "Section N" refers to the comparison report's sections, not this doc. Say so once near line 3.
* ST-7 Nit, 39-40, 94: ZP-23 and ZP-18 unlinked.
* ST-8 Nit, 54-57: "respectively" ambiguous (same as DE-7).
* ST-9 Nit, 112-119: "six-ish" should be the real count.
* ST-10 Nit, 128 and table row 29: translation governance is not an exclusion and has no tracking pointer. Multi-window row 21 should read "excluded (partly covered by the spec)".
* Table vs body consistent on all 11 verdicts; OC-90 vs INDEX consistent on count and ordering. .kiro links unverifiable.

## egm-style

* SY-1 Major, 166-168: log contradicts the body (same as ST-2).
* SY-2 Major, 127-134: "recommended, small, and needs no OpenCode code reading" is an unsupported effort claim; the requirement "require new locales to record their sources" is buried in a descriptive bullet. State it as its own item.
* SY-3 Major, 51-53: "no honest local analog short of adding a crash reporting service" is absolute and subjective; local crash logs exist. Say no local analog for upload-style reporting, and local logging is covered by OC-14.
* SY-4 Major, 113: broken link (same as DE-1).
* SY-5 Nit, 114-119: "six-ish", "roughly", figurative "would outgrow its subject"; "design-check ratchet" undefined.
* SY-6 Nit, 104-105: "honest local analog already exists" has no link; "honest" is filler.
* SY-7 Nit, 40, 94, 78, 93: ZP-23, ZP-18 and ZP-90 section references unlinked.
* SY-8 Nit, 44-45: "the app's existing strength" is subjective; state the behavior.
* SY-9 Nit, 94-95: "feasible engineering but ... deserves its own security design" is opinion; state the decision rule.
* SY-10 Nit, table verdict labels: "remote category", "remote control axis", "excluded permanently" are inconsistent vocabulary.
* SY-11 Nit, 82-83, 153: "legitimately" is filler and "the report itself says" should cite the report section.
* SY-12 Nit, 155-157: stubs and "Rust side owns provider fixtures by convention" uncited.
* SY-13 Nit, 2-9: intro is a run-on of four categories.
* SY-14 Nit, 14 and footer: "*Drafted with EGM*" kept only if house style requires (it matches INDEX).
