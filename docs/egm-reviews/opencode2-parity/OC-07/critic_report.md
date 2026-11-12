# Stage 7: Critic report (OC-07)

Target: OC-07-onboarding-and-splash.md, FeatureRange 1-176. Reviewers egm-design, egm-structural, egm-style, run one round, all completed. Convergence not assessed (no remediation). Standard: feature-spec-standard for design and structural.

## egm-design (DC), standard feature-spec-standard

DC-1 Blocker, no section: no Alternatives Considered section; standard fails a single-path spec. Missing: separate window vs overlay, profile vs app flag, sheet/checklist vs modal flow, state-ready vs timer, status quo. Fix: add the section with concrete rejection reasons.
DC-2 Major, lines 70-73, 104-120: flag decision unresolved while architecture depends on it; flow order vs vault gate undefined; "new chat" while locked undefined. Fix: decide order against the vault gate.
DC-3 Major, lines 28-36, 44-45: fresh-install detection undefined; upgrading users (models and history, no flag), reinstall, and multiple profiles have no rule. Fix: add a migration rule (existing profile, model, or history counts as onboarded) and a CUJ.
DC-4 Major, lines 40-41, 75-77, 124: splash inconsistent (400 ms floor vs "no artificial delay"; Reduce Motion vs floor); no maximum duration or failure path; OC-14 hang watchdog relevant. Fix: reconcile, add a cap and failure path.
DC-5 Major, lines 129-131, no section: no Risks and Mitigations section (crash mid-flow, install in progress or failing, low disk, primer drift, splash covering unlock or error dialog). Fix: add the section.
DC-6 Major, lines 46-56, 62: first-model step underspecified (embed vs navigate away, "fits" undefined, offline policy vs a download). Fix: state embedding, fit criterion, offline behavior.
DC-7 Major, lines 3-9, 19-26: parity claims not backed; upstream data preload and animated splash dropped without stating so; import dependency unverifiable. Fix: list adopted and dropped behaviors; mark dependency unconfirmed.
DC-8 Major, lines 126-127: 21-language requirement unsized versus INDEX "medium". Fix: estimate string count and localization-gate plan.
DC-9 Major, lines 160-163: source-scan test is brittle and misses transitive prompts. Fix: add a runtime check with a recording permissions fake.
DC-10 Nit, lines 32, 34, 137-154: swift links do not resolve. Fix: ../../../swift/ paths.
DC-11 Nit, lines 25, 50: .kiro brief unverifiable. Fix: cite state and commit or soften "defers to".
DC-12 Nit, lines 57-58, 94-95: empty-state prompt in CUJ 2 not in the plan or file list. Fix: add or cite.
DC-13 Nit, lines 133-146: 400-line guidance unlinked. Fix: link.
DC-14 Nit, lines 1-13: upstream links unpinned; INDEX cites commit 7ce35879b8. Fix: pin.

## egm-structural (ST), standard feature-spec-standard

ST-1 Blocker, lines 28-176: no Alternatives Considered (overlay vs window, takeover vs card, flag vs directory inspection per comparison line 90). Fix: add the section with technical rejection reasons.
ST-2 Blocker, lines 28-176: no Risks & Mitigations (flag vs vault ordering, corrupt or deleted flag replays the flow, splash never dismisses, localization cost). Fix: add the section.
ST-3 Major, lines 31-34: swift/ links do not resolve from the doc folder. Fix: ../../../swift/... (or state the repo convention once).
ST-4 Major, lines 25, 50, 85: link to absent .kiro brief, repeated in INDEX.md lines 53 and 82. Fix: mark as "not yet in repo" in plain text.
ST-5 Major, lines 3-4: cites section 7 only; the section 1 finding "Startup ordering: early window and first-launch onboarding" (comparison line 85) is not cited. Fix: add it.
ST-6 Major, lines 104-115: diagram contradicts CUJ 3 and line 44 (returning branch never reaches main surface; flag check drawn before splash). Fix: redraw.
ST-7 Major, lines 29-37, 102-120: plan lacks flag store format, step model type, readiness signal contract, and order versus ProfileUnlockView. Fix: add a short data-flow subsection.
ST-8 Nit, lines 70-77: Open Questions precede the plan that answers them. Fix: move after the plan.
ST-9 Nit, lines 11-13: metadata placement matches INDEX; no change needed unless the template differs.
ST-10 Nit, lines 72-77: no decision owner or deadline; tests depend on the answer. Fix: name owner, mark dependent tests.
ST-11 Nit, lines 126, 159: "21 languages" and "400-line guidance" unlinked. Fix: link INDEX constraints (lines 102-114).
ST-12 Nit, line 172: DESIGN.md section 9 unlinked. Fix: link ../../../DESIGN.md.
ST-13 Nit, lines 133-163: Localizable.xcstrings and test paths not verified. Fix: verify before implementation.

## egm-style (SY)

SY-1 Major, lines 124 vs 77-78, 42: "no artificial delay" vs 400 ms floor vs Reduce Motion instant dismiss. Fix: one rule, e.g. 400 ms floor, none under Reduce Motion, and align line 124.
SY-2 Major, lines 86-88 vs 96-100, 108-109: CUJ 1 says relaunch "shows the hero directly" but splash always shows. Fix: "splash, then the last state".
SY-3 Major, lines 72-74, 105-119, 147-148: vault gate absent from diagram; "keeps its place" and "composition correct" not testable. Fix: add gate to diagram, state order.
SY-4 Major, lines 44-46, 61, 65-66, 89-94: "Skip" ambiguous (step vs all), quit mid-flow undefined. Fix: define Skip step, Skip all, and flag state on quit.
SY-5 Major, lines 6-9, 19-23: parity overstated; OpenCode prefetches provider and model state and opens a draft tab (comparison line 376), no welcome, primer, or import. Fix: say the four-step flow is a TurboSpark design; state migration-progress UI is out of scope.
SY-6 Major, lines 53-56, 67-68: permission rules cited without source. Fix: link.
SY-7 Nit, lines 19, 72: "genuinely" is filler. Fix: drop.
SY-8 Nit, line 40: "quiet pulse" unmeasurable. Fix: give duration or opacity range.
SY-9 Nit, lines 63-64: "brand moment" claim unsupported. Fix: "No outline tracing in v1; pulse and fade only."
SY-10 Nit, lines 40-41, 75-76: "state loads" and "ready" undefined. Fix: name the signal.
SY-11 Nit, lines 50, 156: "wizard exists" and "import spec absent" undefined and inconsistent. Fix: one defined term.
SY-12 Nit, lines 94, 99-100: "last state as today" and empty-state prompt vague and unplanned. Fix: name the view or add to Goals.
SY-13 Nit, lines 45, 58, 97, 65-68: redundancy (never replays x3; non-goals restate goals). Fix: keep one statement each.
SY-14 Nit, lines 32, 34: swift links do not resolve. Fix: ../../../swift/.
SY-15 Nit, lines 146-148, 172: DESIGN.md section 9 and 400-line guidance unlinked. Fix: link.
SY-16 Nit, line 144: hedged "passes the 400-line guidance". Fix: "split if over 400 lines".
SY-17 Nit, line 122 section: no measurable splash budget. Fix: add a target or state none.
