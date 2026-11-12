# Stage 5: Initial zoom out report (OC-07)

Target: docs/designs/opencode2-parity/OC-07-onboarding-and-splash.md (full file, 176 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.
Unverifiable: link to .kiro/specs/desktop-shell-hardening/brief.md (file absent). The two swift/ links do not resolve from the doc folder.

## egm-zoom-out (ZO)

Verdict: Major issues, no Blocker.

ZO-1 Major, lines 31-34, 149-151: ModelInstallGate is a per-install decision function (allowed, confirm, blocked). It has no notion of installed models. Fix: drop item 5, derive "no installed model" from AppModel installed-model state.
ZO-2 Major, lines 19-27, 72-74, 104-120: flag placed before the vault gate while its location is an open question; diagram says flow precedes main surface while line 119 says the vault gate keeps its place. Precedent `BrowserSettings.onboardingDone` is profile-scoped. Fix: state gate order and flag scope, justify against the precedent.
ZO-3 Major, lines 133-163: file plan carries injectable step machinery and an "absent step" rule and mutation check for a step (import) that does not exist. Fix: build three steps now, add import when the wizard lands.
ZO-4 Major, lines 38-45, 75-78, 117-120, 124: splash has no stated need or measured load time; 400 ms floor contradicts "no artificial delay"; shown on every launch (CUJ 3). Fix: show load evidence or limit to first run; fix the contradiction.
ZO-5 Major, lines 24-26, 50-52, 82: import step depends on a spec nobody can check. Fix: state what the wizard is and where its spec lives, or remove step 3.
ZO-6 Nit, lines 31-34, 149-151: swift links resolve relative to the wrong base. Fix: use ../../../swift/... paths.
ZO-7 Nit, lines 126-127, 150-152: no string inventory; CUJ 2 empty-state prompt is an unplanned string. Fix: list strings.
ZO-8 Nit, lines 160-163: source-scan test does not name the prompt APIs scanned. Fix: name the list, reuse RecordingPermissionsTests.
ZO-9 Nit, lines 137-148: flow files under App/ rather than with their feature. Fix: use a FirstRun/ folder.
ZO-10 Nit, lines 3-9, 63-64: parity claim after dropping the animated outline is not restated. Fix: one sentence on what parity means (sequence and gating, not motion).

## egm-inventive-catalyst (IC)

Verdict: no Blocker, five Major, six Nit.

IC-1 Major, lines 75-78, 109, 124: unconditional splash plus 400 ms floor taxes every launch. Fix: show only if state is not ready in about 150-200 ms, short minimum once shown.
IC-2 Major, lines 72-74, 105-120: flag lean conflicts with vault gate (RootView and ProfileUnlockView are siblings chosen by vault state; model step needs AppModel). Fix: state the order relative to the vault gate; split the flow or run it after unlock.
IC-3 Major, lines 32-34, 149-151: ModelInstallGate reuse claim does not match code. Fix: remove the gate modification, read installed-model state from AppModel.
IC-4 Major, lines 50-52, 119-120, 142-145: speculative step machine and dead import slot. Fix: fixed small sequence now, add import step when the wizard ships.
IC-5 Major, lines 53-56, 67-68: primer duplicates the point-of-ask Allow buttons. Fix: fold into the welcome step or the Audio surface.
IC-6 Nit, lines 62-66, 89-94: no handling for flag-write failure, splash that never ends, or flag missing with existing chats. Fix: add timeout fallback and treat existing data as not fresh.
IC-7 Nit, lines 154-158: test and mutation detail in a strategic doc. Fix: move to test plan or drop.
IC-8 Nit, lines 126-127, 152-153: 21-language cost for a flow that will change. Fix: sequence strings after the step list is final.
IC-9 Nit, lines 46, 89-94: persistent empty-state fallback after skipping model step is not stated. Fix: state it.
IC-10 Nit, lines 3-9, 22-26: OpenCode onboarding centers on console sign-in which Non-Goals reject. Fix: say so in Overview.
IC-11 Nit, lines 25, 31, 33: links. Fix: correct swift link depth; brief unverified.
