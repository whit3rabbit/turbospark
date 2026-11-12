# Stage 5: Zoom Out report (OC-90)

Target: docs/designs/opencode2-parity/OC-90-excluded-and-decision-findings.md (full file, 172 lines)
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.
Note: the Zoom Out standard targets feature designs. Context, altitude and anti-weeds vectors apply; code-reuse and YAGNI apply only to the docs-task and reopen items.

## egm-zoom-out

* ZO-1 Major, lines 166-168: "Reopen conditions are recorded per finding above" is false. Only multi-window (46), pairing (96), motion tuning (117), PTY daemon (146) and cassettes (158, weak) have one. Missing: crash reporting, SSH, WSL, cloud usage, OAuth, translation governance.
* ZO-2 Major, lines 166-168: "none permanent except provider OAuth" conflicts with WSL (N/A by construction) and the "do not adopt" items with no reopen path. Name three classes: permanent by decision, N/A, policy exclusion reopenable only if policy changes.
* ZO-3 Nit, line 113: TurboSparkMotion.swift link is repo-root-relative and broken from this folder. Needs `../../../swift/...`.
* ZO-4 Nit, 114-119: "handful of constants" and "six-ish knobs" are loose (file has 9 `static let` Animation members); the reopen threshold ("a dozen animated sites") measures a different quantity than "knobs".
* ZO-5 Nit, 101-105: "analog already exists" overstates; the source report rates token and cost accounting Partial.
* ZO-6 Nit, 127-134: translation governance has no owner or tracking pointer; unclear whether "require new locales to record sources" is a gate or a convention.
* ZO-7 Nit, 3-5: preamble lists four categories; the doc also has a "docs task" class. Multi-window is both "excluded" here and "covered elsewhere" in INDEX.md.
* ZO-8 Nit, 40: ZP-23 is mentioned without a link.
* ZO-9 Weakest justification, 49-59: crash reporting is equated with telemetry; local-only capture is not telemetry. State that local capture is covered and only a remote reporter is excluded.
* Unverifiable: .kiro links (43, 140) and the inspector-terminal contract claims (141-145); "Rust owns provider fixtures by convention" (156-157) is uncited.

## egm-inventive-catalyst

* IC-1 Major, 151-160: "no provider HTTP traffic to record" is stale. ZP-22 (adopted same day) adds Swift-side Anthropic and OpenAI-compatible SSE clients and needs SSE fixtures. Suggest a URLProtocol replay stub with scrubbed fixtures owned by ZP-22.
* IC-2 Major, 49-59: "no honest local analog short of a service" is a false compromise. MetricKit payloads, DiagnosticReports .ips files, or a clean-exit sentinel give local post-mortem evidence with no network. Narrow the decision to "no transmission".
* IC-3 Major, 21, 45-47: the multi-window reopen condition is unsatisfiable, since tabs and split panes never serve a second display. ZP-21's keyed per-tab context store could host a pop-out window; restate the condition.
* IC-4 Major, 99-106: "no cloud tier, nothing to budget" ignores ZP-22. With remote providers the user incurs third-party cost; a user-entered price and soft warning is a ZP-22 follow-up.
* IC-5 Major, 87-97 (also 143): pairing premise "loopback-only" omits ZP-90's loopback-or-single-verified-Tailscale-IPv4 bind rule. Keep the exclusion but cite that transport in the reopen note.
* IC-6 Major, 8-9, 164-168: baseline "no cloud services" contradicts adopted ZP-22 (prompts go to hosted APIs). The real rule is narrower: no app-operated hosted infrastructure or identity; user-directed API calls with user keys are allowed. State it once.
* IC-7 Major, 121-134: prose-only translation rule contradicts the repo's own lesson that only gates stick (SWIFT_LOCALIZATION.md). Suggest a provenance file plus a parity gate. Nit: CLDR is locale data, not a UI terminology authority (line 131).
* IC-8 Major, 136-149: persistence versus no-orphans is a false either/or; respawn with restored cwd, env and scrollback, or user tmux, meets the need. Table (line 30, "revisit after inspector-terminal") and body (line 146, "if users ask") give different triggers.
* IC-9 Nit: same as ZO-1/ZO-2 (reopen conditions, WSL permanence).
* IC-10 Nit: same as ZO-3 (broken link line 113).
* IC-11 Nit, 115-119: motion reopen trigger is unmeasured; add a ratchet count.
* IC-12 Nit, 61-72: multiple named accounts is not OAuth; say that multiple named API-key providers are not excluded.
* Limits: .kiro links unverifiable; TRIZ does not apply to WSL, SSH category, or the motion-tuning exclusion itself.
