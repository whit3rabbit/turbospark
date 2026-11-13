# Planning

## Overview
Model-facing tools for interacting with the user and shaping a session: AskUserQuestion (including the parked interactive path), EnterPlanMode/ExitPlanMode, ReportFindings, ProposeSkills, ProposeGoal, SendFeedback, and the `skill` loader schema. The directory splits into Codable payload types plus OpenAI tool schemas, and static executor enums that parse sloppy argument shapes. Several tools are deliberately not implemented and say so instead of faking success.

## Interface
- Payload types and schemas: `UserQuestionOption`, `UserQuestionItem`, `AskUserQuestionInput/Output`, `EnterPlanModeInput/Output`, `ExitPlanModeInput/Output`, `CodeFindingItem`, `ReportFindingsInput/Output`, `SkillProposalItem`, `ProposeSkillsInput/Output`, `ProposeGoalInput/Output`, `SendFeedbackInput/Output`, and `PlanningInteractiveToolDefinitions` (`askUserQuestion`, `enterPlanMode`, `exitPlanMode`, `reportFindings`, `proposeSkills`, `proposeGoal`, `sendFeedback`, `all`) [PlanningInteractiveTools.swift].
- Executors [PlanningInteractiveExecutors.swift]:
  - `AskUserQuestionExecutor`: `parseQuestions`, `execute`, `executeAwaitingAnswer`, `submitAnswer`, `waitForAnswer`, hooks `onQuestionsAsked` and `answerWaiter`.
  - `PlanModeExecutor`: `enter`, `exit`, `isPlanModeActive(for:)`, hook `onPlanModeChanged`.
  - `ReportFindingsExecutor`: `parseFindings`, `execute`.
  - `ProposeSkillsExecutor.execute(arguments:projectRootURL:)`.
  - `ProposeGoalExecutor.execute` and `SendFeedbackExecutor.execute` (both always throw).
- Skill tool: `SkillInput`, `SkillOutput`, `SkillToolDefinitions.skill`, `.skillTool(projectURL:contextTokens:)` (description lists available skills within a budget), `.all` [SkillTool.swift:SkillToolDefinitions].
- Child directories: none.

## Invariants
- `UserQuestionItem.multiSelect` decodes as optional (default false) so one missing key does not fail the whole questions payload [PlanningInteractiveTools.swift:UserQuestionItem.init(from:)].
- `parseQuestions` accepts `questions`, `questions_json`, or single-question shorthand. Defaults to Yes/No options when none are given, and treats `"true"` or `"1"` as true because the dispatch gate flattens JSON booleans [PlanningInteractiveExecutors.swift:parseQuestions].
- `executeAwaitingAnswer` falls back to the non-blocking `execute` when no `answerWaiter` is installed (tests, subagents). When parked, task cancellation dismisses the question so the loop never waits on a card the user cancelled [PlanningInteractiveExecutors.swift:executeAwaitingAnswer].
- `submitAnswer` is idempotent (no continuation means no-op) so double-clicks and Stop races cannot double-resume. `waitForAnswer` resumes any stale continuation for the same chat key with a dismissed note so a task is never leaked. A nil chat id maps to the key `"default"` [PlanningInteractiveExecutors.swift:submitAnswer, waitForAnswer].
- Static state (`pendingQuestions`, `answerContinuations`, `activePlanModes`, `savedPlans`) is guarded by `NSLock`. Plan mode and pending questions are in-memory only and keyed by chat id string.
- `ProposeSkills` must never overwrite an existing `SKILL.md` (a prompt-injected proposal named after a trusted skill would otherwise replace its instructions); it skips and reports them. Names are trimmed, lowercased, and `/` and `\` replaced, and `.` or `..` or empty names are skipped. Scope is user when `scope == "user"`, there is no project root, or the root is `/` [PlanningInteractiveExecutors.swift:ProposeSkillsExecutor].
- The `ReportFindings` item schema advertises `file`, `line`, `summary`, `failure_scenario`, `category`, and `verdict`. `CodeFindingItem` also decodes `short_summary` and `outcome`, which the executor never renders, so they are not advertised. The executor also accepts flat single-finding arguments and `effort_level`, unadvertised [PlanningInteractiveTools.swift, PlanningInteractiveExecutors.swift:ReportFindingsExecutor].
- `ProposeGoal` and `SendFeedback` throw "not implemented" (codes 34, 35) on purpose: no goal wiring or feedback receiver exists, and a success reply would mislead the model. Do not turn them into success stubs unless the backing feature exists.
- Error codes under domain `TurboSparkTool`: 30 questions, 31 findings, 32 skills, 33 goal parse, 34 goal not implemented, 100 feedback not implemented. Code 35 belongs to `BatchToolExecutor`; `SendFeedback` moved off it so the two stay distinguishable.
- `skillTool(projectURL:contextTokens:)` is dynamic; `skill` in `all` is the static fallback without a skill listing.

## Side Effects
- Disk: `ProposeSkills` creates the skills directory (`SkillManager.shared.defaultUserSkillsDirectory` or `<project>/.turbospark/skills`), per-skill folders, and writes `SKILL.md` atomically [PlanningInteractiveExecutors.swift:ProposeSkillsExecutor].
- Calls `SkillManager.shared.invalidateResolutionCache()` after saving; `skillTool` reads `SkillManager.shared.resolveEffectiveSkills` and `formatSkillsWithinBudget` [SkillTool.swift].
- Static callbacks: `onQuestionsAsked`, `answerWaiter`, `onPlanModeChanged` are invoked to reach the app model.
- Suspends the calling task on a `CheckedContinuation` while awaiting a user answer.
- No network, Keychain, UserDefaults, NotificationCenter, or process spawn.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
