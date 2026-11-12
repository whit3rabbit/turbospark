# Automation

## Overview
Tools that let the model act over time or reach the user outside the turn: a persistent cron scheduler, one-shot wakeups, Sleep, PushNotification, a read-only Config view, and CtxInspect. It also holds Codable payload types and schemas for Monitor, Workflow, RemoteTrigger, notifications, and onboarding picker tools. Only the cron store, the schedule parser, and the Sleep/PushNotification/Config/CtxInspect executors have real behavior here. Most of the Monitor, Workflow, RemoteTrigger, ReadNotifications, ShowOnboardingRolePicker, and ClaudeDesign payload structs have no executor in this directory.

## Interface
- Cron schedule parser: `CronSchedule` (failable init from a 5-field expression; `matches`, `nextFire(after:)`, `describe()`, `localizedDescription()`) [CronScheduler.swift:CronSchedule].
- Cron persistence and firing: `AppCronJob` (with `CronRunRecord`, `humanSchedule`, `localizedHumanSchedule`), `CronScheduler.shared`, `create`, `delete`, `list`, `setEnabled`, `createOneShot`, `fireDueJobs(now:)`, the `fireHandler` hook, and static executors `executeCreate`, `executeDelete`, `executeList`, `executeScheduleWakeup` (each takes an injectable `scheduler`) [CronScheduler.swift:CronScheduler].
- Executors: `SleepExecutor.execute`, `PushNotificationExecutor.execute` plus `onNotificationPushed`, `ConfigToolExecutor.execute` (get and list only), `CtxInspectExecutor.execute` [AutomationExecutors.swift].
- Tool schemas: `MonitoringNotificationDefinitions` (Monitor, PushNotification, Sleep, Config, CtxInspect) [MonitoringNotificationTools.swift:MonitoringNotificationDefinitions]; `WorkflowCronDefinitions` (Workflow, CronCreate, CronDelete, CronList, ScheduleWakeup) [WorkflowCronTools.swift:WorkflowCronDefinitions].
- Payload structs: `MonitorInput/Output`, `RemoteTriggerInput/Output`, `ReadNotificationsInput/Output`, `NotificationItem`, `PushNotificationInput/Output`, `ShowOnboardingRolePickerInput/Output`, `ClaudeDesignInput/Output` [MonitoringNotificationTools.swift]; `WorkflowInput/Output`, `CronCreateInput/Output`, `CronDeleteInput/Output`, `CronListInput`, `CronJobItem`, `CronListOutput`, `ScheduleWakeupInput/Output` [WorkflowCronTools.swift].
- Child directories: none.

## Invariants
- Cron fields accept only `*`, `*/N`, `a-b`, `a-b/N`, and comma lists; names are refused (the parser returns nil) rather than half-supported. Day-of-week 7 is folded into 0 [CronScheduler.swift:parseField, init].
- Day-of-month and day-of-week follow vixie OR semantics only when both fields are restricted (do not start with `*`); otherwise they AND [CronScheduler.swift:dayMatches].
- `nextFire` floors `after` to its minute and starts one minute later (strictly after), skips whole months, days, and hours, and gives up after 366 days. A nil result means the job would never fire, and `create` refuses it [CronScheduler.swift:nextFire, create].
- Matching uses the local calendar, since "09:00" means local time [CronScheduler.swift:matches].
- One-shot jobs (`fireAt` set) keep their absolute time. The `cron` field on them is a placeholder (`* * * * *`) and must never be used to recompute the fire time. Pause/resume and relaunch preserve `fireAt` [CronScheduler.swift:load, setEnabled, createOneShot].
- On load, recurring jobs get a freshly computed `nextFireAt` so a job that fired before quit does not fire immediately [CronScheduler.swift:load].
- Firing is single delivery: `takeDueJobs` reserves ids in `inFlight` under the lock, and `completeFire` advances (recurring) or deletes (non-recurring) and records `fired` or `missed`. The lock is never held across `await`. Run history is capped at `runHistoryLimit` (10), newest first [CronScheduler.swift:fireDueJobs, takeDueJobs, completeFire, recordRun].
- `fireHandler` returns whether the prompt was delivered; false (or no handler) records a missed run [CronScheduler.swift:fireDueJobs].
- Saves are versioned: snapshot taken under `lock`, then `writeLock` plus a `writtenVersion` check so a stale snapshot never overwrites a newer one [CronScheduler.swift:save].
- `AppCronJob` decodes leniently (missing keys default) so a row from a newer build does not sink the file [CronScheduler.swift:AppCronJob.init(from:)].
- `CronCreate` and `ScheduleWakeup` require an active chat id. `ScheduleWakeup` clamps delay to 60...3600 s, and `stop` reports that no wakeup loop exists instead of pretending to stop one [CronScheduler.swift:executeCreate, executeScheduleWakeup].
- `humanSchedule` and `describe()` stay English because CronList returns them to the model and tests pin them. Use `localizedHumanSchedule` for the UI [CronScheduler.swift:describe].
- `Config` is read-only: `set` throws and changes nothing. Do not make it report success for a no-op [AutomationExecutors.swift:ConfigToolExecutor].
- `Sleep` clamps to 0.1...300 s; `PushNotification` reports the true delivery outcome (system notification only when running from an `.app` bundle with a bundle id and permission granted, else in-app only) [AutomationExecutors.swift].
- `CronCreateInput.durable`, `CronJobItem.durable`, and `MonitorInput` fields are schema-level only; the scheduler always persists jobs.
- Error codes under domain `TurboSparkTool`: 60-64 cron (60 invalid or missing cron, 61 missing prompt, 62 no active chat, 63 CronDelete, 64 ScheduleWakeup); 65 PushNotification missing message, 66 Config unknown action, 67 Config set unsupported. `ToolErrorCodeUniquenessTests` fails if two source files reuse a code.

## Side Effects
- Disk: `cron_jobs.json` under `AppStorageRoot.directory` (or an injected directory), loaded in `init` and written after every mutation via `AppJSONStore.load/save` [CronScheduler.swift:init, save].
- UserNotifications: `PushNotificationExecutor` requests authorization (`.alert`, `.sound`) and posts a `UNNotificationRequest` with no trigger [AutomationExecutors.swift:PushNotificationExecutor].
- Calls `onNotificationPushed` (in-app toast) before attempting the system notification.
- Task sleep: `SleepExecutor` suspends the calling task.
- Reads `MacAppSettingsFileStore.load()` for Config and `AppToolCatalog.tools(for: .coder, ...)` for CtxInspect [AutomationExecutors.swift].
- The scheduler does not own a timer; the app's poll timer calls `fireDueJobs`, and `fireHandler` (installed by `AppModel`) submits the prompt on the main actor.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
