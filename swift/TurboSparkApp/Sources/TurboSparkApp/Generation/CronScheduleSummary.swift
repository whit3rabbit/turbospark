import Foundation

/// What a `CronCreate` card says about its schedule (component audit,
/// schedule-card): the plain-language shape and the next runs. Both come
/// from `CronSchedule`, the parser the scheduler fires from, and the
/// arguments are read the way `executeCreate` reads them, so the card cannot
/// promise a time the scheduler would not use.
struct CronScheduleSummary: Equatable {
    static let recurringRunCount = 3

    let expression: String
    /// Nil for a shape with no plain-language name or an expression the
    /// parser refuses (names, @-macros, L, #, out of range): the card then
    /// shows the raw expression only.
    let description: String?
    let recurring: Bool
    /// Empty when the expression does not parse, and for a one-shot whose
    /// call already ran: its one run is fixed in the store, and a time
    /// computed from now would be a run that never happens.
    let nextRuns: [Date]

    static func isCronCreate(_ toolName: String) -> Bool {
        ["croncreate", "cron_create"].contains(toolName.lowercased())
    }

    /// Nil when the call carries no expression at all.
    static func make(
        arguments: [String: String], hasRun: Bool, now: Date, calendar: Calendar = .current
    ) -> CronScheduleSummary? {
        guard let expression = CronScheduler.createCronArgument(arguments) else { return nil }
        let recurring = CronScheduler.createRecurringArgument(arguments)
        guard let schedule = CronSchedule(expression) else {
            return CronScheduleSummary(
                expression: expression, description: nil, recurring: recurring, nextRuns: [])
        }
        let wanted = recurring ? recurringRunCount : (hasRun ? 0 : 1)
        var runs: [Date] = []
        var cursor = now
        while runs.count < wanted, let next = schedule.nextFire(after: cursor, calendar: calendar) {
            runs.append(next)
            cursor = next
        }
        return CronScheduleSummary(
            expression: expression, description: schedule.localizedNamedDescription(),
            recurring: recurring, nextRuns: runs)
    }

    /// One run as "Mon, Nov 23, 9:00 AM" in the UI locale. The weekday is
    /// there because a schedule is usually read by the day it lands on.
    static func runLabel(_ date: Date, locale: Locale, timeZone: TimeZone) -> String {
        var style = Date.FormatStyle(date: .omitted, time: .shortened, locale: locale, timeZone: timeZone)
        style = style.weekday(.abbreviated).month(.abbreviated).day()
        return date.formatted(style)
    }

    /// The zone the runs are shown in, named so a run time is never read
    /// as some other zone's.
    static func zoneLabel(_ timeZone: TimeZone, locale: Locale) -> String {
        timeZone.localizedName(for: .generic, locale: locale) ?? timeZone.identifier
    }
}
