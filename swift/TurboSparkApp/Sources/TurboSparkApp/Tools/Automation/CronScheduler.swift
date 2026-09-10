import Foundation

/// A parsed 5-field cron expression (minute hour day-of-month month
/// day-of-week) with the two operations the scheduler needs: does this
/// moment match, and what is the next matching moment.
///
/// Fields accept `*`, `*/N`, `a-b` ranges, and comma lists, which covers
/// every expression the `CronCreate` tool schema advertises. Named months
/// and weekdays are not accepted -- the tool description says "standard
/// 5-field cron pattern (e.g. '*/5 * * * *')" and a parser that quietly
/// says yes to a vocabulary it does not implement is worse than one that
/// refuses.
public struct CronSchedule: Equatable, Sendable {
    public let minute: Set<Int>
    public let hour: Set<Int>
    public let dayOfMonth: Set<Int>
    public let month: Set<Int>
    public let dayOfWeek: Set<Int>
    /// Whether the day-of-month / day-of-week fields were restricted, which
    /// decides the standard cron OR rule between them.
    private let domRestricted: Bool
    private let dowRestricted: Bool

    /// Parses `expression`; nil on anything malformed or out of range.
    public init?(_ expression: String) {
        let parts = expression.split(whereSeparator: { $0 == " " || $0 == "\t" })
            .map(String.init)
        guard parts.count == 5 else { return nil }
        guard let minuteSet = Self.parseField(parts[0], low: 0, high: 59),
            let hourSet = Self.parseField(parts[1], low: 0, high: 23),
            let domSet = Self.parseField(parts[2], low: 1, high: 31),
            let monthSet = Self.parseField(parts[3], low: 1, high: 12),
            let dowSet = Self.parseField(parts[4], low: 0, high: 7)
        else { return nil }
        // 7 and 0 both mean Sunday; normalize 7 into 0.
        var dowNormalized = dowSet
        if dowNormalized.contains(7) {
            dowNormalized.remove(7)
            dowNormalized.insert(0)
        }
        self.minute = minuteSet
        self.hour = hourSet
        self.dayOfMonth = domSet
        self.month = monthSet
        self.dayOfWeek = dowNormalized
        self.domRestricted = parts[2] != "*"
        self.dowRestricted = parts[4] != "*"
    }

    /// Parses one field into a set of ints in `[low, high]`.
    static func parseField(_ field: String, low: Int, high: Int) -> Set<Int>? {
        var result: Set<Int> = []
        for part in field.split(separator: ",") {
            let pieces = part.split(separator: "/", omittingEmptySubsequences: false)
            let step: Int
            let rangeText: Substring
            if pieces.count == 2 {
                rangeText = pieces[0].isEmpty ? "*" : pieces[0]
                guard let parsed = Int(pieces[1]), parsed >= 1 else { return nil }
                step = parsed
            } else if pieces.count == 1 {
                rangeText = pieces[0]
                step = 1
            } else {
                return nil
            }
            let bounds: (Int, Int)
            if rangeText == "*" {
                bounds = (low, high)
            } else if let single = Int(rangeText) {
                bounds = (single, single)
            } else {
                let dash = rangeText.split(separator: "-")
                guard dash.count == 2, let a = Int(dash[0]), let b = Int(dash[1]) else {
                    return nil
                }
                bounds = (a, b)
            }
            guard bounds.0 >= low, bounds.1 <= high, bounds.0 <= bounds.1 else { return nil }
            var value = bounds.0
            while value <= bounds.1 {
                result.insert(value)
                value += step
            }
        }
        return result.isEmpty ? nil : result
    }

    /// Whether `date` (evaluated in the LOCAL calendar, which is what a user
    /// reading "09:00" means) satisfies every field.
    public func matches(_ date: Date, calendar: Calendar = .current) -> Bool {
        let comps = calendar.dateComponents(
            [.minute, .hour, .day, .month, .weekday], from: date)
        guard let minuteValue = comps.minute, let hourValue = comps.hour,
            let dayValue = comps.day, let monthValue = comps.month,
            let weekdayValue = comps.weekday
        else { return false }
        guard minute.contains(minuteValue), hour.contains(hourValue), month.contains(monthValue)
        else { return false }
        // Calendar.weekday is 1-based with Sunday = 1; cron day-of-week is
        // 0-based with Sunday = 0. (weekday + 6) % 7 is that conversion.
        let cronWeekday = (weekdayValue + 6) % 7
        // Standard cron: when BOTH day fields are restricted, EITHER may
        // match; when only one is, it must.
        let domMatches = dayOfMonth.contains(dayValue)
        let dowMatches = dayOfWeek.contains(cronWeekday)
        switch (domRestricted, dowRestricted) {
        case (true, true): return domMatches || dowMatches
        case (true, false): return domMatches
        case (false, true): return dowMatches
        case (false, false): return true
        }
    }

    /// The next matching moment strictly after `after`, minute resolution,
    /// searched forward for up to 366 days. Computing the match per minute
    /// is at most ~527k cheap set lookups, and callers run this once per job
    /// state change rather than per tick.
    public func nextFire(after: Date, calendar: Calendar = .current) -> Date? {
        var cursor = calendar.date(
            bySetting: .second, value: 0, of: after) ?? after
        // Strictly after: if `after` lands exactly on a match minute, the
        // next fire is a minute past it.
        if matches(cursor, calendar: calendar) {
            cursor.addTimeInterval(60)
        } else {
            cursor.addTimeInterval(60 - Double(calendar.component(.second, from: cursor)))
        }
        let horizon = cursor.addingTimeInterval(366 * 24 * 3600)
        while cursor < horizon {
            if matches(cursor, calendar: calendar) { return cursor }
            cursor.addTimeInterval(60)
        }
        return nil
    }

    /// A human sentence for the schedule, the way `CronList`'s
    /// `humanSchedule` field reads. Falls back to the raw expression for a
    /// shape the simple recognizers do not name.
    public func describe() -> String {
        func single(_ set: Set<Int>) -> Int? { set.count == 1 ? set.first : nil }
        if minute.count == 60 && hour.count == 24 { return "Every minute" }
        // "Every N minutes" covers the evenly-stepped minute field with an
        // otherwise-wild expression.
        if hour.count == 24 {
            let minutes = minute.sorted()
            if minutes.count > 1, minutes == Array(stride(from: minutes[0], to: 60, by: minutes[1] - minutes[0])) {
                return "Every \(minutes[1] - minutes[0]) minutes"
            }
        }
        if let minuteValue = single(minute), hour.count == 24 {
            return "Hourly at :\(String(format: "%02d", minuteValue))"
        }
        if let minuteValue = single(minute), let hourValue = single(hour) {
            let time = String(format: "%02d:%02d", hourValue, minuteValue)
            if month.count == 12 && dayOfMonth.count >= 28 && dayOfWeek.count == 7 {
                return "Daily at \(time)"
            }
            if dayOfWeek.count == 1, let weekday = dayOfWeek.first {
                return "Weekly on \(weekdayName(weekday)) at \(time)"
            }
            if dayOfWeek.count == 5, dayOfWeek == Set([1, 2, 3, 4, 5]) {
                return "Weekdays at \(time)"
            }
        }
        return "Custom schedule"
    }

    private func weekdayName(_ value: Int) -> String {
        let names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return names[value % 7]
    }
}

/// One scheduled prompt: the cron expression, the prompt to submit, and the
/// chat it fires into.
public struct AppCronJob: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var cron: String
    public var prompt: String
    public var recurring: Bool
    public var enabled: Bool
    /// The chat the prompt submits into. Created from the chat the tool ran
    /// in; a job whose chat was deleted is skipped at fire time and its run
    /// history says so.
    public var chatID: UUID
    public var createdAt: Date
    /// Cached next fire; recomputed after every fire, toggle, and load.
    public var nextFireAt: Date?
    /// Last runs, newest first, capped at `runHistoryLimit`.
    public var recentRuns: [CronRunRecord]

    public struct CronRunRecord: Codable, Equatable, Sendable {
        public var date: Date
        /// "fired" or "missed" (target chat gone).
        public var status: String

        public init(date: Date = Date(), status: String) {
            self.date = date
            self.status = status
        }
    }

    public init(
        id: String = UUID().uuidString.prefix(8).lowercased().description,
        cron: String,
        prompt: String,
        recurring: Bool = true,
        enabled: Bool = true,
        chatID: UUID,
        createdAt: Date = Date(),
        nextFireAt: Date? = nil,
        recentRuns: [CronRunRecord] = []
    ) {
        self.id = id
        self.cron = cron
        self.prompt = prompt
        self.recurring = recurring
        self.enabled = enabled
        self.chatID = chatID
        self.createdAt = createdAt
        self.nextFireAt = nextFireAt
        self.recentRuns = recentRuns
    }

    static let runHistoryLimit = 10

    /// Tolerant decode (archive types here all decode leniently; a job row
    /// written by a newer build must not sink the whole file).
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        cron = try container.decodeIfPresent(String.self, forKey: .cron) ?? ""
        prompt = try container.decodeIfPresent(String.self, forKey: .prompt) ?? ""
        recurring = try container.decodeIfPresent(Bool.self, forKey: .recurring) ?? true
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        chatID = try container.decodeIfPresent(UUID.self, forKey: .chatID) ?? UUID()
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        nextFireAt = try container.decodeIfPresent(Date.self, forKey: .nextFireAt)
        recentRuns = try container.decodeIfPresent([CronRunRecord].self, forKey: .recentRuns) ?? []
    }
}

/// The persistent cron store behind the `CronCreate` / `CronList` /
/// `CronDelete` tools and the Scheduled Tasks pane.
///
/// The executor API is STATIC (like `TaskManager`'s) because tool dispatch
/// reaches it without an `AppModel`; firing goes back through the
/// `fireHandler` the app installs at startup, which is the one place that
/// can submit a prompt into a chat.
public final class CronScheduler: @unchecked Sendable {
    public static let shared = CronScheduler()

    private let lock = NSLock()
    private var jobs: [AppCronJob] = []
    /// Delivery suspends before `completeFire` advances or removes the job.
    /// Reserve it under the lock so overlapping polls cannot deliver it twice.
    private var inFlight: Set<String> = []

    /// Installed by `AppModel` at startup. Returns whether the prompt was
    /// actually delivered (a false return records a missed run). ASYNC
    /// because delivery crosses onto the main actor, where `AppModel` and
    /// the chat queue live.
    public var fireHandler: ((AppCronJob) async -> Bool)?

    /// A private directory isolates persisted jobs; a separate instance also
    /// isolates tests from the app's pollers and shared delivery handler.
    private let fileURL: URL

    /// The app uses `shared`; a test passes its own directory so its jobs
    /// cannot reach, or be reached by, anything else.
    public init(directory: URL = AppStorageRoot.directory) {
        self.fileURL = directory.appendingPathComponent("cron_jobs.json")
        load()
    }

    // MARK: - Store

    private func load() {
        guard let archive = AppJSONStore.load(
            [AppCronJob].self, from: fileURL, label: "cron jobs")
        else { return }
        jobs = archive
        // A job that fired before the app quit has a stale nextFireAt; a
        // resume computes it fresh rather than firing immediately.
        lock.lock()
        for index in jobs.indices {
            if jobs[index].enabled, let schedule = CronSchedule(jobs[index].cron) {
                jobs[index].nextFireAt = schedule.nextFire(after: Date())
            }
        }
        lock.unlock()
        save()
    }

    private func save() {
        AppJSONStore.save(jobs, to: fileURL, label: "Cron jobs")
    }

    // MARK: - Mutations (executor + pane surface)

    @discardableResult
    public func create(
        cron: String, prompt: String, recurring: Bool, chatID: UUID
    ) throws -> AppCronJob {
        guard let schedule = CronSchedule(cron) else {
            throw NSError(domain: "TurboSparkTool", code: 60, userInfo: [
                NSLocalizedDescriptionKey:
                    "Invalid cron expression '\(cron)': expected 5 fields "
                    + "(minute hour day month weekday), e.g. '*/5 * * * *'."
            ])
        }
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw NSError(domain: "TurboSparkTool", code: 61, userInfo: [
                NSLocalizedDescriptionKey: "Missing 'prompt': say what to run when the schedule fires."
            ])
        }
        let job = AppCronJob(
            cron: cron, prompt: trimmedPrompt, recurring: recurring, chatID: chatID,
            nextFireAt: schedule.nextFire(after: Date()))
        lock.lock()
        jobs.append(job)
        lock.unlock()
        save()
        return job
    }

    @discardableResult
    public func delete(id: String) -> Bool {
        lock.lock()
        let before = jobs.count
        jobs.removeAll { $0.id == id }
        let removed = jobs.count != before
        lock.unlock()
        if removed { save() }
        return removed
    }

    public func list() -> [AppCronJob] {
        lock.lock()
        defer { lock.unlock() }
        return jobs.sorted {
            ($0.nextFireAt ?? .distantFuture) < ($1.nextFireAt ?? .distantFuture)
        }
    }

    @discardableResult
    public func setEnabled(id: String, enabled: Bool) -> Bool {
        lock.lock()
        var changed = false
        for index in jobs.indices where jobs[index].id == id {
            jobs[index].enabled = enabled
            if enabled, let schedule = CronSchedule(jobs[index].cron) {
                jobs[index].nextFireAt = schedule.nextFire(after: Date())
            } else if !enabled {
                jobs[index].nextFireAt = nil
            }
            changed = true
        }
        lock.unlock()
        if changed { save() }
        return changed
    }

    /// One-shot timer (the `ScheduleWakeup` tool): fires once at `fireAt`
    /// regardless of any cron expression, then removes itself through the
    /// ordinary non-recurring fire path.
    @discardableResult
    public func createOneShot(fireAt: Date, prompt: String, chatID: UUID) -> AppCronJob {
        let job = AppCronJob(
            cron: "* * * * *", prompt: prompt, recurring: false, chatID: chatID,
            nextFireAt: fireAt)
        lock.lock()
        jobs.append(job)
        lock.unlock()
        save()
        return job
    }

    // MARK: - Firing

    /// Fires every due job. Called from the app's poll timer. The lock is
    /// only ever taken from synchronous helpers (`takeDueJobs`,
    /// `completeFire`); the awaiting delivery loop itself never holds it.
    public func fireDueJobs(now: Date = Date()) async {
        for job in takeDueJobs(now: now) {
            let delivered = await fireHandler?(job) ?? false
            completeFire(jobID: job.id, now: now, delivered: delivered)
        }
    }

    private func takeDueJobs(now: Date) -> [AppCronJob] {
        lock.lock()
        defer { lock.unlock() }
        var due: [AppCronJob] = []
        for job in jobs where job.enabled {
            if let fireAt = job.nextFireAt, fireAt <= now, !inFlight.contains(job.id) {
                inFlight.insert(job.id)
                due.append(job)
            }
        }
        return due
    }

    private func completeFire(jobID: String, now: Date, delivered: Bool) {
        recordRun(jobID: jobID, status: delivered ? "fired" : "missed")
        var remove = false
        lock.lock()
        inFlight.remove(jobID)
        if let index = jobs.firstIndex(where: { $0.id == jobID }) {
            if jobs[index].recurring {
                jobs[index].nextFireAt = CronSchedule(jobs[index].cron)?.nextFire(after: now)
            } else {
                remove = true
            }
        }
        lock.unlock()
        if remove { delete(id: jobID) }
        save()
    }

    private func recordRun(jobID: String, status: String) {
        lock.lock()
        if let index = jobs.firstIndex(where: { $0.id == jobID }) {
            jobs[index].recentRuns.insert(AppCronJob.CronRunRecord(status: status), at: 0)
            if jobs[index].recentRuns.count > AppCronJob.runHistoryLimit {
                jobs[index].recentRuns.removeLast(
                    jobs[index].recentRuns.count - AppCronJob.runHistoryLimit)
            }
        }
        lock.unlock()
    }

    // MARK: - Tool Executors (static, AppToolRegistry dispatch surface)
    //
    // `scheduler` defaults to `shared`, which is what dispatch passes. It is
    // a parameter so a test can hand in an isolated store rather than
    // leaving a job in the singleton's file for every later case and every
    // later run to trip over (see `fileURL`).

    public static func executeCreate(
        arguments: [String: String], chatID: UUID?, scheduler: CronScheduler = .shared
    ) throws -> String {
        guard let cron = arguments["cron"] ?? arguments["expression"] ?? arguments["schedule"] else {
            throw NSError(domain: "TurboSparkTool", code: 60, userInfo: [
                NSLocalizedDescriptionKey: "Missing 'cron' argument for CronCreate."
            ])
        }
        guard let prompt = arguments["prompt"] ?? arguments["instruction"] else {
            throw NSError(domain: "TurboSparkTool", code: 61, userInfo: [
                NSLocalizedDescriptionKey: "Missing 'prompt' argument for CronCreate."
            ])
        }
        let recurring = (arguments["recurring"].map { $0.lowercased() != "false" && $0 != "0" }) ?? true
        guard let chatID else {
            throw NSError(domain: "TurboSparkTool", code: 62, userInfo: [
                NSLocalizedDescriptionKey:
                    "CronCreate needs an active chat to deliver its prompt into."
            ])
        }
        let job = try scheduler.create(
            cron: cron, prompt: prompt, recurring: recurring, chatID: chatID)
        let human = CronSchedule(job.cron)?.describe() ?? job.cron
        let recurrence = job.recurring ? "recurring" : "one-shot"
        return "Cron job '\(job.id)' created (\(recurrence), \(human)). "
            + "It will submit into this chat; view or remove it in Settings > Scheduled Tasks."
    }

    public static func executeDelete(
        arguments: [String: String], scheduler: CronScheduler = .shared
    ) throws -> String {
        guard let id = arguments["id"] ?? arguments["jobId"] ?? arguments["job_id"] else {
            throw NSError(domain: "TurboSparkTool", code: 63, userInfo: [
                NSLocalizedDescriptionKey: "Missing 'id' argument for CronDelete."
            ])
        }
        guard scheduler.delete(id: id) else {
            throw NSError(domain: "TurboSparkTool", code: 63, userInfo: [
                NSLocalizedDescriptionKey: "No cron job with id '\(id)'."
            ])
        }
        return "Cron job '\(id)' deleted."
    }

    public static func executeList(
        arguments: [String: String], scheduler: CronScheduler = .shared
    ) -> String {
        let jobs = scheduler.list()
        if jobs.isEmpty {
            return "No scheduled cron jobs. Create one with CronCreate."
        }
        var lines = ["Scheduled cron jobs (\(jobs.count)):"]
        for job in jobs {
            let human = CronSchedule(job.cron)?.describe() ?? job.cron
            let next = job.nextFireAt.map {
                " next \($0.formatted(date: .omitted, time: .shortened))"
            } ?? (job.enabled ? "" : " (paused)")
            lines.append("- [\(job.id)] \(human) (cron: \(job.cron))\(next)")
            lines.append("  Prompt: \(String(job.prompt.prefix(120)))")
        }
        return lines.joined(separator: "\n")
    }

    /// One-shot wakeup (clamped to 60-3600 seconds like the schema says).
    /// `stop` has nothing to stop -- this engine holds no autonomous wakeup
    /// loop -- and says so rather than pretending.
    public static func executeScheduleWakeup(
        arguments: [String: String], chatID: UUID?, scheduler: CronScheduler = .shared
    ) throws -> String {
        if let stop = arguments["stop"], stop.lowercased() == "true" {
            return "No active wakeup loop to stop."
        }
        guard let chatID else {
            throw NSError(domain: "TurboSparkTool", code: 64, userInfo: [
                NSLocalizedDescriptionKey:
                    "ScheduleWakeup needs an active chat to deliver its prompt into."
            ])
        }
        let raw = arguments["delaySeconds"] ?? arguments["delay_seconds"] ?? ""
        guard let requested = Int(raw) else {
            throw NSError(domain: "TurboSparkTool", code: 64, userInfo: [
                NSLocalizedDescriptionKey:
                    "Missing or non-integer 'delaySeconds' for ScheduleWakeup (60 to 3600)."
            ])
        }
        let clamped = min(max(requested, 60), 3600)
        guard let prompt = arguments["prompt"], !prompt.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw NSError(domain: "TurboSparkTool", code: 64, userInfo: [
                NSLocalizedDescriptionKey:
                    "Missing 'prompt': say what to deliver when the wakeup fires."
            ])
        }
        let fireAt = Date().addingTimeInterval(TimeInterval(clamped))
        _ = scheduler.createOneShot(fireAt: fireAt, prompt: prompt, chatID: chatID)
        let clampedNote = clamped != requested ? " (clamped from \(requested)s)" : ""
        return "Wakeup scheduled for \(fireAt.formatted(date: .omitted, time: .shortened)) "
            + "in \(clamped)s\(clampedNote)."
    }
}
