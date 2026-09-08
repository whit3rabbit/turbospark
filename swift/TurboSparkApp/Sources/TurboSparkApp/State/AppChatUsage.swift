import Foundation

/// One day of token usage recorded against a chat, keyed "yyyy-MM-dd" in
/// `AppChatUsageLedger.byDay`.
///
/// Tolerant decode for the reason every archive type here decodes leniently
/// (`swift/CLAUDE.md` Gotcha 13): a ledger written by a later build must not
/// take the chat row down with it.
public struct AppChatUsageDay: Codable, Equatable, Sendable {
    /// Prompt (prefill) tokens across that day's turns.
    public var promptTokens: Int
    /// Generated (decode) tokens across that day's turns.
    public var outputTokens: Int
    /// How many generate calls were recorded.
    public var turns: Int

    public init(promptTokens: Int = 0, outputTokens: Int = 0, turns: Int = 0) {
        self.promptTokens = promptTokens
        self.outputTokens = outputTokens
        self.turns = turns
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        promptTokens = try container.decodeIfPresent(Int.self, forKey: .promptTokens) ?? 0
        outputTokens = try container.decodeIfPresent(Int.self, forKey: .outputTokens) ?? 0
        turns = try container.decodeIfPresent(Int.self, forKey: .turns) ?? 0
    }
}

/// The cumulative token ledger one chat carries on its row.
///
/// Recorded per generate call from the engine's own `GenerationResult`
/// counts, so the usage dashboard and `/stats` report what the model was
/// actually charged rather than a character estimate. Ghost chats never get
/// one: the ghost invariant asserts a ghost row carries no conversation
/// content, and a usage ledger describing a conversation nobody may see on
/// disk is exactly that.
public struct AppChatUsageLedger: Codable, Equatable, Sendable {
    public var totalPromptTokens: Int
    public var totalOutputTokens: Int
    public var turns: Int
    /// Day-bucketed usage, keyed "yyyy-MM-dd" (UTC-day, `dayKey`'s rule).
    public var byDay: [String: AppChatUsageDay]
    public var lastUpdatedAt: Date

    public init(
        totalPromptTokens: Int = 0,
        totalOutputTokens: Int = 0,
        turns: Int = 0,
        byDay: [String: AppChatUsageDay] = [:],
        lastUpdatedAt: Date = Date()
    ) {
        self.totalPromptTokens = totalPromptTokens
        self.totalOutputTokens = totalOutputTokens
        self.turns = turns
        self.byDay = byDay
        self.lastUpdatedAt = lastUpdatedAt
    }

    /// UTC day key. Deliberately not local-time: a ledger that re-buckets
    /// yesterday's tokens because the user flew west reads as a spike, and
    /// the day boundaries only need to be stable, not local.
    public static func dayKey(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: date)
    }

    /// Records one generate call. A non-finite or negative count is treated
    /// as "nothing recorded" rather than trusted: the engine should never
    /// produce one, and a ledger is a terrible place to find out otherwise.
    public mutating func record(
        promptTokens: Int, outputTokens: Int, on date: Date = Date()
    ) {
        let prompt = max(0, promptTokens)
        let output = max(0, outputTokens)
        totalPromptTokens += prompt
        totalOutputTokens += output
        turns += 1
        let key = Self.dayKey(date)
        var day = byDay[key] ?? AppChatUsageDay()
        day.promptTokens += prompt
        day.outputTokens += output
        day.turns += 1
        byDay[key] = day
        lastUpdatedAt = date
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        totalPromptTokens = try container.decodeIfPresent(Int.self, forKey: .totalPromptTokens) ?? 0
        totalOutputTokens = try container.decodeIfPresent(Int.self, forKey: .totalOutputTokens) ?? 0
        turns = try container.decodeIfPresent(Int.self, forKey: .turns) ?? 0
        byDay = try container.decodeIfPresent([String: AppChatUsageDay].self, forKey: .byDay) ?? [:]
        lastUpdatedAt = try container.decodeIfPresent(Date.self, forKey: .lastUpdatedAt) ?? Date()
    }
}

/// A usage total over a set of chats, what the dashboard's headline and
/// charts render from.
public struct AppChatUsageTotals: Equatable, Sendable {
    public var promptTokens: Int
    public var outputTokens: Int
    public var turns: Int
    /// Day-keyed totals across the same chats, dense in the keys the
    /// ledgers actually carry (a day with no usage is absent, not zero).
    public var byDay: [String: AppChatUsageDay]

    public init(
        promptTokens: Int = 0, outputTokens: Int = 0, turns: Int = 0,
        byDay: [String: AppChatUsageDay] = [:]
    ) {
        self.promptTokens = promptTokens
        self.outputTokens = outputTokens
        self.turns = turns
        self.byDay = byDay
    }

    /// Sums the ledger of every chat given. Ghost rows are skipped HERE
    /// rather than left to callers: a ledger on a ghost row is sealed-
    /// conversation metadata by another name, and a caller that forgets the
    /// filter would publish it. Rows without a ledger (chats that predate
    /// recording) contribute nothing; the totals report recorded usage and
    /// never back-fill an estimate.
    public init(chats: [AppChat]) {
        self.init()
        for chat in chats where !chat.isGhost {
            guard let ledger = chat.usage else { continue }
            promptTokens += ledger.totalPromptTokens
            outputTokens += ledger.totalOutputTokens
            turns += ledger.turns
            for (key, day) in ledger.byDay {
                var total = byDay[key] ?? AppChatUsageDay()
                total.promptTokens += day.promptTokens
                total.outputTokens += day.outputTokens
                total.turns += day.turns
                byDay[key] = total
            }
        }
    }

    /// Day totals for the `days` ending at `today` (UTC), oldest first.
    /// Absent days read as zero so a chart line does not skip gaps.
    public static func utcDayKey(_ date: Date) -> String { AppChatUsageLedger.dayKey(date) }

    public func dailySeries(endingAt today: Date, days: Int) -> [(key: String, day: AppChatUsageDay)] {
        // Stepped in UTC so consecutive entries are consecutive UTC days
        // even across a local DST boundary; the bucket keys are UTC too.
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .current
        var series: [(key: String, day: AppChatUsageDay)] = []
        series.reserveCapacity(days)
        for offset in stride(from: -(days - 1), through: 0, by: 1) {
            guard let date = calendar.date(byAdding: .day, value: offset, to: today) else { continue }
            let key = AppChatUsageLedger.dayKey(date)
            series.append((key, byDay[key] ?? AppChatUsageDay()))
        }
        return series
    }
}
