import Foundation

/// Enforced today: `maximumOutputCharacters`, `maximumImageBytes` and
/// `maximumRequestBytes`. RESERVED, NOT ENFORCED: `defaultTimeout`,
/// `maximumTimeout` and `footprintGrowthBudgetBytes` are carried over the
/// wire but nothing reads them. The only deadline in force is the fixed
/// 30 s top-level-await settlement wait
/// (`REPLWorkerContext.defaultSettlementDeadline`), and no supervisor kills
/// a call for time or memory. Do not rely on them until a caller reads them.
struct REPLLimits: Sendable, Equatable, Codable {
    /// Reserved, not enforced (see the type comment).
    var defaultTimeout: TimeInterval
    /// Reserved, not enforced (see the type comment).
    var maximumTimeout: TimeInterval
    var maximumOutputCharacters: Int
    var maximumImageBytes: Int
    var maximumRequestBytes: Int
    /// Reserved, not enforced (see the type comment).
    var footprintGrowthBudgetBytes: Int

    init(
        defaultTimeout: TimeInterval = 30,
        maximumTimeout: TimeInterval = 600,
        maximumOutputCharacters: Int = 30_000,
        maximumImageBytes: Int = 20 * 1_024 * 1_024,
        maximumRequestBytes: Int = 1_024 * 1_024,
        footprintGrowthBudgetBytes: Int = 512 * 1_024 * 1_024
    ) {
        self.defaultTimeout = defaultTimeout
        self.maximumTimeout = maximumTimeout
        self.maximumOutputCharacters = maximumOutputCharacters
        self.maximumImageBytes = maximumImageBytes
        self.maximumRequestBytes = maximumRequestBytes
        self.footprintGrowthBudgetBytes = footprintGrowthBudgetBytes
    }
}

struct REPLSessionConfiguration: Sendable, Equatable, Codable {
    var artifactDirectory: URL
}

struct REPLEmittedImage: Sendable, Equatable, Codable {
    var fileURL: URL
    var label: String?
}

struct REPLTextOutputEvent: Sendable, Equatable, Codable {
    enum Level: String, Sendable, Codable {
        case log
        case info
        case debug
        case warn
        case error
    }

    var level: Level
    var text: String
}

struct REPLCallResult: Sendable, Codable {
    enum Status: String, Sendable, Codable {
        case completed
        case failed
        case parseError
        case unsupportedSyntax
        /// Produced only when a top-level-await program fails to settle
        /// within the fixed settlement deadline. Not produced for a
        /// synchronous infinite loop or a per-call `REPLLimits` timeout.
        case timedOut
        /// Reserved, not produced yet. No cancellation path exists.
        case cancelled
        /// Reserved, not produced yet. `footprintGrowthBudgetBytes` is not enforced.
        case memoryLimit
        /// Reserved, not produced yet. The supervisor serializes callers
        /// through a FIFO mailbox instead of refusing a concurrent call.
        case busy
    }

    var status: Status
    var outputEvents: [REPLTextOutputEvent]
    var consoleText: String
    var errorText: String?
    var completionText: String?
    var images: [REPLEmittedImage]
    var truncated: Bool
    var sessionCreated: Bool
    var sessionReset: Bool
}
