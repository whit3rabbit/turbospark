import Foundation

enum StreamRecoveryEventPresentation: Equatable, Sendable {
    case continued(attempt: Int, tokensSoFar: Int)
    case partialPreserved(rows: Int)
    case tailDiscarded(reason: String?)
    case anchorRecorded(retainedMessages: Int, retriesUsed: Int)
    case retrySucceeded(attempt: Int)
    case retryConflict
    case retryExhausted(limit: Int)

    init(event: StreamRecoveryEvent) {
        switch event {
        case .continued(let attempt, let tokensSoFar):
            self = .continued(attempt: attempt, tokensSoFar: tokensSoFar)
        case .partialPreserved(let rows):
            self = .partialPreserved(rows: rows)
        case .tailDiscarded(let description):
            self = .tailDiscarded(reason: Self.singleLineReason(description))
        case .anchorRecorded(let anchor):
            self = .anchorRecorded(
                retainedMessages: anchor.retainedMessageCount,
                retriesUsed: anchor.retriesUsed)
        case .retrySucceeded(let attempt):
            self = .retrySucceeded(attempt: attempt)
        case .retryConflict:
            self = .retryConflict
        case .retryExhausted:
            self = .retryExhausted(limit: AppStreamRecovery.maxRecoveryRetries)
        }
    }

    var isTerminal: Bool {
        switch self {
        case .retryConflict, .retryExhausted:
            true
        case .continued, .partialPreserved, .tailDiscarded, .anchorRecorded, .retrySucceeded:
            false
        }
    }

    var accessibilityIdentifier: String {
        switch self {
        case .continued:
            "stream-recovery-continued"
        case .partialPreserved:
            "stream-recovery-partial-preserved"
        case .tailDiscarded:
            "stream-recovery-tail-discarded"
        case .anchorRecorded:
            "stream-recovery-anchor-recorded"
        case .retrySucceeded:
            "stream-recovery-retry-succeeded"
        case .retryConflict:
            "stream-recovery-retry-conflict"
        case .retryExhausted:
            "stream-recovery-retry-exhausted"
        }
    }

    var symbolName: String {
        switch self {
        case .continued:
            "arrow.clockwise"
        case .partialPreserved:
            "doc.text"
        case .tailDiscarded:
            "scissors"
        case .anchorRecorded:
            "bookmark"
        case .retryConflict, .retryExhausted:
            "exclamationmark.triangle.fill"
        case .retrySucceeded:
            "checkmark.circle"
        }
    }

    private static func singleLineReason(_ value: String) -> String? {
        let words = value.split(whereSeparator: \.isWhitespace)
        let reason = words.joined(separator: " ")
        guard !reason.isEmpty else { return nil }
        return String(reason.prefix(160))
    }
}
