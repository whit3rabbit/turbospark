import Foundation

/// The browser category extension expected by the shared diagnostics store.
/// Keep this mapping local until AppLogStore lands in the checkout.
enum BrowserAutomationLogCategory: String, Codable, CaseIterable, Sendable {
    case browser
}

enum BrowserAutomationLogLevel: String, Codable, Sendable {
    case info
}

enum BrowserAutomationOutcome: String, Codable, Sendable {
    case completed
    case denied
    case failed
    case cancelled
    case timedOut
    case unsupported
}

/// Closed browser event shape. The category is derived from the event type, matching
/// AppLogStore's documented category-by-event contract.
struct BrowserAutomationDiagnosticEvent: Codable, Equatable, Sendable {
    let actionType: BrowserControlCommandKind
    let canonicalOrigin: String
    let outcome: BrowserAutomationOutcome
    let durationMilliseconds: UInt32

    private enum CodingKeys: String, CodingKey {
        case actionType
        case canonicalOrigin
        case outcome
        case durationMilliseconds
    }

    var category: BrowserAutomationLogCategory { .browser }

    init(
        actionType: BrowserControlCommandKind,
        origin: BrowserOrigin,
        outcome: BrowserAutomationOutcome,
        durationMilliseconds: UInt32
    ) {
        self.actionType = actionType
        self.canonicalOrigin = origin.canonicalString
        self.outcome = outcome
        self.durationMilliseconds = durationMilliseconds
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let canonicalOrigin = try container.decode(String.self, forKey: .canonicalOrigin)
        guard let origin = BrowserOrigin(origin: canonicalOrigin),
              origin.canonicalString == canonicalOrigin
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .canonicalOrigin,
                in: container,
                debugDescription: "Expected a canonical HTTP(S) origin."
            )
        }

        self.actionType = try container.decode(BrowserControlCommandKind.self, forKey: .actionType)
        self.canonicalOrigin = origin.canonicalString
        self.outcome = try container.decode(BrowserAutomationOutcome.self, forKey: .outcome)
        self.durationMilliseconds = try container.decode(
            UInt32.self,
            forKey: .durationMilliseconds
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(actionType, forKey: .actionType)
        try container.encode(canonicalOrigin, forKey: .canonicalOrigin)
        try container.encode(outcome, forKey: .outcome)
        try container.encode(durationMilliseconds, forKey: .durationMilliseconds)
    }
}

/// Local adapter boundary that mirrors AppLogStore.record(_:level:) without duplicating
/// its persistence implementation or binding this feature to an unlanded API.
protocol BrowserAutomationLogSink {
    func record(
        _ event: BrowserAutomationDiagnosticEvent,
        level: BrowserAutomationLogLevel
    ) throws
}

/// Emits one allowlisted diagnostic event for each completed browser transition.
final class BrowserAutomationLogger {
    private let store: any BrowserAutomationLogSink
    private let failureLock = NSLock()
    private var failures = 0

    var writeFailureCount: Int {
        failureLock.lock()
        defer { failureLock.unlock() }
        return failures
    }

    init(store: any BrowserAutomationLogSink) {
        self.store = store
    }

    func recordTransition(
        action: BrowserControlCommandKind,
        origin: BrowserOrigin,
        outcome: BrowserAutomationOutcome,
        durationMilliseconds: UInt32
    ) {
        let event = BrowserAutomationDiagnosticEvent(
            actionType: action,
            origin: origin,
            outcome: outcome,
            durationMilliseconds: durationMilliseconds
        )

        do {
            try store.record(event, level: .info)
        } catch {
            failureLock.lock()
            failures += 1
            failureLock.unlock()
        }
    }
}
