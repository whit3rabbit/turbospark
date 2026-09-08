import Foundation

/// Session-scoped state for Agent mode (`swift/docs/SWIFT_AGENT_MODE.md`):
/// the two fallback counters and the suspension flag.
///
/// **QWEN CODE'S COUNTERS, ONE ROLE EACH.** Consecutive POLICY blocks catch
/// an agent retrying variants of a forbidden action -- after three, the
/// next eligible call asks a human instead of a classifier that keeps
/// saying no. Consecutive UNAVAILABLE results mean the classifier is
/// broken, not disapproving -- after two, later calls skip it entirely so
/// an outage does not add its latency to every ask. An allow verdict or a
/// manual approval resets both; a manual rejection preserves them; selecting
/// Agent mode resets them. Nothing here persists: a launch is a fresh
/// session, and the mode itself lives in project settings.
///
/// Keyed by chat ID, the same session key `SessionApprovalStore` uses, so a
/// card answered in one chat cannot reset another chat's counters.
public actor AgentModeGate {
    public static let shared = AgentModeGate()

    /// Policy blocks in a row before the next eligible call goes to manual
    /// approval.
    public static let maxConsecutiveBlocks = 3
    /// Classifier failures in a row before later calls skip the classifier.
    public static let maxConsecutiveUnavailable = 2

    private var consecutiveBlocks: [String: Int] = [:]
    private var consecutiveUnavailable: [String: Int] = [:]
    /// Sessions where the user chose "approve and suspend Agent" on a
    /// fallback card. Suspended sessions ask for everything, exactly as
    /// `.auto` would; the suspension dies with the session.
    private var suspendedSessions: Set<String> = []

    public init() {}

    // MARK: - Thresholds

    /// True when the classifier must be skipped for this session and the
    /// call routed straight to the manual card.
    public func shouldSkipClassifier(sessionID: String) -> Bool {
        let blocks = consecutiveBlocks[sessionID] ?? 0
        let unavailable = consecutiveUnavailable[sessionID] ?? 0
        return blocks >= Self.maxConsecutiveBlocks
            || unavailable >= Self.maxConsecutiveUnavailable
    }

    /// Why the classifier is being skipped, for the card's notice.
    public func skipReason(sessionID: String) -> String? {
        let blocks = consecutiveBlocks[sessionID] ?? 0
        let unavailable = consecutiveUnavailable[sessionID] ?? 0
        if blocks >= Self.maxConsecutiveBlocks {
            return "Agent mode has blocked \(blocks) calls in a row, so this one asks you directly."
        }
        if unavailable >= Self.maxConsecutiveUnavailable {
            return "The classifier has been unavailable \(unavailable) times in a row, so this one asks you directly."
        }
        return nil
    }

    public func isSuspended(sessionID: String) -> Bool {
        suspendedSessions.contains(sessionID)
    }

    // MARK: - Recording

    /// An allow verdict -- the classifier's or the user's -- breaks both
    /// streaks.
    public func recordAllow(sessionID: String) {
        consecutiveBlocks[sessionID] = 0
        consecutiveUnavailable[sessionID] = 0
    }

    /// Returns the new consecutive-block count.
    @discardableResult
    public func recordBlock(sessionID: String) -> Int {
        let next = (consecutiveBlocks[sessionID] ?? 0) + 1
        consecutiveBlocks[sessionID] = next
        return next
    }

    /// Returns the new consecutive-unavailable count.
    @discardableResult
    public func recordUnavailable(sessionID: String) -> Int {
        let next = (consecutiveUnavailable[sessionID] ?? 0) + 1
        consecutiveUnavailable[sessionID] = next
        return next
    }

    public func suspend(sessionID: String) {
        suspendedSessions.insert(sessionID)
        // A suspended session behaves like .auto for the rest of the
        // session; stale counters would only confuse the notice text if
        // Agent mode is re-selected later without a mode change to reset.
        consecutiveBlocks[sessionID] = 0
        consecutiveUnavailable[sessionID] = 0
    }

    /// Fresh start: called when a session selects Agent mode.
    public func reset(sessionID: String) {
        consecutiveBlocks[sessionID] = 0
        consecutiveUnavailable[sessionID] = 0
        suspendedSessions.remove(sessionID)
    }

    /// Drops a chat's state entirely (chat deleted or reset).
    public func clear(sessionID: String) {
        consecutiveBlocks.removeValue(forKey: sessionID)
        consecutiveUnavailable.removeValue(forKey: sessionID)
        suspendedSessions.remove(sessionID)
    }
}
