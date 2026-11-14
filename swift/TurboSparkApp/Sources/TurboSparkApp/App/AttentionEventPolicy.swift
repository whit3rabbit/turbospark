import Foundation

/// OC-02: the four things that can make a chat need the user while the app is
/// in the background. Raw values are the persisted keys, so renaming a case
/// needs a migration.
enum AttentionEvent: String, CaseIterable, Identifiable, Codable, Sendable {
    case turnFinished
    case needsPermission
    case needsAnswer
    case error

    var id: String { rawValue }

    /// Which pending cue wins when two events meet on one chat. A blocking
    /// cue (the turn is parked on the user) is never replaced by a finish or
    /// an error that lands after it; equal ranks replace, so the newest
    /// blocking event wins.
    var rank: Int {
        switch self {
        case .needsPermission, .needsAnswer: return 2
        case .error: return 1
        case .turnFinished: return 0
        }
    }

    /// Catalog key of the toggle label in Settings.
    var settingsTitle: String {
        switch self {
        case .turnFinished: return "Turn finished"
        case .needsPermission: return "Needs approval"
        case .needsAnswer: return "Has a question"
        case .error: return "Error"
        }
    }
}

/// The named macOS system sounds the picker offers. `NSSound(named:)` finds
/// these in `/System/Library/Sounds`; nothing is bundled, and "none" is the
/// default so enabling an event is a silent banner until the user picks one.
enum AttentionSound: String, CaseIterable, Identifiable, Codable, Sendable {
    case none
    case glass = "Glass"
    case ping = "Ping"
    case pop = "Pop"
    case hero = "Hero"
    case basso = "Basso"

    var id: String { rawValue }

    /// The `NSSound` name, or nil for silence.
    var systemName: String? { self == .none ? nil : rawValue }

    /// Catalog key of the picker label.
    var settingsTitle: String { self == .none ? "No sound" : rawValue }

    /// Tolerant decode: an unknown stored value means silence.
    init(storedValue: String) {
        self = AttentionSound(rawValue: storedValue) ?? .none
    }
}

/// One event's persisted choice. Stored as strings so an unknown sound from a
/// newer build loads as silence instead of failing the whole settings file.
public struct AttentionEventPreference: Codable, Equatable, Sendable {
    public var enabled: Bool = false
    public var sound: String = "none"

    public init(enabled: Bool = false, sound: String = "none") {
        self.enabled = enabled
        self.sound = sound
    }

    var soundChoice: AttentionSound { AttentionSound(storedValue: sound) }
}

/// The persisted matrix. Absent entries read as off with no sound.
public struct AttentionPreferences: Codable, Equatable, Sendable {
    public var events: [String: AttentionEventPreference] = [:]

    public init(events: [String: AttentionEventPreference] = [:]) {
        self.events = events
    }

    subscript(event: AttentionEvent) -> AttentionEventPreference {
        get { events[event.rawValue] ?? AttentionEventPreference() }
        set { events[event.rawValue] = newValue }
    }

    var anyEnabled: Bool { AttentionEvent.allCases.contains { self[$0].enabled } }
}

/// What the policy decided for one event.
struct AttentionDecision: Equatable, Sendable {
    enum Outcome: Equatable, Sendable {
        /// The event's toggle is off.
        case disabled
        /// The user is looking at the app.
        case appActive
        /// A higher-priority cue is already outstanding for this chat.
        case outranked
        /// Post it and play `sound` (nil is silence).
        case deliver
    }

    let outcome: Outcome
    let sound: AttentionSound?

    var shouldPost: Bool { outcome == .deliver }
}

/// The delivery matrix, with no framework types in the decision path so it
/// runs headless: toggles, the focus rule, and one outstanding cue per chat.
///
/// Delivery rule (draft Goals): post only when the app is not active. A chat
/// that needs the user while the app is frontmost gets the in-window cards
/// and sidebar pill (ZP-02) instead. The review's refinement (also post for a
/// background chat inside an active app) was declined, see the draft's
/// implementation record.
///
/// Dedupe: at most one outstanding cue per chat. A newer event replaces it
/// unless it ranks lower (`AttentionEvent.rank`). Slots release when the app
/// becomes active again, when the cue is clicked, and when the chat is
/// deleted, which is one rule with three triggers.
struct AttentionEventPolicy {
    private(set) var slots: [UUID: AttentionEvent] = [:]
    private let isActive: () -> Bool

    init(isActive: @escaping () -> Bool) {
        self.isActive = isActive
    }

    /// Decides and, when delivering, claims the chat's slot.
    mutating func decide(
        _ event: AttentionEvent, chat: UUID, preferences: AttentionPreferences
    ) -> AttentionDecision {
        let preference = preferences[event]
        guard preference.enabled else {
            return AttentionDecision(outcome: .disabled, sound: nil)
        }
        guard !isActive() else {
            return AttentionDecision(outcome: .appActive, sound: nil)
        }
        if let pending = slots[chat], pending.rank > event.rank {
            return AttentionDecision(outcome: .outranked, sound: nil)
        }
        slots[chat] = event
        return AttentionDecision(outcome: .deliver, sound: preference.soundChoice)
    }

    /// Gives a slot back when the post that claimed it did not happen.
    mutating func release(chat: UUID) {
        slots[chat] = nil
    }

    mutating func releaseAll() {
        slots.removeAll()
    }
}
