import Foundation

/// Time-of-day category for contextual greeting classification.
public enum GreetingTimeSlot: String, Codable, CaseIterable, Sendable {
    case earlyMorning
    case morning
    case afternoon
    case evening
    case night
    case anytime
}

/// A localized greeting item representing a distinct greeting phrasing.
public struct Greeting: Identifiable, Codable, Sendable, Equatable {
    public let id: String
    public let category: GreetingTimeSlot
    public let translations: [String: String]

    public init(id: String, category: GreetingTimeSlot, translations: [String: String]) {
        self.id = id
        self.category = category
        self.translations = translations
    }

    /// Resolves the greeting text in the requested AppLanguage with fallback to English.
    public func text(for language: AppLanguage) -> String {
        switch language {
        case .system:
            let code = Locale.current.language.languageCode?.identifier ?? "en"
            if let matched = translations[code] {
                return matched
            }
            return translations["en"] ?? id
        default:
            if let matched = translations[language.rawValue] {
                return matched
            }
            return translations["en"] ?? id
        }
    }
}

/// Container structure matching the greetings.json schema.
private struct GreetingsEnvelope: Codable {
    let greetings: [Greeting]
}

/// Manages the catalog of time-based and general greetings with multi-language resolution.
public final class GreetingProvider: @unchecked Sendable {
    public static let shared = GreetingProvider()

    /// The entire loaded greeting catalog.
    public let allGreetings: [Greeting]

    public init() {
        self.allGreetings = Self.loadCatalog()
    }

    /// Determines the time slot for a given hour (0 to 23).
    public static func slot(forHour hour: Int) -> GreetingTimeSlot {
        switch hour {
        case 4..<7:
            return .earlyMorning
        case 7..<12:
            return .morning
        case 12..<17:
            return .afternoon
        case 17..<22:
            return .evening
        default:
            return .night
        }
    }

    /// Determines the time slot for a given date.
    public static func slot(for date: Date = Date(), calendar: Calendar = .current) -> GreetingTimeSlot {
        let hour = calendar.component(.hour, from: date)
        return slot(forHour: hour)
    }

    /// Returns all greetings eligible for the specified time slot.
    /// A greeting is eligible if it matches the current slot or is marked .anytime.
    public func greetings(forSlot slot: GreetingTimeSlot) -> [Greeting] {
        allGreetings.filter { greeting in
            greeting.category == slot || greeting.category == .anytime
        }
    }

    /// Returns all greetings eligible for the current time of day.
    public func availableGreetings(for date: Date = Date(), calendar: Calendar = .current) -> [Greeting] {
        let current = Self.slot(for: date, calendar: calendar)
        return greetings(forSlot: current)
    }

    /// Returns a randomly chosen greeting appropriate for the current time.
    public func randomGreeting(
        for date: Date = Date(),
        language: AppLanguage = .system,
        calendar: Calendar = .current
    ) -> String {
        let eligible = availableGreetings(for: date, calendar: calendar)
        guard let picked = eligible.randomElement() else {
            return "Welcome!"
        }
        return picked.text(for: language)
    }

    /// Returns a deterministic greeting by index from the eligible pool.
    public func greeting(
        at index: Int,
        for date: Date = Date(),
        language: AppLanguage = .system,
        calendar: Calendar = .current
    ) -> String {
        let eligible = availableGreetings(for: date, calendar: calendar)
        guard !eligible.isEmpty else {
            return "Welcome!"
        }
        let safeIndex = abs(index) % eligible.count
        return eligible[safeIndex].text(for: language)
    }

    /// Resolves a specific greeting by ID.
    public func greeting(byId id: String) -> Greeting? {
        allGreetings.first { $0.id == id }
    }

    /// Loads the bundled greetings catalog or falls back to static defaults.
    private static func loadCatalog() -> [Greeting] {
        if let url = Bundle.module.url(forResource: "greetings", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let envelope = try? JSONDecoder().decode(GreetingsEnvelope.self, from: data),
           !envelope.greetings.isEmpty {
            return envelope.greetings
        }

        // Static fallback if resource bundle is not available
        return [
            Greeting(id: "morning_fallback", category: .morning, translations: ["en": "Good morning"]),
            Greeting(id: "afternoon_fallback", category: .afternoon, translations: ["en": "Good afternoon"]),
            Greeting(id: "evening_fallback", category: .evening, translations: ["en": "Good evening"]),
            Greeting(id: "night_fallback", category: .night, translations: ["en": "Quiet hours, deep focus"]),
            Greeting(id: "anytime_fallback", category: .anytime, translations: ["en": "Welcome back!"])
        ]
    }
}
