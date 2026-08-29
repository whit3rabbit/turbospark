import Foundation

public struct AppPromptPreset: Identifiable, Decodable, Sendable, Equatable {
    public let id: String
    public let title: String
    public let prompt: String

    public init(id: String, title: String, prompt: String) {
        self.id = id
        self.title = title
        self.prompt = prompt
    }

    public static var all: [AppPromptPreset] {
        if let url = Bundle.module.url(forResource: "app-prompts", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let presets = try? JSONDecoder().decode([AppPromptPreset].self, from: data) {
            return presets
        }
        return defaultPresets
    }

    public static var primary: [AppPromptPreset] {
        Array(all.prefix(3))
    }

    public static var secondary: [AppPromptPreset] {
        Array(all.dropFirst(3))
    }

    public static let defaultPresets: [AppPromptPreset] = [
        AppPromptPreset(
            id: "paris",
            title: "Paris",
            prompt: "Plan a memorable first trip to Paris for a curious traveller. Give a compact three-day itinerary with morning, afternoon, and evening ideas; explain the easiest ways to get around; suggest the best seasons and two practical ways to control costs. Use clear headings, distinguish timeless advice from details that should be checked before travel, and keep the whole guide under 450 words."
        ),
        AppPromptPreset(
            id: "fibonacci",
            title: "Write Fibonacci in Python",
            prompt: "Write a complete, executable Python 3 example that defines `fibonacci(n: int) -> list[int]` and returns the first n Fibonacci numbers starting with 0, 1. Reject negative n with `ValueError`, return `[]` for n = 0, use iteration rather than recursion, and include exactly three assertions covering n = 0, n = 1, and n = 7. After the code block, use one plain-text sentence without LaTeX notation to state the time complexity and distinguish output space from extra working space. Do not use external packages."
        ),
        AppPromptPreset(
            id: "fieldfare",
            title: "Meet the fieldfare",
            prompt: "What makes the fieldfare's winter migration to Britain so remarkable? Explain where the birds come from, what they eat, and why they gather in flocks. Focus only on these facts, keep it accurate, and stay under 250 words."
        )
    ]
}
