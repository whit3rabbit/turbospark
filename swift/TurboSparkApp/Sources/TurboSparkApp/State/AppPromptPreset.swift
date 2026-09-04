import Foundation

/// Canned prompt preset used for quick model verification and sample generations in the empty state.
public struct AppPromptPreset: Identifiable, Decodable, Sendable, Equatable {
    /// Unique preset identifier.
    public let id: String
    /// Short display title shown in prompt pills.
    public let title: String
    /// Full prompt text injected into the composer.
    public let prompt: String

    public init(id: String, title: String, prompt: String) {
        self.id = id
        self.title = title
        self.prompt = prompt
    }

    /// The presets, read from the bundle ONCE (state#110).
    ///
    /// **`var` MEANT A FILE READ AND A JSON DECODE PER ACCESS**, and the
    /// accessors below read it twice more; `primary` and `secondary` are
    /// called from SwiftUI bodies, which run per keystroke of the composer
    /// draft. `let` is evaluated lazily and once.
    ///
    /// **AND AN EMPTY ARRAY IS NOT A LIST OF PRESETS.** A `[]` in the bundled
    /// JSON decodes SUCCESSFULLY, so it took the first arm and the empty
    /// state rendered no quick actions at all -- indistinguishable from a
    /// missing resource, which is the case the fallback exists for.
    public static let all: [AppPromptPreset] = {
        if let url = Bundle.module.url(forResource: "app-prompts", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let presets = try? JSONDecoder().decode([AppPromptPreset].self, from: data),
            !presets.isEmpty
        {
            return presets
        }
        return defaultPresets
    }()

    /// Primary quick-action presets shown directly above the empty composer.
    public static var primary: [AppPromptPreset] {
        Array(all.prefix(3))
    }

    /// Additional presets shown in extended menus.
    public static var secondary: [AppPromptPreset] {
        Array(all.dropFirst(3))
    }

    /// Built-in fallback presets covering travel planning, code generation, and factual explanation.
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
