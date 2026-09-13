import Foundation

/// A short, app-wide response-style instruction the user may select.
///
/// The built-ins are ordinary rows rather than protected code-owned presets:
/// people may remove any of them, and a settings file preserves that choice.
/// Their IDs are fixed so the selected row survives a settings round-trip.
public struct AppPersonality: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var name: String
    public var instructions: String

    public init(id: UUID = UUID(), name: String, instructions: String) {
        self.id = id
        self.name = name
        self.instructions = instructions
    }

    /// Missing or malformed fields cost this row's values, not the complete
    /// settings file. The containing array keeps the same element-level
    /// tolerance as the other user-maintained preset lists.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = container.decodeLenient(UUID.self, forKey: .id, fallback: UUID())
        name = container.decodeLenient(String.self, forKey: .name, fallback: "Personality")
        instructions = container.decodeLenient(String.self, forKey: .instructions, fallback: "")
    }

    /// Compact starting points distilled from the supplied examples. Keeping
    /// them short matters on the small local context windows this app targets.
    public static let builtIns: [AppPersonality] = [
        AppPersonality(
            id: UUID(uuidString: "70DBCB8E-A590-4388-9CF2-319A49C17F78")!,
            name: "Formal",
            instructions: "Be precise, professional, and structured. Use relevant domain terms. Do not critique spelling."),
        AppPersonality(
            id: UUID(uuidString: "90533D20-5E9C-4A5D-94A2-89B8B7D6141B")!,
            name: "Friendly",
            instructions: "Be warm, curious, and conversational. Match the user's tone. Be helpful without flattery."),
        AppPersonality(
            id: UUID(uuidString: "B54CE14D-021B-4981-9997-C3B63100D372")!,
            name: "Coach",
            instructions: "Be direct and constructive. Give practical advice, correct mistakes plainly, and encourage progress."),
        AppPersonality(
            id: UUID(uuidString: "D77555E3-F862-4F0D-98CF-33C48D5E4C69")!,
            name: "Creative",
            instructions: "Be playful and imaginative when appropriate. Use fresh language and light humor. Avoid cliches."),
        AppPersonality(
            id: UUID(uuidString: "E1DFE3BA-A207-4DF3-B412-73E05C7FE2DC")!,
            name: "Concise",
            instructions: "Be concise, clear, and complete. Skip small talk, filler, opinions, and unsolicited commentary."),
        AppPersonality(
            id: UUID(uuidString: "FA8F1A31-B9AF-40F9-9993-4C78D5D2D2B5")!,
            name: "Dry Humor",
            instructions: "Be dry, witty, and helpful. Use gentle sarcasm for low-stakes topics; be kind on sensitive ones.")
    ]
}
