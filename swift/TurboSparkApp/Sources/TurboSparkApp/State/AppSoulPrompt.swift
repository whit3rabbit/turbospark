import Foundation

/// A saved SOUL.md entry in the app-wide soul library. The selected row
/// supplies the global SOUL section while SOUL is enabled. Detected external
/// files (Hermes, OpenClaw) are import sources that copy into ordinary rows;
/// they are never consumed on their own.
public struct AppSoulPrompt: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var name: String
    public var content: String

    public init(id: UUID = UUID(), name: String, content: String) {
        self.id = id
        self.name = name
        self.content = content
    }

    /// One malformed row must not discard a user's other saved souls.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = container.decodeLenient(UUID.self, forKey: .id, fallback: UUID())
        name = container.decodeLenient(String.self, forKey: .name, fallback: "SOUL")
        content = container.decodeLenient(String.self, forKey: .content, fallback: "")
    }
}
