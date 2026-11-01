import Foundation

/// Global computer-use preferences stored under `computerUse` in settings.json.
/// Off by default: turning it on is what lets the helper start at all.
public struct ComputerUseSettings: Codable, Equatable, Sendable {
    public var enabled: Bool

    public init(enabled: Bool = false) {
        self.enabled = enabled
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.enabled = container.decodeLenient(Bool.self, forKey: .enabled, fallback: false)
    }
}
