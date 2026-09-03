import Foundation

/// The two things `decodeIfPresent` does NOT tolerate (state#45).
///
/// `swift/CLAUDE.md` Gotcha 13's rule -- "add fields with `decodeIfPresent`
/// and a default" -- is about a key being ABSENT, and that is the only case
/// it covers. Two others are just as reachable in a file that survives across
/// releases, and both are total: the store writes the archive whole, so one
/// bad byte anywhere quarantines every chat or every project.
///
/// - An ENUM decoded through its own `Codable` conformance throws
///   `dataCorrupted` on a raw value this build does not know. A role, a
///   category, a status or an agent type written by a newer release, or a
///   case removed by this one, takes the file down. `decodeIfPresent` does
///   not help: the key is present, the value is the problem.
/// - An ARRAY decodes all-or-nothing. One malformed element throws for the
///   whole array, which throws for its container, which throws for the file.
public enum TolerantDecoding {}

extension KeyedDecodingContainer {
    /// Decodes a raw-representable value, falling back rather than throwing
    /// on a raw value this build does not know.
    ///
    /// The fallback is a decision at each call site and is deliberately not
    /// defaulted: a `ToolRiskLevel` this build cannot read should fall back
    /// to `.high` (it gates an approval card), while a message ROLE should
    /// fall back to `.assistant` (it is display text). Getting those the
    /// wrong way round is worse than throwing.
    func decodeTolerant<T: RawRepresentable>(
        _ type: T.Type, forKey key: Key, fallback: T
    ) -> T where T.RawValue: Decodable {
        guard let raw = try? decodeIfPresent(T.RawValue.self, forKey: key) else { return fallback }
        return T(rawValue: raw) ?? fallback
    }

    /// Decodes an array element by element, dropping the ones that fail.
    ///
    /// Returns `[]` for an absent key. A dropped element is a real loss and
    /// is the smaller one: the alternative is losing the file.
    ///
    /// **IT STILL THROWS WHEN THE KEY IS NOT AN ARRAY AT ALL, AND THAT LINE
    /// IS LOAD-BEARING.** A first version wrapped the whole decode in `try?`,
    /// which turned `{"chats": "this used to be an array"}` into an EMPTY
    /// archive that decoded cleanly -- so nothing was quarantined and the
    /// next atomic save overwrote an intact file, which is precisely the loss
    /// this whole cluster exists to prevent. `StoreDurabilityTests` caught it
    /// on the first run. Element-level tolerance is the goal; container-level
    /// tolerance is the bug.
    func decodeLossyArray<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> [T] {
        guard let raw = try decodeIfPresent([FailableDecodable<T>].self, forKey: key) else {
            return []
        }
        return raw.compactMap(\.value)
    }
}

/// One array element, decoded into `nil` rather than throwing.
struct FailableDecodable<T: Decodable>: Decodable {
    let value: T?

    init(from decoder: any Decoder) throws {
        value = try? T(from: decoder)
    }
}
