import Foundation

/// Makes a fresh server API key: `sk-` plus a random UUID, lowercased.
///
/// 122 bits of randomness is sized for what this server protects -- a
/// loopback socket or a tailnet share -- not for a public endpoint, and it
/// is a key THIS server checks, never a provider credential. Both panes
/// that edit the key read the same engine (`crates/server/src/auth.rs`),
/// so one shape serves both.
enum ServerAPIKeyGenerator {
    static func generate() -> String {
        "sk-" + UUID().uuidString.lowercased()
    }
}
