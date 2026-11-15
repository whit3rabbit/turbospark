import Foundation

/// The single egress point for provider text that may reach a log, a toast,
/// a transcript row or the rollout log (ZP-03).
///
/// It layers two things: the EXACT key the caller holds (so a provider that
/// echoes a rejected key back in an error body cannot leak it, whatever its
/// shape), and `TrajectoryRedactor`'s pattern pass (Authorization and
/// x-api-key header values, `sk-` and `sk-ant-` shaped tokens of 16 or more
/// key characters, bearer tokens, URL credentials). Best effort by design for
/// the patterns; exact for the key.
enum RemoteProviderRedactor {
    /// Matches shorter than this are not treated as a key: replacing a
    /// two-letter "key" would shred ordinary text.
    static let minimumExactKeyLength = 6

    static func redact(_ text: String, keys: [String] = [], maxCharacters: Int = 2_000) -> String {
        var output = String(text.prefix(maxCharacters * 4))
        for key in keys where key.count >= minimumExactKeyLength {
            output = output.replacingOccurrences(of: key, with: TrajectoryRedactor.marker)
        }
        output = TrajectoryRedactor.redact(output)
        if output.count > maxCharacters {
            output = String(output.prefix(maxCharacters)) + "..."
        }
        return output
    }
}
