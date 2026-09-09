import Foundation

/// Parses the pinned-port field, separately from the view so the failure
/// arms can be tested.
///
/// `UInt16(text) ?? 0` was the whole parser before, and it mapped every typo
/// to 0, which the field then captioned "automatic" as though it had been
/// chosen. Empty is the one string that legitimately means automatic.
public enum ServerPortInput {
    public struct ParseError: Error, Equatable {
        public let message: String
    }

    public static func parse(_ text: String) -> Result<UInt16, ParseError> {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return .success(0) }
        guard trimmed.allSatisfy(\.isNumber) else {
            return .failure(ParseError(message: "Digits only"))
        }
        guard let value = Int(trimmed) else {
            // Every character is a digit yet `Int` refused: a run of ASCII
            // digits wider than 64 bits reports the RANGE it broke, not a
            // typo -- the reader of "Digits only" goes looking for a letter
            // that is not there.
            return .failure(
                ParseError(
                    message: trimmed.allSatisfy(\.isASCII)
                        ? "Port must be 1 to 65535"
                        : "Digits only (ASCII)"))
        }
        guard (1...65535).contains(value) else {
            return .failure(ParseError(message: "Port must be 1 to 65535"))
        }
        return .success(UInt16(value))
    }
}
