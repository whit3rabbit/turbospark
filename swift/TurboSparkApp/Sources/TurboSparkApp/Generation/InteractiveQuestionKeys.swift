import Foundation

/// Number keys on a focused question card. Options are numbered 1-9 straight
/// through the card, question by question, so one key names exactly one
/// option even when the card holds several questions; options past the ninth
/// stay click-only. Only ASCII digits act, because the hint glyph each row
/// shows is an ASCII digit.
enum InteractiveQuestionKeys {
    struct Target: Equatable {
        let question: Int
        let option: Int
    }

    /// The option a key picks, or nil for any key that names none.
    static func target(for character: Character, optionCounts: [Int]) -> Target? {
        guard character.isASCII, let digit = character.wholeNumberValue, (1...9).contains(digit) else {
            return nil
        }
        var remaining = digit - 1
        for (question, count) in optionCounts.enumerated() {
            if remaining < count { return Target(question: question, option: remaining) }
            remaining -= max(count, 0)
        }
        return nil
    }

    /// The key that picks an option (1-9), or nil when it has none.
    static func number(question: Int, option: Int, optionCounts: [Int]) -> Int? {
        guard optionCounts.indices.contains(question), option >= 0, option < optionCounts[question] else {
            return nil
        }
        let flat = optionCounts[..<question].reduce(0) { $0 + max($1, 0) } + option + 1
        return flat <= 9 ? flat : nil
    }
}
