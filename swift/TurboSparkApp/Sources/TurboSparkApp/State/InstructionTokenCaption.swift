import Foundation

/// The caption under an instructions editor: what the text costs in tokens
/// and how much of the loaded model's window that is.
///
/// For a local model with a fixed window, instruction tokens are KV cost on
/// every turn, so the figure is the real tokenizer's count when a session is
/// loaded. With no session (or while a turn holds the session's serial
/// queue) it falls back to the shared four-characters-per-token estimate
/// (`ToolPromptCostEstimate`) and says so in the text, never as a number
/// that reads like a measurement.
enum InstructionTokenCaption {
    /// Past this share of the window the caption warns. A system prompt that
    /// alone takes over a quarter of the window leaves little room for the
    /// conversation before auto-compact starts trimming it.
    static let warningFraction = 0.25

    struct Reading: Equatable {
        let tokens: Int
        let isEstimate: Bool
        /// Rounded share of the window, nil when the window is unknown.
        let percent: Int?
        let exceedsWarning: Bool
    }

    /// Nil for text with nothing to count. `measuredTokens` is the real
    /// tokenizer's count (ignored when negative); `windowTokens` is the
    /// loaded model's context, nil or non-positive when unknown.
    static func reading(text: String, measuredTokens: Int?, windowTokens: Int?) -> Reading? {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        let measured = measuredTokens.flatMap { $0 >= 0 ? $0 : nil }
        guard let tokens = measured ?? ToolPromptCostEstimate.tokens(forCharacterCount: text.count)
        else { return nil }
        let window = windowTokens.flatMap { $0 > 0 ? $0 : nil }
        let fraction = window.map { Double(tokens) / Double($0) }
        return Reading(
            tokens: tokens,
            isEstimate: measured == nil,
            percent: fraction.map { Int(($0 * 100).rounded()) },
            exceedsWarning: (fraction ?? 0) > warningFraction)
    }

    /// The caption line in `language`.
    static func text(for reading: Reading, language: AppLanguage) -> String {
        let count = reading.tokens.formatted(.number.locale(language.locale))
        switch (reading.isEstimate, reading.percent) {
        case (false, let percent?):
            return AppStrings.localized(
                "\(count) tokens, \(percent)% of the context window", language: language)
        case (false, nil):
            return AppStrings.localized("\(count) tokens", language: language)
        case (true, let percent?):
            return AppStrings.localized(
                "About \(count) tokens (estimate), \(percent)% of the context window",
                language: language)
        case (true, nil):
            return AppStrings.localized("About \(count) tokens (estimate)", language: language)
        }
    }

    /// The warning beside the caption past `warningFraction`.
    static func warningText(language: AppLanguage) -> String {
        AppStrings.localized(
            "These instructions alone use over a quarter of the context window.",
            language: language)
    }

    /// Every catalog key this type resolves, for the catalog test.
    static let catalogKeys = [
        "%@ tokens, %lld%% of the context window",
        "%@ tokens",
        "About %@ tokens (estimate), %lld%% of the context window",
        "About %@ tokens (estimate)",
        "These instructions alone use over a quarter of the context window.",
    ]
}
