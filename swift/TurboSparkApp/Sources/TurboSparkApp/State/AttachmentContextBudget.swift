import Foundation

/// What the composer's attachments weigh against the context window, for the
/// attachment tray: a per-chip estimate and one budget reading for them all.
///
/// An ESTIMATE, labelled as one in the UI. Documents are priced at the same
/// four characters per token the context ring and the importer charge
/// (`estimatedContextTokens`), through the shared rounding helper the tool
/// picker uses, so no third copy of the arithmetic exists. Images get no
/// figure: the app knows no per-image vision token cost (the ring cannot see
/// them either, one template marker each), and a guessed number would read
/// as a measurement.
enum AttachmentContextBudget {
    /// Where the tray starts warning, as a share of the room the prompt can
    /// use. Earlier than the ring's red (90 percent of the whole window)
    /// because this compares against the window minus the reply reserve.
    static let nearLimitFraction = 0.85

    enum Level: Equatable {
        case fits
        case nearLimit
        case overLimit
    }

    struct Reading: Equatable {
        let level: Level
        /// The attachments' share of the usable window, rounded.
        let attachmentPercent: Int
        /// The whole prompt's share of the usable window, rounded.
        let totalPercent: Int
    }

    /// The estimate for one attachment, or nil when there is nothing to
    /// estimate: an image (no known vision cost) or a document that
    /// extracted no text.
    static func tokens(for attachment: AppPromptAttachment) -> Int? {
        guard !attachment.isImage, attachment.characterCount > 0 else { return nil }
        return ToolPromptCostEstimate.tokens(forCharacterCount: attachment.characterCount)
    }

    /// The sum of the per-chip estimates, so the tray line agrees with the
    /// chips. Nil when no attachment has one.
    static func totalTokens(_ attachments: [AppPromptAttachment]) -> Int? {
        let estimates = attachments.compactMap(tokens(for:))
        return estimates.isEmpty ? nil : estimates.reduce(0, +)
    }

    /// The budget for `attachmentTokens` on top of the rest of the prompt
    /// (`estimatedPromptTokens`: history, system, tools, draft), against the
    /// window minus the reply reserve (the importer's own budget). Nil when
    /// any input is unusable or there is no usable room to compare against.
    static func reading(
        attachmentTokens: Int,
        otherPromptTokens: Int,
        windowTokens: Int,
        reservedForReplyTokens: Int
    ) -> Reading? {
        guard attachmentTokens > 0, otherPromptTokens >= 0, reservedForReplyTokens >= 0 else {
            return nil
        }
        let available = windowTokens - reservedForReplyTokens
        guard available > 0 else { return nil }
        let total = attachmentTokens + otherPromptTokens
        let level: Level
        if total > available {
            level = .overLimit
        } else if Double(total) >= nearLimitFraction * Double(available) {
            level = .nearLimit
        } else {
            level = .fits
        }
        func percent(_ tokens: Int) -> Int {
            Int((Double(tokens) / Double(available) * 100).rounded())
        }
        return Reading(
            level: level,
            attachmentPercent: percent(attachmentTokens),
            totalPercent: percent(total))
    }
}
