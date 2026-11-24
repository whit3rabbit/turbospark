import SwiftUI

/// The token caption under an instructions editor (`InstructionTokenCaption`).
///
/// Counting is debounced through `.task(id:)`: each edit cancels the pending
/// sleep, so only a pause in typing reaches the session. The count runs on
/// the session's own serial queue via `await`, never the main thread. While
/// a turn is generating that queue is held for the whole turn, so the count
/// is skipped and the caption shows the labeled estimate instead
/// (`updateTokenEstimate` skips for the same reason).
@MainActor
struct InstructionTokenCaptionView: View {
    @ObservedObject var model: AppModel
    let text: String

    /// The last real count and the text it measured. Kept while the next
    /// count is pending so the caption does not flip to "About" on every
    /// keystroke; dropped when the session changes.
    @State private var measured: (text: String, tokens: Int, session: ObjectIdentifier)?

    static let debounceNanoseconds: UInt64 = 300_000_000

    private struct CountKey: Equatable {
        let text: String
        let session: ObjectIdentifier?
        let generating: Bool
    }

    private var sessionID: ObjectIdentifier? { model.session.map(ObjectIdentifier.init) }

    private var reading: InstructionTokenCaption.Reading? {
        // A stale count stands in only while a fresh one is coming. During
        // a turn no count runs, so an edited text reads as an estimate.
        let measuredTokens = measured.flatMap { m -> Int? in
            guard m.session == sessionID, m.text == text || !model.generating else { return nil }
            return m.tokens
        }
        return InstructionTokenCaption.reading(
            text: text,
            measuredTokens: measuredTokens,
            windowTokens: model.session == nil ? nil : model.resolvedContextTokens)
    }

    var body: some View {
        Group {
            if let reading {
                let language = AppStrings.currentLanguage
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: InstructionTokenCaption.text(for: reading, language: language))
                        .themedFont(.small)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                    if reading.exceedsWarning {
                        // Icon and text, never color alone.
                        Label {
                            Text(verbatim: InstructionTokenCaption.warningText(language: language))
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .accessibilityHidden(true)
                        }
                        .themedFont(.small)
                        .foregroundStyle(.appWarning)
                    }
                }
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityElement(children: .combine)
            }
        }
        .task(id: CountKey(text: text, session: sessionID, generating: model.generating)) {
            await count()
        }
    }

    private func count() async {
        guard let session = model.session, !model.generating,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        let snapshot = text
        let id = ObjectIdentifier(session)
        try? await Task.sleep(nanoseconds: Self.debounceNanoseconds)
        guard !Task.isCancelled else { return }
        // Raw text, no chat template: the caption prices the instructions
        // themselves, not the few wrapper tokens a template adds.
        guard let tokens = try? await session.countTokens(in: snapshot),
            !Task.isCancelled
        else { return }
        measured = (snapshot, tokens, id)
    }
}
