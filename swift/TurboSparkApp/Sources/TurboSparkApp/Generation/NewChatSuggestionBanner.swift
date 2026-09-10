import SwiftUI

/// The qwen-code "new session suggestion": once the context window is
/// nearly full and the conversation is long enough for the fill to be real,
/// offer a fresh chat. Dismissed per visit (in-memory only -- a suggestion
/// that persisted across relaunches would nag about a conversation the user
/// already chose to continue).
///
/// Threshold is the ring's red tier (90 percent), deliberately ABOVE the
/// auto-compact trigger: while compaction is enabled it is the mechanism
/// that handles a full window, and a banner fighting it would be noise.
/// The banner earns its place when compaction is OFF, or when the window is
/// full even AFTER a compaction.
struct NewChatSuggestionBanner: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    private var shouldSuggest: Bool {
        guard let summary = model.contextUsageSummary,
            summary.fraction >= 0.9,
            model.selectedTurnMessages.count >= 8,
            !model.dismissedNewChatSuggestionChatIDs.contains(model.selectedChatID)
        else { return false }
        return true
    }

    var body: some View {
        if shouldSuggest {
            HStack(spacing: 10) {
                Image(systemName: "arrow.triangle.branch")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.orange)
                    .accessibilityHidden(true)
                Text("This conversation is close to filling its context window. Starting a new chat keeps responses sharp.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                Spacer(minLength: 8)
                Button {
                    model.dismissedNewChatSuggestionChatIDs.insert(model.selectedChatID)
                } label: {
                    Text("Dismiss", bundle: .module)
                        .themedFont(.small)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.appSecondary)
                .help("Hide this suggestion for this chat")
                .accessibilityLabel("Dismiss new chat suggestion")
                Button {
                    model.createChat()
                } label: {
                    Text("New Chat", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(model.isRunning)
                .help("Start a fresh conversation (Cmd+N)")
                .accessibilityLabel("Start a new chat")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.orange.opacity(0.08))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(Color.orange.opacity(0.35), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .transition(.opacity.combined(with: .move(edge: .bottom)))
            .accessibilityElement(children: .contain)
        }
    }
}
