import SwiftUI

/// A compact strip above the transcript naming the agent profile this chat
/// runs as, when that is not the default Coding Agent. The profile changes
/// the system prompt and the tools offered, and otherwise shows only as a
/// glyph on the composer's project pill, so a chat that behaves unlike the
/// rest says why. Read-only: the profile is set in the project's settings.
@MainActor
struct ChatAgentStripView: View {
    @ObservedObject var model: AppModel

    /// The profile to name, or nil to draw nothing. `.coder` is what a chat
    /// gets with no choice made (`AppModel.activeAgentType`), so naming it
    /// would be noise on every chat.
    nonisolated static func shownAgent(_ agentType: AppAgentType?) -> AppAgentType? {
        guard let agentType, agentType != .coder else { return nil }
        return agentType
    }

    var body: some View {
        // `turnProject` is nil in Chat mode, which uses no project profile;
        // it is the same source the tool list reads.
        if let agent = Self.shownAgent(model.turnProject(chatID: model.selectedChatID)?.agentType) {
            HStack(spacing: 6) {
                Image(systemName: agent.systemImage)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Agent", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                // Labels and descriptions are catalog keys (the sidebar and
                // project settings render the same ones).
                Text(LocalizedStringKey(agent.label), bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appText)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(.appElevated, in: Capsule())
            .overlay(Capsule().stroke(.appBorder, lineWidth: 1))
            .help(Text(LocalizedStringKey(agent.descriptionText), bundle: .module))
            .accessibilityElement(children: .combine)
            .frame(maxWidth: .infinity, alignment: .leading)
            .conversationColumn()
            .padding(.vertical, 6)
        }
    }
}
