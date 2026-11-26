import SwiftUI

/// The part of the empty chat screen under the composer: prompt examples as
/// chips, unfinished chats to resume, and the selected project's recent
/// chats. Each section hides itself when it has nothing to show, so a first
/// launch still reads as just the greeting and the composer.
struct EmptyChatHomeView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        let selectedIsGhost = model.selectedChat.isGhost
        let resume = EmptyChatHome.resumeItems(
            chats: model.filteredChats,
            selectedChatID: model.selectedChatID,
            selectedIsGhost: selectedIsGhost,
            isBusy: { model.chatTurnState(for: $0) != .idle })
        let project = model.selectedProject
        let recents = EmptyChatHome.projectRecents(
            chats: model.filteredChats,
            projectID: project?.id,
            selectedChatID: model.selectedChatID,
            selectedIsGhost: selectedIsGhost,
            excluding: Set(resume.map(\.chatID)))

        VStack(alignment: .leading, spacing: 18) {
            if !resume.isEmpty {
                section(
                    header: AppSectionHeader(title: Text("Pick up where you left off", bundle: .module))
                ) {
                    ForEach(resume) { item in
                        row(chatID: item.chatID, title: item.title) {
                            reasonLabel(item.reason)
                        }
                    }
                }
            }
            if let project, !recents.isEmpty {
                section(
                    header: AppSectionHeader(
                        title: Text(verbatim: project.name),
                        subtitle: Text("Recent chats", bundle: .module))
                ) {
                    ForEach(recents) { chat in
                        row(chatID: chat.id, title: chat.title) { EmptyView() }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Examples

    private var examples: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Try an example", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
                .accessibilityAddTraits(.isHeader)
            // Chips are fixed size; a narrow window or a long localized
            // title stacks them instead of clipping.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { exampleChips }
                VStack(alignment: .leading, spacing: 6) { exampleChips }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Prompt examples", bundle: .module))
    }

    @ViewBuilder
    private var exampleChips: some View {
        // Preset titles are bundled data sent to the model as written, not
        // UI chrome, so they render verbatim (as the old grid did).
        ForEach(AppPromptPreset.primary) { preset in
            AppChip(Text(verbatim: preset.title), systemImage: "text.bubble") {
                model.promptText = preset.prompt
            }
            .accessibilityHint(Text("Populates the prompt composer with this example", bundle: .module))
        }
        if !AppPromptPreset.secondary.isEmpty {
            moreExamples
        }
    }

    private var moreExamples: some View {
        Menu {
            ForEach(AppPromptPreset.secondary) { preset in
                Button(preset.title) { model.promptText = preset.prompt }
            }
        } label: {
            Image(systemName: "ellipsis")
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(.appSecondary)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help(Text("More prompt examples", bundle: .module))
        .accessibilityLabel(Text("More prompt examples", bundle: .module))
    }

    // MARK: Rows

    private func section<Content: View>(
        header: AppSectionHeader, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            VStack(spacing: 6) { content() }
        }
        .accessibilityElement(children: .contain)
    }

    private func row<Detail: View>(
        chatID: UUID, title: String, @ViewBuilder detail: () -> Detail
    ) -> some View {
        Button {
            model.selectChat(id: chatID)
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Image(systemName: "bubble.left")
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
                Text(verbatim: title)
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(.appText)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 8)
                detail()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .fill(.appSurface)
                .overlay {
                    RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                        .stroke(theme.border, lineWidth: 1)
                }
        }
        .help(Text("Switch to \(title)", bundle: .module))
        // One element: the title then the reason, read as a single button.
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
    }

    /// Glyph plus words, never color alone (DESIGN.md). The words reuse the
    /// transcript notice's catalog keys so the two places agree.
    private func reasonLabel(_ reason: EmptyChatResumeItem.Reason) -> some View {
        let symbol: String
        let text: Text
        var tone: AnyShapeStyle = AnyShapeStyle(.appSecondary)
        switch reason {
        case .stopped(let notice):
            symbol = notice.symbolName
            switch notice.severity {
            case .neutral: break
            case .warning: tone = AnyShapeStyle(.appWarning)
            case .danger: tone = AnyShapeStyle(.appDanger)
            }
            switch notice {
            case .stoppedByUser: text = Text("Stopped by you", bundle: .module)
            case .engineError: text = Text("Stopped because of an error", bundle: .module)
            case .contextFull: text = Text("Context window full", bundle: .module)
            case .lengthLimit: text = Text("Reached the length limit", bundle: .module)
            case .turnLimit: text = Text("Reached the turn limit", bundle: .module)
            }
        case .goal:
            symbol = "target"
            text = Text("Goal active", bundle: .module)
        case .draft:
            symbol = "pencil"
            text = Text("Draft", bundle: .module)
        }
        return HStack(spacing: 4) {
            Image(systemName: symbol)
                .foregroundStyle(tone)
                .accessibilityHidden(true)
            text.foregroundStyle(.appSecondary)
        }
        .font(theme.ui(.tiny))
        .lineLimit(1)
        .fixedSize()
    }
}
