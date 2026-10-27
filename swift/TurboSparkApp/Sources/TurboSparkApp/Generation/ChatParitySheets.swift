import SwiftUI
import TurboSpark

/// The qwen-code local-command sheets: `/context`, `/tasks`, `/tools`,
/// `/status`, `/rewind` and `/recap`, plus the `/delete` confirmation.
/// Presented from `OutputPaneView`; each mirrors the qwen-code panel of the
/// same name in the parts a native single-model app has data for.
struct ContextUsageSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel

    var body: some View {
        sheetFrame(title: "Context Usage", iconName: "gauge.with.needle", dismiss: dismiss) {
            ContextUsagePopoverView(model: model)
        }
    }
}

/// `/tasks`: the running background agents and shells for this chat, with
/// the same stop/dismiss actions the transcript strips carry, plus the
/// tracked task list. qwen-code renders the same surface from daemon task
/// state; here the data is in-process.
struct TasksStatusSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel

    private var agentRuns: [SubagentRunState] {
        model.backgroundAgentRuns.values
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
            .sorted { $0.startedAt < $1.startedAt }
    }

    private var shells: [BackgroundShellSummary] {
        model.backgroundShellSummaries
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
    }

    var body: some View {
        sheetFrame(title: "Background Tasks", iconName: "square.stack.3d.up", dismiss: dismiss) {
            if agentRuns.isEmpty && shells.isEmpty {
                emptyText("Nothing is running in the background.")
            } else {
                VStack(alignment: .leading, spacing: 14) {
                    if !agentRuns.isEmpty {
                        sectionHeader("Agents")
                        ForEach(agentRuns) { run in
                            AgentRunRow(
                                run: run,
                                onStop: { id in
                                    Task { @MainActor in
                                        _ = try? await model.stopBackgroundAgent(id)
                                    }
                                },
                                onDismiss: { model.dismissBackgroundAgent($0) })
                        }
                    }
                    if !shells.isEmpty {
                        sectionHeader("Shells")
                        ForEach(shells) { shell in
                            shellRow(shell)
                        }
                    }
                }
            }
        }
    }

    /// One agent row, observing ITS OWN run state. `SubagentRunState` is an
    /// ObservableObject and the runs dictionary only publishes on
    /// insert/remove, so a row reading the run through the sheet's model
    /// observation kept the green dot and the Stop button after the run
    /// finished until some unrelated model publish happened by.
    private struct AgentRunRow: View {
        @Environment(\.appTheme) private var theme
        @ObservedObject var run: SubagentRunState
        let onStop: (String) -> Void
        let onDismiss: (String) -> Void

        var body: some View {
            HStack(spacing: 10) {
                Circle()
                    .fill(run.status == "running" ? theme.accent : theme.secondaryText)
                    .frame(width: 8, height: 8)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(run.displayName.isEmpty ? run.agentName : run.displayName)
                        .themedFont(.small, weight: .semibold)
                    Text(run.taskDescription.isEmpty ? run.promptHead : run.taskDescription)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(2)
                }
                Spacer()
                if run.status == "running" {
                    Button {
                        onStop(run.id)
                    } label: { Text("Stop", bundle: .module) }
                    .appAction(.quiet, size: .small)
                } else if run.isRecordedComplete {
                    Button {
                        onDismiss(run.id)
                    } label: { Text("Dismiss", bundle: .module) }
                    .appAction(.quiet, size: .small)
                }
                Text(run.status)
                    .themedCode(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            .accessibilityElement(children: .combine)
        }
    }

    private func shellRow(_ shell: BackgroundShellSummary) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "terminal")
                .themedFont(.small)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(shell.description?.isEmpty == false ? shell.description! : shell.commandHead)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                Text(shell.id)
                    .themedCode(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button {
                model.killBackgroundShell(id: shell.id)
            } label: { Text("Kill", bundle: .module) }
            .appAction(.quiet, size: .small)
        }
        .accessibilityElement(children: .combine)
    }
}

/// `/tools`: the tool catalog the model can call, names and descriptions,
/// the qwen-code `/tools [desc]` panel over the local registry.
struct ToolsListSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel

    private var tools: [OpenAITool] { AppToolCatalog.allTools }

    var body: some View {
        sheetFrame(title: "Tools", iconName: "wrench.and.screwdriver", dismiss: dismiss) {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(tools, id: \.function.name) { tool in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(tool.function.name)
                            .themedCode(.small, weight: .semibold)
                        Text(tool.function.description)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
        }
    }
}

/// `/status`: the About card parity -- app version, platform, chip,
/// thermal and memory pressure, loaded model and its resolved context,
/// permission mode, profile, and this chat's identity.
struct StatusSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel

    var body: some View {
        sheetFrame(title: "Status", iconName: "info.circle", dismiss: dismiss) {
            VStack(alignment: .leading, spacing: 18) {
                statusGroup("App") {
                    statusRow("Version", Bundle.main.object(
                        forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "-")
                    statusRow("Platform", "macOS "
                        + ProcessInfo.processInfo.operatingSystemVersionString)
                    statusRow("Profile", model.currentProfile.name)
                    statusRow("UI language", UserDefaults.standard.string(
                        forKey: AppLanguage.storageKey) ?? AppLanguage.system.rawValue)
                    statusRow("Permission mode", model.effectivePermissionMode.approvalLabelText)
                }
                statusGroup("Engine") {
                    if let alias = model.selected?.alias {
                        statusRow("Model", alias)
                    } else {
                        statusRow("Model", Text("none loaded", bundle: .module))
                    }
                    let contextTokens = model.resolvedContextTokens.formatted(.number.notation(.compactName))
                    statusRow("Context window", Text("\(contextTokens) tokens", bundle: .module))
                    statusRow("Chip", model.telemetry?.chip ?? "-")
                    statusRow("Thermal", model.telemetry?.thermalLevel ?? "-")
                    statusRow("Memory pressure", model.telemetry?.memoryPressure ?? "-")
                }
                statusGroup("Session") {
                    statusRow("Chat", model.selectedChat.title)
                    statusRow("Chat ID", model.selectedChatID.uuidString)
                    statusRow("Messages", String(model.selectedChat.messages.count))
                }
            }
        }
    }
}

/// `/rewind`: the transcript snapshots (each user prompt is one) and the
/// explicit confirm. Rewinding moves the CONVERSATION back, never any
/// files tools wrote -- the same scope line qwen-code's dialog carries.
struct RewindSheet: View {
    @Environment(\.appTheme) private var theme
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel
    @State private var selectedTargetID: UUID?

    var body: some View {
        sheetFrame(title: "Rewind Conversation", iconName: "arrow.counterclockwise", dismiss: dismiss) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Pick a prompt to rewind to. Everything after it is dropped from this conversation; files on disk are not touched.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                let targets = model.rewindTargets
                if targets.isEmpty {
                    emptyText("Nothing to rewind: this conversation has no finished turn yet.")
                } else {
                    ForEach(targets) { target in
                        Button {
                            selectedTargetID = target.id
                        } label: {
                            HStack {
                                Image(systemName: selectedTargetID == target.id
                                    ? "largecircle.fill.circle" : "circle")
                                    .foregroundStyle(selectedTargetID == target.id
                                        ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
                                    .accessibilityHidden(true)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(target.preview)
                                        .themedFont(.small)
                                        .lineLimit(2)
                                    Text(target.createdAt.formatted(
                                        date: .abbreviated, time: .shortened))
                                        .themedFont(.tiny)
                                        .foregroundStyle(.appSecondary)
                                }
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(Text("Rewind to prompt: \(target.preview)", bundle: .module))
                    }
                }
                HStack {
                    Spacer()
                    Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                        .appAction(.secondary)
                        .keyboardShortcut(.cancelAction)
                    // Destructive: everything after the chosen prompt is
                    // dropped from the conversation.
                    Button(role: .destructive, action: rewind) { Text("Rewind", bundle: .module) }
                        .appAction(.destructive)
                        .keyboardShortcut(.defaultAction)
                        .disabled(selectedTargetID == nil)
                }
                .padding(.top, 6)
            }
        }
    }

    private func rewind() {
        guard let selectedTargetID else { return }
        model.rewindTo(anchorMessageID: selectedTargetID)
        dismiss()
    }
}

/// `/recap`: the transient conversation summary. The text dies with the
/// sheet by design -- a recap is a reading of the conversation, not a
/// replacement for it (that is `/compact`).
struct RecapSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel
    @State private var isCopied = false

    var body: some View {
        sheetFrame(title: "Conversation Recap", iconName: "doc.text.magnifyingglass", dismiss: dismiss) {
            VStack(alignment: .leading, spacing: 12) {
                if model.isRecapping && model.recapText.isEmpty {
                    // The bar's label is VoiceOver only, so the words stay visible.
                    HStack(spacing: 8) {
                        AppProgressBar(
                            value: nil,
                            label: Text("Summarizing this conversation...", bundle: .module),
                            size: .compact
                        )
                        .frame(minWidth: 40, idealWidth: 80, maxWidth: 120)
                        Text("Summarizing this conversation...", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .accessibilityHidden(true)
                    }
                } else {
                    ScrollView {
                        ChatMessageMarkdownView(model.recapText)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(minHeight: 200)
                }
                HStack {
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.recapText, forType: .string)
                        isCopied = true
                    } label: {
                        Label {
                            isCopied ? Text("Copied", bundle: .module) : Text("Copy", bundle: .module)
                        } icon: {
                            Image(systemName: "doc.on.doc")
                        }
                    }
                    .appAction(.secondary)
                    // Copying before the recap lands would blank the clipboard.
                    .disabled(model.recapText.isEmpty)
                    .onChange(of: model.recapText) { _, _ in isCopied = false }
                    Spacer()
                    Button { dismiss() } label: { Text("Close", bundle: .module) }
                        .appAction(.primary)
                        .keyboardShortcut(.defaultAction)
                }
            }
        }
    }
}

// MARK: - Shared chrome

/// The frame every parity sheet shares: icon + title + close button, a
/// divider, then scrollable content. One builder rather than five
/// restatements of the `SessionStatsSheet` header.
@MainActor
@ViewBuilder
private func sheetFrame<Content: View>(
    title: LocalizedStringKey, iconName: String, dismiss: DismissAction,
    @ViewBuilder content: () -> Content
) -> some View {
    VStack(alignment: .leading, spacing: 0) {
        HStack {
            Image(systemName: iconName)
                .foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            Text(title, bundle: .module)
                .themedFont(.callout, weight: .semibold)
            Spacer()
            AppIconButton(
                "xmark.circle.fill",
                label: Text("Close \(Text(title, bundle: .module))", bundle: .module)
            ) {
                dismiss()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)

        Divider()

        ScrollView {
            content()
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
    .frame(minWidth: 480, idealWidth: 520, maxWidth: 600, minHeight: 400, idealHeight: 460, maxHeight: 600)
}

private func sectionHeader(_ title: LocalizedStringKey) -> some View {
    Text(title, bundle: .module)
        .themedFont(.small, weight: .bold)
        .foregroundStyle(.appSecondary)
        .textCase(.uppercase)
        .accessibilityAddTraits(.isHeader)
}

private func emptyText(_ text: LocalizedStringKey) -> some View {
    Text(text, bundle: .module)
        .themedFont(.small)
        .foregroundStyle(.appSecondary)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity)
}

@ViewBuilder
private func statusGroup<Content: View>(
    _ title: LocalizedStringKey, @ViewBuilder rows: () -> Content
) -> some View {
    VStack(alignment: .leading, spacing: 6) {
        sectionHeader(title)
        rows()
    }
}

/// A row whose value is data (a version, an id, a name): shown as is.
private func statusRow(_ label: LocalizedStringKey, _ value: String) -> some View {
    statusRow(label, Text(value))
}

/// Every row label is a catalog key.
private func statusRow(_ label: LocalizedStringKey, _ value: Text) -> some View {
    HStack(alignment: .firstTextBaseline) {
        Text(label, bundle: .module)
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
        Spacer(minLength: 12)
        value
            .themedCode(.small)
            .textSelection(.enabled)
            .lineLimit(1)
            .truncationMode(.middle)
    }
}
