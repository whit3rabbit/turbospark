import SwiftUI
import TurboSpark

/// Compact model selector dropdown pill inside the composer footer.
struct PromptModelSelectorPill: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        Menu {
            Section(header: Text("Installed Models", bundle: .module)) {
                // The loadable list only (requirement 3.3).
                let loadable = model.loadableModels
                if loadable.isEmpty {
                    if let statement = model.notConnectedStatement {
                        Text(verbatim: statement)
                    } else {
                        Text("No models installed", bundle: .module)
                    }
                } else {
                    ForEach(loadable) { installed in
                        Button {
                            model.selectModel(installed)
                        } label: {
                            if model.selected?.alias == installed.alias {
                                Label(installed.alias, systemImage: "checkmark")
                            } else {
                                Text(installed.alias)
                            }
                        }
                    }
                }
            }

            Divider()

            Button {
                model.activeSection = .modelManager
            } label: {
                Label { Text("Manage models...", bundle: .module) } icon: { Image(systemName: "internaldrive") }
            }

            Button {
                model.activeSection = .modelHub
            } label: {
                Label { Text("Discover models...", bundle: .module) } icon: { Image(systemName: "shippingbox") }
            }
        } label: {
            HStack(spacing: 3) {
                Group {
                    if let alias = model.selected?.alias {
                        Text(alias)
                    } else {
                        Text("Select Model", bundle: .module)
                    }
                }
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)

                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(.micro, weight: .bold))
                    .foregroundStyle(.appSecondary)
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(Text("Select active model", bundle: .module))
        .accessibilityLabel(Text("Select active model", bundle: .module))
    }
}

/// Project context pill in the prompt composer showing project name, branch, and live git diff stats.
struct PromptProjectContextPill: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        if let project = model.selectedProject {
            HStack(spacing: 6) {
                Image(systemName: project.agentType.systemImage)
                    .font(theme.ui(.tiny, weight: .semibold))
                    .foregroundStyle(.appAccent)

                Text(project.name)
                    .font(theme.ui(.small, weight: .semibold))
                    .lineLimit(1)

                if let worktree = model.worktree, worktree.isGitRepository {
                    Text(worktree.currentBranch)
                        .font(theme.code(.callout, weight: .medium))
                        .foregroundStyle(.appSecondary)

                    if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                        HStack(spacing: 2) {
                            if worktree.totalAdditions > 0 {
                                Text(verbatim: "+\(worktree.totalAdditions)")
                                    .font(theme.ui(.tiny, weight: .bold).monospacedDigit())
                                    .foregroundStyle(.green)
                            }
                            if worktree.totalDeletions > 0 {
                                Text(verbatim: "-\(worktree.totalDeletions)")
                                    .font(theme.ui(.tiny, weight: .bold).monospacedDigit())
                                    .foregroundStyle(.red)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(.appElevated, in: Capsule())
            .overlay(Capsule().stroke(.appBorder, lineWidth: 1))
            .help(Text("Active project: \(project.name)", bundle: .module))
            .accessibilityElement(children: .combine)
            .accessibilityLabel(Text("Active project: \(project.name)", bundle: .module))
        }
    }
}

/// Reasoning effort pill button in the prompt composer footer with quick popup menu selection.
struct PromptReasoningPillControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        let isThinkingActive = model.reasoning != .off

        Menu {
            Section(header: Text("Thinking / Reasoning Effort", bundle: .module)) {
                ForEach(model.availableReasoningLevels) { level in
                    // Level labels and descriptions are catalog keys, the
                    // same lookup the model loader's reasoning menu makes.
                    let title = Text(LocalizedStringKey(model.reasoningLabel(for: level)), bundle: .module)
                        + Text(verbatim: " - ")
                        + Text(LocalizedStringKey(model.reasoningDescription(for: level)), bundle: .module)
                    Button {
                        model.setReasoning(level)
                    } label: {
                        if model.reasoning == level {
                            Label { title } icon: { Image(systemName: "checkmark") }
                        } else {
                            title
                        }
                    }
                }
            }

            Divider()

            Button {
                model.openSettings(tab: .engine)
            } label: {
                Label { Text("Engine settings...", bundle: .module) } icon: { Image(systemName: "gearshape") }
            }
        } label: {
            HStack(spacing: 3) {
                Text(LocalizedStringKey(model.reasoningLabel(for: model.reasoning)), bundle: .module)
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(isThinkingActive ? AnyShapeStyle(.appText) : AnyShapeStyle(.appSecondary))

                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(.micro, weight: .bold))
                    .foregroundStyle(.appSecondary)
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(Text("Adjust reasoning effort", bundle: .module))
        .accessibilityLabel(Text("Reasoning Effort", bundle: .module))
        .accessibilityValue(Text(LocalizedStringKey(model.reasoningLabel(for: model.reasoning)), bundle: .module))
    }
}
