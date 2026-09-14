import SwiftUI
import TurboSpark

/// Compact model selector dropdown pill inside the composer footer.
struct PromptModelSelectorPill: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        Menu {
            Section(header: Text("Installed Models", bundle: .module)) {
                if model.installed.isEmpty {
                    Text("No models installed", bundle: .module)
                } else {
                    ForEach(model.installed) { installed in
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
                Text(model.selected?.alias ?? "Select Model")
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)

                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(.micro, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Select model for generation")
        .accessibilityLabel("Selected model: \(model.selected?.alias ?? "None")")
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
            .background(.appAccent.opacity(0.1), in: Capsule())
            .overlay(Capsule().stroke(.appAccent.opacity(0.25), lineWidth: 0.5))
            .fixedSize()
            .help("Active project: \(project.name)")
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Active project: \(project.name)")
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
                    let title = model.reasoningLabel(for: level)
                        + " - " + model.reasoningDescription(for: level)
                    Button {
                        model.setReasoning(level)
                    } label: {
                        if model.reasoning == level {
                            Label(title, systemImage: "checkmark")
                        } else {
                            Text(title)
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
                Text(model.reasoningLabel(for: model.reasoning))
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(isThinkingActive ? Color.primary : Color.secondary)

                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(.micro, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Reasoning effort level for \(model.selected?.alias ?? "the model"): currently \(model.reasoning.label). Click to change.")
        .accessibilityLabel("Reasoning effort: \(model.reasoning.label)")
        .accessibilityHint("Selects thinking depth for the next response without requiring model reload")
    }
}
