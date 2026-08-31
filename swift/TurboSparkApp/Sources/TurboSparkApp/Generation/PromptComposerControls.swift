import SwiftUI
import TurboSpark

/// Segmented control switching between Chat and Projects modes right in the composer.
struct PromptInteractionModeSegment: View {
    @ObservedObject var model: AppModel

    var body: some View {
        HStack(spacing: 2) {
            ForEach(AppModel.AppInteractionMode.allCases) { mode in
                let isSelected = model.interactionMode == mode
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        model.interactionMode = mode
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: mode.systemImage)
                            .font(.system(size: 10, weight: .semibold))
                        Text(mode.title)
                            .font(.system(size: 11, weight: isSelected ? .semibold : .medium))
                            .lineLimit(1)
                            .fixedSize()
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .foregroundStyle(isSelected ? Color.primary : Color.secondary)
                    .background(
                        isSelected ? Color.primary.opacity(0.12) : Color.clear,
                        in: Capsule()
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Mode: \(mode.title)")
                .accessibilityAddTraits(isSelected ? [.isSelected] : [])
            }
        }
        .padding(2)
        .background(Color.primary.opacity(0.04), in: Capsule())
        .overlay(Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5))
        .fixedSize()
        .help("Switch between conversational Chat mode and agentic Projects / Coding mode")
    }
}

/// Compact model selector dropdown pill inside the composer footer.
struct PromptModelSelectorPill: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Menu {
            Section("Installed Models") {
                if model.installed.isEmpty {
                    Text("No models installed")
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
                Label("Manage models...", systemImage: "internaldrive")
            }

            Button {
                model.activeSection = .modelHub
            } label: {
                Label("Discover models...", systemImage: "shippingbox")
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "sparkles")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(TurboSparkTheme.accentColor)

                Text(model.selected?.alias ?? "Select Model")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Color.primary.opacity(0.04), in: Capsule())
            .overlay(Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5))
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
    @ObservedObject var model: AppModel

    var body: some View {
        if let project = model.selectedProject {
            HStack(spacing: 6) {
                Image(systemName: project.agentType.systemImage)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(TurboSparkTheme.accentColor)

                Text(project.name)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)

                if let worktree = model.worktree, worktree.isGitRepository {
                    Text(worktree.currentBranch)
                        .font(.system(size: 10, weight: .medium).monospaced())
                        .foregroundStyle(.secondary)

                    if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                        HStack(spacing: 2) {
                            if worktree.totalAdditions > 0 {
                                Text("+\(worktree.totalAdditions)")
                                    .font(.system(size: 10, weight: .bold).monospacedDigit())
                                    .foregroundStyle(.green)
                            }
                            if worktree.totalDeletions > 0 {
                                Text("-\(worktree.totalDeletions)")
                                    .font(.system(size: 10, weight: .bold).monospacedDigit())
                                    .foregroundStyle(.red)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(TurboSparkTheme.accentColor.opacity(0.1), in: Capsule())
            .overlay(Capsule().stroke(TurboSparkTheme.accentColor.opacity(0.25), lineWidth: 0.5))
            .fixedSize()
            .help("Active project: \(project.name)")
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Active project: \(project.name)")
        }
    }
}

/// Reasoning effort pill button in the prompt composer footer with quick popup menu selection.
struct PromptReasoningPillControl: View {
    @ObservedObject var model: AppModel

    var body: some View {
        let isThinkingActive = model.reasoning != .off

        Menu {
            Section("Thinking / Reasoning Effort") {
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
                Label("Engine settings...", systemImage: "gearshape")
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isThinkingActive ? "brain.head.profile" : "brain")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(isThinkingActive ? TurboSparkTheme.accentColor : Color.secondary)

                Text("Thinking: \(model.reasoningLabel(for: model.reasoning))")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(isThinkingActive ? Color.primary : Color.secondary)

                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(
                isThinkingActive ? TurboSparkTheme.accentColor.opacity(0.12) : Color.primary.opacity(0.04),
                in: Capsule()
            )
            .overlay(
                Capsule()
                    .stroke(isThinkingActive ? TurboSparkTheme.accentColor.opacity(0.3) : TurboSparkTheme.hairlineColor, lineWidth: 0.5)
            )
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Reasoning effort level for \(model.selected?.alias ?? "the model"): currently \(model.reasoning.label). Click to change.")
        .accessibilityLabel("Reasoning effort: \(model.reasoning.label)")
        .accessibilityHint("Selects thinking depth for the next response without requiring model reload")
    }
}

/// Forge Guardrails status pill button with toggling and settings popover link.
struct ForgeGuardrailsPillControl: View {
    @ObservedObject var model: AppModel

    var body: some View {
        let isEnabled = model.effectiveForgeGuardrailsEnabled
        let isGlobalFixed = model.guardrailsMode == .alwaysOn || model.guardrailsMode == .alwaysOff

        HStack(spacing: 4) {
            Button {
                if !isGlobalFixed {
                    model.setForgeGuardrailsEnabled(!isEnabled)
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: isEnabled ? "shield.checkmark.fill" : "shield.slash")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(isEnabled ? TurboSparkTheme.accentColor : Color.secondary)

                    Text("Guardrails: \(isEnabled ? "On" : "Off")")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(isEnabled ? Color.primary : Color.secondary)
                        .lineLimit(1)
                        .fixedSize()
                }
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(
                    isEnabled ? TurboSparkTheme.accentColor.opacity(0.12) : Color.primary.opacity(0.04),
                    in: Capsule()
                )
                .overlay(
                    Capsule()
                        .stroke(isEnabled ? TurboSparkTheme.accentColor.opacity(0.3) : TurboSparkTheme.hairlineColor, lineWidth: 0.5)
                )
            }
            .buttonStyle(.plain)
            .disabled(isGlobalFixed)
            .help(isGlobalFixed
                  ? "Forge Guardrails is fixed to \(model.guardrailsMode.label) in Settings"
                  : (isEnabled ? "Forge Guardrails active: click to disable" : "Forge Guardrails disabled: click to enable"))
            .accessibilityLabel("Forge Guardrails: \(isEnabled ? "Enabled" : "Disabled")")
            .accessibilityHint(isGlobalFixed ? "Managed by global settings" : "Toggles tool-call guardrails for this context")

            Button {
                model.openSettings(tab: .engine)
            } label: {
                Image(systemName: "info.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Tool-call guardrails are active by default for models supporting tools. You can change this behavior in Settings. Click to open Settings.")
            .accessibilityLabel("Forge Guardrails information")
            .accessibilityHint("Opens Engine Settings to configure Guardrails")
        }
    }
}

/// Popover button providing prompt authoring tips.
struct PromptTipsButton: View {
    let iconButtonSize: CGFloat
    @Binding var showingTips: Bool

    var body: some View {
        Button {
            showingTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: iconButtonSize, height: iconButtonSize)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")
        .accessibilityLabel("Prompt tips")
        .accessibilityHint("Shows a popover with prompt writing guidance")
        .popover(isPresented: $showingTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            PromptTipsGuideView()
        }
    }
}

/// Content inside the prompt tips popover.
struct PromptTipsGuideView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting tips")
                .font(.headline)

            tipSection("Clear task & constraints",
                       "State what you want created, explained, or transformed. Specify length, style, or output structure.")
            tipSection("Provide types & interfaces",
                       "For code tasks, provide signatures, expected inputs/outputs, or small working scaffolds.")
            tipSection("Attach relevant documents",
                       "Attach PDFs, spreadsheets, or code files for local reasoning and question answering.")
        }
        .font(.callout)
        .frame(width: 360, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).fontWeight(.semibold)
            Text(detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Button triggering document file attachment dialog.
struct PromptAttachDocumentButton: View {
    let iconButtonSize: CGFloat
    let isRunning: Bool
    let isExtracting: Bool
    let onAttach: () -> Void

    var body: some View {
        Button(action: onAttach) {
            Group {
                if isExtracting {
                    TaskProgressFlameIcon(size: 16)
                } else {
                    Label("Attach documents", systemImage: "paperclip")
                        .labelStyle(.iconOnly)
                }
            }
            .frame(width: iconButtonSize, height: iconButtonSize)
            .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .disabled(isRunning || isExtracting)
        .help("Attach PDF, Word, Excel, code, or text files")
        .accessibilityLabel(isExtracting
                            ? "Extracting document text"
                            : "Attach documents")
        .accessibilityHint("Opens a file picker to attach documents to this prompt")
    }
}
