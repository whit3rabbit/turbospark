import SwiftUI
import TurboSpark

/// Segmented control switching between Chat and Projects modes in the top window bar.
struct PromptInteractionModeSegment: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        HStack(spacing: 2) {
            ForEach(AppModel.AppInteractionMode.allCases) { mode in
                let isSelected = model.interactionMode == mode
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        model.setInteractionMode(mode)
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: mode.systemImage)
                            .font(theme.ui(points: 10, weight: .semibold))
                        Text(mode.title)
                            .font(theme.ui(points: 12, weight: isSelected ? .semibold : .medium))
                            .lineLimit(1)
                            .fixedSize()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
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
        .background(Color.primary.opacity(0.05), in: Capsule())
        .overlay(Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5))
        .fixedSize()
        .help("Switch between conversational Chat mode and agentic Projects / Coding mode")
    }
}

/// Compact model selector dropdown pill inside the composer footer.
struct PromptModelSelectorPill: View {
    @Environment(\.appTheme) private var theme
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
            HStack(spacing: 3) {
                Text(model.selected?.alias ?? "Select Model")
                    .font(theme.ui(points: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(points: 8, weight: .bold))
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
                    .font(theme.ui(points: 11, weight: .semibold))
                    .foregroundStyle(TurboSparkTheme.accentColor)

                Text(project.name)
                    .font(theme.ui(points: 12, weight: .semibold))
                    .lineLimit(1)

                if let worktree = model.worktree, worktree.isGitRepository {
                    Text(worktree.currentBranch)
                        .font(theme.code(points: 10, weight: .medium))
                        .foregroundStyle(.secondary)

                    if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                        HStack(spacing: 2) {
                            if worktree.totalAdditions > 0 {
                                Text("+\(worktree.totalAdditions)")
                                    .font(theme.ui(points: 10, weight: .bold).monospacedDigit())
                                    .foregroundStyle(.green)
                            }
                            if worktree.totalDeletions > 0 {
                                Text("-\(worktree.totalDeletions)")
                                    .font(theme.ui(points: 10, weight: .bold).monospacedDigit())
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
    @Environment(\.appTheme) private var theme
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
            HStack(spacing: 3) {
                Text(model.reasoningLabel(for: model.reasoning))
                    .font(theme.ui(points: 12, weight: .medium))
                    .foregroundStyle(isThinkingActive ? Color.primary : Color.secondary)

                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(points: 8, weight: .bold))
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

/// Forge Guardrails status pill button with toggling and settings popover link.
struct ForgeGuardrailsPillControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        let isEnabled = model.effectiveForgeGuardrailsEnabled
        let isGlobalFixed = model.guardrailsMode == .alwaysOn || model.guardrailsMode == .alwaysOff
        // **THE INERT CONDITION IS "NO TOOLS WERE OFFERED", NOT "THE MODEL
        // LACKS TOOL MARKUP".** `inspect` accepts unconditionally when the
        // request carried no tools, so that is the one case where this toggle
        // provably changes nothing. Gating on the model's own markup would be
        // backwards: a checkpoint that frames no calls is exactly where the
        // rescue earns its keep, which is why `toolCalling` feeds the tooltip
        // and never the `disabled`.
        let inertReason = model.effectiveGuardrailsInertReason
        let isDisabled = isGlobalFixed || inertReason != nil

        HStack(spacing: 4) {
            Button {
                if !isDisabled {
                    model.setForgeGuardrailsEnabled(!isEnabled)
                }
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isEnabled ? "shield.checkmark.fill" : "shield.slash")
                        .font(theme.ui(points: 10, weight: .semibold))
                        .foregroundStyle(isEnabled ? TurboSparkTheme.accentColor : Color.secondary)

                    Text("Guardrails \(isEnabled ? "On" : "Off")")
                        .font(theme.ui(points: 12, weight: .medium))
                        .foregroundStyle(isEnabled ? Color.primary : Color.secondary)
                        .lineLimit(1)
                        .fixedSize()
                }
            }
            .buttonStyle(.plain)
            // DISABLED WITH A REASON, NEVER HIDDEN (swift/CLAUDE.md Gotchas
            // 23 and 33): a missing control reads as a missing feature.
            .disabled(isDisabled)
            .help(Self.helpText(
                inertReason: inertReason,
                isGlobalFixed: isGlobalFixed,
                modeLabel: model.guardrailsMode.label,
                isEnabled: isEnabled,
                nativeReason: model.toolCallingNativeReason
            ))
            .accessibilityLabel("Forge Guardrails: \(isEnabled ? "Enabled" : "Disabled")")
            .accessibilityHint(
                inertReason ?? (isGlobalFixed
                    ? "Managed by global settings"
                    : "Toggles tool-call guardrails for this context"))

            Button {
                model.openSettings(tab: .engine)
            } label: {
                Image(systemName: "info.circle")
                    .font(theme.ui(points: 10))
                    .foregroundStyle(.tertiary)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Tool-call guardrails rescue a call the decoder could not parse and check its arguments against the schema you sent. Click to open Settings.")
            .accessibilityLabel("Forge Guardrails information")
            .accessibilityHint("Opens Engine Settings to configure Guardrails")
        }
    }
}

extension ForgeGuardrailsPillControl {
    /// One tooltip, assembled from the three things that can be true at once.
    ///
    /// A pure static so it can be asserted without a model (`swift/CLAUDE.md`
    /// Gotcha 26). The ORDER is the point: an inert control explains itself
    /// first, because a user looking at a greyed pill wants to know why it is
    /// grey before anything else.
    static func helpText(
        inertReason: String?,
        isGlobalFixed: Bool,
        modeLabel: String,
        isEnabled: Bool,
        nativeReason: String?
    ) -> String {
        if let inertReason { return inertReason }
        if isGlobalFixed { return "Forge Guardrails is fixed to \(modeLabel) in Settings" }
        var text =
            isEnabled
            ? "Forge Guardrails active: click to disable"
            : "Forge Guardrails disabled: click to enable"
        // A checkpoint that frames no calls of its own is the case the rescue
        // helps MOST, so this reads as a reason to leave guardrails on rather
        // than as a limitation.
        if let nativeReason {
            text += ". This model hands no framed call over (" + nativeReason
                + "), so the rescue is what makes tool calls work here."
        }
        return text
    }
}

/// Directional steering status pill, beside the guardrails one.
///
/// **STEERING RESOLVES AT MODEL OPEN**, so this pill states intent and says
/// when the loaded model is not running it yet. A switch that silently did
/// nothing until the next load is the failure the whole surface exists to
/// avoid.
struct SteeringPillControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        let disabledReason = model.steeringDisabledReason
        let isOn = model.steeringEnabled && disabledReason == nil
        let pending = model.steeringNeedsReload

        HStack(spacing: 4) {
            Button {
                if disabledReason == nil { model.setSteeringEnabled(!model.steeringEnabled) }
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isOn ? "dial.medium.fill" : "dial.medium")
                        .font(theme.ui(points: 10, weight: .semibold))
                        .foregroundStyle(isOn ? TurboSparkTheme.accentColor : Color.secondary)
                    Text(isOn ? "Steering On" : "Steering Off")
                        .font(theme.ui(points: 12, weight: .medium))
                        .foregroundStyle(isOn ? Color.primary : Color.secondary)
                        .lineLimit(1)
                        .fixedSize()
                    // The pending marker is not decoration: without it the
                    // pill reads "On" while the loaded model is unsteered.
                    if pending {
                        Image(systemName: "arrow.clockwise")
                            .font(theme.ui(points: 9, weight: .semibold))
                            .foregroundStyle(Color.orange)
                    }
                }
            }
            .buttonStyle(.plain)
            .disabled(disabledReason != nil)
            .help(Self.helpText(
                disabledReason: disabledReason,
                isOn: isOn,
                needsReload: pending,
                summary: model.activeSteeringSummary
            ))
            .accessibilityLabel("Directional steering: \(isOn ? "On" : "Off")")
            .accessibilityHint(disabledReason ?? "Applied when a model loads")

            Button {
                model.openSettings(tab: .safety)
            } label: {
                Image(systemName: "info.circle")
                    .font(theme.ui(points: 10))
                    .foregroundStyle(.tertiary)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Register and choose a steering direction. Click to open Settings.")
            .accessibilityLabel("Directional steering information")
            .accessibilityHint("Opens Safety and Steering settings")
        }
    }

    /// Pure, for the reason `ForgeGuardrailsPillControl.helpText` is.
    static func helpText(
        disabledReason: String?,
        isOn: Bool,
        needsReload: Bool,
        summary: String?
    ) -> String {
        if let disabledReason { return disabledReason }
        if needsReload {
            return "Reload the model to apply this. Steering is resolved when a model opens, "
                + "so the one loaded now is unchanged."
        }
        if isOn, let summary { return "Running: " + summary }
        if isOn { return "On at the next model load." }
        return "Off. No edit is applied to the residual stream."
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
    @Environment(\.appTheme) private var theme
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting tips")
                .font(theme.ui(points: 13, weight: .semibold))

            tipSection("Clear task & constraints",
                       "State what you want created, explained, or transformed. Specify length, style, or output structure.")
            tipSection("Provide types & interfaces",
                       "For code tasks, provide signatures, expected inputs/outputs, or small working scaffolds.")
            tipSection("Attach relevant documents",
                       "Attach PDFs, spreadsheets, or code files for local reasoning and question answering.")
        }
        .font(theme.ui(points: 12))
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
                    Label("Attach documents", systemImage: "plus")
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
