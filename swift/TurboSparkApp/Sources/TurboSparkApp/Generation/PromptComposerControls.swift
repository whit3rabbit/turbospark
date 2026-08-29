import SwiftUI

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
                    ProgressView().controlSize(.small)
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
