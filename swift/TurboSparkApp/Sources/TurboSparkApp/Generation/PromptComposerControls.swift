import SwiftUI
import TurboSpark

/// Popover button providing prompt authoring tips.
struct PromptTipsButton: View {
    let iconButtonSize: CGFloat
    @Binding var showingTips: Bool

    var body: some View {
        Button {
            showingTips.toggle()
        } label: {
            Label { Text("Prompt tips", bundle: .module) } icon: { Image(systemName: "questionmark.circle") }
                .labelStyle(.iconOnly)
                .frame(width: iconButtonSize, height: iconButtonSize)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.appSecondary)
        .help(Text("Prompt tips", bundle: .module))
        .accessibilityLabel(Text("Prompt tips", bundle: .module))
        .accessibilityHint(Text("Shows a popover with prompt writing guidance", bundle: .module))
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
            Text("Prompting tips", bundle: .module)
                .font(theme.ui(.callout, weight: .semibold))

            tipSection("Clear task & constraints",
                       "State what you want created, explained, or transformed. Specify length, style, or output structure.")
            tipSection("Provide types & interfaces",
                       "For code tasks, provide signatures, expected inputs/outputs, or small working scaffolds.")
            tipSection("Attach relevant documents",
                       "Attach PDFs, spreadsheets, or code files for local reasoning and question answering.")
        }
        .font(theme.ui(.small))
        .frame(width: 360, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: LocalizedStringKey, _ detail: LocalizedStringKey) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title, bundle: .module).fontWeight(.semibold)
            Text(detail, bundle: .module).foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Button triggering document file attachment dialog.
struct PromptAttachDocumentButton: View {
    let iconButtonSize: CGFloat
    let isRunning: Bool
    let isExtracting: Bool
    let onAttach: () -> Void
    @State private var isHovered: Bool = false

    var body: some View {
        Button(action: onAttach) {
            Group {
                if isExtracting {
                    TaskProgressFlameIcon(size: 16)
                } else {
                    Image(systemName: "plus")
                        .themedFont(.callout, weight: .medium)
                }
            }
            .frame(width: iconButtonSize, height: iconButtonSize)
            .background(
                Color.primary.opacity(isHovered ? 0.08 : 0.04),
                in: Circle()
            )
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.appSecondary)
        .onHover { isHovered = $0 }
        .disabled(isRunning || isExtracting)
        .help(Text("Attach PDF, Word, Excel, code, or text files", bundle: .module))
        .accessibilityLabel(isExtracting
                            ? Text("Extracting document text", bundle: .module)
                            : Text("Attach documents", bundle: .module))
        .accessibilityHint(Text("Opens a file picker to attach documents to this prompt", bundle: .module))
    }
}

/// Tools toggle in the composer bar for Chat mode.
///
/// Drawn by the shared `AppChip` matching the composer vocabulary.
struct ChatToolsToggleButton: View {
    @ObservedObject var model: AppModel

    var body: some View {
        AppChip(Text("Tools", bundle: .module), systemImage: "wrench.and.screwdriver", isOn: model.chatToolsEnabled) {
            model.chatToolsEnabled.toggle()
            model.persistSettingsDebounced()
        }
        .help(model.chatToolsEnabled
              ? Text("Tools enabled in Chat", bundle: .module)
              : Text("Enable tools in Chat", bundle: .module))
        .accessibilityValue(model.chatToolsEnabled ? Text("On", bundle: .module) : Text("Off", bundle: .module))
    }
}

/// Web search toggle in the composer bar.
///
/// Drawn by the shared `AppChip` so the chat composer and Humanize offer one
/// chip instead of two copies that drift. The chip's label is its VoiceOver
/// name and the `isSelected` trait carries on/off; the longer tooltip below
/// keeps saying what turning it on actually does.
struct SearchToggleButton: View {
    @ObservedObject var model: AppModel

    var body: some View {
        AppChip(Text("Search", bundle: .module), systemImage: "globe", isOn: model.webSearchEnabled) {
            model.webSearchEnabled.toggle()
        }
        .help(model.webSearchEnabled
              ? Text("Web search enabled: tools will search the web for real-time information", bundle: .module)
              : Text("Enable web search", bundle: .module))
        .accessibilityValue(model.webSearchEnabled ? Text("On", bundle: .module) : Text("Off", bundle: .module))
    }
}

/// Audio dictation button triggering macOS speech dictation into the prompt editor.
struct PromptAudioInputButton: View {
    @FocusState.Binding var promptFocused: Bool
    @State private var isHovered: Bool = false

    var body: some View {
        Button {
            promptFocused = true
            NSApp.sendAction(Selector(("startDictation:")), to: nil, from: nil)
        } label: {
            Image(systemName: "mic")
                .themedFont(.callout, weight: .medium)
                .foregroundStyle(Color.secondary)
                .frame(width: 28, height: 28)
                .background(
                    Color.primary.opacity(isHovered ? 0.08 : 0),
                    in: Circle()
                )
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help(Text("Start voice dictation", bundle: .module))
        .accessibilityLabel(Text("Voice input dictation", bundle: .module))
    }
}
