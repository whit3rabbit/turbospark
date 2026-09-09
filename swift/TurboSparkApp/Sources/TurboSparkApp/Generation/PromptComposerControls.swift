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
        .foregroundStyle(.secondary)
        .onHover { isHovered = $0 }
        .disabled(isRunning || isExtracting)
        .help("Attach PDF, Word, Excel, code, or text files")
        .accessibilityLabel(isExtracting
                            ? "Extracting document text"
                            : "Attach documents")
        .accessibilityHint("Opens a file picker to attach documents to this prompt")
    }
}

/// Web search toggle button in the chat composer bar.
struct SearchToggleButton: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var isHovered: Bool = false

    var body: some View {
        Button {
            model.webSearchEnabled.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "globe")
                    .themedFont(.tiny, weight: .medium)
                Text("Search", bundle: .module)
                    .font(theme.ui(.small, weight: .medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .foregroundStyle(model.webSearchEnabled ? Color.primary : Color.secondary)
            .background(
                model.webSearchEnabled
                    ? TurboSparkTheme.accentColor.opacity(0.14)
                    : Color.primary.opacity(isHovered ? 0.08 : 0.04),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .overlay {
                if model.webSearchEnabled {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(TurboSparkTheme.accentColor.opacity(0.3), lineWidth: 0.5)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help(model.webSearchEnabled ? "Web search enabled: tools will search the web for real-time information" : "Enable web search")
        .accessibilityLabel("Web search: \(model.webSearchEnabled ? "Enabled" : "Disabled")")
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
        .help("Start voice dictation")
        .accessibilityLabel("Voice input dictation")
    }
}
