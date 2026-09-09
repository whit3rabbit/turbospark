import SwiftUI

/// Preset prompt suggestions grid rendered above the composer in an empty conversation state.
struct PromptExamplesView: View {
    @Environment(\.appTheme) private var theme
    let select: (AppPromptPreset) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Try an example", bundle: .module)
                        .font(theme.ui(.callout, weight: .semibold))
                    Text("Choose a prompt, edit it, or write your own.", bundle: .module)
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                moreExamples
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: 10) {
                primaryCards
            }
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 18)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .transition(.opacity.combined(with: .move(edge: .bottom)))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Prompt examples")
    }

    @ViewBuilder
    private var primaryCards: some View {
        ForEach(AppPromptPreset.primary) { preset in
            Button {
                select(preset)
            } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(preset.title)
                        .font(theme.ui(.tiny, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                    Text(preset.prompt)
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                        .lineLimit(3)
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, minHeight: 96, alignment: .topLeading)
                .padding(10)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .background {
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color(nsColor: .windowBackgroundColor))
                    .overlay {
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(.separator.opacity(0.4), lineWidth: 0.5)
                    }
            }
            .help(preset.title)
            .accessibilityLabel("\(preset.title): \(preset.prompt)")
            .accessibilityHint("Populates the prompt composer with this example")
        }
    }

    private var moreExamples: some View {
        Menu {
            ForEach(AppPromptPreset.secondary) { preset in
                Button(preset.title) {
                    select(preset)
                }
            }
        } label: {
            Label("More", systemImage: "ellipsis")
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("More prompt examples")
        .accessibilityLabel("More prompt examples")
    }

    private var columns: [GridItem] {
        [GridItem(.adaptive(minimum: 180, maximum: 240), spacing: 10)]
    }
}
