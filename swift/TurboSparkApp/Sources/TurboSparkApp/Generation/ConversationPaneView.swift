import SwiftUI

@MainActor
struct ConversationPaneView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        if model.hasOutputTranscript {
            VStack(spacing: 0) {
                OutputPaneView(model: model)
                    .frame(maxHeight: .infinity)
                conversationChrome
                    .fixedSize(horizontal: false, vertical: true)
            }
        } else {
            OutputPaneView(model: model)
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 8) {
            ErrorBanner(model: model)
            if model.showsNoModelInstalledBanner {
                NoModelInstalledBanner(model: model)
            }
            PromptComposerView(model: model)
            ModelLoaderControl(model: model, density: .compact)
        }
        .conversationColumn()
        .padding(.top, 8)
        .padding(.bottom, 12)
        .background(.appPage)
    }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Sits above the composer while no model is installed, so a saved chat
/// stays readable and the disabled Send has a stated reason and a way out.
/// Not dismissible: it describes a state, not an event, and disappears on
/// its own once a model is installed or an install starts.
@MainActor
private struct NoModelInstalledBanner: View {
    @ObservedObject var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .themedFont(.title3)
                .foregroundStyle(AppAttention.color)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                Text("No model installed", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Text("Install a model or open an existing model folder to continue this chat.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)

            Spacer(minLength: 8)

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Text("Open model folder\u{2026}", bundle: .module)
                    .themedFont(.small)
            }
            .buttonStyle(.bordered)

            // Discover ranks the catalog by the same hardware-fit probe as
            // the first-run setup screen, so it is the primary path here.
            Button {
                model.openModelHub()
            } label: {
                Text("Discover Models", bundle: .module)
                    .themedFont(.small, weight: .medium)
            }
            .buttonStyle(.borderedProminent)
        }
        .controlSize(.small)
        .padding(12)
        .background(AppAttention.color.opacity(0.08), in: RoundedRectangle(cornerRadius: ConversationLayout.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: ConversationLayout.cardRadius, style: .continuous)
                .stroke(AppAttention.color.opacity(0.35), lineWidth: 1)
        }
        .transition(.opacity)
    }
}
