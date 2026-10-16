import SwiftUI
import TurboSpark

/// The engine line under an audio model's name: what runs it and, when only
/// this app can, an App only mark. Each item is text with a glyph, never color
/// alone, and its longer explanation is both the tooltip and the VoiceOver
/// hint. A narrow window or a long translation stacks the items instead of
/// clipping them.
@MainActor
struct AudioEngineSummary: View {
    let profile: AudioProfile

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) { items }
            VStack(alignment: .leading, spacing: 4) { items }
        }
        .themedFont(.tiny)
        .foregroundStyle(.appSecondary)
    }

    @ViewBuilder private var items: some View {
        let engine = AudioEngineLabel(profile)
        Label { Text(LocalizedStringKey(engine.title), bundle: .module) }
            icon: { Image(systemName: engine.systemImage) }
            .help(Text(LocalizedStringKey(engine.tooltip), bundle: .module))
            .accessibilityHint(Text(LocalizedStringKey(engine.tooltip), bundle: .module))
        if AudioEngineLabel.isAppOnly(profile) {
            Label { Text("App only", bundle: .module) } icon: { Image(systemName: "macwindow") }
                .help(Text("The API server cannot run this model. It runs only in this app.", bundle: .module))
                .accessibilityHint(
                    Text("The API server cannot run this model. It runs only in this app.", bundle: .module))
        }
    }
}
