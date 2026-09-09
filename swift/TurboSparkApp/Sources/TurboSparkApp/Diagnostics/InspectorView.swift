import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct InspectorView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Form {
                modelSection
                // Its own struct rather than a computed property here: the
                // context ladder carries @State (the loaded ladder and its
                // failure flag), which an extension property cannot hold.
                InspectorEssentialsSection(model: model)
                memoryAndPowerSection
                steeringSection
                // Its own struct, not an extension computed property: it
                // carries @State (the edit scope and the preset name field),
                // which an extension property on InspectorView cannot hold.
                GenerationSamplingSection(model: model)
                telemetrySection
                RunnerDiagnosticsSection(diagnostics: model.diagnostics)
                usageDashboardSection
            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
        }
        .background(TurboSparkTheme.pageBackgroundColor)
    }

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "slider.horizontal.3")
                .themedFont(.callout, weight: .semibold)
                .foregroundStyle(TurboSparkTheme.accentColor)
                .help("Model loading, generation, and steering options for the active model")
                .accessibilityHidden(true)
            Text("Model Settings", bundle: .module)
                .themedFont(.callout, weight: .semibold)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .frame(height: AppChromeLayout.topBarHeight)
        .background(TurboSparkTheme.barBackgroundColor)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 0.5)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Model Settings")
    }
}
