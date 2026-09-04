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
                memoryAndPowerSection
                steeringSection
                generationSection
                telemetrySection
                RunnerDiagnosticsSection(diagnostics: model.diagnostics)
            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(TurboSparkTheme.accentColor)
                .help("Model loading, generation, and steering options for the active model")
                .accessibilityHidden(true)
            Text("Model Settings")
                .font(.system(size: 13, weight: .semibold))
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
