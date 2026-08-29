import AppKit
import SwiftUI
import TurboSpark

struct InspectorView: View {
    @ObservedObject var model: AppModel

    var body: some View {
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
        .background(Color(nsColor: .windowBackgroundColor))
    }
}
