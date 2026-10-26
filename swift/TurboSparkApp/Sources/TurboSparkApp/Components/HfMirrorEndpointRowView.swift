import SwiftUI
import TurboSpark

/// The `$HF_ENDPOINT` mirror field inside the Hugging Face authentication
/// card. Its own view so the card stays under the 400-line guideline; the
/// mirror's state never crossed into the token's, so nothing is shared.
// Isolated explicitly: its helpers call each other, and only `body` is
// isolated by the protocol on the macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct HfMirrorEndpointRowView: View {
    @ObservedObject var model: AppModel

    @State private var mirrorEndpointInput: String = ""
    @State private var savedMirrorEndpoint: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Mirror Endpoint ($HF_ENDPOINT)", bundle: .module)
                    .settingsControl("Mirror Endpoint ($HF_ENDPOINT)", pane: .models, timing: .action)
                    .themedFont(.small, weight: .medium)
                Spacer()
                if !savedMirrorEndpoint.isEmpty && savedMirrorEndpoint != "https://huggingface.co" {
                    Button {
                        resetMirrorEndpoint()
                    } label: { Text("Reset to Default", bundle: .module) }
                    .appAction(.quiet, size: .small)
                    .appPointerCursor()
                }
            }
            HStack(spacing: 8) {
                // The placeholder is the default URL itself, which has nothing
                // to translate; the label is the field's VoiceOver name.
                TextField(text: $mirrorEndpointInput, prompt: Text(verbatim: "https://huggingface.co")) {
                    Text("Mirror Endpoint ($HF_ENDPOINT)", bundle: .module)
                }
                .textFieldStyle(.plain)
                .themedCode(.base)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                        .stroke(.appBorder, lineWidth: 1)
                )

                Button {
                    saveMirrorEndpoint()
                } label: { Text("Save Mirror", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(mirrorEndpointInput.trimmingCharacters(in: .whitespacesAndNewlines) == savedMirrorEndpoint)
                .help(mirrorEndpointInput.trimmingCharacters(in: .whitespacesAndNewlines) == savedMirrorEndpoint
                    ? Text("The mirror endpoint is unchanged.", bundle: .module)
                    : Text("Save Mirror", bundle: .module))
                .appPointerCursor()
            }
            Text("Useful for regions with restricted Hugging Face access, e.g. https://hf-mirror.com", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
        .onAppear(perform: loadMirrorEndpoint)
    }

    private func loadMirrorEndpoint() {
        savedMirrorEndpoint = HfEndpointResolution.effectiveEndpoint(from: model.hfEndpointInput)
            ?? HfEndpointResolution.defaultEndpoint
        mirrorEndpointInput = savedMirrorEndpoint
    }

    private func saveMirrorEndpoint() {
        // Reject (rather than silently reset) a mirror the token must not be sent to.
        if HfEndpointResolution.isRejectedInput(mirrorEndpointInput) {
            let reason = AppStrings.localized("use an https:// URL (http only for localhost)")
            model.showToast(AppStrings.localized("Failed to set mirror endpoint: \(reason)"), style: .error)
            return
        }
        let effective = HfEndpointResolution.effectiveEndpoint(from: mirrorEndpointInput)
            ?? HfEndpointResolution.defaultEndpoint
        model.hfEndpointInput = effective
        savedMirrorEndpoint = effective
        do {
            try TurboSparkCatalog.setHfEndpoint(HfEndpointResolution.effectiveEndpoint(from: effective))
            model.showToast(AppStrings.localized("Mirror endpoint updated"), style: .success)
        } catch {
            model.showToast(
                AppStrings.localized("Failed to set mirror endpoint: \(error.localizedDescription)"),
                style: .error)
        }
    }

    private func resetMirrorEndpoint() {
        model.hfEndpointInput = HfEndpointResolution.defaultEndpoint
        mirrorEndpointInput = HfEndpointResolution.defaultEndpoint
        savedMirrorEndpoint = HfEndpointResolution.defaultEndpoint
        do {
            try TurboSparkCatalog.setHfEndpoint(nil)
            model.showToast(AppStrings.localized("Mirror endpoint reset to default"), style: .info)
        } catch {
            model.showToast(
                AppStrings.localized("Failed to reset mirror endpoint: \(error.localizedDescription)"),
                style: .error)
        }
    }
}
