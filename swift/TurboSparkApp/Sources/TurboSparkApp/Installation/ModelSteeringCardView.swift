import SwiftUI
import TurboSpark

/// Steering controls attached to the model the user is inspecting.
///
/// The existing Settings pane remains the expert surface. This card is the
/// short path: choose the model, import a vector, enable it, then reload.
struct ModelSteeringCardView: View {
    @ObservedObject var model: AppModel
    let catalogEntry: CatalogEntry?
    let installedModel: InstalledModel?

    @State private var showingImport = false

    private var descriptor: ModelFeatureDescriptor {
        let sessionInfo: SessionInfo? = {
            guard let installedModel,
                  model.selected?.path == installedModel.path,
                  model.session != nil
            else { return nil }
            return model.info
        }()
        return ModelFeatureDescriptor.resolve(
            installedModel: installedModel,
            catalogEntry: catalogEntry,
            sessionInfo: sessionInfo)
    }

    private var targetHidden: Int? {
        descriptor.hiddenSize
    }

    private var targetLayers: Int? {
        descriptor.layerCount
    }

    private var currentPreset: AppSteeringPreset? {
        guard let id = model.activeSteeringPresetID else { return nil }
        return model.steeringPresets.first { $0.id == id }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            if !descriptor.isSteeringReady {
                Label(
                    "This model family does not support live steering.",
                    systemImage: "xmark.octagon.fill")
                    .foregroundStyle(.secondary)
            } else if installedModel == nil {
                notInstalledMessage
            } else {
                steeringContent
            }
        }
        .themedFont(.small)
        .modelCardStyle()
        .sheet(isPresented: $showingImport) {
            if let installedModel {
                SteeringVectorImportSheet(model: model, installedModel: installedModel)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Label {
                Text("Steering", bundle: .module)
                    .themedFont(.base, weight: .semibold)
            } icon: {
                Image(systemName: "dial.medium.fill")
                    .foregroundStyle(.appAccent)
            }
            Spacer()
            Button {
                model.openSettings(tab: .safety)
            } label: {
                Text("Open Settings", bundle: .module)
            }
            .buttonStyle(.link)
            .themedFont(.small)
        }
    }

    private var notInstalledMessage: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(verbatim: "Install this model first. Then this card can check the vector's shape against its manifest before enabling it.")
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(verbatim: "Live steering edits activations at runtime. The model does not need to be pre-abliterated, but a refusal vector should be extracted for this exact checkpoint.")
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var steeringContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            statusLine

            if model.steeringPresets.isEmpty {
                Text(verbatim: "No control vectors registered for this profile.")
                    .foregroundStyle(.appSecondary)
            } else {
                ForEach(model.steeringPresets) { preset in
                    presetRow(preset)
                }
            }

            Text(verbatim: "The selection is used when a model loads. A currently loaded session changes only after reload.")
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                Button {
                    showingImport = true
                } label: {
                    Label {
                        Text(verbatim: "Download from Hugging Face")
                    } icon: {
                        Image(systemName: "arrow.down.circle")
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(model.generating || model.opening)
                .accessibilityHint("Downloads a GGUF control vector, checks its shape, and selects it for this model")

                Text(verbatim: "or")
                    .foregroundStyle(.tertiary)

                Button {
                    model.openSettings(tab: .safety)
                } label: {
                    Text(verbatim: "Manage local vectors")
                }
                .buttonStyle(.link)
                .controlSize(.small)
            }

            Text(verbatim: "Shape is checked here. Meaning is not: a same-width vector from another checkpoint can still steer the wrong direction.")
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var statusLine: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if model.steeringEnabled, let currentPreset {
                Label {
                    Text(verbatim: "Enabled: " + currentPreset.displayName)
                } icon: {
                    Image(systemName: "checkmark.circle.fill")
                }
                .foregroundStyle(.green)
            } else {
                Label {
                    Text(verbatim: "Steering is off")
                } icon: {
                    Image(systemName: "circle")
                }
                .foregroundStyle(.appSecondary)
            }

            Spacer()

            if isCurrentModel && model.steeringNeedsReload {
                Button {
                    model.reloadForSteering()
                } label: {
                    Label {
                        Text(verbatim: "Reload to apply")
                    } icon: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(.orange)
            }
        }
    }

    private func presetRow(_ preset: AppSteeringPreset) -> some View {
        let compatibility = AppSteeringPolicy.compatibility(
            preset: preset,
            modelHidden: targetHidden,
            modelLayers: targetLayers)
        let isSelected = model.activeSteeringPresetID == preset.id
        let isEnabled = isSelected && model.steeringEnabled

        return HStack(alignment: .top, spacing: 8) {
            Image(systemName: isEnabled ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(isEnabled ? Color.green : Color.secondary)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 3) {
                Text(preset.displayName)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: preset.mode.menuLabel + " at " + String(format: "%.2g", preset.scale))
                    .foregroundStyle(.appSecondary)
                Text(compatibility.summary)
                    .foregroundStyle(compatibility.allowsEnabling ? Color.secondary : Color.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 8)

            Button {
                if isEnabled {
                    model.setSteeringEnabled(false)
                } else {
                    model.selectSteeringPreset(preset.id)
                    model.setSteeringEnabled(true)
                }
            } label: {
                Text(verbatim: isEnabled ? "Disable" : isSelected ? "Enable" : "Use")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(!descriptor.isSteeringReady || !compatibility.allowsEnabling)
            .help(compatibility.summary)
        }
        .padding(.vertical, 4)
    }

    private var isCurrentModel: Bool {
        guard let installedModel else { return false }
        return model.selected?.path == installedModel.path && model.session != nil
    }
}
