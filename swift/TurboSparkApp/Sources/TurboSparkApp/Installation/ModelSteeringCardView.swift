import SwiftUI
import TurboSpark

/// Steering controls attached to the model the user is inspecting.
///
/// The existing Settings pane remains the expert surface. This card is the
/// short path: choose the model, import a vector, enable it, then reload.
struct ModelSteeringCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let catalogEntry: CatalogEntry?
    let installedModel: InstalledModel?

    @State private var showingImport = false
    @State private var showingExtract = false

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
                Label {
                    Text("This model family does not support live steering.", bundle: .module)
                } icon: {
                    Image(systemName: "xmark.octagon.fill")
                }
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
        .sheet(isPresented: $showingExtract) {
            if let installedModel {
                SteeringTraitExtractSheet(model: model, installedModel: installedModel)
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
            .appLinkTint()
            .themedFont(.small)
        }
    }

    private var notInstalledMessage: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Install this model first. Then this card can check the vector's shape against its manifest before enabling it.", bundle: .module)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Live steering edits activations at runtime. The model does not need to be pre-abliterated, but a refusal vector should be extracted for this exact checkpoint.", bundle: .module)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var steeringContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            statusLine

            if model.steeringPresets.isEmpty {
                Text("No control vectors registered for this profile.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            } else {
                ForEach(model.steeringPresets) { preset in
                    presetRow(preset)
                }
            }

            Text("The selection is used when a model loads. A currently loaded session changes only after reload.", bundle: .module)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                Button {
                    showingExtract = true
                } label: {
                    Label {
                        Text("Extract from this model", bundle: .module)
                    } icon: {
                        Image(systemName: "wand.and.stars")
                    }
                }
                .appAction(.secondary, size: .small)
                .disabled(model.generating || model.opening)
                .accessibilityHint(Text(
                    "Captures this model's own activations against a built-in trait and creates a steering preset from them",
                    bundle: .module))

                Button {
                    showingImport = true
                } label: {
                    Label {
                        Text("Download from Hugging Face", bundle: .module)
                    } icon: {
                        Image(systemName: "arrow.down.circle")
                    }
                }
                .appAction(.secondary, size: .small)
                .disabled(model.generating || model.opening)
                .accessibilityHint(Text(
                    "Downloads a GGUF control vector, checks its shape, and selects it for this model",
                    bundle: .module))

                Text(verbatim: "or")
                    .foregroundStyle(.tertiary)

                Button {
                    model.openSettings(tab: .safety)
                } label: {
                    Text("Manage local vectors", bundle: .module)
                }
                .buttonStyle(.link)
                .appLinkTint()
                .controlSize(.small)
            }

            Text("Shape is checked here. Meaning is not: a same-width vector from another checkpoint can still steer the wrong direction.", bundle: .module)
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
                .foregroundStyle(.appSuccess)
            } else {
                Label {
                    Text("Steering is off", bundle: .module)
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
                        Text("Reload to apply", bundle: .module)
                    } icon: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(theme.warning)
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
                .foregroundStyle(isEnabled ? AnyShapeStyle(.appSuccess) : AnyShapeStyle(.appSecondary))
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 3) {
                Text(preset.displayName)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: preset.mode.menuLabel + " at " + String(format: "%.2g", preset.scale))
                    .foregroundStyle(.appSecondary)
                Text(compatibility.summary)
                    .foregroundStyle(compatibility.allowsEnabling ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.appDanger))
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
            // Only gate enabling: an already-enabled preset must stay disableable
            // even when it does not fit the model being viewed.
            .disabled(!isEnabled && (!descriptor.isSteeringReady || !compatibility.allowsEnabling))
            .help(compatibility.summary)
        }
        .padding(.vertical, 4)
    }

    private var isCurrentModel: Bool {
        guard let installedModel else { return false }
        return model.selected?.path == installedModel.path && model.session != nil
    }
}
