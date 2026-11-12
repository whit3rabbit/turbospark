import SwiftUI
import TurboSpark

/// Directional steering in the composer, beside the guardrails pill: a menu
/// that turns steering off, picks a registered direction, or starts a
/// built-in trait extraction for the selected model.
///
/// **STEERING RESOLVES AT MODEL OPEN**, so the label states intent and says
/// when the loaded model is not running it yet, and the menu carries the
/// reload. A switch that silently did nothing until the next load is the
/// failure the whole surface exists to avoid.
///
/// Disabled with the engine's reason, never hidden, when the loaded model's
/// family does not dispatch steering (swift/CLAUDE.md Gotchas 23 and 33): a
/// missing control reads as a missing feature.
@MainActor
struct SteeringPillControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var extractRequest: SteeringExtractRequest?

    /// The family refusal, read off the LOADED session only. Before a load
    /// the menu stays usable: a selection is intent for the next open.
    private var familyRefusal: String? {
        guard model.steeringFamilySupported == false else { return nil }
        return model.info?.steering.reason
            ?? AppStrings.localized("This model family does not support live steering.")
    }

    var body: some View {
        let disabledReason = model.steeringDisabledReason
        let isOn = model.steeringEnabled && disabledReason == nil
        let pending = model.steeringNeedsReload

        HStack(spacing: 4) {
            Menu {
                menuContent(pending: pending)
            } label: {
                label(isOn: isOn, pending: pending)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .disabled(familyRefusal != nil)
            .help(familyRefusal ?? Self.helpText(
                disabledReason: disabledReason,
                isOn: isOn,
                needsReload: pending,
                summary: model.activeSteeringSummary
            ))
            .accessibilityLabel(Text("Directional steering", bundle: .module))
            .accessibilityValue(accessibilityValue(isOn: isOn))
            .accessibilityHint(familyRefusal.map { Text($0) }
                ?? Text("Applied when a model loads", bundle: .module))

            Button {
                model.openSettings(tab: .safety)
            } label: {
                Image(systemName: "info.circle")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(Text("Register and choose a steering direction. Click to open Settings.", bundle: .module))
            .accessibilityLabel(Text("Directional steering information", bundle: .module))
            .accessibilityHint(Text("Opens Safety and Steering settings", bundle: .module))
        }
        .sheet(item: $extractRequest) { request in
            SteeringTraitExtractSheet(
                model: model,
                installedModel: request.installedModel,
                initialTraitID: request.traitID)
        }
    }

    private func label(isOn: Bool, pending: Bool) -> some View {
        HStack(spacing: 3) {
            Image(systemName: isOn ? "dial.medium.fill" : "dial.medium")
                .font(theme.ui(.tiny, weight: .semibold))
                .foregroundStyle(isOn ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
            (isOn ? Text("Steering On", bundle: .module) : Text("Steering Off", bundle: .module))
                .font(theme.ui(.small, weight: .medium))
                .foregroundStyle(isOn ? AnyShapeStyle(.appText) : AnyShapeStyle(.appSecondary))
                .lineLimit(1)
            // The pending marker is not decoration: without it the pill
            // reads "On" while the loaded model is unsteered.
            if pending {
                Image(systemName: "arrow.clockwise")
                    .font(theme.ui(.micro, weight: .semibold))
                    .foregroundStyle(AppAttention.color)
            }
            Image(systemName: "chevron.down")
                .themedFont(.micro, weight: .bold)
                .foregroundStyle(.appSecondary)
        }
    }

    @ViewBuilder
    private func menuContent(pending: Bool) -> some View {
        Toggle(isOn: Binding(
            get: { !model.steeringEnabled },
            set: { if $0 { model.setSteeringEnabled(false) } }
        )) {
            Text("Off", bundle: .module)
        }

        let rows = SteeringComposerMenuModel.presetRows(
            presets: model.steeringPresets,
            activeID: model.activeSteeringPresetID,
            enabled: model.steeringEnabled,
            target: model.selected)
        if !rows.isEmpty {
            Section {
                ForEach(rows) { row in
                    Toggle(isOn: Binding(
                        get: { row.isActive },
                        set: { on in
                            guard on else { return }
                            model.selectSteeringPreset(row.id)
                            model.setSteeringEnabled(true)
                        }
                    )) {
                        // A preset name is user data (or a trait name
                        // localized when the preset was created).
                        Text(verbatim: row.name)
                    }
                    .disabled(!row.canEnable)
                    .help(row.blockReason.map { Text($0) } ?? Text(verbatim: row.name))
                }
            } header: {
                Text("Directions", bundle: .module)
            }
        }

        if let selected = model.selected {
            let traits = SteeringComposerMenuModel.unextractedTraits(
                presets: model.steeringPresets, alias: selected.alias)
            if !traits.isEmpty {
                Section {
                    ForEach(traits) { trait in
                        Button {
                            extractRequest = SteeringExtractRequest(
                                installedModel: selected, traitID: trait.rawValue)
                        } label: {
                            Text(LocalizedStringKey(trait.title), bundle: .module)
                        }
                    }
                } header: {
                    Text("Extract from this model", bundle: .module)
                }
            }
        }

        Divider()

        if pending {
            Button {
                model.reloadForSteering()
            } label: {
                Text("Reload to apply", bundle: .module)
            }
            .disabled(!model.canReloadModel)
        }
        Button {
            model.openSettings(tab: .safety)
        } label: {
            Text("Steering Settings...", bundle: .module)
        }
    }

    private func accessibilityValue(isOn: Bool) -> Text {
        guard isOn, let preset = model.resolvedSteeringPreset else {
            return Text("Off", bundle: .module)
        }
        return Text(verbatim: preset.displayName)
    }

    /// Pure, for the reason `ForgeGuardrailsPillControl.helpText` is, and
    /// nonisolated so a synchronous test can call it off the main actor.
    nonisolated static func helpText(
        disabledReason: String?,
        isOn: Bool,
        needsReload: Bool,
        summary: String?
    ) -> String {
        if let disabledReason { return disabledReason }
        if needsReload {
            return AppStrings.localized(
                "Reload the model to apply this. Steering is resolved when a model opens, so the one loaded now is unchanged.")
        }
        if isOn, let summary { return AppStrings.localized("Running: \(summary)") }
        if isOn { return AppStrings.localized("On at the next model load.") }
        return AppStrings.localized("Off. No edit is applied to the residual stream.")
    }
}

/// Which install and trait the extract sheet opens for.
struct SteeringExtractRequest: Identifiable {
    let installedModel: InstalledModel
    let traitID: String
    var id: String { installedModel.path + "|" + traitID }
}

/// The menu's rows, kept pure so the ordering and gating are testable
/// without a model (swift/CLAUDE.md Gotcha 26).
enum SteeringComposerMenuModel {
    struct PresetRow: Identifiable, Equatable {
        let id: UUID
        let name: String
        let isActive: Bool
        let canEnable: Bool
        let blockReason: String?
    }

    /// Every registered preset, checked against the selected install's
    /// shape through the same policy the open uses. A preset whose shape
    /// the engine would refuse is listed but cannot be picked.
    static func presetRows(
        presets: [AppSteeringPreset],
        activeID: UUID?,
        enabled: Bool,
        target: InstalledModel?
    ) -> [PresetRow] {
        let descriptor = target.map { ModelFeatureDescriptor.resolve(installedModel: $0) }
        return presets.map { preset in
            let compatibility = AppSteeringPolicy.compatibility(
                preset: preset,
                modelHidden: descriptor?.hiddenSize,
                modelLayers: descriptor?.layerCount)
            return PresetRow(
                id: preset.id,
                name: preset.displayName,
                isActive: enabled && preset.id == activeID,
                canEnable: compatibility.allowsEnabling,
                blockReason: compatibility.allowsEnabling ? nil : compatibility.summary)
        }
    }

    /// The built-in traits with no preset yet for this install, found by the
    /// deterministic output path the extraction writes. An extracted trait
    /// is already a row under Directions, so it is not offered twice.
    static func unextractedTraits(
        presets: [AppSteeringPreset], alias: String
    ) -> [SteeringTrait] {
        let paths = Set(presets.map {
            URL(fileURLWithPath: $0.vectorPath).standardizedFileURL.path
        })
        return SteeringTrait.allCases.filter { trait in
            let output = SteeringTraitPresetFactory.outputURL(
                modelAlias: alias, traitID: trait.rawValue)
            return !paths.contains(output.standardizedFileURL.path)
        }
    }
}
