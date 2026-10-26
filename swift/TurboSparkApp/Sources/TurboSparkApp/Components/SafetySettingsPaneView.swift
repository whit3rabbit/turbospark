import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Directional steering: registered directions, which one is active, and
/// whether the loaded model is running it.
///
/// **NOTHING SHIPS A DIRECTION, AND THIS PANE SAYS SO IN ITS FIRST
/// PARAGRAPH.** The engine is agnostic to what a vector encodes -- the same
/// code path serves concept steering, style vectors, interpretability probes
/// and refusal-direction work -- so provenance is the operator's and the app
/// has no behaviour to advertise. A control labelled as though it shipped one
/// would be claiming work that does not exist (`swift/CLAUDE.md` Gotcha 23:
/// check the work exists before adding the control that claims to do it).
@MainActor
public struct SafetySettingsPaneView: View {
    @ObservedObject var model: AppModel

    @State private var showingEditor = false
    @State private var editingPreset: AppSteeringPreset?
    @State private var presetToDelete: AppSteeringPreset?
    @State private var isConfirmingDeletePreset = false

    public init(model: AppModel) {
        self.model = model
    }

    // Each section is its own property rather than an inline `Section` in one
    // `Form`: five inline sections is one expression and exceeds the macOS 14
    // SDK type-checker's budget (Gotcha 45).
    //
    // Order is purpose and selected state first, then what is loaded, then
    // the registry. The extraction and implementation notes that used to
    // open the pane sit behind "Details" at the end of the first section.
    public var body: some View {
        Form {
            activationSection
            statusSection
            presetsSection
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .sheet(isPresented: $showingEditor) {
            SteeringPresetEditorSheet(
                model: model,
                preset: editingPreset ?? AppSteeringPreset()
            ) { saved in
                model.upsertSteeringPreset(saved)
                if model.activeSteeringPresetID == nil {
                    model.selectSteeringPreset(saved.id)
                }
            }
        }
        .confirmationDialog(
            Text("Delete Preset", bundle: .module),
            isPresented: $isConfirmingDeletePreset,
            presenting: presetToDelete
        ) { preset in
            Button(role: .destructive) {
                model.deleteSteeringPreset(preset.id)
            } label: {
                Text("Delete '\(preset.displayName)'", bundle: .module)
            }
            Button(role: .cancel) {} label: {
                Text("Cancel", bundle: .module)
            }
        } message: { preset in
            Text("Are you sure you want to delete '\(preset.displayName)' from disk? This action cannot be undone.", bundle: .module)
        }
    }

    private var activationSection: some View {
        Section(header: Text("Activation", bundle: .module)) {
            // Resolved once: it walks the same compatibility check the
            // policy does, and both the tooltip and the row read it.
            let disabledReason = SteeringText.disabledReason(model)
            // The purpose line keeps the one claim this pane must make before
            // anything else: nothing ships a direction.
            Text("Steer how the loaded model writes with a control vector you supply. TurboSpark ships no directions.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            Toggle(isOn: steeringBinding) {
                Text("Apply steering when a model loads", bundle: .module)
            }
            .settingsControl("Apply steering when a model loads", pane: .safety, timing: .nextTurn)
            .disabled(model.steeringDisabledReason != nil)
            .help(disabledReason ?? Text("Applied at the next model load.", bundle: .module))

            Picker(selection: presetBinding) {
                Text("None", bundle: .module)
                    .settingsControl("None", pane: .safety, timing: .nextTurn).tag(UUID?.none)
                ForEach(model.steeringPresets) { preset in
                    Text(verbatim: preset.displayName).tag(UUID?.some(preset.id))
                }
            } label: { Text("Active direction", bundle: .module) }
            .settingsControl("Active direction", pane: .safety, timing: .nextTurn)
            .disabled(model.steeringPresets.isEmpty)

            if let reason = disabledReason {
                Label { reason } icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            // **STEERING RESOLVES ONCE, AT OPEN.** Without this note the
            // toggle looks like it did something and did not, which is the
            // exact failure this whole surface exists to avoid.
            if model.steeringNeedsReload {
                HStack {
                    Label {
                        Text("The loaded model is not running this. Steering is applied when a model opens.", bundle: .module)
                    } icon: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .themedFont(.small)
                    Spacer()
                    Button { model.reloadForSteering() } label: { Text("Reload model", bundle: .module) }
                        .appAction(.secondary, size: .small)
                        .disabled(model.selected == nil || model.generating || model.opening)
                }
            }

            DisclosureGroup {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Directional steering applies a rank-1 edit to the model's residual stream at every layer, while it generates. Nothing is written to the model files and the edit is reversible between two turns.", bundle: .module)
                    Text("TurboSpark ships no directions. A direction is a .gguf control vector in llama.cpp layout that you extract or supply yourself, and this engine has no opinion about what one encodes: the same code path serves concept steering, style vectors and refusal-direction work.", bundle: .module)
                        .foregroundStyle(.appSecondary)
                    Text("Extract one with scripts/extract_direction.py, or use a published set. docs/OBLITERATION.md has the measurements, including what happens at full strength.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                .fixedSize(horizontal: false, vertical: true)
                .padding(.vertical, 2)
            } label: {
                Text("Details", bundle: .module)
            }
        }
        .settingsControl("Activation", pane: .safety, timing: .nextTurn)
    }

    private var presetsSection: some View {
        Section(header: Text("Directions", bundle: .module)) {
            if model.steeringPresets.isEmpty {
                AppEmptyState(
                    systemImage: "arrow.triangle.swap",
                    title: Text("No directions registered.", bundle: .module),
                    message: Text("Directional steering applies a rank-1 edit to the model's residual stream at every layer, while it generates. Nothing is written to the model files and the edit is reversible between two turns.", bundle: .module)
                ) {
                    Button {
                        editingPreset = nil
                        showingEditor = true
                    } label: {
                        Label { Text("Add a direction", bundle: .module) } icon: { Image(systemName: "plus") }
                    }
                    .appAction(.primary, size: .small)
                }
            } else {
                ForEach(model.steeringPresets) { preset in
                    SteeringPresetRow(
                        preset: preset,
                        compatibility: AppSteeringPolicy.compatibility(
                            preset: preset,
                            modelHidden: model.selectedModelHiddenSize,
                            modelLayers: model.selectedModelLayerCount
                        ),
                        onEdit: {
                            editingPreset = preset
                            showingEditor = true
                        },
                        onDelete: {
                            presetToDelete = preset
                            isConfirmingDeletePreset = true
                        }
                    )
                }
                Button {
                    editingPreset = nil
                    showingEditor = true
                } label: {
                    Label { Text("Add a direction", bundle: .module) } icon: { Image(systemName: "plus") }
                }
                .appAction(.secondary, size: .small)
            }
        }
        .settingsControl("Directions", pane: .safety, timing: .nextTurn)
    }

    /// What the LOADED session reports, read back rather than restated. The
    /// engine's own summary line is the authority on what is running.
    private var statusSection: some View {
        Section(header: Text("Loaded model", bundle: .module)) {
            if model.session == nil {
                Text("No model loaded.", bundle: .module)
                    .settingsControl("No model loaded.", pane: .safety, timing: .nextTurn).foregroundStyle(.appSecondary)
            } else if let summary = model.activeSteeringSummary {
                LabeledContent {
                    // The engine's own summary line, read back verbatim.
                    Text(verbatim: summary)
                } label: {
                    Text("Steering", bundle: .module)
                }
            } else if model.steeringFamilySupported == false {
                LabeledContent {
                    if let reason = model.info?.steering.reason {
                        Text(verbatim: reason)
                    } else {
                        Text("Not supported by this model's family", bundle: .module)
                    }
                } label: {
                    Text("Steering", bundle: .module)
                }
            } else {
                LabeledContent {
                    Text("Off", bundle: .module)
                } label: {
                    Text("Steering", bundle: .module)
                }
            }
        }
    }

    private var steeringBinding: Binding<Bool> {
        Binding(
            get: { model.steeringEnabled },
            set: { model.setSteeringEnabled($0) }
        )
    }

    private var presetBinding: Binding<UUID?> {
        Binding(
            get: { model.activeSteeringPresetID },
            set: { model.selectSteeringPreset($0) }
        )
    }
}

// Isolated explicitly for Gotcha 45's reason.
/// One registered direction, with its shape checked against the selected
/// model.
@MainActor
struct SteeringPresetRow: View {
    let preset: AppSteeringPreset
    let compatibility: AppSteeringPolicy.Compatibility
    let onEdit: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(verbatim: preset.displayName).fontWeight(.medium)
                Spacer()
                Button(action: onEdit) { Text("Edit", bundle: .module) }
                    .appAction(.quiet, size: .small)
                Button(role: .destructive, action: onDelete) { Text("Delete", bundle: .module) }
                    .appAction(.destructive, size: .small)
            }
            detailLine
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Label { SteeringText.compatibility(compatibility) } icon: { Image(systemName: compatibilityIcon) }
                .themedFont(.small)
                .foregroundStyle(compatibility.allowsEnabling ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(AppAttention.color))
        }
        .padding(.vertical, 2)
    }

    /// Mode, strength, layer band and width, joined by a separator. Each
    /// part is its own catalog key so a translation can reorder within it.
    private var detailLine: Text {
        let scale = preset.scale.formatted(.number.precision(.fractionLength(0...2)))
        var parts: [Text] = [SteeringText.mode(preset.mode), Text("scale \(scale)", bundle: .module)]
        if !preset.layers.isEmpty { parts.append(Text("layers \(preset.layers)", bundle: .module)) }
        if let hidden = preset.vectorHidden { parts.append(Text("\(hidden) wide", bundle: .module)) }
        return parts.dropFirst().reduce(parts[0]) { $0 + Text(verbatim: " | ") + $1 }
    }

    private var compatibilityIcon: String {
        switch compatibility {
        case .shapeMatches: return "checkmark.circle"
        case .unknown: return "questionmark.circle"
        case .widthMismatch, .layerOverrun, .noVector: return "exclamationmark.triangle"
        }
    }
}
