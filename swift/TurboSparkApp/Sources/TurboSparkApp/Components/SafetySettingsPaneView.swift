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

    public init(model: AppModel) {
        self.model = model
    }

    // Each section is its own property rather than an inline `Section` in one
    // `Form`: five inline sections is one expression and exceeds the macOS 14
    // SDK type-checker's budget (Gotcha 45).
    public var body: some View {
        Form {
            explanationSection
            activationSection
            presetsSection
            statusSection
        }
        .formStyle(.grouped)
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
    }

    private var explanationSection: some View {
        Section("What this is") {
            VStack(alignment: .leading, spacing: 8) {
                Text(
                    "Directional steering applies a rank-1 edit to the model's residual stream "
                        + "at every layer, while it generates. Nothing is written to the model "
                        + "files and the edit is reversible between two turns."
                )
                Text(
                    "TurboSpark ships no directions. A direction is a .gguf control vector in "
                        + "llama.cpp layout that you extract or supply yourself, and this engine "
                        + "has no opinion about what one encodes: the same code path serves "
                        + "concept steering, style vectors and refusal-direction work."
                )
                .foregroundStyle(.appSecondary)
                Text(
                    "Extract one with scripts/extract_direction.py, or use a published set. "
                        + "docs/OBLITERATION.md has the measurements, including what happens at "
                        + "full strength."
                )
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
            .padding(.vertical, 2)
        }
    }

    private var activationSection: some View {
        Section("Activation") {
            Toggle("Apply steering when a model loads", isOn: steeringBinding)
            .settingsControl("Apply steering when a model loads", pane: .safety, timing: .nextTurn)
                .disabled(model.steeringDisabledReason != nil)
                .help(model.steeringDisabledReason ?? "Applied at the next model load.")

            Picker("Active direction", selection: presetBinding) {
                Text("None", bundle: .module)
                    .settingsControl("None", pane: .safety, timing: .nextTurn).tag(UUID?.none)
                ForEach(model.steeringPresets) { preset in
                    Text(preset.displayName).tag(UUID?.some(preset.id))
                }
            }
            .settingsControl("Active direction", pane: .safety, timing: .nextTurn)
            .disabled(model.steeringPresets.isEmpty)

            if let reason = model.steeringDisabledReason {
                Label(reason, systemImage: "exclamationmark.triangle")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            // **STEERING RESOLVES ONCE, AT OPEN.** Without this note the
            // toggle looks like it did something and did not, which is the
            // exact failure this whole surface exists to avoid.
            if model.steeringNeedsReload {
                HStack {
                    Label(
                        "The loaded model is not running this. Steering is applied when a model "
                            + "opens.",
                        systemImage: "arrow.clockwise"
                    )
                    .themedFont(.small)
                    Spacer()
                    Button("Reload model") { model.reloadForSteering() }
                        .disabled(model.selected == nil || model.generating || model.opening)
                }
            }
        }
            .settingsControl("Activation", pane: .safety, timing: .nextTurn)
    }

    private var presetsSection: some View {
        Section("Directions") {
            if model.steeringPresets.isEmpty {
                Text("No directions registered.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
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
                    onDelete: { model.deleteSteeringPreset(preset.id) }
                )
            }
            Button {
                editingPreset = nil
                showingEditor = true
            } label: {
                Label("Add a direction", systemImage: "plus")
            }
        }
            .settingsControl("Directions", pane: .safety, timing: .nextTurn)
    }

    /// What the LOADED session reports, read back rather than restated. The
    /// engine's own summary line is the authority on what is running.
    private var statusSection: some View {
        Section("Loaded model") {
            if model.session == nil {
                Text("No model loaded.", bundle: .module)
                    .settingsControl("No model loaded.", pane: .safety, timing: .nextTurn).foregroundStyle(.appSecondary)
            } else if let summary = model.activeSteeringSummary {
                LabeledContent("Steering", value: summary)
            } else if model.steeringFamilySupported == false {
                LabeledContent(
                    "Steering",
                    value: model.info?.steering.reason ?? "Not supported by this model's family")
            } else {
                LabeledContent("Steering", value: "Off")
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
                Text(preset.displayName).fontWeight(.medium)
                Spacer()
                Button("Edit", action: onEdit).buttonStyle(.link)
                Button("Delete", action: onDelete).buttonStyle(.link)
            }
            Text(detailLine)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Label(compatibility.summary, systemImage: compatibilityIcon)
                .themedFont(.small)
                .foregroundStyle(compatibility.allowsEnabling ? Color.secondary : Color.red)
        }
        .padding(.vertical, 2)
    }

    private var detailLine: String {
        var parts = [preset.mode.menuLabel, "scale \(preset.scale)"]
        if !preset.layers.isEmpty { parts.append("layers \(preset.layers)") }
        if let hidden = preset.vectorHidden { parts.append("\(hidden) wide") }
        return parts.joined(separator: " | ")
    }

    private var compatibilityIcon: String {
        switch compatibility {
        case .shapeMatches: return "checkmark.circle"
        case .unknown: return "questionmark.circle"
        case .widthMismatch, .layerOverrun, .noVector: return "exclamationmark.triangle"
        }
    }
}
