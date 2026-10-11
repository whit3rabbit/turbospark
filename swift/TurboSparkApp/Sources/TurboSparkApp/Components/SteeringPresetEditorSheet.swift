import SwiftUI
import TurboSpark
import UniformTypeIdentifiers

// Isolated explicitly for `swift/CLAUDE.md` Gotcha 45's reason.
/// Registers or edits one steering direction.
///
/// Picking a file READS ITS HEADER through the engine's own parser
/// (`TurboSparkCatalog.controlVectorInfo`), so the sheet can say what the
/// vector actually covers and whether its shape fits the selected model
/// before anything is loaded. This app never parses the GGUF itself: a second
/// reader would be free to disagree with the one the open uses.
@MainActor
struct SteeringPresetEditorSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var draft: AppSteeringPreset
    @State private var vectorInfo: ControlVectorInfo?
    @State private var readError: String?
    // Set once the user touches the Mode picker; from then on the file's
    // declared mode must not overwrite the explicit choice.
    @State private var modeTouched = false

    private let onSave: (AppSteeringPreset) -> Void

    init(
        model: AppModel,
        preset: AppSteeringPreset,
        onSave: @escaping (AppSteeringPreset) -> Void
    ) {
        self.model = model
        self._draft = State(initialValue: preset)
        self.onSave = onSave
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            Form {
                identitySection
                vectorSection
                editSection
            }
            .formStyle(.grouped)
            Divider()
            footer
        }
        .frame(width: 560, height: 620)
        .onAppear { readVector(applyDeclaredMode: false) }
    }

    private var header: some View {
        HStack {
            Group {
                if draft.name.isEmpty {
                    Text("Add a direction", bundle: .module)
                } else {
                    Text("Edit direction", bundle: .module)
                }
            }
            .themedFont(.base, weight: .semibold)
            .accessibilityAddTraits(.isHeader)
            Spacer()
        }
        .padding()
    }

    private var identitySection: some View {
        Section(header: Text("Identity", bundle: .module)) {
            TextField(text: $draft.name) {
                Text("Name", bundle: .module)
            }
            .help(Text("What this direction encodes. Only you know: nothing here inspects it.", bundle: .module))
            TextField(text: $draft.notes, axis: .vertical) {
                Text("Notes", bundle: .module)
            }
            .lineLimit(2...4)
        }
    }

    private var vectorSection: some View {
        Section(header: Text("Control vector", bundle: .module)) {
            HStack {
                TextField(text: $draft.vectorPath) {
                    Text("Path to a .gguf control vector", bundle: .module)
                }
                .truncationMode(.head)
                Button { pickFile() } label: { Text("Choose...", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
            .onChange(of: draft.vectorPath) { _, _ in readVector() }

            if let info = vectorInfo {
                // What the FILE says, so a mismatch is visible as two numbers
                // rather than as a failed load minutes later.
                LabeledContent {
                    Text(verbatim: "\(info.hidden)")
                } label: {
                    Text("Width", bundle: .module)
                }
                LabeledContent {
                    coverageText(info)
                } label: {
                    Text("Blocks", bundle: .module)
                }
                if let declared = info.declaredArch {
                    LabeledContent {
                        // An architecture identifier from the file header.
                        Text(verbatim: declared)
                    } label: {
                        Text("Declared for", bundle: .module)
                    }
                    .help(Text("Advisory only: nothing validates against this, which is exactly why it is worth reading.", bundle: .module))
                }
            }
            if let readError {
                Label { Text(verbatim: readError) } icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.small)
                    .foregroundStyle(.red)
            }

            let compatibility = AppSteeringPolicy.compatibility(
                preset: draft,
                modelHidden: model.selectedModelHiddenSize,
                modelLayers: model.selectedModelLayerCount
            )
            Label { SteeringText.compatibility(compatibility) } icon: { Image(systemName: "info.circle") }
                .themedFont(.small)
                .foregroundStyle(compatibility.allowsEnabling ? Color.secondary : Color.red)
        }
    }

    private var editSection: some View {
        Section(header: Text("Edit", bundle: .module)) {
            Picker(selection: Binding(
                get: { draft.mode },
                set: { draft.mode = $0; modeTouched = true }
            )) {
                ForEach(AppSteeringModeOption.allCases) { mode in
                    SteeringText.mode(mode).tag(mode)
                }
            } label: { Text("Mode", bundle: .module) }
            HStack {
                Text("Strength", bundle: .module)
                Slider(value: $draft.scale, in: -2...2, step: 0.05)
                    .accessibilityLabel(Text("Strength", bundle: .module))
                Text(draft.scale, format: .number.precision(.fractionLength(2)))
                    .monospacedDigit()
                    .frame(width: 48, alignment: .trailing)
            }
            // The measured warning, stated where the number is set rather
            // than in a doc nobody opens. `docs/OBLITERATION.md`: ablating at
            // strength 1.0 over every layer collapsed the turn on a real 27B
            // install -- immediate end-of-turn, no text -- and the usable
            // band was measured at 0.4 on one direction and 0.8 on another.
            // A band read off one direction is not an operating point for
            // another, which is why this warns rather than clamping.
            if abs(draft.scale) > 0.8 {
                Label {
                    Text("Measured usable strengths on real installs ran 0.4 to 0.8, and full ablation over every layer collapsed the turn entirely. Restrict the layer band, or lower this.", bundle: .module)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .themedFont(.small)
                .foregroundStyle(.orange)
            }
            TextField(text: $draft.layers) {
                Text("Layers, e.g. 20:45 (blank = every covered block)", bundle: .module)
            }
            if draft.mode == .clamp {
                TextField(value: $draft.target, format: .number) {
                    Text("Clamp target", bundle: .module)
                }
            }
            TextField(value: $draft.gate, format: .number) {
                Text("Gate threshold (0 always fires)", bundle: .module)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            Spacer()
            Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
            Button {
                onSave(draft)
                dismiss()
            } label: { Text("Save", bundle: .module) }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
            .disabled(draft.vectorPath.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding()
    }

    private func coverageText(_ info: ControlVectorInfo) -> Text {
        // The SPAN is what a layer-count check compares against, and the
        // range is what tells a reader that a missing block 0 is the
        // convention rather than a gap.
        if let low = info.minLayer, let high = info.maxLayer {
            return Text("\(info.coveredLayers) of \(info.spannedLayers) (\(low) to \(high))", bundle: .module)
        }
        return Text("\(info.coveredLayers) of \(info.spannedLayers)", bundle: .module)
    }

    private func readVector(applyDeclaredMode: Bool = true) {
        let result = AppModel.readingVectorShape(
            into: draft, applyDeclaredMode: applyDeclaredMode && !modeTouched)
        draft = result.preset
        vectorInfo = result.info
        readError = result.error
    }

    private func pickFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]
        if panel.runModal() == .OK, let url = panel.url {
            draft.vectorPath = url.path
        }
    }
}
