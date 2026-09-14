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
        .onAppear { readVector() }
    }

    private var header: some View {
        HStack {
            Text(draft.name.isEmpty ? "Add a direction" : "Edit direction")
                .themedFont(.base, weight: .semibold)
            Spacer()
        }
        .padding()
    }

    private var identitySection: some View {
        Section(header: Text("Identity", bundle: .module)) {
            TextField("Name", text: $draft.name)
                .help("What this direction encodes. Only you know: nothing here inspects it.")
            TextField("Notes", text: $draft.notes, axis: .vertical)
                .lineLimit(2...4)
        }
    }

    private var vectorSection: some View {
        Section(header: Text("Control vector", bundle: .module)) {
            HStack {
                TextField("Path to a .gguf control vector", text: $draft.vectorPath)
                    .truncationMode(.head)
                Button { pickFile() } label: { Text("Choose...", bundle: .module) }
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
                    Text(coverageText(info))
                } label: {
                    Text("Blocks", bundle: .module)
                }
                if let declared = info.declaredArch {
                    LabeledContent {
                        Text(declared)
                    } label: {
                        Text("Declared for", bundle: .module)
                    }
                        .help(
                            "Advisory only: nothing validates against this, which is exactly "
                                + "why it is worth reading.")
                }
            }
            if let readError {
                Label(readError, systemImage: "exclamationmark.triangle")
                    .themedFont(.small)
                    .foregroundStyle(.red)
            }

            let compatibility = AppSteeringPolicy.compatibility(
                preset: draft,
                modelHidden: model.selectedModelHiddenSize,
                modelLayers: model.selectedModelLayerCount
            )
            Label(compatibility.summary, systemImage: "info.circle")
                .themedFont(.small)
                .foregroundStyle(compatibility.allowsEnabling ? Color.secondary : Color.red)
        }
    }

    private var editSection: some View {
        Section(header: Text("Edit", bundle: .module)) {
            Picker(selection: $draft.mode) {
                ForEach(AppSteeringModeOption.allCases) { mode in
                    Text(mode.menuLabel).tag(mode)
                }
            } label: { Text("Mode", bundle: .module) }
            HStack {
                Text("Strength", bundle: .module)
                Slider(value: $draft.scale, in: -2...2, step: 0.05)
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
                Label(
                    "Measured usable strengths on real installs ran 0.4 to 0.8, and full "
                        + "ablation over every layer collapsed the turn entirely. Restrict the "
                        + "layer band, or lower this.",
                    systemImage: "exclamationmark.triangle"
                )
                .themedFont(.small)
                .foregroundStyle(.orange)
            }
            TextField("Layers, e.g. 20:45 (blank = every covered block)", text: $draft.layers)
            if draft.mode == .clamp {
                TextField("Clamp target", value: $draft.target, format: .number)
            }
            TextField("Gate threshold (0 always fires)", value: $draft.gate, format: .number)
        }
    }

    private var footer: some View {
        HStack {
            Spacer()
            Button { dismiss() } label: { Text("Cancel", bundle: .module) }
            Button {
                onSave(draft)
                dismiss()
            } label: { Text("Save", bundle: .module) }
            .keyboardShortcut(.defaultAction)
            .disabled(draft.vectorPath.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding()
    }

    private func coverageText(_ info: ControlVectorInfo) -> String {
        // The SPAN is what a layer-count check compares against, and the
        // range is what tells a reader that a missing block 0 is the
        // convention rather than a gap.
        let range: String
        if let low = info.minLayer, let high = info.maxLayer {
            range = " (\(low) to \(high))"
        } else {
            range = ""
        }
        return "\(info.coveredLayers) of \(info.spannedLayers)\(range)"
    }

    private func readVector() {
        let result = AppModel.readingVectorShape(into: draft)
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
