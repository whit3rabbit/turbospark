import SwiftUI
import TurboSpark

/// Extracts one built-in trait direction from an installed model, on this
/// Mac, through the `turbospark-steer` CLI, and turns the result into the
/// app's named preset.
///
/// The display copy for a trait comes from `SteeringTrait`, not the CLI's
/// report: a Rust-side pack without app copy would otherwise ship English
/// under every language, so unknown pack ids are hidden rather than
/// rendered.
@MainActor
struct SteeringTraitExtractSheet: View {
    @ObservedObject var model: AppModel
    let installedModel: InstalledModel
    /// The trait to preselect, when the sheet is opened for one (the
    /// composer's steering menu). Nil preselects the first shown trait.
    var initialTraitID: String? = nil
    @Environment(\.dismiss) private var dismiss

    @State private var reports: [SteeringTraitPackReport] = []
    @State private var loadError: String?
    @State private var selectedID: String?
    @State private var scaleText = ""
    @State private var extractTask: Task<Void, Never>?
    @State private var errorMessage: String?
    /// Set on the first Extract press. Missing input is flagged only after
    /// the user tried to go on (DESIGN.md, action hierarchy).
    @State private var attempted = false
    @StateObject private var controller = SteeringTraitExtractionController()

    private var descriptor: ModelFeatureDescriptor {
        ModelFeatureDescriptor.resolve(installedModel: installedModel)
    }

    /// The CLI's reports joined with this build's copy. A pack id the app
    /// does not carry copy for is left out, on the reasoning in the type doc.
    private var traits: [(report: SteeringTraitPackReport, trait: SteeringTrait)] {
        reports.compactMap { report in
            SteeringTrait(rawValue: report.id).map { (report, $0) }
        }
    }

    private var selectedReport: SteeringTraitPackReport? {
        reports.first { $0.id == selectedID }
    }

    /// The field shows the pack's scale with a dot ("0.3"); a user in a
    /// comma-decimal locale types "0,3". Both parse.
    private var parsedScale: Double? {
        let text = scaleText.trimmingCharacters(in: .whitespaces)
        let formatter = NumberFormatter()
        formatter.locale = .current
        formatter.numberStyle = .decimal
        guard let value = Double(text) ?? formatter.number(from: text)?.doubleValue,
            value.isFinite
        else { return nil }
        return abs(value) <= 2 ? value : nil
    }

    /// Extraction opens its own copy of the model in a child process, twice
    /// (one open per side). With the same install resident in the app that
    /// is two copies at once, which a large model on a unified-memory Mac
    /// cannot afford, so the run waits for an unload.
    private var isLoadedHere: Bool {
        model.session != nil && model.selected?.path == installedModel.path
    }

    private var validationError: String? {
        guard selectedReport != nil else {
            return AppStrings.localized("Choose a trait first.")
        }
        guard parsedScale != nil else {
            return AppStrings.localized("Strength must be between -2 and 2.")
        }
        return nil
    }

    /// Why Extract cannot run right now, from the app's own state. Nil when
    /// only input is missing (that is flagged on the press instead).
    private var blockingReason: Text? {
        if isRunning { return Text("Extracting...", bundle: .module) }
        if !descriptor.isSteeringReady {
            return Text("This model family does not support live steering.", bundle: .module)
        }
        if isLoadedHere {
            return Text(
                "This model is loaded. Unload it before extracting; extraction opens its own copy of the model.",
                bundle: .module)
        }
        if model.generating || model.opening {
            return Text("Wait for the current generation or model load to finish.", bundle: .module)
        }
        return nil
    }

    private var isRunning: Bool {
        if case .capturing = controller.phase { return true }
        if case .finishing = controller.phase { return true }
        return false
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            form
            Divider()
            footer
        }
        .frame(minWidth: 560, minHeight: 480)
        .onAppear { loadPacks() }
        .onDisappear {
            extractTask?.cancel()
            controller.cancel()
        }
    }

    private var header: some View {
        AppAdaptiveHeader {
            VStack(alignment: .leading, spacing: 3) {
                Text("Extract steering for \(installedModel.alias)", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text(
                    "Capture this model's own activations against a built-in trait and turn them into a steering preset. Nothing is downloaded and no weights are modified.",
                    bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } actions: {
            Button { dismiss() } label: {
                Text("Done", bundle: .module)
            }
            .appAction(.secondary)
        }
        .padding(16)
    }

    private var form: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if !descriptor.isSteeringReady {
                    Label {
                        Text("This model family does not support live steering.", bundle: .module)
                    } icon: {
                        Image(systemName: "xmark.octagon.fill")
                    }
                    .foregroundStyle(.appDanger)
                }

                if isLoadedHere {
                    HStack(alignment: .top, spacing: 8) {
                        Label {
                            Text(
                                "This model is loaded. Unload it before extracting; extraction opens its own copy of the model.",
                                bundle: .module)
                                .fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "memorychip")
                        }
                        .foregroundStyle(AppAttention.color)
                        Spacer(minLength: 8)
                        Button {
                            model.unloadModel()
                        } label: {
                            Text("Unload Model", bundle: .module)
                        }
                        .appAction(.secondary, size: .small)
                        .disabled(!model.canUnloadModel)
                    }
                }

                traitSection

                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Strength", bundle: .module)
                            Spacer()
                            TextField(text: $scaleText, prompt: Text(verbatim: scalePlaceholder)) {
                                Text("Strength", bundle: .module)
                            }
                                .textFieldStyle(.roundedBorder)
                                .frame(minWidth: 70, idealWidth: 90, maxWidth: 120)
                        }
                        Text(
                            "Start at the trait's recommended 0.3. Past 0.6 an ablation can collapse generation entirely.",
                            bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if isRunning, case let .capturing(fraction) = controller.phase {
                    AppProgressBar(
                        value: fraction,
                        label: Text("Capturing prompts...", bundle: .module))
                }

                Text(
                    "The direction is extracted from this exact checkpoint on this Mac. Shape is checked; meaning is judged by reading paired responses afterwards.",
                    bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(16)
        }
    }

    private var traitSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Text("Trait", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                if let loadError {
                    Label(loadError, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.appDanger)
                        .fixedSize(horizontal: false, vertical: true)
                } else if reports.isEmpty {
                    Text("Looking for the steer tool...", bundle: .module)
                        .foregroundStyle(.appSecondary)
                } else {
                    ForEach(traits, id: \.report.id) { trait in
                        traitRow(trait)
                    }
                }
            }
        }
    }

    private func traitRow(
        _ trait: (report: SteeringTraitPackReport, trait: SteeringTrait)
    ) -> some View {
        let isSelected = selectedID == trait.report.id
        return Button {
            guard !isRunning else { return }
            selectedID = trait.report.id
            scaleText = Self.format(trait.report.recommendedScale)
        } label: {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(
                        isSelected ? AnyShapeStyle(.appSuccess) : AnyShapeStyle(.appSecondary))
                    .padding(.top, 2)
                VStack(alignment: .leading, spacing: 3) {
                    Text(LocalizedStringKey(trait.trait.title), bundle: .module)
                        .themedFont(.small, weight: .medium)
                        .foregroundStyle(.primary)
                    Text(LocalizedStringKey(trait.trait.summary), bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    // The mode goes through its own localized menu label;
                    // the CLI's raw spelling ("ablate") is never shown.
                    (Text(LocalizedStringKey(trait.report.presetMode.menuLabel), bundle: .module)
                        + Text(verbatim: " | ")
                        + Text(
                            "\(trait.report.positivePrompts + trait.report.negativePrompts) prompts, recommended strength \(Self.format(trait.report.recommendedScale))",
                            bundle: .module))
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer(minLength: 8)
            }
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
        .disabled(isRunning)
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.appDanger)
                    .fixedSize(horizontal: false, vertical: true)
            } else if attempted, let validationError {
                Label(validationError, systemImage: "info.circle")
                    .foregroundStyle(AppAttention.color)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Spacer()
                Button { dismiss() } label: {
                    Text("Cancel", bundle: .module)
                }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)
                Button {
                    startExtraction()
                } label: {
                    if isRunning {
                        ProgressView()
                            .controlSize(.small)
                        Text("Extracting...", bundle: .module)
                    } else {
                        Label {
                            Text("Extract", bundle: .module)
                        } icon: {
                            Image(systemName: "wand.and.stars")
                        }
                    }
                }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
                // Missing input (trait, strength) keeps the button live and
                // is flagged on the press; only the app's own state disables
                // it, and the tooltip names that state.
                .disabled(blockingReason != nil)
                .help(blockingReason ?? Text("Extract", bundle: .module))
            }
        }
        .padding(16)
    }

    // MARK: actions

    private func loadPacks() {
        guard let cli = SteeringTraitTool.findSteerCLI() else {
            loadError = AppStrings.localized(
                "The steer tool could not be found beside the app or on PATH.")
            return
        }
        Task.detached(priority: .userInitiated) {
            let result = SteeringTraitExtractionController.listPacks(cliURL: cli)
            await MainActor.run {
                switch result {
                case let .success(packs):
                    reports = packs
                    // Preselect the first trait the sheet SHOWS (the refusal
                    // pack, the trait the steering surface exists for): a
                    // pack id without app copy is hidden and must not be the
                    // invisible selection.
                    if let first = traits.first(where: { $0.report.id == initialTraitID })?.report
                        ?? traits.first?.report
                    {
                        selectedID = first.id
                        scaleText = Self.format(first.recommendedScale)
                    }
                case let .failure(error):
                    loadError = (error as? SteeringTraitError)?.errorDescription
                        ?? error.localizedDescription
                }
            }
        }
    }

    private func startExtraction() {
        attempted = true
        guard let report = selectedReport, let scale = parsedScale, !isRunning else { return }
        errorMessage = nil
        let output = SteeringTraitPresetFactory.outputURL(
            modelAlias: installedModel.alias, traitID: report.id)
        let alias = installedModel.alias
        let traitID = report.id
        let traitName = localizedTraitName(traitID)
        let modelPath = installedModel.path
        extractTask = Task {
            do {
                guard let cli = SteeringTraitTool.findSteerCLI() else {
                    throw SteeringTraitError.toolFailed(
                        status: 127,
                        message: AppStrings.localized(
                            "The steer tool could not be found beside the app or on PATH."))
                }
                try await controller.extract(
                    cliURL: cli,
                    modelPath: modelPath,
                    traitID: traitID,
                    outputURL: output)
                let preset = SteeringTraitPresetFactory.makePreset(
                    traitID: traitID,
                    traitName: traitName,
                    report: report,
                    scale: scale,
                    outputURL: output,
                    modelAlias: alias,
                    existing: model.steeringPresets)
                model.upsertSteeringPreset(preset)
                model.selectSteeringPreset(preset.id)
                model.setSteeringEnabled(true)
                model.showToast(
                    AppStrings.localized(
                        "Direction extracted and selected. Reload the model to apply steering."),
                    style: .success,
                    duration: 7.0)
                dismiss()
            } catch is CancellationError {
                // Backed out mid-run: change nothing.
            } catch {
                errorMessage = (error as? SteeringTraitError)?.errorDescription
                    ?? error.localizedDescription
            }
        }
    }

    private var scalePlaceholder: String {
        Self.format(selectedReport?.recommendedScale ?? 0.3)
    }

    /// Two significant digits with a dot, the spelling the CLI and the
    /// docs use; `parsedScale` accepts the locale's decimal mark too.
    static func format(_ scale: Double) -> String {
        String(format: "%.2g", scale)
    }

    /// The preset's display name, localized at creation time like every
    /// other persisted-but-human string in this surface. An unknown trait
    /// falls back to its id, which is still a name the user chose to run.
    private func localizedTraitName(_ id: String) -> String {
        SteeringTrait(rawValue: id)?.localizedTitle ?? id
    }
}
