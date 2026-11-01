import SwiftUI
import TurboSpark

/// Extracts one built-in trait direction from an installed model, on this
/// Mac, through the `turbospark-steer` CLI, and turns the result into the
/// app's named preset.
///
/// The display copy for a trait lives HERE as localized literals keyed by
/// the pack id, not in the CLI's report: a Rust-side pack without app copy
/// would otherwise ship English under every language, so unknown pack ids
/// are hidden rather than rendered.
@MainActor
struct SteeringTraitExtractSheet: View {
    @ObservedObject var model: AppModel
    let installedModel: InstalledModel
    @Environment(\.dismiss) private var dismiss

    @State private var reports: [SteeringTraitPackReport] = []
    @State private var loadError: String?
    @State private var selectedID: String?
    @State private var scaleText = ""
    @State private var extractTask: Task<Void, Never>?
    @State private var errorMessage: String?
    @StateObject private var controller = SteeringTraitExtractionController()

    private var descriptor: ModelFeatureDescriptor {
        ModelFeatureDescriptor.resolve(installedModel: installedModel)
    }

    /// The CLI's reports joined with this build's localized copy. A pack id
    /// the app does not carry copy for is left out, on the reasoning in the
    /// type doc.
    private var traits: [(report: SteeringTraitPackReport, title: LocalizedStringKey, summary: LocalizedStringKey)] {
        reports.compactMap { report in
            switch report.id {
            case "refusals":
                return (
                    report,
                    "Refusal suppression",
                    "Steers away from refusals by contrasting refusal-prone prompts with ordinary requests."
                )
            case "slop":
                return (
                    report,
                    "Slop reduction",
                    "Steers away from cliches and purple prose by contrasting forced-flowery and plain writing."
                )
            case "humor":
                return (
                    report,
                    "Humor reduction",
                    "Steers away from forced jokes by contrasting joke-seeking and informational requests."
                )
            case "sycophancy":
                return (
                    report,
                    "Sycophancy reduction",
                    "Steers away from praise openers by contrasting flattery-conditioned and factual answers."
                )
            case "emotions":
                return (
                    report,
                    "Emotional tone",
                    "Softens heavy negative affect by contrasting loss- and joy-framed writing."
                )
            default:
                return nil
            }
        }
    }

    private var selectedReport: SteeringTraitPackReport? {
        reports.first { $0.id == selectedID }
    }

    private var parsedScale: Double? {
        guard let value = Double(scaleText), value.isFinite else { return nil }
        return abs(value) <= 2 ? value : nil
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
            .keyboardShortcut(.cancelAction)
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

                traitSection

                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Ablate strength", bundle: .module)
                            Spacer()
                            TextField(text: $scaleText, prompt: Text(verbatim: "0.3")) {
                                Text("Ablate strength", bundle: .module)
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
        _ trait: (report: SteeringTraitPackReport, title: LocalizedStringKey, summary: LocalizedStringKey)
    ) -> some View {
        let isSelected = selectedID == trait.report.id
        return Button {
            guard !isRunning else { return }
            selectedID = trait.report.id
            scaleText = String(format: "%.2g", trait.report.recommendedScale)
        } label: {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(
                        isSelected ? AnyShapeStyle(.appSuccess) : AnyShapeStyle(.appSecondary))
                    .padding(.top, 2)
                VStack(alignment: .leading, spacing: 3) {
                    Text(trait.title, bundle: .module)
                        .themedFont(.small, weight: .medium)
                        .foregroundStyle(.primary)
                    Text(trait.summary, bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(
                        "\(trait.report.positivePrompts + trait.report.negativePrompts) prompts, recommended \(trait.report.recommendedMode) at \(String(format: "%.2g", trait.report.recommendedScale))",
                        bundle: .module)
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
            } else if let validationError {
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
                .disabled(
                    isRunning || validationError != nil || !descriptor.isSteeringReady
                        || model.generating || model.opening)
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
                    // Preselect the refusal pack, the trait the steering
                    // surface exists for, and its recommended strength.
                    selectedID = packs.first?.id
                    if let first = packs.first {
                        scaleText = String(format: "%.2g", first.recommendedScale)
                    }
                case let .failure(error):
                    loadError = (error as? SteeringTraitError)?.errorDescription
                        ?? error.localizedDescription
                }
            }
        }
    }

    private func startExtraction() {
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

    /// The preset's display name, localized at creation time like every
    /// other persisted-but-human string in this surface.
    private func localizedTraitName(_ id: String) -> String {
        switch id {
        case "refusals": return AppStrings.localized("Refusal suppression")
        case "slop": return AppStrings.localized("Slop reduction")
        case "humor": return AppStrings.localized("Humor reduction")
        case "sycophancy": return AppStrings.localized("Sycophancy reduction")
        case "emotions": return AppStrings.localized("Emotional tone")
        // An unknown trait falls back to its id, which is still a name the
        // user chose to run.
        default: return id
        }
    }
}
