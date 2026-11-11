import SwiftUI
import TurboSpark

/// The detail pane for a Hugging Face result: the compatibility verdict, the
/// quantization, the local name, and the install.
///
/// It replaces the old flow's four manual steps. Selecting a row runs the
/// check (one explicit click, a few seconds of header reads, no download),
/// the alias is proposed instead of demanded, and a base-model tokenizer is
/// found without the user naming a sidecar repository.
@MainActor
struct HubResultDetailView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: HubSearchController
    let repoId: String
    /// `nil` when the user checked an exact repository instead of picking a row.
    let result: HubSearchResult?

    @Environment(\.appTheme) private var theme
    /// The quantization the user picked. `nil` follows the check's default
    /// (the best one with kernels), which needs no second header read.
    @State private var selectedFile: String?
    @State private var alias = ""
    /// Once the user edits the alias it is theirs; a later quantization change
    /// must not overwrite it.
    @State private var aliasEdited = false
    @State private var destination: String?
    @State private var showsConfirm = false

    /// The identity of what the pane is checking, so a new selection restarts it.
    private struct CheckTask: Equatable {
        let repo: String
        let file: String?
        let configuration: String
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                verdictSection
                quantizationSection
                installSection
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .task(id: CheckTask(repo: repoId, file: selectedFile, configuration: configuration)) {
            await controller.check(request(file: selectedFile), configuration: configuration)
        }
        .onChange(of: repoId) {
            selectedFile = nil
            aliasEdited = false
            refreshAlias()
        }
        .onChange(of: quantLabel) { refreshAlias() }
        .onAppear { refreshAlias() }
        .alert(Text("Install anyway?", bundle: .module), isPresented: $showsConfirm) {
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
            Button { performInstall() } label: { Text("Install", bundle: .module) }
        } message: {
            Text(decision.reason ?? "")
        }
    }

    // MARK: - State

    private var configuration: String { model.fitRecommendationConfigurationID }

    private func key(file: String?) -> HubSearchController.CheckKey {
        .init(repo: repoId, file: file, configuration: configuration)
    }

    private func request(file: String?) -> HubSearchController.HubCheckRequest {
        .init(
            repo: repoId,
            sidecarHint: result?.baseModel,
            file: file,
            context: model.activeFitContext,
            slots: model.activeCacheSlots,
            loadGuard: model.activeLoadGuard)
    }

    /// The check for the repository's default file. It owns the quantization
    /// list, which does not change when the user picks one.
    private var defaultCheck: HubCheck? {
        if case .done(let check) = controller.checkPhase(key(file: nil)) { return check }
        return nil
    }

    /// What the pane is currently showing a verdict for.
    private var currentPhase: HubSearchController.CheckPhase? {
        controller.checkPhase(key(file: selectedFile))
    }

    private var currentCheck: HubCheck? {
        if case .done(let check) = currentPhase { return check }
        return nil
    }

    private var variants: [RepoVariant] { defaultCheck?.variants ?? [] }

    private var selectedVariant: RepoVariant? {
        let file = selectedFile ?? defaultCheck?.file
        return variants.first { $0.file == file }
    }

    private var quantLabel: String? {
        currentCheck?.kind == "gguf" ? selectedVariant?.quantLabel : nil
    }

    private var chosenLibrary: String? {
        model.installDestinations.libraryForInstall(selection: destination)
    }

    private var decision: ModelInstallDecision {
        guard let check = currentCheck else {
            return .blocked(AppStrings.localized("Checking compatibility..."))
        }
        return HubInstallGating.decision(
            check: check,
            selectedVariant: check.kind == "gguf" ? selectedVariant : nil,
            destination: model.installDestinations.gateDestination(library: chosenLibrary))
    }

    private var takenAliases: Set<String> {
        Set(model.installed.map(\.alias)).union(model.catalog.map(\.alias))
    }

    private func refreshAlias() {
        guard !aliasEdited else { return }
        alias = ModelAlias.suggest(repo: repoId, quantLabel: quantLabel, taken: takenAliases)
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(verbatim: repoId)
                .themedFont(.title3, weight: .semibold)
                .foregroundStyle(.appText)
                .textSelection(.enabled)
                .accessibilityAddTraits(.isHeader)
            if let result {
                HStack(spacing: 12) {
                    if let downloads = result.downloads {
                        stat("arrow.down.circle", downloads)
                    }
                    if let likes = result.likes {
                        stat("heart", likes)
                    }
                    if result.hasGguf { formatChip("GGUF") }
                    if result.hasMlx { formatChip("MLX") }
                }
                if !result.bundledAliases.isEmpty {
                    Text("This repository is also in the curated catalog as \(result.bundledAliases.joined(separator: ", ")). Use that entry for a verified install.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
        }
    }

    private func stat(_ symbol: String, _ value: UInt64) -> some View {
        HStack(spacing: 4) {
            Image(systemName: symbol).accessibilityHidden(true)
            Text(verbatim: value.formatted(.number.notation(.compactName))).monospacedDigit()
        }
        .themedFont(.small)
        .foregroundStyle(.appSecondary)
    }

    private func formatChip(_ label: String) -> some View {
        Text(verbatim: label)
            .themedFont(.tiny, weight: .semibold)
            .foregroundStyle(.appSecondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 4))
    }

    // MARK: - Verdict

    @ViewBuilder private var verdictSection: some View {
        switch currentPhase {
        case .none, .checking:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Checking compatibility...", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            .accessibilityElement(children: .combine)
        case .failed(let message):
            callout(Text(verbatim: message), tone: theme.danger, icon: "xmark.octagon.fill")
            retryButton
        case .done(let check):
            verdictCard(check)
        }
    }

    private func verdictCard(_ check: HubCheck) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            switch check.state {
            case .runnable:
                runnableHeader(check)
                identityRows(check)
            case .refused:
                verdictLabel(Text("Would not run here", bundle: .module), ok: false)
                callout(Text(verbatim: check.reason ?? AppStrings.localized("This port has no decode flow for this checkpoint.")), tone: theme.danger, icon: "xmark.octagon.fill")
            case .needsAuth:
                callout(Text("This repository is gated or needs a Hugging Face sign-in. Add a token in Settings, then check again.", bundle: .module), tone: theme.warning, icon: "lock.fill")
                HStack(spacing: 8) {
                    Button { model.openSettings(tab: .models) } label: { Text("Open Settings", bundle: .module) }
                        .appAction(.secondary, size: .small)
                    retryButton
                }
            case .unavailable:
                callout(Text(verbatim: check.reason ?? AppStrings.localized("no answer")), tone: theme.warning, icon: "wifi.exclamationmark")
                retryButton
            }
            ForEach(Array(check.warnings.enumerated()), id: \.offset) { _, warning in
                callout(Text(verbatim: warning), tone: theme.warning, icon: "exclamationmark.triangle.fill")
            }
        }
        .padding(14)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }

    private func runnableHeader(_ check: HubCheck) -> some View {
        HStack(spacing: 8) {
            verdictLabel(Text("Would run here", bundle: .module), ok: true)
            Spacer()
            // A second question, shown only when something answered it.
            if let fit = check.fit { ModelFitVerdictPill(verdict: fit.verdict) }
        }
    }

    private func verdictLabel(_ text: Text, ok: Bool) -> some View {
        Label {
            text
        } icon: {
            Image(systemName: ok ? "checkmark.seal.fill" : "xmark.octagon.fill")
        }
        .themedFont(.small, weight: .semibold)
        .foregroundStyle(ok ? AnyShapeStyle(.appSuccess) : AnyShapeStyle(.appDanger))
    }

    private func identityRows(_ check: HubCheck) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let file = check.file { specRow(Text("File", bundle: .module), Text(verbatim: file)) }
            if let architecture = check.architecture {
                specRow(Text("Declares", bundle: .module), Text(verbatim: architecture))
            }
            if let bytes = selectedVariant?.bytes ?? check.downloadBytes {
                specRow(Text("Download", bundle: .module), Text(verbatim: MetricFormat.storage(bytes)))
            }
            if let repo = check.sidecarRepo {
                specRow(Text("Tokenizer from", bundle: .module), Text(verbatim: repo))
            }
        }
    }

    private func specRow(_ title: Text, _ value: Text) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            title.themedFont(.small).foregroundStyle(.appSecondary)
            Spacer(minLength: 8)
            value.themedFont(.small).foregroundStyle(.appText)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
    }

    private var retryButton: some View {
        Button {
            Task { await controller.check(request(file: selectedFile), configuration: configuration, force: true) }
        } label: { Text("Check again", bundle: .module) }
        .appAction(.secondary, size: .small)
    }

    private func callout(_ text: Text, tone: Color, icon: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: icon).themedFont(.small).accessibilityHidden(true)
            text.themedFont(.small).fixedSize(horizontal: false, vertical: true)
        }
        .foregroundStyle(tone)
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tone.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Quantization

    /// Shown only when there is a choice. The fit is per file, so picking one
    /// re-reads that file's header rather than reusing the default's verdict.
    @ViewBuilder private var quantizationSection: some View {
        if variants.count > 1, defaultCheck?.kind == "gguf" {
            VStack(alignment: .leading, spacing: 6) {
                LabeledContent {
                    Picker(selection: fileBinding) {
                        ForEach(variants) { variant in
                            variant.pickerLabel.tag(variant.file)
                        }
                    } label: { Text("Quantization", bundle: .module) }
                    .labelsHidden()
                } label: { Text("Quantization", bundle: .module) }
                if let skipped = defaultCheck?.shardedSkipped, skipped > 0 {
                    Text("Multi-part files not listed: \(skipped). This port cannot walk a shard set.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
        }
    }

    private var fileBinding: Binding<String> {
        Binding(
            get: { selectedFile ?? defaultCheck?.file ?? "" },
            set: { newValue in
                // The default file's verdict is already held; a second read of
                // the same header would only restart the spinner.
                selectedFile = (newValue == defaultCheck?.file) ? nil : newValue
            })
    }

    // MARK: - Install

    private var installSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            LabeledContent {
                TextField(text: aliasBinding, prompt: Text("my-model", bundle: .module)) {
                    Text("Save as", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
            } label: { Text("Save as", bundle: .module) }

            if let problem = aliasProblem {
                Label(problem, systemImage: "exclamationmark.triangle.fill")
                    .themedFont(.small)
                    .foregroundStyle(.appWarning)
            }

            InstallDestinationPicker(model: model, selection: $destination)

            if let reason = decision.reason, currentCheck != nil {
                Label(reason, systemImage: decision.isBlocked ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                    .themedFont(.small)
                    .foregroundStyle(decision.isBlocked ? AnyShapeStyle(.appDanger) : AnyShapeStyle(AppAttention.color))
                    .fixedSize(horizontal: false, vertical: true)
            }

            installProgress

            HStack {
                Spacer()
                Button { requestInstall() } label: { Text("Install Model", bundle: .module) }
                    .appAction(.primary)
                    .disabled(!canInstall)
                    .help(Text("Downloads and installs this repository", bundle: .module))
            }
        }
    }

    private var aliasBinding: Binding<String> {
        Binding(
            get: { alias },
            set: { newValue in
                alias = newValue
                aliasEdited = true
            })
    }

    /// Why the alias cannot be used, in words, or `nil`.
    private var aliasProblem: String? {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return AppStrings.localized("Choose a name for this model.") }
        if !AppModel.isSafeInstallAlias(trimmed) {
            return AppStrings.localized("Use letters, numbers, dots, dashes and underscores only.")
        }
        if ModelAlias.isTaken(trimmed, in: takenAliases) {
            return AppStrings.localized("A model with this name already exists.")
        }
        return nil
    }

    private var canInstall: Bool {
        currentCheck != nil && !decision.isBlocked && aliasProblem == nil
            && !model.isRunning && model.canInstall(alias: alias)
    }

    @ViewBuilder private var installProgress: some View {
        if model.isInstallingModel, model.installingAlias == alias {
            AppProgressBar(
                value: model.installProgressFraction,
                label: Text("Downloading", bundle: .module),
                size: .compact)
        }
    }

    private func requestInstall() {
        switch decision {
        case .blocked: return
        case .confirm: showsConfirm = true
        case .allowed: performInstall()
        }
    }

    private func performInstall() {
        guard let check = currentCheck else { return }
        model.installRepo(
            repo: repoId,
            alias: alias.trimmingCharacters(in: .whitespacesAndNewlines),
            file: check.kind == "gguf" ? check.file : nil,
            sidecarRepo: check.sidecarRepo,
            library: chosenLibrary)
    }
}
