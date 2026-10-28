import SwiftUI
import TurboSpark

/// Sheet for probing and installing arbitrary Hugging Face repositories.
@MainActor
struct ModelProbeSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme

    @State private var repo = ""
    @State private var alias = ""
    @State private var ggufFile = ""
    @State private var sidecarRepo = ""
    @State private var isProbing = false
    @State private var report: ProbeReport? = nil
    @State private var probeError: String? = nil

    @State private var variants: RepoVariants? = nil
    @State private var isListingVariants = false
    @State private var showsConfirm = false
    // Bumped per request so a slower earlier probe or variant listing cannot
    // overwrite the result of a newer one.
    @State private var probeGeneration = 0
    @State private var variantsGeneration = 0
    /// The library the user picked for this install. Nil follows the default
    /// (requirement 2.2). Per install: it is not remembered across launches.
    @State private var destination: String?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .frame(minWidth: 560, minHeight: 500)
        .alert(Text("Install anyway?", bundle: .module), isPresented: $showsConfirm) {
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
            Button { performInstall() } label: { Text("Install", bundle: .module) }
        } message: {
            Text(decision.reason ?? "")
        }
    }

    nonisolated static func sanitizeRepo(_ input: String) -> String {
        ModelProbeGating.sanitizeRepo(input)
    }

    /// What the gate says about installing what is currently selected.
    private var decision: ModelInstallDecision {
        ModelProbeGating.installDecision(
            repo: repo,
            probeError: probeError,
            report: report,
            variants: variants,
            ggufFile: ggufFile,
            selectedVariant: selectedVariant,
            destination: model.installDestinations.gateDestination(library: chosenLibrary)
        )
    }

    /// The `library:` the install takes for the menu's choice: nil for This Mac.
    private var chosenLibrary: String? {
        model.installDestinations.libraryForInstall(selection: destination)
    }

    private var selectedVariant: RepoVariant? {
        variants?.variants.first { $0.file == ggufFile }
    }

    private var isGgufRepo: Bool {
        repo.lowercased().contains("gguf")
            || ggufFile.lowercased().hasSuffix(".gguf")
            || (report?.file?.lowercased().hasSuffix(".gguf") ?? false)
            || (variants != nil && !(variants?.variants.isEmpty ?? true))
            || (report?.sidecarsMissing.contains("tokenizer.json") ?? false)
    }

    private var header: some View {
        AppAdaptiveHeader {
            VStack(alignment: .leading, spacing: 2) {
                Text("Probe Hugging Face Repository", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text("Check compatibility by reading checkpoint headers without downloading.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
        } actions: {
            Button { dismiss() } label: { Text("Done", bundle: .module) }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)
        }
        .padding(16)
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                formFields
                if isProbing {
                    VStack(alignment: .leading, spacing: 6) {
                        AppProgressBar(value: nil, label: Text("Probing repository headers...", bundle: .module), size: .compact)
                        Text("Probing repository headers...", bundle: .module)
                            .themedFont(.base)
                            .foregroundStyle(.appSecondary)
                    }
                    .padding(.vertical, 8)
                }
                if let err = probeError {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(err)
                            .themedFont(.small)
                            .foregroundStyle(.appDanger)

                        if err.contains("401") || err.contains("403") || err.contains("gated") || err.localizedCaseInsensitiveContains("unauthorized") || err.localizedCaseInsensitiveContains("forbidden") {
                            HStack(spacing: 6) {
                                Image(systemName: "key.fill")
                                    .themedFont(.tiny)
                                    .foregroundStyle(AppAttention.color)
                                Text("This repository may require Hugging Face authentication.", bundle: .module)
                                    .themedFont(.tiny)
                                    .foregroundStyle(.appSecondary)
                                // Hugging Face sign-in lives in Models & Storage.
                                Button {
                                    model.openSettings(tab: .models)
                                } label: { Text("Open Settings", bundle: .module) }
                                .appAction(.quiet, size: .small)
                            }
                            .padding(.top, 2)
                        }
                    }
                    .padding(10)
                    .background(.appDangerTint, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                }
                if let report {
                    ProbeReportCardView(report: report)
                }
            }
            .padding(16)
        }
    }

    // MARK: - Form

    private var formFields: some View {
        VStack(alignment: .leading, spacing: 10) {
            LabeledContent {
                TextField(text: $repo, prompt: example("mlx-community/Qwen3.6-35B-A3B-4bit")) {
                    Text("Repository", bundle: .module)
                }
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: repo) {
                        let clean = Self.sanitizeRepo(repo)
                        if clean != repo { repo = clean }
                        // A report, variant list, or filename from the
                        // previous repo must not gate the install of this one.
                        // Bump the generations so an in-flight probe or
                        // listing for the old repo cannot land afterwards.
                        report = nil
                        probeError = nil
                        variants = nil
                        ggufFile = ""
                        probeGeneration += 1
                        variantsGeneration += 1
                        isProbing = false
                        isListingVariants = false
                    }
                    .onSubmit { listVariants() }
            } label: {
                Text("Repository", bundle: .module)
            }
            LabeledContent {
                TextField(text: $alias, prompt: example("my-qwen-model")) {
                    Text("Local Alias", bundle: .module)
                }
                    .textFieldStyle(.roundedBorder)
            } label: {
                Text("Local Alias", bundle: .module)
            }
            quantizationField
            LabeledContent {
                VStack(alignment: .leading, spacing: 4) {
                    TextField(text: $sidecarRepo, prompt: example("original-org/model-tokenizer")) {
                        Text("Sidecar Repo (optional)", bundle: .module)
                    }
                        .textFieldStyle(.roundedBorder)
                        .onChange(of: sidecarRepo) {
                            let clean = Self.sanitizeRepo(sidecarRepo)
                            if clean != sidecarRepo { sidecarRepo = clean }
                        }
                        .onSubmit { reprobeIfProbed() }

                    if isGgufRepo {
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "info.circle.fill")
                                .themedFont(.tiny)
                                .foregroundStyle(.appAccent)
                            Text("GGUF files usually lack tokenizer.json. If the probe reports missing sidecars, enter the upstream Hugging Face repo (e.g. Qwen/Qwen3.8-27B) here.", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                        .padding(.top, 2)
                    }
                }
            } label: {
                Text("Sidecar Repo (optional)", bundle: .module)
            }
        }
    }

    /// The quantization picker.
    ///
    /// It falls back to the free-text field it replaced rather than hiding
    /// it: the listing recognizes the names this port's ladder knows, and a
    /// repository is free to name a file something else.
    @ViewBuilder
    private var quantizationField: some View {
        LabeledContent {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    if let listed = variants, !listed.variants.isEmpty {
                        Picker(selection: $ggufFile) {
                            Text("Let the installer choose", bundle: .module).tag("")
                            ForEach(listed.variants) { v in
                                variantLabel(v).tag(v.file)
                            }
                        } label: {
                            Text("Quantization", bundle: .module)
                        }
                        .labelsHidden()
                        .onChange(of: ggufFile) { reprobeIfProbed() }
                    } else {
                        TextField(text: $ggufFile, prompt: example("model-q4_k_m.gguf")) {
                            Text("GGUF filename (optional)", bundle: .module)
                        }
                            .textFieldStyle(.roundedBorder)
                            // A probe of a different filename says nothing
                            // about this one.
                            .onChange(of: ggufFile) {
                                report = nil
                                probeError = nil
                            }
                    }
                    Button {
                        listVariants()
                    } label: {
                        Label {
                            Text("List the .gguf files this repository publishes", bundle: .module)
                        } icon: {
                            if isListingVariants {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "arrow.clockwise")
                            }
                        }
                        .labelStyle(.iconOnly)
                    }
                    .appAction(.secondary)
                    .disabled(repo.isEmpty || isListingVariants)
                    .help(Text("List the .gguf files this repository publishes", bundle: .module))
                }

                if let listed = variants, listed.variants.count > 1 && ggufFile.isEmpty {
                    Text("This repository publishes \(listed.variants.count) GGUF variants. Select a specific file to probe and install.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(AppAttention.color)
                }

                if let v = selectedVariant, !v.executable {
                    Text("This port has no kernels for the type this filename names. Probe it to see what the header actually contains.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(AppAttention.color)
                }
                // Nonzero shards is why the list may be short or empty. Say
                // so, or an empty picker reads as "no GGUF here".
                if let listed = variants, listed.shardedSkipped > 0 {
                    Text("Multi-part files not listed: \(listed.shardedSkipped). This port cannot walk a shard set.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
        } label: { Text("Quantization", bundle: .module) }
    }

    /// Quantization, size, kernel status and file name. The quantization
    /// name, size and file are data; only the two status words translate.
    private func variantLabel(_ v: RepoVariant) -> Text {
        var label = v.quantLabel.map { Text(verbatim: $0) } ?? Text("unrecognized", bundle: .module)
        if let bytes = v.bytes {
            label = label + Text(verbatim: " - \(MetricFormat.storage(bytes))")
        }
        if !v.executable {
            label = label + Text(verbatim: " (") + Text("no kernels", bundle: .module) + Text(verbatim: ")")
        }
        return label + Text(verbatim: " - \(v.file)")
    }

    /// A placeholder that shows an example value: the words are a catalog
    /// key, the example is data.
    private func example(_ value: String) -> Text {
        Text("e.g. \(value)", bundle: .module)
    }

    // MARK: - Footer

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            // The reason sits BESIDE the button rather than only in a
            // tooltip: a disabled control with no visible cause reads as a
            // broken control.
            if let reason = decision.reason, !repo.isEmpty {
                Label(reason, systemImage: decision.isBlocked ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                    .themedFont(.small)
                    .foregroundStyle(decision.isBlocked ? AnyShapeStyle(.appDanger) : AnyShapeStyle(AppAttention.color))
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Where the model will be installed: only when there is a choice
            // (requirement 2.1), and the gate above counts that library's room.
            InstallDestinationPicker(model: model, selection: $destination)
            HStack {
                Button { runProbe() } label: { Text("Run Probe", bundle: .module) }
                    .appAction(.secondary)
                    .disabled(repo.isEmpty || isProbing)
                    .help(Text("Fetches checkpoint headers without downloading the model", bundle: .module))

                Spacer()

                Button { requestInstall() } label: { Text("Install Model", bundle: .module) }
                    .appAction(.primary)
                    .disabled(
                        repo.isEmpty || alias.isEmpty || isProbing || model.isRunning
                            || decision.isBlocked || !model.canInstall(alias: alias)
                    )
                    .help(Text("Downloads and installs the probed repository", bundle: .module))
            }
        }
        .padding(16)
    }

    // MARK: - Actions

    private func listVariants() {
        let clean = Self.sanitizeRepo(repo)
        if clean != repo { repo = clean }
        guard !clean.isEmpty else { return }
        isListingVariants = true
        variants = nil
        variantsGeneration += 1
        let generation = variantsGeneration
        // Blocking network FFI (up to the HTTP timeout): keep it off the
        // main actor so the spinner animates and the app stays responsive.
        Task {
            let outcome = await Task.detached(priority: .userInitiated) {
                Result { try TurboSparkCatalog.variants(repo: clean) }
            }.value
            guard generation == variantsGeneration else { return }
            switch outcome {
            case .success(let found):
                variants = found
            case .failure(let error):
                // Surface it (a 401/404/offline failure otherwise looks like
                // "this repo has no GGUF variants"); reuse probeError so the
                // existing sign-in hint applies.
                variants = nil
                probeError = AppStrings.localized("Variant listing failed: \(error.localizedDescription)")
            }
            isListingVariants = false
        }
    }

    /// Re-probe after a variant change, but only if a probe already ran.
    /// Selecting from the menu should not start network traffic nobody asked
    /// for.
    private func reprobeIfProbed() {
        if report != nil { runProbe() }
    }

    private func runProbe() {
        let cleanRepo = Self.sanitizeRepo(repo)
        let cleanSidecar = Self.sanitizeRepo(sidecarRepo)
        if cleanRepo != repo { repo = cleanRepo }
        if cleanSidecar != sidecarRepo { sidecarRepo = cleanSidecar }
        isProbing = true
        probeError = nil
        report = nil
        probeGeneration += 1
        let generation = probeGeneration
        let file = ggufFile.isEmpty ? nil : ggufFile
        let sidecar = cleanSidecar.isEmpty ? nil : cleanSidecar
        // The configuration this app will OPEN under, so the probe's fit and
        // a session's allocation are one answer. Captured on the main actor;
        // the blocking network FFI itself runs detached.
        let context = model.activeFitContext
        let slots = model.activeCacheSlots
        let loadGuard = model.activeLoadGuard
        Task {
            let outcome = await Task.detached(priority: .userInitiated) {
                Result {
                    try TurboSparkCatalog.probe(
                        repo: cleanRepo,
                        file: file,
                        sidecarRepo: sidecar,
                        context: context,
                        expertCacheSlots: slots,
                        loadGuard: loadGuard)
                }
            }.value
            guard generation == probeGeneration else { return }
            switch outcome {
            case .success(let probed):
                report = probed
                _ = AccessibilityNotification.Announcement.post(
                    .init(AppStrings.localized("Probe complete. Report is available."))
                )
            case .failure(let error):
                let msg = AppStrings.localized("Probe failed: \(error.localizedDescription)")
                probeError = msg
                _ = AccessibilityNotification.Announcement.post(.init(msg))
            }
            isProbing = false
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
        let cleanRepo = Self.sanitizeRepo(repo)
        let cleanSidecar = Self.sanitizeRepo(sidecarRepo)
        model.installRepo(
            repo: cleanRepo,
            alias: alias,
            file: ggufFile.isEmpty ? nil : ggufFile,
            sidecarRepo: cleanSidecar.isEmpty ? nil : cleanSidecar,
            library: chosenLibrary)
        dismiss()
    }
}
