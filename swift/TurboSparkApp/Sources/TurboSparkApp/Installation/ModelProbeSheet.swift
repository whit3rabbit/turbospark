import SwiftUI
import TurboSpark

/// Sheet for probing and installing arbitrary Hugging Face repositories.
@MainActor
struct ModelProbeSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

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

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .frame(minWidth: 560, minHeight: 500)
        .alert("Install anyway?", isPresented: $showsConfirm) {
            Button("Cancel", role: .cancel) {}
            Button("Install") { performInstall() }
        } message: {
            Text(decision.reason ?? "")
        }
    }

    /// What the gate says about installing what is currently selected.
    private var decision: ModelInstallDecision {
        ModelInstallGate.decide(
            probeRunnable: report?.runnable,
            refusedBecause: report?.refusedBecause,
            verdict: report?.fit?.verdict,
            installBytes: report?.downloadBytes ?? selectedVariant?.bytes,
            freeDiskBytes: ModelInstallGate.freeSpace(at: AppStorageRoot.directory))
    }

    private var selectedVariant: RepoVariant? {
        variants?.variants.first { $0.file == ggufFile }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Probe Hugging Face Repository", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text("Check compatibility by reading checkpoint headers without downloading.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button("Done") { dismiss() }
                .keyboardShortcut(.cancelAction)
                .accessibilityLabel("Done")
                .accessibilityHint("Closes the probe sheet")
        }
        .padding(16)
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                formFields
                if isProbing {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
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
                            .foregroundStyle(.red)

                        if err.contains("401") || err.contains("403") || err.contains("gated") || err.localizedCaseInsensitiveContains("unauthorized") || err.localizedCaseInsensitiveContains("forbidden") {
                            HStack(spacing: 6) {
                                Image(systemName: "key.fill")
                                    .themedFont(.tiny)
                                    .foregroundStyle(.orange)
                                Text("This repository may require Hugging Face authentication.", bundle: .module)
                                    .themedFont(.tiny)
                                    .foregroundStyle(.appSecondary)
                                Button("Open Settings") {
                                    model.openSettings(tab: .general)
                                }
                                .buttonStyle(.link)
                                .themedFont(.tiny)
                            }
                            .padding(.top, 2)
                        }
                    }
                    .padding(10)
                    .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
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
            LabeledContent("Repository") {
                TextField("e.g. mlx-community/Qwen3.6-35B-A3B-4bit", text: $repo)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Hugging Face repository to probe")
                    .onSubmit { listVariants() }
            }
            LabeledContent("Local Alias") {
                TextField("e.g. my-qwen-model", text: $alias)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Local model alias")
            }
            quantizationField
            LabeledContent("Sidecar Repo (optional)") {
                TextField("e.g. original-org/model-tokenizer", text: $sidecarRepo)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Sidecar repository for tokenizer (optional)")
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
        LabeledContent("Quantization") {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    if let listed = variants, !listed.variants.isEmpty {
                        Picker("", selection: $ggufFile) {
                            Text("Let the installer choose", bundle: .module).tag("")
                            ForEach(listed.variants) { v in
                                Text(variantLabel(v)).tag(v.file)
                            }
                        }
                        .labelsHidden()
                        .accessibilityLabel("Quantization to install")
                        .onChange(of: ggufFile) { reprobeIfProbed() }
                    } else {
                        TextField("e.g. model-q4_k_m.gguf", text: $ggufFile)
                            .textFieldStyle(.roundedBorder)
                            .accessibilityLabel("GGUF filename (optional)")
                    }
                    Button {
                        listVariants()
                    } label: {
                        if isListingVariants {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "arrow.clockwise")
                        }
                    }
                    .disabled(repo.isEmpty || isListingVariants)
                    .help("List the .gguf files this repository publishes")
                }

                if let v = selectedVariant, !v.executable {
                    Text("This port has no kernels for the type this filename names. Probe it to see what the header actually contains.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.orange)
                }
                // Nonzero shards is why the list may be short or empty. Say
                // so, or an empty picker reads as "no GGUF here".
                if let listed = variants, listed.shardedSkipped > 0 {
                    Text("\(listed.shardedSkipped) multi-part file\(listed.shardedSkipped == 1 ? "" : "s") not listed: this port cannot walk a shard set.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
        }
    }

    private func variantLabel(_ v: RepoVariant) -> String {
        let size = v.bytes.map { " - \(MetricFormat.storage($0))" } ?? ""
        let quant = v.quantLabel ?? "unrecognized"
        let blocked = v.executable ? "" : " (no kernels)"
        return "\(quant)\(size)\(blocked) - \(v.file)"
    }

    // MARK: - Footer

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            // The reason sits BESIDE the button rather than only in a
            // tooltip: a disabled control with no visible cause reads as a
            // broken control.
            if let reason = decision.reason {
                Label(reason, systemImage: decision.isBlocked ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                    .themedFont(.small)
                    .foregroundStyle(decision.isBlocked ? Color.red : Color.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Run Probe") { runProbe() }
                    .disabled(repo.isEmpty || isProbing)
                    .help("Fetch repository headers and inspect model compatibility")
                    .accessibilityHint("Fetches checkpoint headers without downloading the model")

                Spacer()

                Button("Install Model") { requestInstall() }
                    .buttonStyle(.borderedProminent)
                    .disabled(
                        repo.isEmpty || alias.isEmpty || isProbing || model.isInstallingModel
                            || decision.isBlocked
                    )
                    .help("Download and install probed repository")
                    .accessibilityHint("Downloads and installs the probed repository")
            }
        }
        .padding(16)
    }

    // MARK: - Actions

    private func listVariants() {
        guard !repo.isEmpty else { return }
        isListingVariants = true
        variants = nil
        Task {
            variants = try? TurboSparkCatalog.variants(repo: repo)
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
        isProbing = true
        probeError = nil
        report = nil
        Task {
            do {
                report = try TurboSparkCatalog.probe(
                    repo: repo,
                    file: ggufFile.isEmpty ? nil : ggufFile,
                    sidecarRepo: sidecarRepo.isEmpty ? nil : sidecarRepo,
                    // The configuration this app will OPEN under, so the
                    // probe's fit and a session's allocation are one answer.
                    context: model.activeFitContext,
                    expertCacheSlots: model.activeCacheSlots,
                    loadGuard: model.activeLoadGuard)
                _ = AccessibilityNotification.Announcement.post(
                    .init("Probe complete. Report is available.")
                )
            } catch {
                let msg = "Probe failed: \(error.localizedDescription)"
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
        model.installRepo(
            repo: repo,
            alias: alias,
            file: ggufFile.isEmpty ? nil : ggufFile,
            sidecarRepo: sidecarRepo.isEmpty ? nil : sidecarRepo)
        dismiss()
    }
}
