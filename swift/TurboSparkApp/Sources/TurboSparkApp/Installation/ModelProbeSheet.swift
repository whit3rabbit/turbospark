import SwiftUI
import TurboSpark

/// Sheet for probing and installing arbitrary Hugging Face repositories.
struct ModelProbeSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var repo = ""
    @State private var alias = ""
    @State private var ggufFile = ""
    @State private var sidecarRepo = ""
    @State private var isProbing = false
    @State private var probeReportJSON: String? = nil
    @State private var probeError: String? = nil

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .frame(minWidth: 540, minHeight: 460)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Probe Hugging Face Repository")
                    .font(.headline)
                    .accessibilityAddTraits(.isHeader)
                Text("Check compatibility by reading checkpoint headers without downloading.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
                        Text("Probing repository headers...")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 8)
                }

                if let err = probeError {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(10)
                        .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                }

                if let report = probeReportJSON {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Probe Report")
                            .font(.subheadline.weight(.semibold))
                            .accessibilityAddTraits(.isHeader)
                        ScrollView(.horizontal) {
                            Text(report)
                                .font(.caption.monospaced())
                                .padding(10)
                                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
            }
            .padding(16)
        }
    }

    private var formFields: some View {
        VStack(alignment: .leading, spacing: 10) {
            LabeledContent("Repository") {
                TextField("e.g. mlx-community/Qwen3.6-35B-A3B-4bit", text: $repo)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Hugging Face repository to probe")
            }
            LabeledContent("Local Alias") {
                TextField("e.g. my-qwen-model", text: $alias)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Local model alias")
            }
            LabeledContent("GGUF File (optional)") {
                TextField("e.g. model-q4_k_m.gguf", text: $ggufFile)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("GGUF filename (optional)")
            }
            LabeledContent("Sidecar Repo (optional)") {
                TextField("e.g. original-org/model-tokenizer", text: $sidecarRepo)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Sidecar repository for tokenizer (optional)")
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Run Probe") {
                runProbe()
            }
            .disabled(repo.isEmpty || isProbing)
            .accessibilityHint("Fetches checkpoint headers without downloading the model")

            Spacer()

            Button("Install Model") {
                installCustom()
            }
            .buttonStyle(.borderedProminent)
            .disabled(repo.isEmpty || alias.isEmpty || isProbing || model.isInstallingModel)
            .accessibilityHint("Downloads and installs the probed repository")
        }
        .padding(16)
    }

    private func runProbe() {
        isProbing = true
        probeError = nil
        probeReportJSON = nil
        Task {
            do {
                let fileArg = ggufFile.isEmpty ? nil : ggufFile
                let sidecarArg = sidecarRepo.isEmpty ? nil : sidecarRepo
                let json = try TurboSparkCatalog.probe(repo: repo, file: fileArg, sidecarRepo: sidecarArg)
                probeReportJSON = json
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

    private func installCustom() {
        let fileArg = ggufFile.isEmpty ? nil : ggufFile
        let sidecarArg = sidecarRepo.isEmpty ? nil : sidecarRepo
        model.installRepo(repo: repo, alias: alias, file: fileArg, sidecarRepo: sidecarArg)
        dismiss()
    }
}
