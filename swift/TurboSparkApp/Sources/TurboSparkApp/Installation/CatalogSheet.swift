import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Modal sheet for browsing the curated catalog, viewing local installs, and streaming Hugging Face models.
@MainActor
struct CatalogSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var recommendations: [ModelRecommendation] = []
    @State private var customRepo = ""
    @State private var customAlias = ""
    @State private var isCustomExpanded = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .frame(minWidth: 640, minHeight: 480)
        .task {
            if let recs = try? TurboSparkCatalog.recommend(loadGuard: model.activeLoadGuard) {
                recommendations = recs
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Model Catalog")
                    .font(.headline)
                    .accessibilityAddTraits(.isHeader)
                Text("Install models locally from the curated catalog or Hugging Face.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
                .accessibilityLabel("Done")
                .accessibilityHint("Closes the catalog sheet")
        }
        .padding(14)
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if model.isInstallingModel {
                    installProgressBox
                }

                installedSection
                curatedSection
                customSection
            }
            .padding(14)
        }
    }

    private var installProgressBox: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(model.installStageText ?? "Installing...")
                    .font(.callout.weight(.medium))
                Spacer()
                Button("Cancel") {
                    model.cancelInstall()
                }
                .controlSize(.small)
                .accessibilityLabel("Cancel installation")
                .accessibilityHint("Stops the in-progress model installation")
            }

            if let fraction = model.installProgressFraction {
                ProgressView(value: fraction)
                    .accessibilityLabel("Installation progress")
                    .accessibilityValue(MetricFormat.percent(fraction * 100))
                HStack {
                    if let downloaded = model.installDownloadedBytes, let total = model.installTotalBytes {
                        Text("\(MetricFormat.storage(downloaded)) / \(MetricFormat.storage(total))")
                    }
                    Spacer()
                    Text(MetricFormat.percent(fraction * 100))
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            } else {
                ProgressView()
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
    }

    private var installedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Installed Models")
                .font(.subheadline.weight(.semibold))
                .accessibilityAddTraits(.isHeader)

            if model.installed.isEmpty {
                Text("No models currently installed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(model.installed) { item in
                    let visuals = ModelFamilyVisuals.resolve(alias: item.alias, family: item.family, name: item.alias)
                    HStack {
                        HStack(spacing: 8) {
                            ModelLogoView(visuals: visuals, size: 24, cornerRadius: 6)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.alias)
                                    .font(.callout.weight(.medium))
                                Text("\(visuals.family) | \(MetricFormat.storage(item.installBytes))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        if model.selected?.alias == item.alias && model.session != nil {
                            Text("Active")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(TurboSparkTheme.accentColor)
                        }
                        Button("Delete", role: .destructive) {
                            model.deleteModel(item)
                        }
                        .controlSize(.small)
                        .disabled(model.isInstallingModel || !model.canDeleteModel)
                        .help("Delete model from disk")
                        .accessibilityLabel("Delete \(item.alias)")
                        .accessibilityHint("Removes this installed model from disk")
                    }
                    .padding(8)
                    .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    private var curatedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Available in Catalog")
                .font(.subheadline.weight(.semibold))
                .accessibilityAddTraits(.isHeader)

            ForEach(model.catalog) { entry in
                let visuals = ModelFamilyVisuals.resolve(alias: entry.alias, family: entry.family, name: entry.name)
                HStack {
                    HStack(spacing: 8) {
                        ModelLogoView(visuals: visuals, size: 24, cornerRadius: 6)

                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(entry.alias)
                                    .font(.callout.weight(.medium))
                                HStack(spacing: 3) {
                                    Image(systemName: visuals.iconSystemName)
                                        .font(.system(size: 8))
                                    Text(visuals.family)
                                }
                                .font(.caption2)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1.5)
                                .background(visuals.accentColor.opacity(0.15), in: Capsule())
                                .foregroundStyle(visuals.accentColor)
                            }
                            Text(entry.name)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("Download: \(MetricFormat.storage(entry.downloadBytes)) | Disk: \(MetricFormat.storage(entry.installBytes))")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    Spacer()
                    if entry.installed {
                        Text("Installed")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Button("Install") {
                            model.installModel(alias: entry.alias)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(model.isInstallingModel || model.isRunning)
                        .help("Install \(entry.alias)")
                        .accessibilityLabel("Install \(entry.alias)")
                        .accessibilityHint("Downloads and installs \(entry.name)")
                    }
                }
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private var customSection: some View {
        DisclosureGroup("Install from Hugging Face Repository", isExpanded: $isCustomExpanded) {
            VStack(alignment: .leading, spacing: 10) {
                TextField("Hugging Face repo (e.g. owner/model)", text: $customRepo)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Hugging Face repository")
                TextField("Local alias (e.g. my-model)", text: $customAlias)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Local alias")

                Button("Install Custom Repo") {
                    model.installRepo(repo: customRepo, alias: customAlias)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(customRepo.isEmpty || customAlias.isEmpty || model.isInstallingModel)
                .accessibilityLabel("Install custom repository")
                .accessibilityHint("Downloads and builds model from specified Hugging Face repository")
            }
            .padding(.top, 6)
        }
        .font(.subheadline)
    }

    private var footer: some View {
        HStack {
            Button("Choose Local Folder...") {
                ModelLocationPicker.choose(for: model)
            }
            .help("Select an existing model directory")
            .accessibilityHint("Opens a folder picker for an already-installed model")

            Spacer()

            Button("Refresh") {
                model.refreshModels()
            }
            .help("Reload model catalog")
            .accessibilityHint("Reloads the model catalog from disk")
        }
        .padding(12)
    }
}
