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
            recommendations = model.fitRecommendations()
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Model Catalog", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text("Install models locally from the curated catalog or Hugging Face.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
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
                    .themedFont(.base, weight: .medium)
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
                        Text("\(MetricFormat.storage(downloaded)) / \(MetricFormat.storage(total))", bundle: .module)
                    }
                    Spacer()
                    Text(MetricFormat.percent(fraction * 100))
                }
                .themedFont(.small).monospacedDigit()
                .foregroundStyle(.appSecondary)
            } else {
                ProgressView()
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
    }

    private var installedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Installed Models", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            if model.installed.isEmpty {
                Text("No models currently installed.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                ForEach(model.installed) { item in
                    let visuals = ModelFamilyVisuals.resolve(alias: item.alias, family: item.family, name: item.alias)
                    HStack {
                        HStack(spacing: 8) {
                            ModelLogoView(visuals: visuals, size: 24, cornerRadius: 6)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.alias)
                                    .themedFont(.base, weight: .medium)
                                Text("\(visuals.family) | \(MetricFormat.storage(item.installBytes))", bundle: .module)
                                    .themedFont(.small)
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        Spacer()
                        if model.selected?.alias == item.alias && model.session != nil {
                            Text("Active", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                                .foregroundStyle(.appAccent)
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
                    .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    private var curatedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Available in Catalog", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            ForEach(model.catalog) { entry in
                let visuals = ModelFamilyVisuals.resolve(alias: entry.alias, family: entry.family, name: entry.name)
                HStack {
                    HStack(spacing: 8) {
                        ModelLogoView(visuals: visuals, size: 24, cornerRadius: 6)

                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(entry.alias)
                                    .themedFont(.base, weight: .medium)
                                HStack(spacing: 3) {
                                    Image(systemName: visuals.iconSystemName)
                                        .themedFont(.micro)
                                    Text(visuals.family)
                                }
                                .themedFont(.tiny)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1.5)
                                .background(visuals.accentColor.opacity(0.15), in: Capsule())
                                .foregroundStyle(visuals.accentColor)
                            }
                            Text(entry.name)
                                .themedFont(.small)
                                .foregroundStyle(.appSecondary)
                            Text("Download: \(MetricFormat.storage(entry.downloadBytes)) | Disk: \(MetricFormat.storage(entry.installBytes))", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    Spacer()
                    if entry.installed {
                        Text("Installed", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
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
                .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
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
        .themedFont(.small)
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
