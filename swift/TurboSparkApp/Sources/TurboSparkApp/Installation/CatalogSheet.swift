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
            Button { dismiss() } label: { Text("Done", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
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
                Group {
                    if let stage = model.installStageText {
                        Text(stage)
                    } else {
                        Text("Installing", bundle: .module)
                    }
                }
                .themedFont(.base, weight: .medium)
                Spacer()
                Button {
                    model.cancelInstall()
                } label: { Text("Cancel", bundle: .module) }
                .appAction(.secondary, size: .small)
                .accessibilityLabel(Text("Cancel Download", bundle: .module))
            }

            if let fraction = model.installProgressFraction {
                AppProgressBar(value: fraction, label: Text("Download progress", bundle: .module),
                               animated: !model.isInstallPaused)
                HStack {
                    if let downloaded = model.installDownloadedBytes, let total = model.installTotalBytes {
                        Text(verbatim: "\(MetricFormat.storage(downloaded)) / \(MetricFormat.storage(total))")
                    }
                    Spacer()
                    Text(verbatim: MetricFormat.percent(fraction * 100))
                }
                .themedFont(.small).monospacedDigit()
                .foregroundStyle(.appSecondary)
            } else {
                AppProgressBar(value: nil, label: Text("Download progress", bundle: .module))
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
                                Text(verbatim: item.alias)
                                    .themedFont(.base, weight: .medium)
                                Text(verbatim: "\(visuals.family) | \(MetricFormat.storage(item.installBytes))")
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
                        Button(role: .destructive) {
                            model.deleteModel(item)
                        } label: { Text("Delete", bundle: .module) }
                        .appAction(.destructive, size: .small)
                        .disabled(model.isInstallingModel || !model.canDeleteModel)
                        .help(Text("Delete model from disk", bundle: .module))
                        .accessibilityLabel(Text("Delete \(item.alias)", bundle: .module))
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
                            Text(
                                "Download: \(MetricFormat.storage(entry.downloadBytes)) | Disk: \(MetricFormat.storage(entry.installBytes))",
                                bundle: .module)
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
                        Button {
                            model.installModel(alias: entry.alias)
                        } label: { Text("Install", bundle: .module) }
                        .appAction(.secondary, size: .small)
                        .disabled(model.isRunning || !model.canInstall(alias: entry.alias))
                        .help(Text("Install \(entry.alias)", bundle: .module))
                        .accessibilityLabel(Text("Install \(entry.alias)", bundle: .module))
                    }
                }
                .padding(8)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private var customSection: some View {
        DisclosureGroup(isExpanded: $isCustomExpanded) {
            VStack(alignment: .leading, spacing: 10) {
                TextField(text: $customRepo, prompt: Text("Hugging Face repo (e.g. owner/model)", bundle: .module)) {
                    Text("Repository", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
                TextField(text: $customAlias, prompt: Text("Local alias (e.g. my-model)", bundle: .module)) {
                    Text("Local Alias", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)

                Button {
                    model.installRepo(repo: customRepo, alias: customAlias)
                } label: { Text("Install Custom Repo", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(
                    customRepo.isEmpty || customAlias.isEmpty || model.isRunning
                        || !model.canInstall(alias: customAlias))
            }
            .padding(.top, 6)
        } label: {
            Text("Install from Hugging Face Repository", bundle: .module)
        }
        .themedFont(.small)
    }

    private var footer: some View {
        HStack {
            Button {
                ModelLocationPicker.choose(for: model)
            } label: { Text("Choose Local Folder...", bundle: .module) }
            .appAction(.secondary)
            .help(Text("Select an existing model directory", bundle: .module))

            Spacer()

            Button {
                model.refreshModels()
            } label: { Text("Refresh", bundle: .module) }
            .appAction(.secondary)
            .help(Text("Reload model catalog", bundle: .module))
        }
        .padding(12)
    }
}
