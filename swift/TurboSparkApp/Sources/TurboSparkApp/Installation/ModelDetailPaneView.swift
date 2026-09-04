import AppKit
import SwiftUI
import TurboSpark

/// Right-column detail inspector and action pane for a selected model.
struct ModelDetailPaneView: View {
    @ObservedObject var model: AppModel
    let entry: CatalogEntry
    let installedModel: InstalledModel?
    let recommendation: ModelRecommendation?
    @State private var showingDeleteConfirm = false

    private var visuals: ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(alias: entry.alias, family: entry.family, name: entry.name)
    }

    private var isInstalled: Bool {
        installedModel != nil || entry.installed
    }

    private var isActive: Bool {
        model.selected?.alias == entry.alias && model.session != nil
    }

    private var isDownloadingThis: Bool {
        model.isInstallingModel
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                heroHeader
                capabilityTags
                actionCard
                ModelHardwareFitCardView(
                    recommendation: recommendation,
                    entry: entry
                )
                ModelTechnicalSpecsCardView(
                    entry: entry,
                    installedModel: installedModel,
                    visuals: visuals
                )
            }
            .padding(24)
        }
        .confirmationDialog(
            "Delete \(entry.alias)?",
            isPresented: $showingDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button("Delete Model", role: .destructive) {
                if let inst = installedModel ?? model.installed.first(where: { $0.alias == entry.alias }) {
                    model.deleteModel(inst)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will remove the downloaded model files from disk.")
        }
    }

    private var heroHeader: some View {
        HStack(alignment: .top, spacing: 16) {
            ModelLogoView(visuals: visuals, size: 56, cornerRadius: 14)

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(entry.name)
                        .font(.title2.weight(.bold))
                        .lineLimit(2)

                    Text("CATALOG")
                        .font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(TurboSparkTheme.accentColor.opacity(0.16), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(TurboSparkTheme.accentColor)

                    statusSeal
                }

                HStack(spacing: 8) {
                    Text(entry.alias)
                        .font(.subheadline.monospaced().weight(.medium))
                        .foregroundStyle(.secondary)

                    Text("\u{2022}")
                        .foregroundStyle(.tertiary)

                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .font(.caption2)
                            .foregroundStyle(visuals.accentColor)
                        Text(visuals.family)
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .help("Model Family: \(visuals.family)")

                    Text("\u{2022}")
                        .foregroundStyle(.tertiary)

                    Text(entry.status.capitalized)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(statusTint)
                }
            }

            Spacer()

            copyAliasButton
        }
    }

    /// The catalog's own verdict on the row.
    ///
    /// This was an unconditional seal captioned "Verified TurboSpark Model",
    /// which contradicted the `status` text rendered two lines below it on any
    /// row that is not verified. Same defect as `ModelCardView`'s, and the
    /// same fix: read the field (`swift/CLAUDE.md` Gotcha 22).
    @ViewBuilder
    private var statusSeal: some View {
        switch entry.status {
        case "verified":
            Image(systemName: "checkmark.seal.fill")
                .font(.title3)
                .foregroundStyle(TurboSparkTheme.accentColor)
                .help("Verified: this port has run this row end to end")
                .accessibilityHidden(true)
        case "caveat":
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title3)
                .foregroundStyle(.orange)
                .help("Runs with a caveat; see the notes below")
                .accessibilityHidden(true)
        default:
            EmptyView()
        }
    }

    private var statusTint: Color {
        switch entry.status {
        case "verified": return .green
        case "caveat": return .orange
        default: return .secondary
        }
    }

    private var copyAliasButton: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(entry.alias, forType: .string)
        } label: {
            Label("Copy Alias", systemImage: "doc.on.doc")
                .font(.caption)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .help("Copy model alias to clipboard")
        .accessibilityLabel("Copy alias")
        .accessibilityHint("Copies the model alias \(entry.alias) to the clipboard")
    }

    private var capabilityTags: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                HStack(spacing: 4) {
                    Image(systemName: visuals.iconSystemName)
                        .font(.caption2)
                        .foregroundStyle(visuals.accentColor)
                    Text(visuals.family)
                        .font(.caption.weight(.semibold))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(visuals.accentColor.opacity(0.15), in: Capsule())
                .overlay(Capsule().stroke(visuals.accentColor.opacity(0.3), lineWidth: 0.5))

                ForEach(visuals.capabilities, id: \.self) { tag in
                    Text(tag)
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color(nsColor: .controlBackgroundColor), in: Capsule())
                        .overlay(Capsule().stroke(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 0.5))
                }
            }
        }
    }

    private var actionCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Quantization & Format")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 6) {
                        Text(visuals.formatLabel)
                            .font(.callout.weight(.semibold))
                        Text("\u{2022}")
                            .foregroundStyle(.tertiary)
                        Text("\(MetricFormat.storage(entry.downloadBytes)) download")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                actionButtons
            }

            if isDownloadingThis {
                downloadProgressArea
            }
        }
        .modelCardStyle()
    }

    @ViewBuilder
    private var actionButtons: some View {
        HStack(spacing: 10) {
            if isDownloadingThis {
                Button("Cancel Download", role: .cancel) {
                    model.cancelInstall()
                }
                .controlSize(.regular)
            } else if isActive {
                Button {
                    model.activeSection = .chat
                } label: {
                    Label("Active in Chat", systemImage: "bubble.left.and.bubble.right.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .accessibilityHint("Opens the chat view with this model loaded")

                Button("Unload") {
                    model.unloadModel()
                }
                .controlSize(.regular)
                .accessibilityHint("Unloads the model from memory")
            } else if isInstalled {
                Button {
                    if let inst = installedModel ?? model.installed.first(where: { $0.alias == entry.alias }) {
                        model.openChatWithModel(inst)
                    }
                } label: {
                    Label("Load & Chat", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(model.opening || model.isRunning)
                .accessibilityHint("Loads the model into memory and opens chat")

                Button(role: .destructive) {
                    showingDeleteConfirm = true
                } label: {
                    Image(systemName: "trash")
                        .accessibilityLabel("Delete \(entry.alias)")
                }
                .controlSize(.regular)
                .disabled(!model.canDeleteModel)
                .help("Delete model from disk")
                .accessibilityHint("Removes the model files from disk after confirmation")
            } else {
                Button {
                    model.installModel(alias: entry.alias)
                } label: {
                    Label("Download & Install", systemImage: "arrow.down.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(model.isInstallingModel || model.isRunning)
                .accessibilityHint("Downloads and installs \(entry.alias)")
            }
        }
    }

    private var downloadProgressArea: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()

            HStack {
                Text(model.installStageText ?? "Downloading weights...")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.primary)

                Spacer()

                if let fraction = model.installProgressFraction {
                    Text(MetricFormat.percent(fraction * 100))
                        .font(.caption.monospacedDigit().weight(.semibold))
                        .foregroundStyle(Color.accentColor)
                }
            }
            // Combine the stage text and percentage into one announcement
            // so VoiceOver reads "Downloading weights, 42%" rather than two
            // disjoint strings.
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(model.installStageText ?? "Downloading weights")
            .accessibilityValue(model.installProgressFraction.map {
                MetricFormat.percent($0 * 100)
            } ?? "in progress")

            if let fraction = model.installProgressFraction {
                ProgressView(value: fraction)
                    .tint(Color.accentColor)
                    .accessibilityLabel("Download progress")
                    .accessibilityValue(MetricFormat.percent(fraction * 100))
            } else {
                ProgressView()
                    .accessibilityLabel("Download progress")
            }

            HStack {
                if let downloaded = model.installDownloadedBytes, let total = model.installTotalBytes {
                    Text("\(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))")
                }
                Spacer()
                if let eta = model.installETAText {
                    Text(eta)
                }
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
    }
}
