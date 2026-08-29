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
                if let rec = recommendation {
                    hardwareFitCard(rec)
                }
                technicalSpecificationsCard
                if let notes = entry.notes, !notes.isEmpty {
                    notesCard(notes)
                }
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

                    Image(systemName: "checkmark.seal.fill")
                        .font(.title3)
                        .foregroundStyle(Color.accentColor)
                        .help("Verified TurboSpark Model")
                        .accessibilityHidden(true)
                }

                HStack(spacing: 8) {
                    Text(entry.alias)
                        .font(.subheadline.monospaced().weight(.medium))
                        .foregroundStyle(.secondary)

                    Text("•")
                        .foregroundStyle(.tertiary)

                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .font(.caption2)
                            .foregroundStyle(visuals.accentColor)
                        Text(visuals.family)
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                    Text("•")
                        .foregroundStyle(.tertiary)

                    Text(entry.status.capitalized)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(entry.status == "verified" ? .green : .secondary)
                }
            }

            Spacer()

            copyAliasButton
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
                        Text("•")
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
                .disabled(model.opening || model.isRunning)
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

    private func hardwareFitCard(_ rec: ModelRecommendation) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "gauge.with.needle")
                    .font(.subheadline)
                    .foregroundStyle(Color.accentColor)
                Text("Hardware Compatibility")
                    .font(.headline)

                Spacer()

                verdictPill(rec.verdict)
            }

            Text(rec.verdictSummary)
                .font(.callout)
                .foregroundStyle(.secondary)

            Divider()

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 110, maximum: 200), spacing: 12)], spacing: 12) {
                metricCell(title: "Unified Memory", value: MetricFormat.storage(rec.countedBytes))
                metricCell(title: "On-Disk Storage", value: MetricFormat.storage(rec.installBytes))
                metricCell(title: "Max Context", value: "\(rec.largestContext.formatted()) tokens")

                if rec.slotCacheSlots > 0 {
                    metricCell(title: "Expert Slots", value: "\(rec.slotCacheSlots) slots")
                }

                if let minS = rec.toksPerSecondMin, let maxS = rec.toksPerSecondMax {
                    metricCell(title: "Expected Speed", value: "\(String(format: "%.1f", minS))–\(String(format: "%.1f", maxS)) tok/s")
                }
            }
        }
        .modelCardStyle()
    }

    private func metricCell(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.monospacedDigit().weight(.medium))
        }
    }

    @ViewBuilder
    private func verdictPill(_ verdict: ModelRecommendation.FitVerdict) -> some View {
        switch verdict {
        case .resident:
            Text("Full Resident Fit")
                .font(.caption.weight(.bold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.green.opacity(0.18), in: Capsule())
                .foregroundStyle(.green)
        case .streams:
            Text("Fast MoE Streaming")
                .font(.caption.weight(.bold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.blue.opacity(0.18), in: Capsule())
                .foregroundStyle(.blue)
        case .tight:
            Text("Tight Memory Headroom")
                .font(.caption.weight(.bold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.orange.opacity(0.18), in: Capsule())
                .foregroundStyle(.orange)
        case .refused:
            Text("Exceeds Memory Limit")
                .font(.caption.weight(.bold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.red.opacity(0.18), in: Capsule())
                .foregroundStyle(.red)
        case .unknown:
            EmptyView()
        }
    }

    private var technicalSpecificationsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: visuals.iconSystemName)
                    .foregroundStyle(visuals.accentColor)
                Text("Technical Overview")
                    .font(.headline)
            }

            VStack(spacing: 8) {
                specRow(label: "Architecture", value: visuals.architectureType)
                specRow(label: "Format & Quant", value: visuals.formatLabel)
                specRow(label: "Download Size", value: MetricFormat.storage(entry.downloadBytes))
                specRow(label: "Estimated Install", value: MetricFormat.storage(entry.installBytes))
                if let inst = installedModel {
                    specRow(label: "Installed Path", value: inst.path)
                    specRow(label: "Installed On", value: inst.installedOn)
                }
            }
        }
        .modelCardStyle()
    }

    private func specRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private func notesCard(_ notes: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Benchmark & Architecture Notes")
                .font(.headline)

            Text(notes)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .modelCardStyle()
    }
}

private extension View {
    func modelCardStyle() -> some View {
        self
            .padding(16)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5))
    }
}
