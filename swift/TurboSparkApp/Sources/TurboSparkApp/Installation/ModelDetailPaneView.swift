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
                hardwareFitCard
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

                    statusSeal
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
                    .help("Model Family: \(visuals.family)")

                    Text("•")
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

    /// Whether this machine has actually sized the row.
    ///
    /// A `.unknown` verdict means the sizing could not be determined, and the
    /// numeric fields that come with it are zeros rather than measurements.
    private var fitIsKnown: Bool {
        guard let recommendation else { return false }
        return recommendation.verdict != .unknown
    }

    @ViewBuilder
    private var hardwareFitCard: some View {
        if let recommendation, fitIsKnown {
            knownFitCard(recommendation)
        } else {
            unprobedFitCard
        }
    }

    private func knownFitCard(_ rec: ModelRecommendation) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "gauge.with.needle")
                    .font(.subheadline)
                    .foregroundStyle(Color.accentColor)
                    .help("Hardware compatibility estimate")
                Text("Hardware Compatibility")
                    .font(.headline)

                Spacer()

                verdictPill(rec.verdict)
            }

            Text(rec.verdictSummary)
                .font(.callout)
                .foregroundStyle(.secondary)

            Divider()

            // EVERY CELL IS GUARDED ON BEING NON-ZERO. A zero here is an
            // absent reading, not a measurement of zero, and it rendered as
            // "Zero KB" and "0 tokens" -- which read like facts about the
            // model rather than like missing data.
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 110, maximum: 200), spacing: 12)], spacing: 12) {
                if rec.countedBytes > 0 {
                    metricCell(title: "Unified Memory", value: MetricFormat.storage(rec.countedBytes))
                }
                if rec.installBytes > 0 {
                    metricCell(title: "On-Disk Storage", value: MetricFormat.storage(rec.installBytes))
                }
                if rec.largestContext > 0 {
                    metricCell(title: "Max Context", value: "\(rec.largestContext.formatted()) tokens")
                }
                if rec.slotCacheSlots > 0 {
                    metricCell(title: "Expert Slots", value: "\(rec.slotCacheSlots) slots")
                }
                if let minS = rec.toksPerSecondMin, let maxS = rec.toksPerSecondMax {
                    metricCell(
                        title: "Expected Speed",
                        value: "\(String(format: "%.1f", minS))-\(String(format: "%.1f", maxS)) tok/s")
                }
            }
        }
        .modelCardStyle()
    }

    /// Shown when the row has no sizing for this machine.
    ///
    /// The previous version rendered the recommendation's zeros through the
    /// same grid, so an unsized row advertised "Unified Memory: Zero KB",
    /// "Max Context: 0 tokens" and "Expert Slots: 16 slots". The last one is
    /// the worst of the three and is a documented trap: with no architecture
    /// read there is no expert stride, `Auto` divides by nothing and returns
    /// `DEFAULT_CACHE_SLOTS`, so 16 is the answer arrived at BY IGNORANCE and
    /// looks identical to a real one (root `CLAUDE.md` Gotcha 58).
    ///
    /// What is shown instead is only what the catalog itself states, plus the
    /// command that would produce the missing numbers. There is deliberately
    /// no Probe button: `ts_recommend_json` is offline-only and this binding
    /// exposes no probing call, so a button here could not do the work.
    private var unprobedFitCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "gauge.with.needle")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
                Text("Hardware Compatibility")
                    .font(.headline)

                Spacer()

                Text("Not sized")
                    .font(.caption.weight(.bold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.secondary.opacity(0.16), in: Capsule())
                    .foregroundStyle(.secondary)
            }

            Text("This row has no memory sizing for this machine, so whether it fits, how much it would pin, and its largest usable context are all unknown.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 110, maximum: 200), spacing: 12)], spacing: 12) {
                if entry.installBytes > 0 {
                    metricCell(title: "On-Disk Storage", value: MetricFormat.storage(entry.installBytes))
                }
                if entry.downloadBytes > 0 {
                    metricCell(title: "Download", value: MetricFormat.storage(entry.downloadBytes))
                }
            }

            Text("Read the headers to size it: turbospark-model recommend --probe")
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .modelCardStyle()
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Hardware compatibility: not sized on this machine")
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
                    .help("\(visuals.family) architecture specifications")
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
