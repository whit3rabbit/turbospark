import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Right-column detail inspector and action pane for a selected model.
@MainActor
struct ModelDetailPaneView: View {
    @ObservedObject var model: AppModel
    let entry: CatalogEntry
    let installedModel: InstalledModel?
    let recommendation: ModelRecommendation?
    @State private var showingDeleteConfirm = false
    @State private var showsInstallConfirm = false

    /// What this machine says about installing this row.
    ///
    /// No probe here: a curated row is curated, so the memory verdict and the
    /// free disk are the two questions left. `ModelProbeSheet` adds the probe
    /// arm for an arbitrary repository.
    private var installDecision: ModelInstallDecision {
        ModelInstallGate.decide(
            probeRunnable: nil,
            refusedBecause: nil,
            verdict: recommendation?.verdict,
            installBytes: recommendation?.installBytes ?? entry.installBytes,
            freeDiskBytes: ModelInstallGate.freeSpace(at: AppStorageRoot.directory))
    }

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
                ModelSteeringCardView(
                    model: model,
                    catalogEntry: entry,
                    installedModel: installedModel
                )
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
            Button(role: .destructive) {
                if let inst = installedModel ?? model.installed.first(where: { $0.alias == entry.alias }) {
                    model.deleteModel(inst)
                }
            } label: { Text("Delete Model", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This will remove the downloaded model files from disk.", bundle: .module)
        }
    }

    private var heroHeader: some View {
        HStack(alignment: .top, spacing: 16) {
            ModelLogoView(visuals: visuals, size: 56, cornerRadius: 14)

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(entry.name)
                        .themedFont(.title2, weight: .bold)
                        .lineLimit(2)

                    Text("CATALOG", bundle: .module)
                        .themedFont(.micro, weight: .bold)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(.appAccent.opacity(0.16), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.appAccent)

                    statusSeal
                }

                HStack(spacing: 8) {
                    Text(entry.alias)
                        .themedCode(.small, weight: .medium)
                        .foregroundStyle(.appSecondary)

                    Text(verbatim: "\u{2022}")
                        .foregroundStyle(.tertiary)

                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .themedFont(.tiny)
                            .foregroundStyle(visuals.accentColor)
                        Text(visuals.family)
                    }
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .help("Model Family: \(visuals.family)")

                    Text(verbatim: "\u{2022}")
                        .foregroundStyle(.tertiary)

                    Text(entry.status.capitalized)
                        .themedFont(.small, weight: .medium)
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
                .themedFont(.title3)
                .foregroundStyle(.appAccent)
                .help("Verified: this port has run this row end to end")
                .accessibilityHidden(true)
        case "caveat":
            Image(systemName: "exclamationmark.triangle.fill")
                .themedFont(.title3)
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
            Label { Text("Copy Alias", bundle: .module) } icon: { Image(systemName: "doc.on.doc") }
                .themedFont(.small)
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
                        .themedFont(.tiny)
                        .foregroundStyle(visuals.accentColor)
                    Text(visuals.family)
                        .themedFont(.small, weight: .semibold)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(visuals.accentColor.opacity(0.15), in: Capsule())
                .overlay(Capsule().stroke(visuals.accentColor.opacity(0.3), lineWidth: 0.5))

                ForEach(visuals.capabilities, id: \.self) { tag in
                    Text(tag)
                        .themedFont(.small, weight: .medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.appSurface, in: Capsule())
                        .overlay(Capsule().stroke(.appBorder.opacity(0.6), lineWidth: 0.5))
                }
            }
        }
    }

    private var actionCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Quantization & Format", bundle: .module)
                        .themedFont(.small, weight: .medium)
                        .foregroundStyle(.appSecondary)
                    HStack(spacing: 6) {
                        Text(visuals.formatLabel)
                            .themedFont(.base, weight: .semibold)
                        Text(verbatim: "\u{2022}")
                            .foregroundStyle(.tertiary)
                        Text("\(MetricFormat.storage(entry.downloadBytes)) download", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    }
                }

                Spacer()

                actionButtons
            }

            // The cause sits BESIDE the control, in the card's vertical
            // stack rather than the button row. A disabled button whose
            // reason lives only in a tooltip reads as a broken button.
            if let reason = installDecision.reason, !isInstalled {
                Label(
                    reason,
                    systemImage: installDecision.isBlocked
                        ? "xmark.octagon.fill" : "exclamationmark.triangle.fill"
                )
                .themedFont(.small)
                .foregroundStyle(installDecision.isBlocked ? Color.red : Color.orange)
                .fixedSize(horizontal: false, vertical: true)
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
                Button(role: .cancel) {
                    model.cancelInstall()
                } label: { Text("Cancel Download", bundle: .module) }
                .controlSize(.regular)
            } else if isActive {
                Button {
                    model.activeSection = .chat
                } label: {
                    Label { Text("Active in Chat", bundle: .module) } icon: { Image(systemName: "bubble.left.and.bubble.right.fill") }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .accessibilityHint("Opens the chat view with this model loaded")

                Button {
                    model.unloadModel()
                } label: { Text("Unload", bundle: .module) }
                .controlSize(.regular)
                // state#85: `ModelLoaderControl` already read this predicate
                // and the two detail panes did not.
                .disabled(!model.canUnloadModel)
                .accessibilityHint("Unloads the model from memory")
            } else if isInstalled {
                Button {
                    if let inst = installedModel ?? model.installed.first(where: { $0.alias == entry.alias }) {
                        model.openChatWithModel(inst)
                    }
                } label: {
                    Label { Text("Load & Chat", bundle: .module) } icon: { Image(systemName: "play.fill") }
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
                    if case .confirm = installDecision {
                        showsInstallConfirm = true
                    } else {
                        model.installModel(alias: entry.alias)
                    }
                } label: {
                    Label { Text("Download & Install", bundle: .module) } icon: { Image(systemName: "arrow.down.circle.fill") }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(model.isInstallingModel || model.isRunning || installDecision.isBlocked)
                .accessibilityHint("Downloads and installs \(entry.alias)")
                .alert(Text("Install anyway?", bundle: .module), isPresented: $showsInstallConfirm) {
                    Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
                    Button { model.installModel(alias: entry.alias) } label: { Text("Install", bundle: .module) }
                } message: {
                    Text(installDecision.reason ?? "")
                }
            }
        }
    }

    private var downloadProgressArea: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()

            HStack {
                Text(model.installStageText ?? "Downloading weights...")
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appText)

                Spacer()

                if let fraction = model.installProgressFraction {
                    Text(MetricFormat.percent(fraction * 100))
                        .themedFont(.small, weight: .semibold).monospacedDigit()
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
                    Text("\(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))", bundle: .module)
                }
                Spacer()
                if let eta = model.installETAText {
                    Text(eta)
                }
            }
            .themedFont(.tiny).monospacedDigit()
            .foregroundStyle(.appSecondary)
        }
    }
}
