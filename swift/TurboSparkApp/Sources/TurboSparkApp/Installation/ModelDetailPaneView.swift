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
            downloadBytes: entry.downloadBytes,
            freeDiskBytes: ModelInstallGate.freeSpaceOnModelStore(),
            unmountedStoreVolume: ModelInstallGate.unmountedStoreVolume())
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
        model.isInstallingModel && model.installingAlias == entry.alias
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
            Text("Delete \(entry.alias)?", bundle: .module),
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
                    Text(verbatim: entry.name)
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
                    Text(verbatim: entry.alias)
                        .themedCode(.small, weight: .medium)
                        .foregroundStyle(.appSecondary)

                    Text(verbatim: "\u{2022}")
                        .foregroundStyle(.tertiary)

                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .themedFont(.tiny)
                            .foregroundStyle(visuals.accentColor)
                        Text(verbatim: visuals.family)
                    }
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .help(Text("Model Family: \(visuals.family)", bundle: .module))

                    Text(verbatim: "\u{2022}")
                        .foregroundStyle(.tertiary)

                    statusText
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
                .help(Text("Verified: this port has run this row end to end", bundle: .module))
                .accessibilityHidden(true)
        case "caveat":
            Image(systemName: "exclamationmark.triangle.fill")
                .themedFont(.title3)
                .foregroundStyle(.orange)
                .help(Text("Runs with a caveat; see the model notes.", bundle: .module))
                .accessibilityHidden(true)
        default:
            EmptyView()
        }
    }

    /// The catalog row's `status` in words. An unexpected value is data
    /// from `models.json` and is shown as written.
    private var statusText: Text {
        switch entry.status {
        case "verified": return Text("Verified", bundle: .module)
        case "caveat": return Text("Caveat", bundle: .module)
        case "runs": return Text("Runs", bundle: .module)
        default: return Text(verbatim: entry.status.capitalized)
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
        }
        .appAction(.secondary, size: .small)
        .help(Text("Copy model alias to clipboard", bundle: .module))
    }

    private var capabilityTags: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                HStack(spacing: 4) {
                    Image(systemName: visuals.iconSystemName)
                        .themedFont(.tiny)
                        .foregroundStyle(visuals.accentColor)
                    Text(verbatim: visuals.family)
                        .themedFont(.small, weight: .semibold)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(visuals.accentColor.opacity(0.15), in: Capsule())
                .overlay(Capsule().stroke(visuals.accentColor.opacity(0.3), lineWidth: 0.5))

                ForEach(visuals.capabilities, id: \.self) { tag in
                    // Capability words are catalog keys; a name with no key
                    // renders as written.
                    Text(LocalizedStringKey(tag), bundle: .module)
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
                        Text(verbatim: visuals.formatLabel)
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
                .appAction(.secondary)
            } else if isActive {
                Button {
                    model.activeSection = .chat
                } label: {
                    Label { Text("Active in Chat", bundle: .module) } icon: { Image(systemName: "bubble.left.and.bubble.right.fill") }
                }
                .appAction(.primary)
                .accessibilityHint(Text("Opens the chat view with this model loaded", bundle: .module))

                Button {
                    model.unloadModel()
                } label: { Text("Unload", bundle: .module) }
                .appAction(.secondary)
                // state#85: `ModelLoaderControl` already read this predicate
                // and the two detail panes did not.
                .disabled(!model.canUnloadModel)
                .accessibilityHint(Text("Unloads the model from memory", bundle: .module))
            } else if isInstalled {
                Button {
                    if let inst = installedModel ?? model.installed.first(where: { $0.alias == entry.alias }) {
                        model.openChatWithModel(inst)
                    }
                } label: {
                    Label { Text("Load & Chat", bundle: .module) } icon: { Image(systemName: "play.fill") }
                }
                .appAction(.primary)
                .disabled(model.opening || model.isRunning)
                .accessibilityHint(Text("Loads the model into memory and opens chat", bundle: .module))

                AppIconButton("trash", label: Text("Delete Model", bundle: .module), role: .destructive) {
                    showingDeleteConfirm = true
                }
                .appAction(.destructive)
                .disabled(!model.canDeleteModel)
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
                .appAction(.primary)
                .disabled(
                    model.isRunning || installDecision.isBlocked
                        || !model.canInstall(alias: entry.alias))
                .accessibilityHint(Text("Downloads and installs \(entry.alias)", bundle: .module))
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
                stageText
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appText)

                Spacer()

                if let fraction = model.installProgressFraction {
                    Text(verbatim: MetricFormat.percent(fraction * 100))
                        .themedFont(.small, weight: .semibold).monospacedDigit()
                        .foregroundStyle(Color.accentColor)
                }
            }
            // Combine the stage text and percentage into one announcement
            // so VoiceOver reads "Downloading weights, 42%" rather than two
            // disjoint strings.
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(stageText)
            .accessibilityValue(
                model.installProgressFraction.map { Text(verbatim: MetricFormat.percent($0 * 100)) }
                    ?? Text("In progress", bundle: .module))

            if let fraction = model.installProgressFraction {
                AppProgressBar(value: fraction, label: Text("Download progress", bundle: .module),
                               animated: !model.isInstallPaused)
            } else {
                AppProgressBar(value: nil, label: Text("Download progress", bundle: .module))
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

    /// The installer's live stage line (already a display string), or the
    /// catalog's placeholder before the first stage arrives.
    private var stageText: Text {
        if let stage = model.installStageText { return Text(stage) }
        return Text("Downloading model files", bundle: .module)
    }
}
