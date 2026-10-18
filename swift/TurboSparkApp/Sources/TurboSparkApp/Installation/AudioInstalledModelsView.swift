import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Installed > Audio: every audio model on this Mac and every pinned one it
/// can download, as a list beside the selected model's detail, with the same
/// lifecycle as Text and Image models.
@MainActor
struct AudioInstalledModelsView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @State private var selectedID: String?
    /// The row Discover Models asked the list to scroll to.
    @State private var scrollTarget: String?
    @Environment(\.appTheme) private var theme

    private var entries: [AudioInstalledEntry] { controller.installedAudioEntries }

    /// The selected row, or the same model in its new group after a download,
    /// adoption or delete moved it. The first row until the user picks one,
    /// since the catalog can arrive after the view does.
    private var selectedEntry: AudioInstalledEntry? {
        guard let selectedID else { return entries.first }
        if let exact = entries.first(where: { $0.id == selectedID }) { return exact }
        let alias = selectedID.split(separator: ":", maxSplits: 1).last.map(String.init)
        return entries.first { $0.alias == alias }
    }

    var body: some View {
        let scale = AppMetrics.textScale(theme)
        VStack(spacing: 0) {
            AudioInstalledHeaderView(controller: controller, entries: entries) {
                guard let first = entries.first(where: { $0.group == .available }) else { return }
                selectedID = first.id
                scrollTarget = first.id
            }
            Divider()
            HStack(spacing: 0) {
                list
                    .frame(minWidth: 260 * scale, idealWidth: 320 * scale, maxWidth: 380 * scale)
                    .frame(maxHeight: .infinity)
                Rectangle().fill(.appBorder).frame(width: AppChromeLayout.dividerWidth)
                detail
                    .frame(minWidth: 280 * scale, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task {
            if controller.profiles.isEmpty && !controller.isRefreshingModels { controller.refreshModels() }
        }
    }

    private var list: some View {
        ScrollViewReader { proxy in
            listContent
                .onChange(of: scrollTarget) { _, target in
                    guard let target else { return }
                    TSMotion.animate(TSMotion.pane) { proxy.scrollTo(target, anchor: .top) }
                    scrollTarget = nil
                }
        }
    }

    private var listContent: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4) {
                if entries.isEmpty {
                    emptyList
                } else {
                    ForEach(AudioInstalledCatalog.grouped(entries), id: \.group) { section in
                        Section {
                            ForEach(section.entries) { entry in
                                AudioInstalledRow(model: model, controller: controller, entry: entry,
                                                  isSelected: selectedEntry?.id == entry.id) { selectedID = entry.id }
                                    .id(entry.id)
                            }
                        } header: {
                            HStack {
                                Text(LocalizedStringKey(section.group.titleKey), bundle: .module)
                                    .themedFont(.small, weight: .bold).foregroundStyle(.appSecondary)
                                    .accessibilityAddTraits(.isHeader)
                                Spacer()
                                Text(verbatim: "\(section.entries.count)")
                                    .themedFont(.tiny).monospacedDigit().foregroundStyle(.appSecondary)
                            }
                            .padding(.horizontal, 8).padding(.top, 10).padding(.bottom, 2)
                        }
                    }
                }
            }
            .padding(8)
        }
    }

    @ViewBuilder private var emptyList: some View {
        if controller.isRefreshingModels {
            ProgressView().controlSize(.small).frame(maxWidth: .infinity).padding(.top, 24)
        } else if let error = controller.modelCatalogError {
            AppEmptyState(systemImage: "exclamationmark.triangle", title: Text("No Models Installed", bundle: .module),
                          message: Text(verbatim: error)) {
                Button { controller.refreshModels() } label: { Text("Retry", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
        } else {
            AppEmptyState(systemImage: "waveform", title: Text("No Models Installed", bundle: .module),
                          message: Text("Rescan to read the audio catalog again.", bundle: .module)) {
                Button { controller.refreshModels() } label: { Text("Rescan", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
        }
    }

    @ViewBuilder private var detail: some View {
        if let entry = selectedEntry {
            AudioInstalledModelDetailView(model: model, controller: controller, entry: entry)
                .id(entry.id)
        } else {
            AppEmptyState(
                systemImage: "internaldrive",
                title: Text("No Model Selected", bundle: .module),
                message: Text("Select an installed model to view specifications, runtime options, or load it into memory.", bundle: .module))
        }
    }
}

/// One list row: the model, its size and the modes that use it, and its state.
@MainActor
struct AudioInstalledRow: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let entry: AudioInstalledEntry
    let isSelected: Bool
    let onSelect: () -> Void
    @Environment(\.appTheme) private var theme

    var body: some View {
        Button(action: onSelect) {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: Self.symbol(entry.profile?.identity.task))
                    .themedFont(.callout, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .frame(width: 30, height: 30)
                    .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: entry.name).themedFont(.callout, weight: .semibold).lineLimit(1)
                    if let size = sizeText { Text(verbatim: size).themedFont(.tiny).monospacedDigit().foregroundStyle(.appSecondary) }
                    if let usedBy { usedBy.themedFont(.tiny).foregroundStyle(.appSecondary).lineLimit(2) }
                }
                Spacer(minLength: 4)
                AudioModelStateLabel(state: controller.entryState(entry, download: entry.profile.flatMap { model.audioDownload(for: $0) }))
            }
            .padding(.horizontal, 8).padding(.vertical, 6)
            .contentShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .background(isSelected ? theme.selection : Color.clear,
                    in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .help(Text(verbatim: entry.alias))
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var sizeText: String? {
        if let bytes = entry.installBytes, bytes > 0 { return MetricFormat.storage(bytes) }
        if let bytes = entry.profile?.downloadBytes, bytes > 0 { return MetricFormat.storage(bytes) }
        return nil
    }

    private var usedBy: Text? {
        let modes = controller.speechToTextModes(using: entry.alias)
        guard !modes.isEmpty else { return nil }
        let names = modes.map { AppStrings.localized(String.LocalizationValue($0.titleKey)) }.joined(separator: ", ")
        return Text("Used by \(names)", bundle: .module)
    }

    static func symbol(_ task: AudioTask?) -> String {
        switch task {
        case .speechToText: return "text.bubble"
        case .textToSpeech: return "speaker.wave.2"
        case .music: return "music.note"
        case .diarization: return "person.2.wave.2"
        default: return "waveform"
        }
    }
}

/// The Installed header for audio: the same title block and action roles as
/// Text, with Rescan, Add Local Model... and Discover Models.
@MainActor
struct AudioInstalledHeaderView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let entries: [AudioInstalledEntry]
    /// Shows the first pinned model this Mac can download.
    let discover: () -> Void

    var body: some View {
        AppAdaptiveHeader {
            ModelWorkspaceTitle(
                systemImage: "waveform", tint: .teal,
                title: Text("Installed Models", bundle: .module),
                badge: Text("LOCAL STORAGE", bundle: .module),
                summary: summary)
        } actions: {
            rescanButton.appAction(.secondary)
            addLocalMenu
            discoverButton
        } compactActions: {
            discoverButton
            Menu {
                rescanButton
                addLocalMenu
            } label: {
                Label { Text("More", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
            }
            .appMenuAction()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var summary: Text {
        let onDisk = entries.filter { $0.group == .installed }
        let bytes = onDisk.reduce(UInt64(0)) { $0 + ($1.installBytes ?? 0) }
        let separator = Text(verbatim: " \u{2022} ")
        var line = Text("Audio models: \(onDisk.count)", bundle: .module)
        if bytes > 0 { line = line + separator + Text("On disk: \(MetricFormat.storage(bytes))", bundle: .module) }
        if let alias = controller.loadedModelAlias { line = line + separator + Text("Active: \(alias)", bundle: .module) }
        return line
    }

    private var rescanButton: some View {
        Button { controller.refreshModels() } label: {
            Label { Text("Rescan", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
        }
        .disabled(controller.isRefreshingModels)
        .help(Text("Rescan storage directories for models", bundle: .module))
    }

    /// Models whose engine can open a folder the user picks. FluidAudio
    /// loads only verified catalog installs, so its models are not listed.
    private var addLocalMenu: some View {
        Menu {
            ForEach(controller.profiles.filter { AudioRuntimeProvider($0).acceptsLocalFolder && AudioRuntimeProvider($0) != .unsupported && $0.identity.task.isRunnable }) { profile in
                Button { controller.chooseModelFolder(for: profile) } label: { Text(verbatim: profile.displayName) }
            }
        } label: {
            Label { Text("Add Local Model...", bundle: .module) } icon: { Image(systemName: "folder.badge.plus") }
        }
        .appMenuAction()
        .disabled(controller.isBusy || controller.hasRecordingActivity)
        .help(Text("Use a checkpoint folder instead of the pinned download. Its files are not checked.", bundle: .module))
    }

    private var discoverButton: some View {
        Button(action: discover) {
            Label { Text("Discover Models", bundle: .module) } icon: { Image(systemName: "shippingbox.fill") }
        }
        .appAction(.primary)
        .disabled(!entries.contains { $0.group == .available })
        .help(Text("Show the pinned audio models this Mac can download", bundle: .module))
    }
}
