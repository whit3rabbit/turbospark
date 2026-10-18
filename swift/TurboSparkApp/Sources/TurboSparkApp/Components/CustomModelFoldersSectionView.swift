import AppKit
import SwiftUI
import TurboSpark

/// Settings section for managing additional external folders to scan for models.
// Isolated explicitly: its helpers call each other, and only `body` is
// isolated by the protocol on the macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct CustomModelFoldersSectionView: View {
    @ObservedObject var model: AppModel
    /// **OBSERVED DIRECTLY**, like the Model libraries section: `AppModel` does
    /// not publish it (a mount would redraw every screen), and a mount or an
    /// unmount is exactly when a row here changes.
    @ObservedObject private var libraryState: LibraryState

    /// Read so the strings built below follow the language picker: they come
    /// from `AppStrings`, which has no dependency SwiftUI can see.
    @Environment(\.locale) private var locale

    /// What the last `reread()` found per folder: whether it can be read and how
    /// many models are in it.
    ///
    /// `ModelStorageManager.scanModels` walks the whole directory tree with
    /// a `FileManager.enumerator`, stat-ing every entry -- real I/O, not a
    /// property read. It used to run inline in `body`, so SwiftUI re-ran the
    /// walk, for every configured folder, on every body evaluation this view
    /// received for any reason (`swift/docs/SWIFT_SETTINGS_AUDIT.md`). `.task(id:)`
    /// below reruns this exactly when the folder list changes or a volume
    /// comes or goes.
    @State private var readings: [String: ScanFolderReading] = [:]

    init(model: AppModel) {
        self.model = model
        _libraryState = ObservedObject(wrappedValue: model.libraryState)
    }

    /// Why `reread()` runs: the folder list, or a drive coming or going.
    ///
    /// **THE VOLUME COUNT IS THE TRIGGER, NOT THE LIBRARY LIST.** A scan folder
    /// on a drive that holds no library changes nothing in `libraries`, and the
    /// row must still flip when the drive is plugged in. `LibraryState` already
    /// counts the mounts, unmounts and renames the process-wide volume monitor
    /// reports, so no second workspace observer is added here.
    private struct ReadTrigger: Equatable {
        let folders: [String]
        let volumeChanges: Int
    }

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppAdaptiveHeader {
                Label { Text("Additional Model Folders", bundle: .module) } icon: { Image(systemName: "folder.badge.plus") }
                    .settingsControl("Additional Model Folders", pane: .models, timing: .action)
                    .themedFont(.base, weight: .semibold)
            } actions: {
                Button {
                    addCustomFolder()
                } label: {
                    Label {
                        Text("Add Folder\u{2026}", bundle: .module)
                    } icon: {
                        Image(systemName: "plus")
                    }
                }
                .appAction(.secondary, size: .small)
            }

            VStack(alignment: .leading, spacing: 10) {
                if model.customModelDirectories.isEmpty {
                    HStack {
                        Spacer()
                        VStack(spacing: 6) {
                            Image(systemName: "folder.badge.questionmark")
                                .themedFont(.title2)
                                .foregroundStyle(.appSecondary)
                                .accessibilityHidden(true)
                            Text("No additional model scan folders configured.", bundle: .module)
                                .themedFont(.small)
                                .foregroundStyle(.appSecondary)
                        }
                        .padding(.vertical, 16)
                        Spacer()
                    }
                } else {
                    ForEach(model.customModelDirectories, id: \.self) { dir in
                        folderRow(dir)
                    }
                }

                Text("Add folders on external drives or secondary locations to scan for .gturbo bundles and .gguf models.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            .modelsSettingsCard()
        }
        .task(id: ReadTrigger(folders: model.customModelDirectories, volumeChanges: libraryState.volumeChangeCount)) {
            await reread()
        }
    }

    private func folderRow(_ dir: String) -> some View {
        ScanFolderRowView(
            row: ScanFolderRowModel(path: dir, reading: readings[dir], language: language),
            reveal: { ModelStorageManager.revealInFinder(path: dir) },
            remove: { removeCustomFolder(dir) })
    }

    /// Reads every folder off the main actor: a folder on an external or network
    /// drive can take seconds, and the walk used to block the UI.
    private func reread() async {
        guard let fresh = await Self.readOffMainActor(model.customModelDirectories) else { return }
        readings = fresh
    }

    /// `readFolders` on a utility thread. Nil when the caller was cancelled (the
    /// folder list changed again, or a second volume event arrived): a reading
    /// that was superseded is never applied.
    ///
    /// The detached task is cancelled with the caller. A detached task does not
    /// inherit cancellation, and without this a hub with three drives would start
    /// three full walks that all run to the end.
    nonisolated static func readOffMainActor(
        _ folders: [String],
        status: @escaping @Sendable (String) -> ScanFolderStatus = { ModelStorageManager.scanFolderStatus($0) },
        count: @escaping @Sendable (String) -> Int = { CustomModelFoldersSectionView.modelCount(in: $0) }
    ) async -> [String: ScanFolderReading]? {
        let work = Task.detached(priority: .utility) {
            CustomModelFoldersSectionView.readFolders(folders, status: status, count: count)
        }
        let result = await withTaskCancellationHandler {
            await work.value
        } onCancel: {
            work.cancel()
        }
        return Task.isCancelled ? nil : result
    }

    /// Whether each folder can be read and, for a connected one, how many
    /// models it holds.
    ///
    /// **A FOLDER THAT CANNOT BE READ IS NOT WALKED** (requirement 3.9): the
    /// status is a stat of the mount point, and the count, a recursive walk, runs
    /// only for a folder that exists. A repeated path is read once.
    nonisolated static func readFolders(
        _ folders: [String],
        status: (String) -> ScanFolderStatus = { ModelStorageManager.scanFolderStatus($0) },
        count: (String) -> Int = { CustomModelFoldersSectionView.modelCount(in: $0) }
    ) -> [String: ScanFolderReading] {
        var statuses: [String: ScanFolderStatus] = [:]
        var connected: [String] = []
        for folder in folders where statuses[folder] == nil {
            if Task.isCancelled { return [:] }
            let found = status(folder)
            statuses[folder] = found
            if found == .connected { connected.append(folder) }
        }
        let counts = folderCounts(for: connected, count: count)
        return statuses.reduce(into: [:]) { result, entry in
            result[entry.key] = ScanFolderReading(status: entry.value, modelCount: counts[entry.key] ?? 0)
        }
    }

    /// `customModelDirectories` is decoded leniently and never deduplicated,
    /// so a repeated path must not trap the dictionary build.
    nonisolated static func folderCounts(
        for folders: [String], count: (String) -> Int = { CustomModelFoldersSectionView.modelCount(in: $0) }
    ) -> [String: Int] {
        var result: [String: Int] = [:]
        for folder in folders where result[folder] == nil {
            result[folder] = count(folder)
        }
        return result
    }

    /// Image installs live in the same folders and are found by the image scan,
    /// not the text-model one, so the count adds both.
    nonisolated static func modelCount(in folder: String) -> Int {
        ModelStorageManager.scanModels(in: folder, sourceTag: "Custom").count
            + ((try? TurboSparkCatalog.imageInstalled(in: ModelStorageManager.expandPath(folder)))?.count ?? 0)
    }

    private func addCustomFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = AppStrings.localized("Add Scan Folder")
        panel.message = AppStrings.localized("Select a folder containing models to scan")

        if panel.runModal() == .OK, let url = panel.url {
            let path = url.path
            if !model.customModelDirectories.contains(path) {
                model.customModelDirectories.append(path)
                model.persistSettings()
                model.refreshModels()
                model.showToast(AppStrings.localized("Added model scan folder: \(url.lastPathComponent)"), style: .success)
            }
        }
    }

    private func removeCustomFolder(_ folder: String) {
        model.customModelDirectories.removeAll { $0 == folder }
        model.persistSettings()
        model.refreshModels()
        model.showToast(AppStrings.localized("Removed model scan folder"), style: .info)
    }
}
