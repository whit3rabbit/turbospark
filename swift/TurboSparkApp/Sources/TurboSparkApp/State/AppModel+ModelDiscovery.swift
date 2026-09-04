import Foundation
import TurboSpark

/// What models exist: the catalog's rows, the scanned ones, keeping
/// `selected` pointed at something that is still there, and the two identity
/// questions a delete has to ask.
///
/// **THE SCAN IS OFF THE MAIN ACTOR AND CANCELLABLE, AND BOTH HALVES OF THAT
/// HAVE BEEN WRONG.** It is a recursive `FileManager.enumerator` per root
/// plus a second full walk per bundle found, which beachballed every launch
/// until it moved -- and the catalog rows still land synchronously, because
/// every caller expects `installed` to be usable when `refreshModels`
/// returns. state#51 then found the cancel sitting below the early return,
/// so a disabled setting still landed its rows, and state#87 found the
/// cancel reaching a wrapper rather than the walk itself.
///
/// Split from the LOADING half next door, which is about what is resident
/// rather than about what is on disk.
extension AppModel {
    func uniqueAlias(base: String, suffix: String, existing: Set<String>) -> String {
        var candidate = "\(base) (\(suffix))"
        var counter = 2
        while existing.contains(candidate) {
            candidate = "\(base) (\(suffix) \(counter))"
            counter += 1
        }
        return candidate
    }

    /// Re-reads the installed-model list and re-scans the external
    /// directories.
    ///
    /// **THE DIRECTORY WALKS RUN OFF THE MAIN ACTOR.** This is called from
    /// `AppModel.init()`, and the LM Studio scan (on by default) plus every
    /// custom folder is a recursive `FileManager.enumerator`, with
    /// `directorySize(at:)` a SECOND full walk per bundle found. On a large
    /// library, or one on an external drive, that beachballed every launch.
    /// The catalog rows still land synchronously -- they are one small JSON
    /// read, and every caller expects `installed` to be usable when this
    /// returns -- and the scanned rows merge in when they arrive.
    public func refreshModels() {
        do {
            // **`try?` HID THE ONE ERROR THAT CHANGES BEHAVIOUR.** A corrupt
            // or unreadable `installed.json` presented as "no models
            // installed", and `deleteModel` then told the user a file had
            // been left on disk for a model this app did install.
            var catalogRows: [InstalledModel] = []
            do {
                catalogRows = try TurboSparkCatalog.installed()
            } catch {
                self.error =
                    "Could not read the installed-model list: \(error.localizedDescription). "
                    + "Installed models may be missing from this list."
            }

            installed = catalogRows
            catalog = try TurboSparkCatalog.available()
            telemetry = TurboSparkSession.systemTelemetry
            reconcileSelection()

            let lmPath =
                enableLMStudioDetection
                ? (lmStudioDirectory.isEmpty
                    ? ModelStorageManager.defaultLMStudioModelsDirectory : lmStudioDirectory)
                : nil
            let customDirs = customModelDirectories.filter {
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            // **CANCEL FIRST, THEN DECIDE WHETHER TO SCAN** (state#51). The
            // cancel used to sit BELOW this guard, so turning LM Studio
            // detection off (or clearing the last custom folder) returned
            // here with the previous scan still running -- and it landed its
            // rows through `mergeScannedModels` afterwards, re-adding
            // exactly the models the setting had just excluded.
            modelScanTask?.cancel()
            modelScanTask = nil
            guard lmPath != nil || !customDirs.isEmpty else { return }

            // **THE STORED TASK IS THE ONE DOING THE WALK** (state#87).
            // `modelScanTask` used to hold a MainActor wrapper awaiting an
            // inner `Task.detached`, and cancelling a parent does not cancel
            // a DETACHED child -- so `modelScanTask?.cancel()` above stopped
            // nothing at all and the walk ran to completion whatever the
            // settings said. Detached is still what keeps it off the main
            // actor (this is a recursive `FileManager.enumerator` plus a
            // second full walk per bundle found), so the fix is to store
            // that task rather than to wrap it.
            modelScanTask = Task.detached(priority: .utility) { [weak self] in
                let scanned = ModelStorageManager.scanExternal(
                    lmStudioPath: lmPath, customPaths: customDirs)
                guard !Task.isCancelled, let self else { return }
                await MainActor.run {
                    self.mergeScannedModels(scanned, into: catalogRows)
                }
            }
        } catch {
            self.error = "\(error)"
        }
    }

    /// Folds scanned rows into the catalog rows, de-duplicating by PATH and
    /// uniquifying aliases (state#14: a scanned row's alias is not unique
    /// across sources the way the catalog's own is).
    private func mergeScannedModels(
        _ scanned: [(model: InstalledModel, sourceTag: String)], into catalogRows: [InstalledModel]
    ) {
        var combined = catalogRows
        var existingPaths = Set(
            catalogRows.map { URL(fileURLWithPath: $0.path).standardizedFileURL.path })
        var existingAliases = Set(catalogRows.map { $0.alias })

        let excluded = ModelOrganizationStore.shared.excludedScanPaths
        for entry in scanned {
            let stdPath = URL(fileURLWithPath: entry.model.path).standardizedFileURL.path
            guard !existingPaths.contains(stdPath) else { continue }
            // A row the user removed from TurboSpark stays removed (state#88).
            // The scan finds it again on every refresh, so the exclusion is
            // the only thing that can make the button mean anything.
            guard !excluded.contains(stdPath) else { continue }
            existingPaths.insert(stdPath)
            var resolvedAlias = entry.model.alias
            if existingAliases.contains(resolvedAlias) {
                resolvedAlias = uniqueAlias(
                    base: entry.model.alias, suffix: entry.sourceTag, existing: existingAliases)
            }
            existingAliases.insert(resolvedAlias)
            combined.append(
                InstalledModel(
                    alias: resolvedAlias,
                    repo: entry.model.repo,
                    revision: entry.model.revision,
                    path: entry.model.path,
                    family: entry.model.family,
                    installBytes: entry.model.installBytes,
                    installedOn: entry.model.installedOn
                ))
        }
        installed = combined
        reconcileSelection()
    }

    /// Keeps `selected` pointing at a row that still exists.
    ///
    /// Keyed on `path`, not `alias` (state#14/U5/U6): `InstalledModel.id` is
    /// the alias, and a scanned LM Studio/Custom row's alias is not
    /// guaranteed unique across sources the way the catalog's own is, so
    /// alias-based lookups can silently rebind to the WRONG row once two rows
    /// happen to share one. Path is the one field that actually identifies a
    /// single file on disk.
    private func reconcileSelection() {
        if let currentPath = selected?.path, !installed.contains(where: { $0.path == currentPath }) {
            selected = installed.first
        } else if selected == nil {
            selected = installed.first
        }
        if let selected {
            modelPathText = selected.path
        }
    }

    /// Whether this app installed `model` itself, and may therefore delete
    /// its bytes.
    ///
    /// **MATCHED ON PATH ALONE.** This used to be `entryPath == stdPath ||
    /// entry.alias == model.alias`, so a SCANNED row (LM Studio, a custom
    /// folder) whose alias happened to collide with a catalog install passed
    /// -- and `TurboSparkCatalog.delete(model.alias)` then removed the
    /// CATALOG's copy, at a different path, while the row the user clicked
    /// stayed on disk. `refreshModels` de-duplicates by path and uniquifies
    /// aliases only for the rows it merges IN, so a collision between a
    /// scanned row and a catalog row is reachable. Path is the one field that
    /// identifies a single thing on disk.
    ///
    /// A pure static so it can be tested against a fixture: the inline
    /// version could only be exercised against whatever `installed.json`
    /// happened to be on the machine.
    /// Whether the running server is holding this model (state#43).
    ///
    /// A pure static for the reason `isCatalogTracked` is one: the live form
    /// reads `serverAttachedSessions`, whose values are real
    /// `TurboSparkSession`s, so nothing about it could be asserted without a
    /// Metal device and an install (`swift/CLAUDE.md` Gotcha 26).
    ///
    /// The server keys an attachment by the id it serves it under, which is
    /// the alias for a model attached from the Server pane and the path for
    /// one attached by path, so both are checked -- unlike state#14's
    /// identity question, where `path` alone is correct because it names a
    /// thing on disk. Here the question is "is this string in the server's
    /// table", and a false positive costs a refused delete while a false
    /// negative deletes a served model's bytes.
    static func isAttachedToServer(model: InstalledModel, servedIDs: Set<String>) -> Bool {
        servedIDs.contains(model.alias) || servedIDs.contains(model.path)
    }

    static func isCatalogTracked(model: InstalledModel, in catalogRows: [InstalledModel]) -> Bool {
        let stdPath = URL(fileURLWithPath: model.path).standardizedFileURL.path
        return catalogRows.contains { entry in
            URL(fileURLWithPath: entry.path).standardizedFileURL.path == stdPath
        }
    }
}
