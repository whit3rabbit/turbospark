import Foundation
import TurboSpark

extension AppModel {
    /// Appends `(suffix)` to `base`, and if that still collides, keeps
    /// numbering (`(suffix 2)`, `(suffix 3)`, ...) until it finds a name not
    /// already in `existing`. The one-shot version of this (try the plain
    /// suffix once, insert whatever comes out) collided silently on a THIRD
    /// model sharing the same base alias -- `existingAliases.insert` on an
    /// already-present value is a no-op, so two distinct `InstalledModel`
    /// rows ended up with the identical alias, which is exactly the field
    /// `selectModel`/`deleteModel` used to key off of (state#14).
    // `internal` rather than `private`: exercised directly by
    // ModelIdentityTests via `@testable import`.
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
            guard lmPath != nil || !customDirs.isEmpty else { return }

            modelScanTask?.cancel()
            modelScanTask = Task {
                let scanned = await Task.detached(priority: .utility) {
                    ModelStorageManager.scanExternal(lmStudioPath: lmPath, customPaths: customDirs)
                }.value
                guard !Task.isCancelled else { return }
                self.mergeScannedModels(scanned, into: catalogRows)
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

        for entry in scanned {
            let stdPath = URL(fileURLWithPath: entry.model.path).standardizedFileURL.path
            guard !existingPaths.contains(stdPath) else { continue }
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

    public func selectModel(_ model: InstalledModel) {
        // Keyed on path (state#14): two distinct rows can share an alias
        // when one is a scanned LM Studio/Custom entry, so alias equality
        // here could read "already selected" for a DIFFERENT model on disk
        // and silently refuse to switch to it.
        guard !generating, selected?.path != model.path else { return }
        selected = model
        modelPathText = model.path
        Task {
            await open(model)
        }
    }

    public func openModelHub() {
        activeSection = .modelHub
    }

    public func openChatWithModel(_ model: InstalledModel) {
        if selected?.path != model.path {
            selectModel(model)
        }
        activeSection = .chat
    }

    /// Re-reads machine telemetry (thermal state, memory pressure, Low Power
    /// Mode). Cheap: two `NSProcessInfo` sends and a sysctl, no allocation.
    ///
    /// **Assigns only when the value CHANGED.** `telemetry` is `@Published`
    /// on a `@MainActor` model, so an unconditional write on every poll would
    /// fire `objectWillChange` twice a second and re-evaluate the whole
    /// window body for a reading that did not move.
    public func refreshTelemetry() {
        let fresh = TurboSparkSession.systemTelemetry
        if fresh != telemetry {
            telemetry = fresh
        }
    }

    /// The guard both the loader and the model hub must use.
    ///
    /// **One accessor rather than each caller building its own.** The ranking
    /// and the loader's refusal share one memory budget by construction, and
    /// that is what makes a hub verdict worth showing; a hub ranking under
    /// `.relaxed` while sessions open under `.strict` promises a fit the
    /// loader then refuses, where the user cannot see the two disagree.
    public var activeLoadGuard: OpenOptions.LoadGuard {
        runtimeOptions.loadGuard.loadGuard(customBytes: runtimeOptions.loadGuardCustomBytes)
    }

    public func buildOpenOptions() -> OpenOptions {
        var options = OpenOptions()
        if maxContextTokens > 0 {
            options.maxContext = .fixed(UInt32(maxContextTokens))
        }
        if runtimeOptions.expertCacheSlots > 0 {
            options.expertCacheSlots = .fixed(UInt32(runtimeOptions.expertCacheSlots))
        }
        options.powerProfile = runtimeOptions.powerProfile.powerProfile
        // Always set, including `.relaxed`, so the value the hub RANKED under
        // and the value a session OPENS under are the same object rather than
        // two defaults that happen to agree today.
        options.loadGuard = activeLoadGuard
        if runtimeOptions.minAutoContextTokens > 0 {
            options.minAutoContext = runtimeOptions.minAutoContextTokens
        }
        options.speculation = runtimeOptions.speculation.speculation
        options.speculativeDrafter = runtimeOptions.speculativeDrafter.drafter
        if runtimeOptions.maxTokensPerSec > 0 {
            options.maxTokensPerSec = runtimeOptions.maxTokensPerSec
        }
        let sPath = (runtimeOptions.steeringPath ?? steeringPath)?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let sPath, !sPath.isEmpty {
            options.steering = sPath
            options.steeringMode = runtimeOptions.steeringMode.steeringMode
            options.steeringScale = runtimeOptions.steeringScale
            let layers = runtimeOptions.steeringLayers.trimmingCharacters(in: .whitespacesAndNewlines)
            if !layers.isEmpty {
                options.steeringLayers = layers
            }
            if runtimeOptions.steeringMode == .clamp {
                options.steeringTarget = runtimeOptions.steeringTarget
            }
            if runtimeOptions.steeringGate > 0 {
                options.steeringGate = runtimeOptions.steeringGate
            }
        }
        return options
    }

    public func restoreReasoningPreference(for model: InstalledModel) {
        if let saved = modelReasoningDefaults[model.alias],
           let level = GenerateOptions.Reasoning(rawValue: saved) {
            self.reasoning = level
        } else if let saved = modelReasoningDefaults[model.path],
                  let level = GenerateOptions.Reasoning(rawValue: saved) {
            self.reasoning = level
        }
    }

    public func open(_ model: InstalledModel) async {
        guard !generating else { return }
        opening = true
        error = nil
        // **DETACHED RATHER THAN MERELY DROPPED, and detached rather than
        // stopping the whole server.** `session = nil` alone leaves the OLD
        // model resident and served: the server holds its own reference
        // (`TurboSparkServer`'s doc). Stopping the server was the right move
        // when it could serve only one model, and is now too big a hammer --
        // it would take every OTHER attached model down to swap this one.
        detachChatSessionFromServer()
        session = nil
        defer { opening = false }

        do {
            let options = buildOpenOptions()
            session = try await TurboSparkSession(modelPath: model.path, options: options)
            selected = model
            modelPathText = model.path
            restoreReasoningPreference(for: model)
            // A restored preference is scoped to the PREVIOUS model's alias,
            // and the accepted set is per checkpoint (root Gotcha 56). Clamp
            // to the nearest rung this template can express rather than
            // carrying an unsupported level forward.
            //
            // **AND SAY SO.** This dropped straight to `.off` before, which
            // silently turns thinking off for a user who explicitly turned it
            // on -- the level moved for a real reason and a level that moves
            // without a word is the silent no-op this whole feature exists to
            // avoid.
            let wanted = self.reasoning
            let expressible = nearestAvailableReasoning(to: wanted)
            self.reasoning = expressible
            updateTokenEstimate()
            if expressible != wanted {
                showToast(
                    "Loaded \(model.alias). Reasoning \(wanted.label) is not available here, using \(expressible.label).",
                    style: .info
                )
            } else {
                showToast("Loaded \(model.alias)", style: .success)
            }
        } catch {
            let msg = "Failed to load model: \(error.localizedDescription)"
            self.error = msg
            showToast(msg, style: .error)
        }
    }

    public func loadModel() {
        guard let selected else { return }
        Task {
            await open(selected)
        }
    }

    public func reloadModel() {
        guard let selected else { return }
        Task {
            await open(selected)
        }
    }

    public func unloadModel() {
        guard !generating else { return }
        // Or the model stays resident and served, with the Chat pane showing
        // nothing loaded. See `detachChatSessionFromServer`'s own note.
        detachChatSessionFromServer()
        session = nil
        showToast("Model unloaded", style: .info)
    }

    public func setModelURL(_ url: URL) {
        guard !generating else { return }
        let path = url.standardizedFileURL.path
        modelPathText = path
        detachChatSessionFromServer()
        session = nil
        opening = true
        Task {
            // `defer` here, inside the async work, not around this whole
            // (synchronous) function: the previous placement fired the
            // moment `setModelURL` returned -- immediately after spawning
            // this Task -- so `opening` flipped back to `false` before the
            // awaited `TurboSparkSession` init had even started, and the
            // loading indicator never rendered (state#11).
            defer { self.opening = false }
            do {
                let options = self.buildOpenOptions()
                self.session = try await TurboSparkSession(modelPath: path, options: options)
                self.refreshModels()
                self.showToast("Opened model at \(url.lastPathComponent)", style: .success)
            } catch {
                let msg = "Failed to open custom model path: \(error.localizedDescription)"
                self.error = msg
                self.showToast(msg, style: .error)
            }
        }
    }

    public func deleteModel(_ model: InstalledModel) {
        // **THE BYTES BEING DELETED MAY BE MAPPED RIGHT NOW** (state#43).
        // This had no `!generating` guard, and `unloadModel()` returns
        // SILENTLY while a turn is running -- so `selected` was nilled anyway
        // and the `.gturbo` directory removed out from under a live mmap and
        // a decode loop reading it.
        guard !generating, !submitting else {
            showToast(
                "Cannot delete '\(model.alias)' while a turn is running. Stop it first.",
                style: .warning)
            return
        }
        // And a SERVED model has a second holder the Chat pane cannot see
        // (`swift/CLAUDE.md` Gotcha 26): the server keeps it resident and
        // keeps answering for it, so removing its files leaves a server
        // serving a model whose bytes are gone.
        if Self.isAttachedToServer(model: model, servedIDs: Set(serverAttachedSessions.keys)) {
            showToast(
                "'\(model.alias)' is attached to the running server. Detach it there first.",
                style: .warning)
            return
        }
        // **MATCHED ON PATH ALONE** (state#14): `alias` is not unique once a
        // scanned row exists, so an `alias ||` arm unloads the session of a
        // DIFFERENT model that happens to share the name.
        if selected?.path == model.path {
            unloadModel()
            selected = nil
        }

        // Only a model this app actually installed through `TurboSparkCatalog`
        // may have its file bytes removed. A row that only ever came from
        // scanning LM Studio's library or a user's custom folder
        // (`refreshModels()`'s LM Studio/Custom merge) is not in the
        // catalog's own install list; deleting it used to fall through to an
        // unconditional `removeItem` on whatever path the scan found, which
        // destroys files the app never wrote and only discovered by walking
        // a directory it was pointed at -- the opposite of the "without
        // copying any bytes" promise that scan makes.
        let catalogInstalled: [InstalledModel]
        do {
            catalogInstalled = try TurboSparkCatalog.installed()
        } catch {
            // Refusing beats guessing: an unreadable list cannot distinguish
            // "this app installed it" from "it was only scanned", and the two
            // answers differ by whether bytes get removed.
            self.error =
                "Could not read the installed-model list, so '\(model.alias)' was not deleted: "
                + error.localizedDescription
            return
        }
        let isCatalogTracked = Self.isCatalogTracked(model: model, in: catalogInstalled)

        guard isCatalogTracked else {
            ModelOrganizationStore.shared.removeMetadata(for: model.alias, path: model.path)
            refreshModels()
            showToast("Removed '\(model.alias)' from TurboSpark. The file was left on disk at \(model.path).", style: .info)
            return
        }

        var deleted = false
        var lastError: Error?

        // 1. Try TurboSparkCatalog delete
        do {
            try TurboSparkCatalog.delete(model.alias)
            deleted = true
        } catch {
            lastError = error
        }

        // 2. If catalog delete failed or model is an external file, try direct filesystem removal
        if !deleted {
            let expanded = ModelStorageManager.expandPath(model.path)
            if FileManager.default.fileExists(atPath: expanded) {
                do {
                    try FileManager.default.removeItem(atPath: expanded)
                    deleted = true
                    lastError = nil
                } catch {
                    lastError = error
                }
            }
        }

        ModelOrganizationStore.shared.removeMetadata(for: model.alias, path: model.path)
        refreshModels()

        if deleted {
            showToast("Deleted model '\(model.alias)'", style: .info)
        } else if let err = lastError {
            let msg = "Failed to delete model: \(err.localizedDescription)"
            self.error = msg
            showToast(msg, style: .error)
        }
    }
}

