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

    public func refreshModels() {
        do {
            var combinedInstalled = (try? TurboSparkCatalog.installed()) ?? []
            var existingPaths = Set(combinedInstalled.map { (try? URL(fileURLWithPath: $0.path).standardizedFileURL.path) ?? $0.path })
            var existingAliases = Set(combinedInstalled.map { $0.alias })

            // Scan LM Studio library if enabled
            if enableLMStudioDetection {
                let lmPath = lmStudioDirectory.isEmpty ? ModelStorageManager.defaultLMStudioModelsDirectory : lmStudioDirectory
                let lmModels = ModelStorageManager.scanModels(in: lmPath, sourceTag: "LM Studio")
                for m in lmModels {
                    let stdPath = (try? URL(fileURLWithPath: m.path).standardizedFileURL.path) ?? m.path
                    if !existingPaths.contains(stdPath) {
                        existingPaths.insert(stdPath)
                        // Make sure alias is unique
                        var resolvedAlias = m.alias
                        if existingAliases.contains(resolvedAlias) {
                            resolvedAlias = uniqueAlias(base: m.alias, suffix: "LM Studio", existing: existingAliases)
                        }
                        existingAliases.insert(resolvedAlias)
                        let adjustedModel = InstalledModel(
                            alias: resolvedAlias,
                            repo: m.repo,
                            revision: m.revision,
                            path: m.path,
                            family: m.family,
                            installBytes: m.installBytes,
                            installedOn: m.installedOn
                        )
                        combinedInstalled.append(adjustedModel)
                    }
                }
            }

            // Scan any configured custom directories
            for dir in customModelDirectories where !dir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let customModels = ModelStorageManager.scanModels(in: dir, sourceTag: "Custom")
                for m in customModels {
                    let stdPath = (try? URL(fileURLWithPath: m.path).standardizedFileURL.path) ?? m.path
                    if !existingPaths.contains(stdPath) {
                        existingPaths.insert(stdPath)
                        var resolvedAlias = m.alias
                        if existingAliases.contains(resolvedAlias) {
                            resolvedAlias = uniqueAlias(base: m.alias, suffix: "Custom", existing: existingAliases)
                        }
                        existingAliases.insert(resolvedAlias)
                        let adjustedModel = InstalledModel(
                            alias: resolvedAlias,
                            repo: m.repo,
                            revision: m.revision,
                            path: m.path,
                            family: m.family,
                            installBytes: m.installBytes,
                            installedOn: m.installedOn
                        )
                        combinedInstalled.append(adjustedModel)
                    }
                }
            }

            installed = combinedInstalled
            catalog = try TurboSparkCatalog.available()
            telemetry = TurboSparkSession.systemTelemetry
            // Keyed on `path`, not `alias` (state#14/U5/U6): `InstalledModel.id`
            // is the alias, and a scanned LM Studio/Custom row's alias is not
            // guaranteed unique across sources the way the catalog's own is,
            // so alias-based lookups can silently rebind to the WRONG row
            // once two rows happen to share one. Path is the one field that
            // actually identifies a single file on disk.
            if let currentPath = selected?.path, !installed.contains(where: { $0.path == currentPath }) {
                selected = installed.first
            } else if selected == nil {
                selected = installed.first
            }
            if let selected {
                modelPathText = selected.path
            }
        } catch {
            self.error = "\(error)"
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
        session = nil
        defer { opening = false }

        do {
            let options = buildOpenOptions()
            session = try await TurboSparkSession(modelPath: model.path, options: options)
            selected = model
            modelPathText = model.path
            restoreReasoningPreference(for: model)
            if session?.info.reasoningSupport == SessionInfo.ReasoningSupport.none {
                self.reasoning = .off
            }
            updateTokenEstimate()
            showToast("Loaded \(model.alias)", style: .success)
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
        session = nil
        showToast("Model unloaded", style: .info)
    }

    public func setModelURL(_ url: URL) {
        guard !generating else { return }
        let path = url.standardizedFileURL.path
        modelPathText = path
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
        if selected?.alias == model.alias || selected?.path == model.path {
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
        let stdPath = (try? URL(fileURLWithPath: model.path).standardizedFileURL.path) ?? model.path
        let catalogInstalled = (try? TurboSparkCatalog.installed()) ?? []
        let isCatalogTracked = catalogInstalled.contains { entry in
            let entryPath = (try? URL(fileURLWithPath: entry.path).standardizedFileURL.path) ?? entry.path
            return entryPath == stdPath || entry.alias == model.alias
        }

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

