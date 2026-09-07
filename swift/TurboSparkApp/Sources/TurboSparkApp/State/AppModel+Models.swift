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
    public func selectModel(_ model: InstalledModel) {
        // Keyed on path (state#14): two distinct rows can share an alias
        // when one is a scanned LM Studio/Custom entry, so alias equality
        // here could read "already selected" for a DIFFERENT model on disk
        // and silently refuse to switch to it.
        // **"ALREADY SELECTED" IS NOT "ALREADY LOADED"** (state#84). This
        // returned whenever the paths matched, session or no session -- so
        // after an unload, or after an open that FAILED, both "Load Model"
        // and "Load & Chat" were dead for the very model the user was
        // looking at, with nothing on screen to say why. The refusal is for
        // a redundant re-open, which needs a live session to be redundant.
        guard !generating, !opening,
            !(selected?.path == model.path && session != nil)
        else { return }
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

    /// The expert-cache slot policy both the loader and the model hub must
    /// use, for `activeLoadGuard`'s reason one term over.
    ///
    /// A footprint is `slots x layers x expert stride`, so a hub ranking at
    /// one slot count beside a session opening at another is not an
    /// approximation of the same answer, it is a different configuration.
    /// Gemma 4 reads 2,175 MiB at 16 slots and 3,654 at 32.
    ///
    /// `0` is this app's spelling of automatic (`allowedSlotCounts`), which
    /// the binding spells `.auto`.
    public var activeCacheSlots: OpenOptions.Sizing {
        runtimeOptions.expertCacheSlots > 0
            ? .fixed(UInt32(runtimeOptions.expertCacheSlots)) : .auto
    }

    /// The context window a fit should be computed at.
    ///
    /// **Read off the SESSION when there is one.** `SessionInfo` holds what
    /// was RESOLVED, and under automatic sizing nothing was asked for (swift
    /// Gotcha 6), so the Inspector's setting is the request and `info` is the
    /// answer. Falls back to the request, then to the engine's own default.
    public var activeFitContext: UInt32 {
        if let resolved = session?.info.maxContext, resolved > 0 {
            return UInt32(resolved)
        }
        return maxContextTokens > 0 ? UInt32(maxContextTokens) : 4096
    }

    /// Curated rows ranked for this machine, at the configuration this app
    /// will actually open under.
    ///
    /// One call for every caller: `ModelHubView`, `CatalogSheet`,
    /// `ModelInstallView` and the installed-model detail pane all read the
    /// same rows, so a fifth assembling its own would compile and be wrong
    /// only once the user moved a setting off its default.
    /// **Ranked**, and the order is the answer: `rank_recommendations` puts
    /// fitting rows above non-fitting ones and frozen evidence above
    /// estimates. Callers that need lookup take `fitRecommendationsByAlias`
    /// rather than rebuilding a dictionary and losing it.
    public func fitRecommendations() -> [ModelRecommendation] {
        (try? TurboSparkCatalog.recommend(
            context: activeFitContext,
            expertCacheSlots: activeCacheSlots,
            loadGuard: activeLoadGuard)) ?? []
    }

    /// The same rows keyed by alias, for a view that joins rather than lists.
    public func fitRecommendationsByAlias() -> [String: ModelRecommendation] {
        Dictionary(fitRecommendations().map { ($0.alias, $0) }, uniquingKeysWith: { a, _ in a })
    }

    /// This machine's fit for one installed model, or `nil` when the catalog
    /// has no row for it (a scanned or side-loaded install).
    public func fit(for alias: String?) -> ModelRecommendation? {
        guard let alias, !alias.isEmpty else { return nil }
        return fitRecommendationsByAlias()[alias]
    }

    /// - Parameter modelPath: the install about to be opened, for the
    ///   `.auto` kv-bits eligibility check below. Defaults to `selected?.path`
    ///   for every existing call site that already has one; `setModelURL`
    ///   passes the raw path it is about to open instead, since it has no
    ///   `InstalledModel` in hand at the point it calls this.
    public func buildOpenOptions(modelPath: String? = nil) -> OpenOptions {
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
        // `.auto` asks for 4-bit ONLY on a checkpoint the manifest itself
        // says would accept it (`ModelFeatureDescriptor.kvQuantEligible`,
        // mirroring the engine's own `--kv-bits` refusal rule) -- an
        // explicit width bypasses the check and is sent unconditionally,
        // same as any other named request in this app.
        switch runtimeOptions.kvBits {
        case .auto:
            let path = modelPath ?? selected?.path ?? ""
            if ModelFeatureDescriptor.supportsKvQuant(atInstallPath: path) {
                options.kvBits = .four
            }
        case .off:
            break
        default:
            options.kvBits = runtimeOptions.kvBits.kvBits
        }
        if runtimeOptions.maxTokensPerSec > 0 {
            options.maxTokensPerSec = runtimeOptions.maxTokensPerSec
        }
        // STEERING IS GATED ON THE SWITCH, not on a path being present.
        // `resolvedSteeringPreset` folds the named preset and the Inspector's
        // raw knobs (the implicit "Custom" preset) into one value, so there
        // is one answer to "what would this session steer with" rather than
        // two the UI and the open could disagree about.
        if steeringEnabled, let preset = resolvedSteeringPreset {
            let sPath = preset.vectorPath.trimmingCharacters(in: .whitespacesAndNewlines)
            if !sPath.isEmpty {
                options.steering = sPath
                options.steeringMode = preset.mode.steeringMode
                options.steeringScale = preset.scale
                let layers = preset.layers.trimmingCharacters(in: .whitespacesAndNewlines)
                if !layers.isEmpty {
                    options.steeringLayers = layers
                }
                if preset.mode == .clamp {
                    options.steeringTarget = preset.target
                }
                if preset.gate > 0 {
                    options.steeringGate = preset.gate
                }
            }
        }
        return options
    }

    /// **PATH FIRST, ALIAS AS THE LEGACY FALLBACK** (state#96). `setReasoning`
    /// writes the path now, and reading the alias first would let an entry
    /// written before that change shadow the fresh one for as long as the
    /// alias entry exists -- which is forever, since nothing rewrites it.
    public func restoreReasoningPreference(for model: InstalledModel) {
        if let saved = modelReasoningDefaults[model.path],
            let level = GenerateOptions.Reasoning(rawValue: saved) {
            self.reasoning = level
        } else if let saved = modelReasoningDefaults[model.alias],
            let level = GenerateOptions.Reasoning(rawValue: saved) {
            self.reasoning = level
        }
    }

    public func open(_ model: InstalledModel) async {
        // **`!opening` AS WELL AS `!generating`** (state#50). Two overlapping
        // opens each ran `defer { opening = false }`, so the first to finish
        // cleared the flag for both, and `selected` and `session` could end
        // up naming different models -- the pane says one thing and the turn
        // runs another, with no error anywhere.
        guard !generating, !opening else { return }
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
            let options = buildOpenOptions(modelPath: model.path)
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
        // **THE PREDICATE THE UI DRAWS FROM** (state#85). This checked one of
        // its three terms, and two panes called it with no `.disabled` at
        // all, so Unload was live during a submission and during an open --
        // where it clears `session` under a load that is about to publish
        // one, leaving the pane and the engine disagreeing about what is
        // resident.
        guard canUnloadModel else { return }
        // Or the model stays resident and served, with the Chat pane showing
        // nothing loaded. See `detachChatSessionFromServer`'s own note.
        detachChatSessionFromServer()
        session = nil
        showToast("Model unloaded", style: .info)
    }

    public func setModelURL(_ url: URL) {
        guard !generating, !opening else { return }
        let path = url.standardizedFileURL.path
        modelPathText = path
        // **`selected` FOLLOWS THE PATH, OR `reconcileSelection` UNDOES THIS**
        // (state#50). This left `selected` pointing at the PREVIOUS install,
        // and the `refreshModels()` below then ran `reconcileSelection`,
        // which sets `modelPathText = selected.path` -- so the field reverted
        // to the old model's path the moment the new one finished loading.
        // A row that matches the path is selected; anything else is a
        // manually-opened path with no row, and nil is the honest answer.
        selected = installed.first {
            URL(fileURLWithPath: $0.path).standardizedFileURL.path == path
        }
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
                let options = self.buildOpenOptions(modelPath: path)
                self.session = try await TurboSparkSession(modelPath: path, options: options)
                self.refreshModels()
                // `refreshModels` runs `reconcileSelection`, which rewrites
                // `modelPathText` from `selected`. With no matching row that
                // would blank the field the user just filled in.
                self.modelPathText = path
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
        // **AND AN OPEN IN FLIGHT IS THE SAME HAZARD ONE STEP EARLIER**
        // (state#73). `ts_session_open` runs on the binding's private queue
        // for tens of seconds on a real install, and `unloadModel()` refuses
        // while `opening` -- so Delete during a load removed the directory
        // under a mapping that was still being established, and the open's
        // tail then published a session for a model that no longer exists.
        // `opening` is not per model, so this refuses during ANY load; that
        // is the honest bound, since the load in flight has no row here.
        guard !opening else {
            showToast(
                "Cannot delete '\(model.alias)' while a model is loading. Wait for it to finish.",
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
            // **EXCLUDED, NOT FORGOTTEN** (state#88). This dropped the row's
            // notes, tags and favorite and left the bytes alone -- and the
            // next scan walked the same directory and put the row straight
            // back, now stripped of everything the user had written on it.
            // So the button destroyed exactly the data it had no business
            // touching and failed at the one thing it claimed to do.
            ModelOrganizationStore.shared.excludeFromScan(path: model.path)
            refreshModels()
            showToast(
                "Removed '\(model.alias)' from TurboSpark. The file was left on disk at "
                    + "\(model.path); restore it under Settings > Models.",
                style: .info, duration: 5.0)
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
        // The remembered reasoning level goes with it (state#96). Nothing
        // pruned this, so `settings.json` accumulated an entry per model ever
        // deleted -- and once the key is the PATH, a later install at the
        // same path silently inherits a level its own user never chose.
        modelReasoningDefaults.removeValue(forKey: model.path)
        persistSettings()
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

