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
    /// Returns whether a load was actually started.
    ///
    /// **THROUGH THE CHOKE POINT** (task 6.7). A warned model asks first
    /// (`requestLoad`) and a model that is not there is refused, so this returns
    /// false for both: nothing started.
    @discardableResult
    public func selectModel(_ model: InstalledModel) -> Bool {
        requestLoad(model, purpose: .chat(switchToChat: false)) == .started
    }

    /// The load itself, once the gate has said yes.
    ///
    /// Keyed on path (state#14): two distinct rows can share an alias
    /// when one is a scanned LM Studio/Custom entry, so alias equality
    /// here could read "already selected" for a DIFFERENT model on disk
    /// and silently refuse to switch to it.
    /// **"ALREADY SELECTED" IS NOT "ALREADY LOADED"** (state#84). This
    /// returned whenever the paths matched, session or no session -- so
    /// after an unload, or after an open that FAILED, both "Load Model"
    /// and "Load & Chat" were dead for the very model the user was
    /// looking at, with nothing on screen to say why. The refusal is for
    /// a redundant re-open, which needs a live session to be redundant
    /// (`requestLoad` makes it against what the session was opened with).
    func beginChatLoad(_ model: InstalledModel) {
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
        // Compared against what the session was OPENED with, not `selected`:
        // an install that finishes mid-turn selects the new row while the
        // old model stays loaded, and "already selected" then skipped the load.
        // The Chat pane opens once the load has started, or at once for the model
        // already in memory; a warned model keeps the pane until it is answered.
        requestLoad(model, purpose: .chat(switchToChat: true))
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
    /// One configuration source for every caller: browser views, catalog
    /// sheets, onboarding and installed-model detail all fit against these
    /// same active settings. Interactive browser views use
    /// `loadFitRecommendations` below so errors stay visible and header
    /// probing remains off the main actor.
    /// **Ranked**, and the order is the answer: `rank_recommendations` puts
    /// fitting rows above non-fitting ones and frozen evidence above
    /// estimates. Callers that need lookup take `fitRecommendationsByAlias`
    /// rather than rebuilding a dictionary and losing it.
    public func fitRecommendations(probe: Bool = false) -> [ModelRecommendation] {
        (try? TurboSparkCatalog.recommend(
            context: activeFitContext,
            expertCacheSlots: activeCacheSlots,
            loadGuard: activeLoadGuard,
            probe: probe)) ?? []
    }

    /// Reuse this machine's saved ranking. On a cache miss, try the offline
    /// catalog before probing checkpoint headers, without blocking the UI.
    public func loadFitRecommendations(
        probeIfNeeded: Bool = true,
        onProgress: @escaping (UInt32, UInt32) -> Void = { _, _ in }
    ) async throws
        -> [ModelRecommendation]
    {
        let context = activeFitContext
        let slots = activeCacheSlots
        let guardTier = activeLoadGuard
        let key = ModelRecommendationCache.Key(
            telemetry: telemetry ?? TurboSparkSession.systemTelemetry,
            configuration: fitRecommendationConfigurationID,
            catalog: try TurboSparkCatalog.recommendationCatalogFingerprint(),
            probeIfNeeded: probeIfNeeded)
        return try await ModelRecommendationCache.shared.load(key: key, onProgress: onProgress) { progress in
            let offline = try await Task.detached(priority: .userInitiated) {
                try TurboSparkCatalog.recommend(
                    context: context,
                    expertCacheSlots: slots,
                    loadGuard: guardTier)
            }.value
            guard probeIfNeeded, !offline.contains(where: \.runs) else { return offline }

            for try await event in TurboSparkCatalog.recommendWithProgress(
                context: context,
                expertCacheSlots: slots,
                loadGuard: guardTier
            ) {
                switch event {
                case .progress(let completed, let total):
                    progress(completed, total)
                case .finished(let rows):
                    return rows
                }
            }
            // An interrupted stream must never turn an incomplete pass into
            // a persistent ranking.
            throw CancellationError()
        }
    }

    /// Changes whenever a setting that affects fit arithmetic changes. Views
    /// use it as their task identity so recommendations cannot stay frozen at
    /// the configuration that happened to be active when the browser opened.
    public var fitRecommendationConfigurationID: String {
        [
            String(activeFitContext),
            String(runtimeOptions.expertCacheSlots),
            runtimeOptions.loadGuard.rawValue,
            String(runtimeOptions.loadGuardCustomBytes),
        ].joined(separator: ":")
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

    /// **INSTALL ID FIRST, THEN THE LEGACY PATH, THEN THE ALIAS** (task 5.5,
    /// requirement 8.1; state#96 had made it path then alias). The level lives
    /// in `ModelOrganizationStore` under the model's `install_id`, so it is
    /// still found after the model moves or its drive reconnects under another
    /// name. `modelReasoningDefaults` is the legacy table: the first path or
    /// alias entry found there is copied to the install id (unless the id
    /// already has a level, which wins) and removed from the table here. The
    /// alias stays last because an alias entry shadows a fresher path entry for
    /// as long as it exists, and nothing rewrites it.
    public func restoreReasoningPreference(for model: InstalledModel) {
        let legacyBefore = modelReasoningDefaults
        let saved = ModelOrganizationStore.shared.reasoningDefault(
            for: model, migratingLegacyFrom: &modelReasoningDefaults)
        // The retired legacy key reaches disk with the next settings save.
        if modelReasoningDefaults != legacyBefore { persistSettingsDebounced() }
        if let saved, let level = GenerateOptions.Reasoning(rawValue: saved) {
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
        // **THE NET UNDER EVERY LOAD PATH** (task 6.7, requirement 3.3): a model
        // whose drive is away or whose folder is gone is never opened, whatever
        // asked, including the paths that cannot ask a question (the load after
        // an install, a reload). The interactive paths were refused earlier by
        // `requestLoad`; this one says why in a toast and never in `error`.
        guard canOpenNow(model) else { return }
        if let steeringReason = steeringOpenBlockReason(for: model) {
            let message = AppStrings.localized(
                "Cannot load \(model.alias) with steering enabled: \(steeringReason)")
            error = message
            showToast(message, style: .error, duration: 8.0)
            return
        }
        loadObserver?(model)
        opening = true
        defer { opening = false }
        await openSession(model)
    }

    /// The load itself, shared by `open(_:)` and `setModelURL(_:)` so the
    /// steering check, reasoning clamp, server re-attach and estimate refresh
    /// cannot drift apart again. The caller owns `opening`.
    func openSession(_ model: InstalledModel) async {
        error = nil
        // **DETACHED RATHER THAN MERELY DROPPED, and detached rather than
        // stopping the whole server.** `session = nil` alone leaves the OLD
        // model resident and served: the server holds its own reference
        // (`TurboSparkServer`'s doc). Stopping the server was the right move
        // when it could serve only one model, and is now too big a hammer --
        // it would take every OTHER attached model down to swap this one.
        detachChatSessionFromServer()
        session = nil

        do {
            let options = buildOpenOptions(modelPath: model.path)
            let steeringConfiguration = AppSteeringPolicy.configuration(
                enabled: steeringEnabled, preset: resolvedSteeringPreset)
            session = try await TurboSparkSession(modelPath: model.path, options: options)
            loadedSteeringConfiguration = steeringConfiguration
            sessionModelPath = model.path
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
                    AppStrings.localized("Loaded \(model.alias). Reasoning \(wanted.label) is not available here, using \(expressible.label)."),
                    style: .info
                )
            } else {
                showToast(AppStrings.localized("Loaded \(model.alias)"), style: .success)
            }
            if self.server != nil {
                self.attachChatSession()
            }
        } catch {
            let msg = AppStrings.localized("Failed to load model: \(error.localizedDescription)")
            self.error = msg
            showToast(msg, style: .error)
        }
    }

    /// Loads the selected model: the Load button and the Model menu's Load Chat
    /// Model. Through the choke point, so a warned model asks first.
    public func loadModel() {
        guard let selected else { return }
        requestLoad(selected, purpose: .chat(switchToChat: false))
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
        showToast(AppStrings.localized("Model unloaded"), style: .info)
    }

    /// Unloads either the active workspace's model or whichever model is currently resident.
    public func unloadActiveOrAnyModel() {
        if activeSection == .images {
            if canUnloadImageModel {
                unloadImageModel()
            } else if canUnloadModel {
                unloadModel()
            }
        } else {
            if canUnloadModel {
                unloadModel()
            } else if canUnloadImageModel {
                unloadImageModel()
            }
        }
    }

    public func setModelURL(_ url: URL) {
        guard !generating, !opening else { return }
        let path = url.standardizedFileURL.path
        modelPathText = path
        // A row that matches the path is used as-is; a manually opened path
        // with no row gets a transient one, so the load goes through the same
        // `openSession` as every catalog open (steering check, reasoning
        // clamp, server re-attach) and `selected` names what is really
        // loaded instead of being rebound to `installed.first`.
        let row =
            installed.first {
                URL(fileURLWithPath: $0.path).standardizedFileURL.path == path
            }
            ?? InstalledModel(
                alias: url.lastPathComponent, repo: "", path: path, family: "")
        // A folder inside a library on a slow drive asks like any other load;
        // a folder nobody installed (no registry row) has no library to advise
        // on and loads at once.
        requestLoad(row, purpose: .chosenFolder)
    }

    /// Opens the folder the user chose, once the gate has said yes.
    func beginOpeningFolder(_ row: InstalledModel) {
        let path = URL(fileURLWithPath: row.path).standardizedFileURL.path
        if let steeringReason = steeringOpenBlockReason(for: row) {
            let message = AppStrings.localized(
                "Cannot load \(row.alias) with steering enabled: \(steeringReason)")
            error = message
            showToast(message, style: .error, duration: 8.0)
            return
        }
        loadObserver?(row)
        // Set synchronously: the loading indicator keys off it, and it must
        // be true the moment this returns (state#11).
        opening = true
        Task {
            defer { self.opening = false }
            await self.openSession(row)
            guard self.session != nil else { return }
            self.refreshModels()
            // `refreshModels` runs `reconcileSelection`; keep the field the
            // user just filled in.
            self.modelPathText = path
        }
    }

    /// Deletes `model` from disk, or removes a scanned row from the list.
    ///
    /// **EVERY ENGINE ROW IS DELETED BY THE ENGINE, THROUGH ITS INSTALL ID, OFF
    /// THE MAIN ACTOR** (task 6.4, BINDING from the 4.6 review). The old code
    /// fell back to `FileManager.removeItem(model.path)` after any error from
    /// the engine's alias delete, which for a library row deleted the recorded
    /// path of a drive that was away (or another drive mounted there), followed
    /// links planted on the drive, bypassed the engine's `busy` refusal and left
    /// the registry and the library marker listing a model that was gone. That
    /// fallback does not exist any more: the app never removes a path it
    /// recorded for a row the engine owns. A row only a scan found keeps today's
    /// rule below.
    public func deleteModel(_ model: InstalledModel) {
        _ = startModelDelete(model)
    }

    /// `deleteModel`, returning the work it started so a test can wait for it.
    /// Nil when it was refused, finished at once (a scanned row) or is already
    /// running for this row.
    @discardableResult
    func startModelDelete(_ model: InstalledModel) -> Task<Void, Never>? {
        // **THE BYTES BEING DELETED MAY BE MAPPED RIGHT NOW** (state#43).
        // This had no `!generating` guard, and `unloadModel()` returns
        // SILENTLY while a turn is running -- so `selected` was nilled anyway
        // and the `.gturbo` directory removed out from under a live mmap and
        // a decode loop reading it.
        guard !generating, !submitting else {
            showToast(
                AppStrings.localized("Cannot delete '\(model.alias)' while a turn is running. Stop it first."),
                style: .warning)
            return nil
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
                AppStrings.localized("Cannot delete '\(model.alias)' while a model is loading. Wait for it to finish."),
                style: .warning)
            return nil
        }
        // And a SERVED model has a second holder the Chat pane cannot see
        // (`swift/CLAUDE.md` Gotcha 26): the server keeps it resident and
        // keeps answering for it, so removing its files leaves a server
        // serving a model whose bytes are gone.
        if Self.isAttachedToServer(model: model, servedIDs: Set(serverAttachedSessions.keys).union(serverInfo?.models ?? [])) {
            showToast(
                AppStrings.localized("'\(model.alias)' is attached to the running server. Detach it there first."),
                style: .warning)
            return nil
        }
        // Humanize owns a separate mapping that the Chat session cannot see.
        // Keep this guard here so deletion from Installed Models is safe too.
        // It is released with the chat model, after the delete has succeeded.
        let humanizeHoldsIt = humanizeWorkspace.usesModel(at: model.path)
        if humanizeHoldsIt {
            guard !humanizeWorkspace.isBusy else {
                showToast(AppStrings.localized("Stop Humanize before deleting its model."), style: .warning)
                return nil
            }
        }

        // **A ROW ONLY A SCAN FOUND IS NEVER THE ENGINE'S, AND ITS FILES ARE
        // NEVER THE APP'S TO DELETE.** A row that came from scanning LM Studio's
        // library or a user's custom folder (`refreshModels()`'s merge) carries
        // the placeholder identity the app's scanners build
        // (`ModelIdentityKey.isPlaceholder`). Deleting it used to fall through to
        // an unconditional `removeItem` on whatever path the scan found, which
        // destroys files the app never wrote and only discovered by walking a
        // directory it was pointed at -- the opposite of the "without copying
        // any bytes" promise that scan makes. It leaves the list and stays out
        // of the next scan; nothing is deleted, so nothing can fail, and the
        // model can be released at once.
        guard !ModelIdentityKey.isPlaceholder(model) else {
            if humanizeHoldsIt { humanizeWorkspace.unload() }
            // **MATCHED ON PATH ALONE** (state#14): `alias` is not unique once
            // a scanned row exists, so an `alias ||` arm unloads the session of
            // a DIFFERENT model that happens to share the name.
            if selected?.path == model.path {
                unloadModel()
                selected = nil
            }
            // **EXCLUDED, NOT FORGOTTEN** (state#88). This dropped the row's
            // notes, tags and favorite and left the bytes alone -- and the
            // next scan walked the same directory and put the row straight
            // back, now stripped of everything the user had written on it.
            // So the button destroyed exactly the data it had no business
            // touching and failed at the one thing it claimed to do.
            ModelOrganizationStore.shared.excludeFromScan(path: model.path)
            refreshModels()
            showToast(
                AppStrings.localized(
                    "Removed '\(model.alias)' from TurboSpark. The file was left on disk at \(model.path); restore it under Settings > Models."),
                style: .info, duration: 5.0)
            return nil
        }

        // A model whose drive is away has nothing to delete: the engine would
        // refuse (`not_connected`) and the recorded path may now be another
        // drive. The row offers no removal until the drive is connected.
        guard model.availability != .notConnected else {
            showToast(
                ModelDeleteText.notConnected(alias: model.alias, libraryName: model.connectName ?? model.libraryName),
                style: .warning)
            return nil
        }
        guard !modelsBeingChanged.contains(model.installId) else { return nil }
        modelsBeingChanged.insert(model.installId)

        // **THE ENGINE CALL IS OFF THE MAIN ACTOR AND THE SESSION STAYS.** It
        // can read the mount table, remove many gigabytes and write
        // `installed.json`, and it answers `busy` while an install or a move
        // runs. A refused delete changes nothing, so the loaded model, the
        // selection and the notes stay as they were until the engine has said
        // yes.
        let service = libraryState.service
        let installId = model.installId
        return Task { [weak self] in
            let outcome = await Task.detached(priority: .userInitiated) {
                Result { try service.deleteInstall(installId: installId) }
            }.value
            await self?.finishModelDelete(model, outcome: outcome)
        }
    }

    /// What follows an answer from the engine's delete.
    private func finishModelDelete(_ model: InstalledModel, outcome: Result<Void, Error>) async {
        modelsBeingChanged.remove(model.installId)
        switch outcome {
        case .failure(let error):
            // A warning toast, not `error`: a refusal is an answer to what the
            // user asked, and an unplugged drive is never an availability
            // banner (requirement 3.4).
            showToast(
                ModelDeleteText.failure(
                    error, alias: model.alias, libraryName: model.connectName ?? model.libraryName),
                style: .warning)
            // `busy` changed nothing. Any other answer may mean the lists are
            // out of date (`unknown_install`: someone else removed it).
            if (error as? TurboSparkError)?.libraryFault?.code != .busy {
                await libraryState.refresh()
            }
        case .success:
            if humanizeWorkspace.usesModel(at: model.path) { humanizeWorkspace.unload() }
            // **MATCHED ON PATH ALONE** (state#14), and a model that cannot be
            // unloaded because a turn started during the delete stays selected:
            // `selected` nil beside a live session is the state#43 mismatch.
            if selected?.path == model.path {
                unloadModel()
                if session == nil { selected = nil }
            }
            // The remembered reasoning level is part of the same record (task
            // 5.5), so it goes with the notes and tags. A legacy path entry that
            // was never read is pruned too: nothing used to prune it, so
            // `settings.json` accumulated an entry per model ever deleted, and a
            // later install at the same path silently inherited a level its own
            // user never chose (state#96).
            ModelOrganizationStore.shared.removeMetadata(for: model)
            modelReasoningDefaults.removeValue(forKey: model.path)
            persistSettings()
            refreshModels()
            await libraryState.refresh()
            showToast(AppStrings.localized("Deleted model '\(model.alias)'"), style: .info)
        }
    }

    // MARK: - Remove from list

    /// Stops listing a model the engine knows, without touching a file: what a
    /// model whose folder is missing offers besides Delete (requirement 3.7). A
    /// model on a drive that is away offers it no longer, because the engine
    /// forgets a row only with its library connected.
    public func removeModelFromList(_ model: InstalledModel) {
        _ = startRemoveFromList(model)
    }

    /// `removeModelFromList`, returning the work it started so a test can wait.
    ///
    /// The engine's forget needs the model's library connected and writable (its
    /// marker must stop listing the model too), so for a drive that is away it
    /// answers `not_connected`; the toast says so and the row stays.
    @discardableResult
    func startRemoveFromList(_ model: InstalledModel) -> Task<Void, Never>? {
        // A row a scan found has no registry entry: its "remove" is Delete.
        guard !ModelIdentityKey.isPlaceholder(model) else { return nil }
        guard !modelsBeingChanged.contains(model.installId) else { return nil }
        modelsBeingChanged.insert(model.installId)

        let service = libraryState.service
        let installId = model.installId
        return Task { [weak self] in
            let outcome = await Task.detached(priority: .userInitiated) {
                Result { try service.forget(installId: installId) }
            }.value
            await self?.finishRemoveFromList(model, outcome: outcome)
        }
    }

    private func finishRemoveFromList(_ model: InstalledModel, outcome: Result<Void, Error>) async {
        modelsBeingChanged.remove(model.installId)
        switch outcome {
        case .failure(let error):
            showToast(
                ModelDeleteText.failure(
                    error, alias: model.alias, libraryName: model.connectName ?? model.libraryName),
                style: .warning)
            if (error as? TurboSparkError)?.libraryFault?.code != .busy {
                await libraryState.refresh()
            }
        case .success:
            // Its notes and tags stay: the files may still be on the drive, and
            // a library added again brings the model back under the same id.
            if selected?.path == model.path, loadedModelPath != model.path {
                selected = nil
            }
            await libraryState.refresh()
            showToast(ModelDeleteText.removedFromList(alias: model.alias), style: .info)
        }
    }
}
