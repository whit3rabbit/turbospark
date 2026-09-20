import CTurboSpark
import CryptoKit
import Foundation

// THE TWO TYPES BELOW DECODE snake_case WHERE EVERYTHING ELSE IN THIS
// PACKAGE DECODES camelCase, and that asymmetry is deliberate rather than an
// oversight.
//
// The rest of this binding's JSON is a WIRE shape invented for it, so the
// Rust side spells it camelCase and no CodingKeys are needed anywhere. These
// two are not: they are the engine's own on-disk formats -- `models.json`
// (checked into the repository) and `~/.turbospark/installed.json` (written
// by text-model installs) -- passed through unchanged so a GUI's
// rows and the CLI's rows are provably the same data. Renaming them to suit
// Swift would either fork the format or rewrite files this binding does not
// own.
//
// The drift this guards against is real and was caught here: the first draft
// assumed camelCase, and `SurfaceTests` failed on `downloadBytes`. That is
// the Swift test target doing the one job no Rust test can.

/// A row of the curated model table.
public struct CatalogEntry: Decodable, Sendable, Identifiable, Equatable {
    public var id: String { alias }

    public let alias: String
    public let name: String
    public let family: String
    /// Bytes read off the network during an install.
    public let downloadBytes: UInt64
    /// Approximate bytes on disk afterwards.
    public let installBytes: UInt64
    public let status: String
    public let notes: String?
    /// Whether this row is already installed, so a list needs one call
    /// rather than two and a join. Added by the binding, not by the
    /// catalog file.
    public let installed: Bool

    enum CodingKeys: String, CodingKey {
        case alias, name, family, status, notes, installed
        case downloadBytes = "download_bytes"
        case installBytes = "install_bytes"
    }
}

/// A model already on disk.
public struct InstalledModel: Decodable, Sendable, Identifiable, Equatable {
    public var id: String { alias }

    public let alias: String
    /// The weights repository, `owner/name`.
    public let repo: String
    /// The revision streamed. For a floating row this is the literal `main`
    /// and therefore says less than it looks like it does.
    public let revision: String
    public let path: String
    public let family: String
    /// `text`, `image`, or `audio`. Older rows decode as `text`.
    public let modality: String
    public let installBytes: UInt64
    /// `YYYY-MM-DD`. Whole days only.
    public let installedOn: String

    public init(
        alias: String,
        repo: String,
        revision: String = "main",
        path: String,
        family: String,
        modality: String = "text",
        installBytes: UInt64 = 0,
        installedOn: String = ""
    ) {
        self.alias = alias
        self.repo = repo
        self.revision = revision
        self.path = path
        self.family = family
        self.modality = modality
        self.installBytes = installBytes
        self.installedOn = installedOn
    }

    enum CodingKeys: String, CodingKey {
        case alias, repo, revision, path, family, modality
        case installBytes = "install_bytes"
        case installedOn = "installed_on"
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.alias = try values.decode(String.self, forKey: .alias)
        self.repo = try values.decode(String.self, forKey: .repo)
        self.revision = try values.decode(String.self, forKey: .revision)
        self.path = try values.decode(String.self, forKey: .path)
        self.family = try values.decode(String.self, forKey: .family)
        self.modality = try values.decodeIfPresent(String.self, forKey: .modality) ?? "text"
        self.installBytes = try values.decode(UInt64.self, forKey: .installBytes)
        self.installedOn = try values.decode(String.self, forKey: .installedOn)
    }
}

/// A valid image-generation install. Image artifacts intentionally have a
/// separate type and listing from text models.
public struct ImageInstalledModel: Decodable, Sendable, Identifiable, Equatable {
    public var id: String { path }

    public let alias: String
    public let modelID: String
    public let revision: String
    public let path: String
    public let width: UInt32
    public let height: UInt32
    public let schedulerSteps: UInt32
    public let quantization: String

    public init(
        alias: String,
        modelID: String,
        revision: String,
        path: String,
        width: UInt32,
        height: UInt32,
        schedulerSteps: UInt32,
        quantization: String = "unknown"
    ) {
        self.alias = alias
        self.modelID = modelID
        self.revision = revision
        self.path = path
        self.width = width
        self.height = height
        self.schedulerSteps = schedulerSteps
        self.quantization = quantization
    }

    private enum CodingKeys: String, CodingKey {
        case alias, modelID, revision, path, width, height, schedulerSteps, quantization
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.alias = try values.decode(String.self, forKey: .alias)
        self.modelID = try values.decode(String.self, forKey: .modelID)
        self.revision = try values.decode(String.self, forKey: .revision)
        self.path = try values.decode(String.self, forKey: .path)
        self.width = try values.decode(UInt32.self, forKey: .width)
        self.height = try values.decode(UInt32.self, forKey: .height)
        self.schedulerSteps = try values.decode(UInt32.self, forKey: .schedulerSteps)
        self.quantization = try values.decodeIfPresent(String.self, forKey: .quantization) ?? "unknown"
    }
}

/// A curated image-generation source that can be downloaded into the shared
/// image store. This is separate from `CatalogEntry` because image installs
/// are not text-model installs.
public struct ImageCatalogEntry: Decodable, Sendable, Identifiable, Equatable {
    public var id: String { alias }

    public let alias: String
    public let modelID: String
    public let revision: String
    public let quantization: String
}

/// What an install will cost, before it starts.
public struct InstallCost: Decodable, Sendable, Equatable {
    public let downloadBytes: UInt64
    public let installBytes: UInt64
}

/// Result of relocating the managed TurboSpark store.
public struct StoreRelocation: Decodable, Sendable, Equatable {
    public let source: String
    public let destination: String
    public let bytes: UInt64
    public let files: UInt64
}

/// One install-progress event.
public enum InstallEvent: Sendable, Equatable {
    case stage(String)
    case bytes(done: UInt64, total: UInt64)
    case finished(InstalledModel)
}

/// Progress from a curated image source download and pack.
public enum ImageInstallEvent: Sendable, Equatable {
    case stage(String)
    case bytes(done: UInt64, total: UInt64)
    case finished(ImageInstalledModel)
}

/// Progress from a managed store relocation.
public enum StoreRelocationEvent: Sendable, Equatable {
    case bytes(done: UInt64, total: UInt64)
    case finished(StoreRelocation)
}

/// Progress while curated checkpoint headers are ranked for this machine.
public enum RecommendationEvent: Sendable, Equatable {
    case progress(completed: UInt32, total: UInt32)
    case finished([ModelRecommendation])
}

/// Browsing, probing and installing models.
///
/// Available on every platform, including ones that cannot then RUN a model:
/// the artifact is the same either way, and refusing to list a catalog on a
/// machine that cannot decode would be a restriction with no reason behind
/// it.
public enum TurboSparkCatalog {
    /// The active process-local model store root.
    public static func storeRoot() throws -> String {
        try takeString { out in ts_store_root_get(out) }
    }

    /// Sets or clears the active process-local model store root. Passing nil
    /// restores the environment-derived default.
    public static func setStoreRoot(_ root: String?) throws {
        try check(withOptionalCString(root) { ts_store_root_set($0) })
    }

    /// Relocates the managed store on a dedicated thread. The source is only
    /// removed after the destination and install-record paths verify.
    public static func relocateStore(to destination: String) -> AsyncThrowingStream<StoreRelocationEvent, Error> {
        AsyncThrowingStream { continuation in
            let thread = Thread {
                let box = StoreRelocationBox(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<StoreRelocationBox>.fromOpaque(userdata).release() }
                do {
                    let json = try takeString { out in
                        destination.withCString {
                            ts_store_relocate($0, storeRelocationCallback, userdata, out)
                        }
                    }
                    continuation.yield(.finished(try decode(StoreRelocation.self, from: json)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            thread.name = "com.turbospark.store-relocation"
            thread.start()
        }
    }

    /// The curated table, each row carrying whether it is installed.
    public static func available() throws -> [CatalogEntry] {
        try decode([CatalogEntry].self, from: try takeString { ts_catalog_json($0) })
    }

    /// Includes checkpoint revisions and measured evidence, which the display
    /// rows omit. Installing a model does not change its hardware fit.
    public static func recommendationCatalogFingerprint() throws -> String {
        let json = try takeString { ts_catalog_json($0) }
        return try recommendationCatalogFingerprint(json: Data(json.utf8))
    }

    static func recommendationCatalogFingerprint(json: Data) throws -> String {
        guard var rows = try JSONSerialization.jsonObject(with: json) as? [[String: Any]] else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: [], debugDescription: "Expected a catalog row array"))
        }
        for index in rows.indices { rows[index].removeValue(forKey: "installed") }
        rows.sort { ($0["alias"] as? String ?? "") < ($1["alias"] as? String ?? "") }
        let data = try JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys])
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// What text models are installed in `~/.turbospark/models/text`.
    public static func installed() throws -> [InstalledModel] {
        try decode([InstalledModel].self, from: try takeString { ts_installed_json($0) })
    }

    /// Valid image-generation installs in the shared machine store.
    public static func imageInstalled() throws -> [ImageInstalledModel] {
        try decode(
            [ImageInstalledModel].self,
            from: try takeString { ts_image_installed_json($0) })
    }

    /// The curated pinned image sources, including the Z-Image MLX variants.
    public static func imageAvailable() throws -> [ImageCatalogEntry] {
        try decode(
            [ImageCatalogEntry].self,
            from: try takeString { ts_image_catalog_json($0) })
    }

    /// What installing `alias` will cost. Call this before `install` to show
    /// a determinate bar and warn about disk space.
    public static func cost(of alias: String) throws -> InstallCost {
        try decode(
            InstallCost.self,
            from: try takeString { out in
                alias.withCString { ts_install_bytes_json($0, out) }
            })
    }

    /// Resolves a model alias or relative directory path to its canonical on-disk install path.
    ///
    /// - Parameter modelOrAlias: The catalog alias (e.g. "gemma4") or directory path.
    /// - Returns: The canonical path if the model exists on disk, or nil if unresolved.
    public static func resolvePath(for modelOrAlias: String) throws -> String? {
        do {
            return try takeString { out in
                modelOrAlias.withCString { ts_model_resolve_path($0, out) }
            }
        } catch let error as TurboSparkError where error.code == .open {
            return nil
        }
    }

    /// Returns the install path for an alias if installed, or nil if not installed.
    public static func path(of alias: String) throws -> String? {
        try resolvePath(for: alias)
    }

    /// Checks whether a model alias or directory path is installed locally.
    public static func isInstalled(_ aliasOrPath: String) throws -> Bool {
        try resolvePath(for: aliasOrPath) != nil
    }

    /// Deletes a validated image install from the shared image store by its
    /// listed path. An alias remains accepted when it identifies one row.
    public static func deleteImage(_ path: String) throws {
        try check(path.withCString { ts_image_delete($0) })
    }

    /// Returns the catalog entry for a given alias from the curated table, if present.
    public static func entry(for alias: String) throws -> CatalogEntry? {
        try available().first { $0.alias == alias }
    }

    /// Probes a Hugging Face repository by header alone: kilobytes and
    /// seconds, no download.
    ///
    /// `repo` is `owner/name` or `owner/name@revision`.
    ///
    /// **`context`, `expertCacheSlots` and `loadGuard` MUST be what your
    /// sessions will OPEN with**, for `recommend`'s reason: `report.fit` is
    /// an answer at one configuration, and probing under one while opening
    /// under another promises a fit the loader then refuses.
    public static func probe(
        repo: String,
        file: String? = nil,
        sidecarRepo: String? = nil,
        context: UInt32 = 4096,
        expertCacheSlots: OpenOptions.Sizing? = nil,
        loadGuard: OpenOptions.LoadGuard? = nil
    ) throws -> ProbeReport {
        let options = ProbeOptions(
            contextWindow: context == 0 ? nil : context,
            loadGuard: loadGuard,
            expertCacheSlots: expertCacheSlots)
        let json = try encodeOptions(options)
        let raw = try takeString { out in
            repo.withCString { r in
                withOptionalCString(file) { f in
                    withOptionalCString(sidecarRepo) { s in
                        withOptionalCString(json) { o in
                            ts_probe_json(r, f, s, o, out)
                        }
                    }
                }
            }
        }
        return try decode(ProbeReport.self, from: raw)
    }

    /// What a longer context window would cost an INSTALLED model.
    ///
    /// A different question from `recommend`'s and not derivable from it: KV
    /// is not linear in the window, so multiplying one figure is 3.5x high on
    /// a sliding-window family. `rungs` is EMPTY when the install's shape
    /// could not be read, which is a question nothing answered rather than a
    /// model with no memory cost.
    public static func contextLadder(
        modelPath: String,
        expertCacheSlots: OpenOptions.Sizing? = nil,
        loadGuard: OpenOptions.LoadGuard? = nil
    ) throws -> ContextLadder {
        let json = try encodeOptions(
            ProbeOptions(
                contextWindow: nil, loadGuard: loadGuard,
                expertCacheSlots: expertCacheSlots))
        return try decode(
            ContextLadder.self,
            from: try takeString { out in
                modelPath.withCString { p in
                    withOptionalCString(json) { o in
                        ts_context_ladder_json(p, o, out)
                    }
                }
            })
    }

    /// Every `.gguf` a repository publishes, best quality first. One API
    /// call, no header reads, no download.
    ///
    /// **It carries no fit**, deliberately: a fit needs the checkpoint's
    /// shape, which needs a header read PER FILE. Use this to fill a
    /// quantization picker and `probe(repo:file:)` on the one the user picks.
    public static func variants(repo: String) throws -> RepoVariants {
        try decode(
            RepoVariants.self,
            from: try takeString { out in
                repo.withCString { ts_repo_variants_json($0, out) }
            })
    }

    /// What a `.gguf` control vector declares, read from the file alone: no
    /// model, no session, no network. A vector is around 1.3 MB, so this is
    /// milliseconds.
    ///
    /// Call it BEFORE offering a vector against an install, so a UI can say
    /// "this file is 4096 wide and your model is 5120" instead of letting
    /// `TurboSparkSession.init` fail minutes into a load. It reads the same
    /// parser the open reads, so the two cannot disagree about what a file
    /// means.
    public static func controlVectorInfo(path: String) throws -> ControlVectorInfo {
        let json = try takeString { out in
            path.withCString { p in ts_control_vector_info_json(p, out) }
        }
        return try decode(ControlVectorInfo.self, from: json)
    }

    /// Deletes an installed model from `~/.turbospark` and removes its directory.
    public static func delete(_ alias: String) throws {
        try check(alias.withCString { ts_model_delete($0) })
    }

    /// Ranks curated models by hardware fit for this machine at `context`.
    ///
    /// **`loadGuard` MUST match what your sessions will OPEN with.** This
    /// ranking and the loader's refusal share one memory budget by
    /// construction, which is what makes a recommendation worth showing;
    /// ranking under `.relaxed` while opening under `.strict` promises a fit
    /// the loader then refuses, in the one place a user cannot see the two
    /// disagree. `nil` means `.relaxed`, the default on both sides.
    /// **`expertCacheSlots` must match too**, for the same reason one term
    /// over: a footprint is `slots x layers x expert stride`, so a ranking at
    /// one slot count and an open at another are two configurations rather
    /// than one approximation.
    public static func recommend(
        context: UInt32 = 4096,
        expertCacheSlots: OpenOptions.Sizing? = nil,
        loadGuard: OpenOptions.LoadGuard? = nil,
        probe: Bool = false
    ) throws -> [ModelRecommendation] {
        let json = try encodeOptions(
            RecommendOptions(
                loadGuard: loadGuard,
                expertCacheSlots: expertCacheSlots,
                probe: probe ? true : nil))
        return try decode(
            [ModelRecommendation].self,
            from: try takeString { out in
                withOptionalCString(json) { ts_recommend_json(context, $0, out) }
            })
    }

    /// Ranks curated models on a dedicated thread and reports real header
    /// probe progress. Use this for interactive surfaces when `probe` is
    /// true; the synchronous overload remains the cheap offline path.
    public static func recommendWithProgress(
        context: UInt32 = 4096,
        expertCacheSlots: OpenOptions.Sizing? = nil,
        loadGuard: OpenOptions.LoadGuard? = nil,
        probe: Bool = true
    ) -> AsyncThrowingStream<RecommendationEvent, Error> {
        AsyncThrowingStream { continuation in
            let thread = Thread {
                let box = RecommendationBox(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<RecommendationBox>.fromOpaque(userdata).release() }
                do {
                    let json = try encodeOptions(
                        RecommendOptions(
                            loadGuard: loadGuard,
                            expertCacheSlots: expertCacheSlots,
                            probe: probe ? true : nil))
                    let result = try takeString { out in
                        withOptionalCString(json) {
                            ts_recommend_progress_json(
                                context,
                                $0,
                                recommendationCallback,
                                userdata,
                                out)
                        }
                    }
                    continuation.yield(.finished(
                        try decode([ModelRecommendation].self, from: result)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            thread.name = "com.turbospark.recommend"
            thread.start()
        }
    }

    /// Encodes an options bag, or `nil` when every field is absent.
    ///
    /// NULL and `{}` mean the same thing to the ABI, so sending nothing when
    /// there is nothing to send keeps the no-options call byte-identical to
    /// what it was before these knobs existed.
    private static func encodeOptions<T: Encodable>(_ options: T) throws -> String? {
        let data = try JSONEncoder().encode(options)
        let text = String(decoding: data, as: UTF8.self)
        return text == "{}" ? nil : text
    }

    /// The options bag `ts_recommend_json` takes. Private because the
    /// only caller is `recommend` above; a JSON blob rather than a second C
    /// argument for the reason every other options bag in this ABI is one --
    /// a knob added later is a field rather than a break.
    private struct RecommendOptions: Encodable {
        let loadGuard: OpenOptions.LoadGuard?
        let expertCacheSlots: OpenOptions.Sizing?
        /// Omitted for the fast offline path so existing callers keep the
        /// byte-for-byte options shape they used before probing was exposed.
        let probe: Bool?
    }

    /// The options bag `ts_probe_json` takes: `RecommendOptions` plus the
    /// window, because a probe reports a FIT and a fit is only meaningful at
    /// a stated context and slot count.
    private struct ProbeOptions: Encodable {
        let contextWindow: UInt32?
        let loadGuard: OpenOptions.LoadGuard?
        let expertCacheSlots: OpenOptions.Sizing?
    }

    /// Installs a catalog row, streaming progress.
    ///
    /// **A failed or cancelled walk CANNOT RESUME**: it streams gigabytes without writing the
    /// checkpoint to disk whole, and a failure restarts it from the
    /// beginning. Tell the user before starting; the first `.stage` event
    /// says so. Pause/resume preserves the current worker while the app stays open.
    ///
    /// Byte events arrive from several download threads at once and may go
    /// backwards in wall-clock order. Take the maximum rather than the last
    /// if you drive a progress bar from them.
    ///
    /// **CANCELLABLE SINCE `ts_install_cancel` EXISTED ON THE RUST SIDE**
    /// (`cancelInstall()` below): a cancelled walk dies the same death a
    /// network failure gives it -- nothing of the partial install is kept,
    /// and the error carries the "install cancelled" text.
    public static func install(_ alias: String) -> AsyncThrowingStream<InstallEvent, Error> {
        AsyncThrowingStream { continuation in
            // A dedicated thread, not a global queue slot: this blocks for
            // tens of minutes, and parking a shared concurrent-queue worker
            // for that long starves everything else in the process.
            let thread = Thread {
                let box = InstallBox(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<InstallBox>.fromOpaque(userdata).release() }
                do {
                    let json = try takeString { out in
                        alias.withCString { ts_install($0, installCallback, userdata, out) }
                    }
                    continuation.yield(.finished(try decode(InstalledModel.self, from: json)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            thread.name = "com.turbospark.install"
            thread.start()
        }
    }

    /// Downloads and packs one curated image source into the shared image
    /// store. The source is staged temporarily and removed after publication.
    public static func installImage(_ alias: String) -> AsyncThrowingStream<ImageInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let thread = Thread {
                let box = ImageInstallBox(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<ImageInstallBox>.fromOpaque(userdata).release() }
                do {
                    let json = try takeString { out in
                        alias.withCString { ts_image_install($0, imageInstallCallback, userdata, out) }
                    }
                    continuation.yield(.finished(
                        try decode(ImageInstalledModel.self, from: json)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            thread.name = "com.turbospark.image-install"
            thread.start()
        }
    }

    /// Probes and installs an arbitrary Hugging Face repository, streaming progress.
    ///
    /// `repo` is `owner/name` or `owner/name@revision`. `alias` is the local name.
    ///
    /// Supports pause/resume while the worker stays alive; cancellation,
    /// failure, or quitting requires a fresh install.
    public static func install(
        repo: String,
        alias: String,
        file: String? = nil,
        sidecarRepo: String? = nil
    ) -> AsyncThrowingStream<InstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let thread = Thread {
                let box = InstallBox(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<InstallBox>.fromOpaque(userdata).release() }
                do {
                    let json = try takeString { out in
                        repo.withCString { r in
                            alias.withCString { a in
                                withOptionalCString(file) { f in
                                    withOptionalCString(sidecarRepo) { s in
                                        ts_install_repo(r, a, f, s, installCallback, userdata, out)
                                    }
                                }
                            }
                        }
                    }
                    continuation.yield(.finished(try decode(InstalledModel.self, from: json)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            thread.name = "com.turbospark.install"
            thread.start()
        }
    }

    /// Signals every in-flight install walk to stop.
    ///
    /// Returns true when at least one walk was running and has been
    /// signalled. The walk does not die ON this call: it fails at its next
    /// ranged chunk read with the "install cancelled" error, seconds later,
    /// and keeps nothing (the walk cannot resume, so a cancelled install is
    /// a dead install, exactly like one that failed on the network).
    /// Documented safe from any thread.
    @discardableResult
    public static func cancelInstall() -> Bool {
        ts_install_cancel() != 0
    }

    /// Pauses at the next download boundary without discarding current work.
    /// An in-flight request may finish first. The app must remain open.
    @discardableResult
    public static func pauseInstall() -> Bool {
        ts_install_pause() != 0
    }

    /// Continues a paused install in this process. Cannot revive a cancelled
    /// or failed install, or recover an install after quitting the app.
    @discardableResult
    public static func resumeInstall() -> Bool {
        ts_install_resume() != 0
    }

    /// How many install walks have finished (success, failure, or cancel)
    /// since process start. Read twice around `cancelInstall()` to confirm
    /// a cancelled walk actually exited rather than being wedged inside a
    /// blocking read.
    public static func installsFinished() -> UInt32 {
        ts_installs_finished()
    }
}

/// Carries an install continuation across the C boundary.
///
/// The byte callback fires from several download threads at once, which the
/// C header states as an obligation on the caller. `AsyncThrowingStream`'s
/// continuation is documented `Sendable` and safe to yield to concurrently,
/// so no lock is needed here -- but a caller's own progress state does need
/// one, which is why `install`'s docs say to take the maximum.
private final class InstallBox: @unchecked Sendable {
    let continuation: AsyncThrowingStream<InstallEvent, Error>.Continuation
    init(_ continuation: AsyncThrowingStream<InstallEvent, Error>.Continuation) {
        self.continuation = continuation
    }
}

private final class RecommendationBox: @unchecked Sendable {
    let continuation: AsyncThrowingStream<RecommendationEvent, Error>.Continuation

    init(_ continuation: AsyncThrowingStream<RecommendationEvent, Error>.Continuation) {
        self.continuation = continuation
    }
}

private let recommendationCallback: TsRecommendCallback = { userdata, completed, total in
    guard let userdata else { return }
    let box = Unmanaged<RecommendationBox>.fromOpaque(userdata).takeUnretainedValue()
    box.continuation.yield(.progress(completed: completed, total: total))
}

private let installCallback: TsInstallCallback = { userdata, kind, text, len, done, total in
    guard let userdata else { return }
    let box = Unmanaged<InstallBox>.fromOpaque(userdata).takeUnretainedValue()
    switch kind {
    case TS_INSTALL_STAGE:
        guard let text, len > 0,
            let s = String(bytes: UnsafeRawBufferPointer(start: text, count: len), encoding: .utf8)
        else { return }
        box.continuation.yield(.stage(s))
    case TS_INSTALL_BYTES:
        box.continuation.yield(.bytes(done: done, total: total))
    default:
        return
    }
}

private final class ImageInstallBox: @unchecked Sendable {
    let continuation: AsyncThrowingStream<ImageInstallEvent, Error>.Continuation

    init(_ continuation: AsyncThrowingStream<ImageInstallEvent, Error>.Continuation) {
        self.continuation = continuation
    }
}

private let imageInstallCallback: TsInstallCallback = { userdata, kind, text, len, done, total in
    guard let userdata else { return }
    let box = Unmanaged<ImageInstallBox>.fromOpaque(userdata).takeUnretainedValue()
    switch kind {
    case TS_INSTALL_STAGE:
        guard let text, len > 0,
            let stage = String(
                bytes: UnsafeRawBufferPointer(start: text, count: len), encoding: .utf8)
        else { return }
        box.continuation.yield(.stage(stage))
    case TS_INSTALL_BYTES:
        box.continuation.yield(.bytes(done: done, total: total))
    default:
        return
    }
}

private final class StoreRelocationBox: @unchecked Sendable {
    let continuation: AsyncThrowingStream<StoreRelocationEvent, Error>.Continuation

    init(_ continuation: AsyncThrowingStream<StoreRelocationEvent, Error>.Continuation) {
        self.continuation = continuation
    }
}

private let storeRelocationCallback: TsInstallCallback = { userdata, kind, _, _, done, total in
    guard let userdata, kind == TS_INSTALL_BYTES else { return }
    let box = Unmanaged<StoreRelocationBox>.fromOpaque(userdata).takeUnretainedValue()
    box.continuation.yield(.bytes(done: done, total: total))
}

/// Calls `body` with a C string, or NULL when the value is absent.
///
/// Written out rather than using `?.withCString`, which cannot express the
/// null case without duplicating the body.
private func withOptionalCString<R>(
    _ value: String?,
    _ body: (UnsafePointer<CChar>?) -> R
) -> R {
    guard let value else { return body(nil) }
    return value.withCString { body($0) }
}

/// What a `.gguf` control vector declares about itself.
///
/// **A SHAPE MATCH IS NOT A SEMANTIC MATCH, and this reports the shape only.**
/// The engine refuses a width or layer-count mismatch and refuses NOTHING
/// else: a vector extracted for a different checkpoint of the same hidden size
/// opens, steers, and changes behaviour in a direction nobody asked for,
/// silently. A UI rendering these fields owes its user that sentence.
public struct ControlVectorInfo: Decodable, Sendable, Equatable {
    /// The hidden size every direction in the file declares. Must equal the
    /// install's own `arch.hiddenSize` or the open refuses the set.
    public let hidden: Int
    /// How many blocks actually carry a direction.
    public let coveredLayers: Int
    /// Lowest block index covered.
    ///
    /// **NORMALLY 1, NOT 0, AND THAT IS NOT A GAP.** Block 0 is not
    /// expressible in a file this engine writes and llama.cpp never applies a
    /// direction there, so a "31 of 32" reading is that convention working.
    public let minLayer: Int?
    /// Highest block index covered.
    public let maxLayer: Int?
    /// The span the directions cover, which is what the install's own
    /// `arch.numLayers` is compared against. A gapped vector covering 4 blocks
    /// across 8 spans 8: comparing the COUNT instead would call it compatible
    /// with a 5-layer model.
    public let spannedLayers: Int
    /// The mode the file itself declares, used when a caller names none.
    public let declaredMode: String?
    /// The architecture the file was extracted against, when it carries one.
    /// **Advisory only** -- nothing validates against it, which is exactly why
    /// it is worth showing.
    public let declaredArch: String?
}
