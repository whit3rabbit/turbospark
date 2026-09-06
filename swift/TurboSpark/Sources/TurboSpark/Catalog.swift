import CTurboSpark
import Foundation

// THE TWO TYPES BELOW DECODE snake_case WHERE EVERYTHING ELSE IN THIS
// PACKAGE DECODES camelCase, and that asymmetry is deliberate rather than an
// oversight.
//
// The rest of this binding's JSON is a WIRE shape invented for it, so the
// Rust side spells it camelCase and no CodingKeys are needed anywhere. These
// two are not: they are the engine's own on-disk formats -- `models.json`
// (checked into the repository) and `~/.turbospark/installed.json` (written
// by every `turbospark-model pull`) -- passed through unchanged so a GUI's
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
    public let installBytes: UInt64
    /// `YYYY-MM-DD`. Whole days only.
    public let installedOn: String

    public init(
        alias: String,
        repo: String,
        revision: String = "main",
        path: String,
        family: String,
        installBytes: UInt64 = 0,
        installedOn: String = ""
    ) {
        self.alias = alias
        self.repo = repo
        self.revision = revision
        self.path = path
        self.family = family
        self.installBytes = installBytes
        self.installedOn = installedOn
    }

    enum CodingKeys: String, CodingKey {
        case alias, repo, revision, path, family
        case installBytes = "install_bytes"
        case installedOn = "installed_on"
    }
}

/// What an install will cost, before it starts.
public struct InstallCost: Decodable, Sendable, Equatable {
    public let downloadBytes: UInt64
    public let installBytes: UInt64
}

/// One install-progress event.
public enum InstallEvent: Sendable, Equatable {
    case stage(String)
    case bytes(done: UInt64, total: UInt64)
    case finished(InstalledModel)
}

/// Browsing, probing and installing models.
///
/// Available on every platform, including ones that cannot then RUN a model:
/// the artifact is the same either way, and refusing to list a catalog on a
/// machine that cannot decode would be a restriction with no reason behind
/// it.
public enum TurboSparkCatalog {
    /// The curated table, each row carrying whether it is installed.
    public static func available() throws -> [CatalogEntry] {
        try decode([CatalogEntry].self, from: try takeString { ts_catalog_json($0) })
    }

    /// What is installed in `~/.turbospark`.
    public static func installed() throws -> [InstalledModel] {
        try decode([InstalledModel].self, from: try takeString { ts_installed_json($0) })
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
        loadGuard: OpenOptions.LoadGuard? = nil
    ) throws -> [ModelRecommendation] {
        let json = try encodeOptions(
            RecommendOptions(loadGuard: loadGuard, expertCacheSlots: expertCacheSlots))
        return try decode(
            [ModelRecommendation].self,
            from: try takeString { out in
                withOptionalCString(json) { ts_recommend_json(context, $0, out) }
            })
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

    /// The one-key options bag `ts_recommend_json` takes. Private because the
    /// only caller is `recommend` above; a JSON blob rather than a second C
    /// argument for the reason every other options bag in this ABI is one --
    /// a knob added later is a field rather than a break.
    private struct RecommendOptions: Encodable {
        let loadGuard: OpenOptions.LoadGuard?
        let expertCacheSlots: OpenOptions.Sizing?
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
    /// **The walk CANNOT RESUME**: it streams gigabytes without writing the
    /// checkpoint to disk whole, and a failure restarts it from the
    /// beginning. Tell the user before starting; the first `.stage` event
    /// says so.
    ///
    /// Byte events arrive from several download threads at once and may go
    /// backwards in wall-clock order. Take the maximum rather than the last
    /// if you drive a progress bar from them.
    ///
    /// **AND IT CANNOT BE CANCELLED.** Dropping the consuming task ends
    /// DELIVERY and nothing else: `ts_install` blocks its thread for the
    /// whole walk and the C ABI exposes no install-cancel call, so the
    /// download keeps running to completion or failure on a thread nobody is
    /// listening to. Do not build a Stop button on this that claims
    /// otherwise. Making it real needs a `ts_install_cancel` on the Rust side
    /// first, which does not exist today.
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

    /// Probes and installs an arbitrary Hugging Face repository, streaming progress.
    ///
    /// `repo` is `owner/name` or `owner/name@revision`. `alias` is the local name.
    ///
    /// Cannot resume and CANNOT BE CANCELLED, for the reasons the catalog
    /// overload above states in full.
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
