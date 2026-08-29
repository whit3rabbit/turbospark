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
    /// seconds, no download. Returns the raw JSON, since a probe report is
    /// for display and its shape follows what the engine learns to read.
    ///
    /// `repo` is `owner/name` or `owner/name@revision`.
    public static func probe(
        repo: String,
        file: String? = nil,
        sidecarRepo: String? = nil
    ) throws -> String {
        try takeString { out in
            repo.withCString { r in
                withOptionalCString(file) { f in
                    withOptionalCString(sidecarRepo) { s in
                        ts_probe_json(r, f, s, out)
                    }
                }
            }
        }
    }

    /// Deletes an installed model from `~/.turbospark` and removes its directory.
    public static func delete(_ alias: String) throws {
        try check(alias.withCString { ts_model_delete($0) })
    }

    /// Ranks curated models by hardware fit for this machine at `context`.
    public static func recommend(context: UInt32 = 4096) throws -> [ModelRecommendation] {
        try decode([ModelRecommendation].self, from: try takeString { ts_recommend_json(context, $0) })
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
