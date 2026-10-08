import CTurboSpark
import Foundation

/// Facts about the linked library itself, rather than any model or session.
public enum TurboSparkRuntime {
    /// What `ts_build_info_json` reports.
    public struct BuildInfo: Decodable, Sendable, Equatable {
        public let abiVersion: UInt32
        /// The turbospark-ffi crate version.
        public let version: String
        /// True in an unoptimized build, whose timings must not be quoted as
        /// performance.
        public let debugAssertions: Bool
    }

    /// The ABI revision of the header this package compiled against.
    public static var headerABIVersion: UInt32 { UInt32(TS_ABI_VERSION) }

    /// The ABI revision of the linked archive.
    public static var libraryABIVersion: UInt32 { ts_abi_version() }

    /// Build information for a diagnostics pane or a bug report.
    public static func buildInfo() throws -> BuildInfo {
        try decode(BuildInfo.self, from: takeString { ts_build_info_json($0) })
    }

    /// Throws if the staged header and archive disagree about the ABI.
    ///
    /// Both are gitignored copies staged by `make swift-lib`, so a stale one
    /// is easy to end up with and otherwise surfaces as a missing JSON key far
    /// from its cause. The engine opens and servers call this once per
    /// process through `verifyABIOnce()`.
    public static func verifyABI() throws {
        try verify(header: headerABIVersion, library: libraryABIVersion)
    }

    /// Separated so the comparison is testable without a skewed archive.
    static func verify(header: UInt32, library: UInt32) throws {
        guard header == library else {
            throw TurboSparkError(
                code: .unknown,
                message: "TurboSpark header is ABI \(header) but the linked library is ABI "
                    + "\(library). Run `make swift-lib` to restage both from the same build.")
        }
    }

    private static let onceResult: Result<Void, Error> = Result { try verifyABI() }

    /// `verifyABI()`, evaluated once and cached for the life of the process.
    public static func verifyABIOnce() throws {
        try onceResult.get()
    }
}
