import TurboSpark

/// What the menu bar shows about a running daemon's engine version (OC-13).
///
/// **THE DAEMON OUTLIVES THE APP.** A daemon started by the build before a
/// Sparkle update keeps serving the old engine, so the app compares the
/// version stamped in the daemon's meta file against its own minimum and
/// says so, rather than letting a request-shape mismatch surface later as an
/// unrelated error. The fix is always the same: restart it from this app.
///
/// Pure, so the choice is asserted without a daemon.
enum DaemonVersionLine: Equatable {
    /// Compatible: show the version.
    case version(String)
    /// Incompatible or malformed: name the reported version and the minimum.
    case mismatch(daemon: String, minimum: String)
    /// Started by a build that predates the version stamp.
    case unknown

    static func make(
        for status: TurboSparkDaemonStatus,
        minimum: String = DaemonEngineCompatibility.minimumEngineVersion
    ) -> DaemonVersionLine {
        switch status.compatibility(minimum: minimum) {
        case .compatible:
            return .version(status.engineVersion ?? "")
        case .unknownVersion:
            return .unknown
        case .unparseable(let value):
            return .mismatch(daemon: value, minimum: minimum)
        case .majorMismatch(let daemon, let min), .tooOld(let daemon, let min):
            return .mismatch(daemon: daemon, minimum: min)
        }
    }

    /// True for the states that should lead the user to Restart.
    var isWarning: Bool {
        if case .version = self { return false }
        return true
    }
}
