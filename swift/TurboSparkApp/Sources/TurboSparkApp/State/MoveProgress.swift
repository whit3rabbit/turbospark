import Foundation

/// What a move in flight is doing, for the sheet that watches it (requirement 7.2).
///
/// `AppModel.activeMove` holds one from the moment a move starts until it ends in
/// ANY way (finished, failed, cancelled, refused by the engine, shut down), so a
/// control that asks "is a move running?" has one place to look.
///
/// **THE BYTES COME FROM THE ENGINE'S CALLBACK AND ONLY FROM IT.** The first
/// callback is `(0, total)`, then the cumulative bytes copied. Until it arrives the
/// bar is indeterminate (`fraction` is nil): the app does not know the total, and a
/// bar at zero would claim it did. A callback that arrives late with a smaller
/// figure never moves the bar back (the Swift package says to take the maximum).
struct MoveProgress: Equatable, Sendable {
    /// The line under the bar. The titles are catalog keys, rendered through
    /// `Text(LocalizedStringKey(...), bundle: .module)`; `LocalizationParityTests`
    /// holds each one to a translation in every language.
    enum Phase: String, CaseIterable, Sendable {
        /// No callback yet: the engine is checking and creating the staging folder.
        case starting = "Preparing to move..."
        case copying = "Copying..."
        /// Every byte is copied; the engine compares the copy with the original,
        /// lists it and removes the original. There is no callback for these.
        case finishing = "Verifying and finishing..."
        /// Cancel was pressed and the engine has not answered yet.
        case cancelling = "Cancelling..."

        var title: String { rawValue }
    }

    let installId: String
    let alias: String
    /// Both names are display text (This Mac in the UI language, a library name
    /// made safe to show).
    let sourceName: String
    let destinationName: String
    /// The folder the model is being moved FROM. A loader that is handed a path and
    /// not a row (an image or an audio session, Humanize, a chosen folder) asks
    /// `AppModel.isMoving(path:)`, which reads this. Empty only for a progress made
    /// without one (a test of the bar alone).
    let path: String
    private(set) var done: UInt64 = 0
    private(set) var total: UInt64 = 0
    /// True once the engine's first callback has arrived.
    private(set) var hasProgress = false
    /// Cancel was pressed. The engine may still finish the move (a cancel that
    /// arrives after the copy was published cannot undo it).
    var cancelRequested = false

    init(installId: String, alias: String, sourceName: String, destinationName: String, path: String = "") {
        self.installId = installId
        self.alias = alias
        self.sourceName = sourceName
        self.destinationName = destinationName
        self.path = path
    }

    var phase: Phase {
        if cancelRequested { return .cancelling }
        if !hasProgress { return .starting }
        return total > 0 && done >= total ? .finishing : .copying
    }

    /// 0...1, or nil while the total is not known.
    var fraction: Double? {
        guard hasProgress, total > 0 else { return nil }
        return min(1, Double(done) / Double(total))
    }

    mutating func apply(done: UInt64, total: UInt64) {
        hasProgress = true
        if total > 0 { self.total = total }
        self.done = max(self.done, done)
    }
}

/// Forwards at most one progress figure per interval, plus the first and the last.
///
/// The engine calls back between files and chunks, which on a fast drive is
/// hundreds of times a second from the move's own thread. Each forwarded figure is
/// a published change on the main actor, so a flood would redraw the sheet far
/// faster than anyone reads it. The first figure starts the bar and the last one
/// (`done >= total`) is never dropped, because it is the one that says "finishing".
final class MoveProgressThrottle: @unchecked Sendable {
    private let interval: TimeInterval
    private let now: @Sendable () -> TimeInterval
    private let lock = NSLock()
    private var lastForwarded: TimeInterval?

    init(
        minimumInterval: TimeInterval = 0.1,
        now: @escaping @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
    ) {
        interval = minimumInterval
        self.now = now
    }

    func shouldForward(done: UInt64, total: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let current = now()
        let isLast = total > 0 && done >= total
        if let lastForwarded, current - lastForwarded < interval, !isLast { return false }
        lastForwarded = current
        return true
    }
}

/// A model that was just moved, for the screens that remember a model by path.
/// The Installed Models list keeps its own selected path, which names a folder
/// that is gone after a move.
struct MovedModelEvent: Equatable, Sendable {
    let id = UUID()
    let oldInstallId: String
    let newInstallId: String
    let oldPath: String
    /// Where the model is listed now. Nil when the listing did not hold it yet.
    let newPath: String?

    /// The path a selection should hold after the move: the new one when it was
    /// on the moved model, otherwise as it was.
    func follow(selectedPath: String?) -> String? {
        guard let selectedPath else { return nil }
        if selectedPath == oldPath, let newPath { return newPath }
        return selectedPath
    }
}
