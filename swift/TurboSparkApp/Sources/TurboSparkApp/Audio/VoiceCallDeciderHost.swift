import Foundation
import TurboSpark

/// OC-08 follow-up: a dedicated small model for the auto-send decider.
///
/// "Automatic" keeps the first-pass behavior (the loaded chat model). A user
/// who picks an installed model instead gets a SECOND resident session,
/// owned here, the same way Humanize owns one beside the chat session
/// (`HumanizeController`). No engine change is needed: each
/// `TurboSparkSession` is its own engine.
///
/// The load is lazy and bounded:
///
/// - It starts only when a call is active AND auto-send is on AND a model was
///   chosen. Never at launch, never from a download.
/// - It has its OWN budget (`loadBudget`). Until it finishes, every decision
///   fails closed to draft-first, immediately, so a cold load never eats the
///   decision's timeout and never delays the user's draft.
/// - It unloads when the call ends, when auto-send is turned off, when the
///   choice changes, when its files are about to be deleted or moved, and
///   after `idleTimeout` without a decision (a later decision reloads it,
///   failing closed meanwhile).
/// - It never runs beside a chat turn: a decision refuses to start while a
///   turn is busy, and one in flight is cancelled and discarded when a turn
///   starts.

/// What the host needs of a loaded decider model. A seam so the state machine
/// is testable without a model. `TurboSparkSession` is the production one.
protocol VoiceCallDeciderEngine: AnyObject, Sendable {
    func complete(_ messages: [ChatMessage], options: GenerateOptions) async throws -> String
    func cancel()
}

/// Production engine: one `TurboSparkSession` read to the end as a string.
final class VoiceCallSessionEngine: VoiceCallDeciderEngine {
    private let session: TurboSparkSession
    init(session: TurboSparkSession) { self.session = session }

    func complete(_ messages: [ChatMessage], options: GenerateOptions) async throws -> String {
        var response = ""
        for try await event in session.generate(messages, options: options) {
            try Task.checkCancellation()
            if case .content(let chunk) = event { response += chunk }
        }
        return response
    }

    func cancel() { session.cancel() }
}

enum VoiceCallDeciderError: Error, Equatable {
    /// No dedicated model is wanted right now (call off, auto-send off, or
    /// the choice is Automatic).
    case notWanted
    /// Not loaded yet. The caller falls back to draft-first.
    case notReady
    /// A chat turn is running; the decider never overlaps one.
    case turnBusy
}

/// A model the user may pick, reduced to what the decision needs. Built from
/// `InstalledModel` in one place so tests do not need an install.
struct VoiceCallDeciderCandidate: Equatable, Sendable, Identifiable {
    let installId: String
    let alias: String
    let path: String
    let bytes: UInt64
    var id: String { installId }
}

/// Which model answers, decided from the saved choice and what is installed.
enum VoiceCallDeciderResolution: Equatable {
    enum Note: Equatable {
        /// The saved model is gone (deleted, drive away, no longer a text
        /// model). The choice behaves as Automatic until the user picks again.
        case savedModelUnavailable
    }

    /// The loaded chat model answers (the first-pass behavior).
    case automatic(note: Note?)
    /// A second resident session on this model answers.
    case dedicated(VoiceCallDeciderCandidate)

    var dedicatedCandidate: VoiceCallDeciderCandidate? {
        if case .dedicated(let candidate) = self { return candidate }
        return nil
    }

    var note: Note? {
        if case .automatic(let note) = self { return note }
        return nil
    }

    /// `savedID` nil means Automatic. An id that is no longer in `available`
    /// falls back to Automatic WITH a note, and never to a different model.
    /// A model that is already the loaded chat model needs no second copy:
    /// it is Automatic (same answer, no extra memory).
    static func resolve(
        savedID: String?, available: [VoiceCallDeciderCandidate], loadedChatPath: String?
    ) -> VoiceCallDeciderResolution {
        guard let savedID else { return .automatic(note: nil) }
        guard let match = available.first(where: { $0.installId == savedID }) else {
            return .automatic(note: .savedModelUnavailable)
        }
        if let loadedChatPath, loadedChatPath == match.path { return .automatic(note: nil) }
        return .dedicated(match)
    }
}

/// Budgets. The decision timeout is per model: a bigger model needs longer to
/// prefill the rubric, and a fixed 2 s would fail closed on every decision.
/// The load has its own, much larger, budget.
enum VoiceCallDeciderBudget {
    static let loadBudget: Duration = .seconds(90)
    static let idleTimeout: Duration = .seconds(180)

    /// Never shorter than the Automatic budget.
    static func decisionTimeout(modelBytes: UInt64) -> Duration {
        let gb = UInt64(1) << 30
        switch modelBytes {
        case ...(3 * gb): return VoiceCallTuning.decisionTimeout
        case ...(6 * gb): return .milliseconds(3_000)
        default: return .milliseconds(4_000)
        }
    }

    /// A second resident model must leave the chat model room: at most a
    /// quarter of physical memory (9 GB on a 36 GB Mac).
    static func fitsBesideChatModel(bytes: UInt64, physicalMemory: UInt64) -> Bool {
        bytes <= physicalMemory / 4
    }
}

@MainActor
final class VoiceCallDeciderHost: ObservableObject {
    enum Status: Equatable {
        case unloaded
        case loading
        case ready
        case failed(Failure)
    }

    enum Failure: Equatable {
        case loadTimedOut
        case loadFailed
        case tooLarge
    }

    typealias Sleep = @Sendable (Duration) async throws -> Void

    @Published private(set) var status: Status = .unloaded

    /// Opens the engine at a path. Replaced in tests.
    var open: @Sendable (String) async throws -> VoiceCallDeciderEngine = { _ in
        throw VoiceCallDeciderError.notReady
    }
    /// Whether a chat turn is running now. Set by the owner.
    var turnBusy: @MainActor () -> Bool = { false }
    var sleep: Sleep = { try await Task.sleep(for: $0) }
    var loadBudget: Duration = VoiceCallDeciderBudget.loadBudget
    var idleTimeout: Duration = VoiceCallDeciderBudget.idleTimeout
    var physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory

    private(set) var candidate: VoiceCallDeciderCandidate?
    private(set) var wanted = false
    private var engine: VoiceCallDeciderEngine?
    /// Bumped by every load, unload and change, so a late open, timer or
    /// decision from an older generation is dropped.
    private var epoch = 0
    private var loadTimer: Task<Void, Never>?
    private var idleTimer: Task<Void, Never>?

    /// The path whose files this host maps or is opening, for the delete and
    /// move guards.
    var heldPath: String? {
        switch status {
        case .loading, .ready: return candidate?.path
        case .unloaded, .failed: return nil
        }
    }

    /// The dedicated model to use, or nil for Automatic. A change unloads the
    /// old one; nothing loads until `setWanted(true)`.
    func configure(candidate new: VoiceCallDeciderCandidate?) {
        guard new != candidate else { return }
        candidate = new
        unload()
        applyWanted()
    }

    /// True while a call is active, auto-send is on, and a model is chosen.
    func setWanted(_ value: Bool) {
        guard value != wanted else { return }
        wanted = value
        if value {
            // A new call is a fresh chance after a failed load.
            if case .failed = status { status = .unloaded }
            applyWanted()
        } else {
            unload()
        }
    }

    private func applyWanted() {
        guard wanted, let candidate, status == .unloaded else { return }
        guard VoiceCallDeciderBudget.fitsBesideChatModel(bytes: candidate.bytes, physicalMemory: physicalMemory) else {
            status = .failed(.tooLarge)
            return
        }
        startLoad(candidate)
    }

    private func startLoad(_ target: VoiceCallDeciderCandidate) {
        epoch += 1
        let mine = epoch
        status = .loading
        let open = self.open
        let sleep = self.sleep
        let budget = loadBudget
        let path = target.path
        Task { [weak self] in
            do {
                let opened = try await open(path)
                self?.loaded(opened, epoch: mine)
            } catch {
                self?.loadFailed(.loadFailed, epoch: mine)
            }
        }
        loadTimer?.cancel()
        loadTimer = Task { [weak self] in
            // The load budget. A sleep that throws (cancelled) must not
            // count as an expiry.
            do { try await sleep(budget) } catch { return }
            self?.loadFailed(.loadTimedOut, epoch: mine)
        }
    }

    private func loaded(_ opened: VoiceCallDeciderEngine, epoch mine: Int) {
        // Native opening can finish after an unload, a change or a timeout.
        // It is dropped, never published.
        guard epoch == mine, status == .loading, wanted else { return }
        loadTimer?.cancel()
        loadTimer = nil
        engine = opened
        status = .ready
        scheduleIdle()
    }

    private func loadFailed(_ failure: Failure, epoch mine: Int) {
        guard epoch == mine, status == .loading else { return }
        epoch += 1
        loadTimer?.cancel()
        loadTimer = nil
        status = .failed(failure)
    }

    /// Releases the model's memory. Safe at any time; a decision in flight
    /// fails closed.
    func unload() {
        epoch += 1
        loadTimer?.cancel()
        loadTimer = nil
        idleTimer?.cancel()
        idleTimer = nil
        engine?.cancel()
        engine = nil
        if status != .unloaded { status = .unloaded }
    }

    /// Unloads when this host holds the files at `path`. Called before a
    /// delete or a move, so the mapping never outlives the files.
    func release(path: String) {
        guard let held = heldPath, held == path else { return }
        unload()
    }

    /// Cancels a decision in flight (a turn started, or the call ended).
    func cancelInFlight() {
        engine?.cancel()
    }

    private func scheduleIdle() {
        idleTimer?.cancel()
        let mine = epoch
        let sleep = self.sleep
        let timeout = idleTimeout
        idleTimer = Task { [weak self] in
            do { try await sleep(timeout) } catch { return }
            self?.idleExpired(epoch: mine)
        }
    }

    private func idleExpired(epoch mine: Int) {
        guard epoch == mine, status == .ready else { return }
        unload()
    }

    /// One decision on the dedicated model. Throws, never guesses: every
    /// throw becomes draft-first in the caller.
    func complete(_ messages: [ChatMessage], options: GenerateOptions) async throws -> String {
        guard wanted, candidate != nil else { throw VoiceCallDeciderError.notWanted }
        guard !turnBusy() else { throw VoiceCallDeciderError.turnBusy }
        guard status == .ready, let engine else {
            // Idle expiry or a failed earlier load: try again for the NEXT
            // utterance. This one fails closed at once; the load never
            // counts against its budget.
            if status == .unloaded { applyWanted() }
            throw VoiceCallDeciderError.notReady
        }
        let mine = epoch
        idleTimer?.cancel()
        defer { if epoch == mine, status == .ready { scheduleIdle() } }
        let reply = try await engine.complete(messages, options: options)
        // A turn, an unload or a change landed while it ran: discard.
        guard epoch == mine, wanted, !turnBusy() else { throw VoiceCallDeciderError.turnBusy }
        return reply
    }
}

/// Picks the decider per decision, so a setting change applies to the next
/// utterance without rebuilding the call controller.
struct VoiceCallRoutedDecider: VoiceCallSubmitDeciding {
    let automatic: VoiceCallSubmitDeciding
    let dedicated: VoiceCallSubmitDeciding
    let usesDedicated: @Sendable () async -> Bool

    func decide(_ utterance: String) async throws -> VoiceCallSubmitDecision {
        if await usesDedicated() { return try await dedicated.decide(utterance) }
        return try await automatic.decide(utterance)
    }
}
