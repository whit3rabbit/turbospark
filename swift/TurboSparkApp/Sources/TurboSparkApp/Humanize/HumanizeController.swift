import Foundation
import TurboSpark

/// What the controller needs of a loaded humanizer model.
///
/// A seam so the state machine is testable without a 12 GB install.
/// `TurboSparkSession` is the only production conformer.
protocol HumanizeEngine: AnyObject, Sendable {
    func humanize(
        _ draft: String, options: HumanizeOptions
    ) -> AsyncThrowingStream<GenerationEvent, Error>
    func cancel()
}

extension TurboSparkSession: HumanizeEngine {}

/// State for the Humanize section: paste text, press a button, read the
/// rewrite and its audit.
///
/// **IT OWNS ITS OWN SESSION.** The humanizer is a raw-completion rewriter, so
/// the chat session (whatever model it holds, with its chat template) cannot
/// serve it; the rewrite needs `humanizer-12b` opened at the model's 8,192
/// window. Opening is expensive (it maps the weights and compiles Metal
/// pipelines), so the engine stays resident until `unload()`.
@MainActor
final class HumanizeController: ObservableObject {
    /// The catalog alias this section serves. The wrapper and sampling recipe
    /// inside the library are specific to this model.
    static let modelAlias = "humanizer-12b"
    /// The model's window. Opening wider only reserves KV it can never use.
    static let contextTokens = UInt32(HumanizePlanner.window)

    enum Phase: Equatable {
        case idle
        case loadingModel
        case running
    }

    @Published var draft = ""
    /// Saved on every change; a run uses the rules as they were when it started.
    @Published var rules: HumanizeRules {
        didSet { if rules != oldValue { rulesStore.save(rules) } }
    }
    private let rulesStore: HumanizeRulesStore
    @Published private(set) var rewrite = ""
    /// The draft the current rewrite was made from, for highlighting.
    @Published private(set) var sourceDraft = ""
    /// Which draft text produced which rewrite text, finished parts only.
    @Published private(set) var segments: [HumanizeSegment] = []
    @Published private(set) var plan: HumanizePlan?
    @Published private(set) var contextError: String?
    private var countedDraft: String?
    private var countedPath: String?
    private let tokenCounter = HumanizeTokenCounter()
    lazy var countDraftTokens: (String, String) async throws -> Int = { [tokenCounter] draft, path in
        try await tokenCounter.count(draft, at: path)
    }
    /// Nil until every part of a finished rewrite reported its audit.
    @Published private(set) var checks: HumanizeChecks?
    /// Nil until a rewrite finishes every part.
    @Published private(set) var ruleFindings: HumanizeRuleFindings?
    @Published private(set) var stats: HumanizeStats?
    @Published private(set) var stopReason: GenerationResult.StopReason?
    @Published private(set) var phase: Phase = .idle
    /// Prompt tokens evaluated so far, as a 0...1 fraction; nil outside prefill.
    @Published private(set) var prefillProgress: Double?
    /// The part being rewritten; nil for a draft that fits in one part.
    @Published private(set) var partProgress: PartProgress?
    @Published private(set) var errorMessage: String?
    @Published private(set) var engineLoaded = false

    struct PartProgress: Equatable {
        let current: Int
        let total: Int
    }

    /// The installed model's path, or nil when it is not installed. Set by
    /// the owner so this type never imports `AppModel`.
    var installedPath: () -> String? = { nil }
    /// Starts the catalog download of the model.
    var installHandler: () -> Void = {}
    /// The resource policy to open under (power profile, load guard). Only
    /// quality-neutral settings belong here: the chat session's steering,
    /// KV quantization, speculation and rate cap would all move a model
    /// whose quality gates were frozen at the defaults.
    var baseOptions: () -> OpenOptions = { OpenOptions() }
    /// Opens the engine at a path. Replaced in tests.
    lazy var openEngine: (String) async throws -> HumanizeEngine = { [weak self] path in
        // Read at open time, so the owner's setup order does not matter.
        var options = self?.baseOptions() ?? OpenOptions()
        options.maxContext = .fixed(HumanizeController.contextTokens)
        return try await TurboSparkSession(modelPath: path, options: options)
    }

    init(defaults: UserDefaults = .standard) {
        rulesStore = HumanizeRulesStore(defaults: defaults)
        rules = rulesStore.load()
    }

    private var engine: HumanizeEngine?
    private var enginePath: String?
    private var operationPath: String?
    private var task: Task<Void, Never>?

    var isInstalled: Bool { installedPath() != nil }
    var isBusy: Bool { phase != .idle }
    var hasDraft: Bool { !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    /// The plan for the draft as it is now; nil while a count is pending.
    var currentPlan: HumanizePlan? {
        guard countedDraft == draft, countedPath == installedPath() else { return nil }
        return plan
    }
    /// Fit is judged per part, not on the whole draft: a long draft is
    /// rewritten in parts, so only a single passage too long for the window
    /// is refused.
    var canRun: Bool { isInstalled && hasDraft && !isBusy && currentPlan?.fits != false }

    func usesModel(at path: String) -> Bool {
        enginePath == path || operationPath == path
    }

    func refreshContext() async {
        let text = draft
        guard let path = installedPath(), hasDraft else {
            plan = nil
            contextError = nil
            return
        }
        do {
            let counter = countDraftTokens
            let plan = try await HumanizePlanner.plan(text) { try await counter($0, path) }
            try Task.checkCancellation()
            guard draft == text, installedPath() == path else { return }
            countedDraft = text
            countedPath = path
            self.plan = plan
            contextError = nil
        } catch is CancellationError {
        } catch {
            guard draft == text, installedPath() == path else { return }
            plan = nil
            contextError = error.localizedDescription
        }
    }

    /// An explicit load lets the user prepare the model before writing a draft.
    func load() {
        guard isInstalled, !isBusy, !engineLoaded else { return }
        phase = .loadingModel
        operationPath = installedPath()
        task = Task { [weak self] in
            guard let self else { return }
            defer { finishOperation() }
            errorMessage = nil
            do { _ = try await ensureEngine() }
            catch is CancellationError { }
            catch { errorMessage = error.localizedDescription }
        }
    }

    private func finishOperation() {
        phase = .idle
        prefillProgress = nil
        partProgress = nil
        operationPath = nil
        task = nil
    }
    /// The rewrite ended on the length budget rather than the model's own
    /// end, so its last sentence may be cut off.
    var wasTruncated: Bool { stopReason == .maxTokens }

    /// Rewrites the current draft. A repeat call is a new sample: the recipe
    /// draws at temperature 1.0 and no seed is sent.
    func run() {
        guard canRun else { return }
        // Synchronous, before the task is scheduled: `canRun` reads this, so
        // a second click in the gap would otherwise start a second rewrite
        // against the one engine.
        phase = engine == nil ? .loadingModel : .running
        operationPath = installedPath()
        let source = draft, rules = rules
        task = Task { [weak self] in await self?.perform(draft: source, rules: rules) }
    }

    /// Stops the rewrite in flight. The partial text stays visible.
    func cancel() {
        task?.cancel()
        engine?.cancel()
    }

    /// Cancels any rewrite and releases the model's memory.
    func unload() {
        cancel()
        engine = nil
        enginePath = nil
        engineLoaded = false
    }

    private func perform(draft: String, rules: HumanizeRules) async {
        errorMessage = nil
        rewrite = ""
        sourceDraft = draft
        segments = []
        checks = nil
        ruleFindings = nil
        stats = nil
        stopReason = nil
        prefillProgress = nil
        defer { finishOperation() }
        do {
            guard let path = operationPath else { throw HumanizeError.notInstalled }
            let counter = countDraftTokens
            let plan = try await HumanizePlanner.plan(draft) { try await counter($0, path) }
            try Task.checkCancellation()
            self.plan = plan
            countedDraft = draft
            countedPath = path
            // Refused before mapping 12 GB of weights, and never truncated.
            guard plan.fits else { throw HumanizeError.passageTooLong }
            let engine = try await ensureEngine()
            try Task.checkCancellation()
            phase = .running
            try await rewrite(plan.parts, of: draft, rules: rules, with: engine)
        } catch is CancellationError {
            // A stopped rewrite is partial, so its audit would report numbers
            // "missing" that were never reached.
            stopReason = .cancelled
            checks = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Rewrites each part in order. Finished parts stay on screen when a later
    /// one fails or is stopped.
    private func rewrite(
        _ parts: [HumanizePart], of draft: String, rules: HumanizeRules, with engine: HumanizeEngine
    ) async throws {
        let source = draft as NSString
        var output = ""
        var combined = HumanizeChecks()
        var audited = true
        var totals = HumanizeStats()
        var truncated = false
        var finished = true
        // One seed for every part: repeatable output means the same draft
        // gives the same rewrite, part for part.
        let options = HumanizeOptions(seed: rules.fixedSeed ? rules.seed : nil)
        for (index, part) in parts.enumerated() {
            if index > 0 {
                // The draft's own break between two parts, kept as written.
                let previous = NSMaxRange(parts[index - 1].range)
                output += source.substring(with: NSRange(location: previous, length: part.range.location - previous))
            }
            partProgress = parts.count > 1 ? PartProgress(current: index + 1, total: parts.count) : nil
            prefillProgress = nil
            let prefix = output
            var streamed = ""
            var result: GenerationResult?
            // Parts are sized so the engine's automatic budget (2.5x, capped
            // at 2,048) always covers them; no per-call limit is sent.
            for try await event in engine.humanize(source.substring(with: part.range), options: options) {
                try Task.checkCancellation()
                switch event {
                case .prefill(let done, let total):
                    prefillProgress = total > 0 ? Double(done) / Double(total) : nil
                case .content(let delta):
                    prefillProgress = nil
                    streamed += delta
                    rewrite = prefix + streamed
                case .finished(let finished):
                    result = finished
                default:
                    break
                }
            }
            // Cancelling the consuming task ends the stream quietly rather
            // than throwing, so the loop above can exit with no `.finished`.
            if Task.isCancelled || result?.stopReason == .cancelled { throw CancellationError() }
            guard let result else {
                audited = false
                finished = false
                break
            }
            // The result's content is authoritative; the stream is only its
            // progressive view. Replacements apply before the segment is
            // recorded, so highlights line up with the text the user copies.
            let text = rules.applyReplacements(
                to: result.content.trimmingCharacters(in: .whitespacesAndNewlines))
            let start = (output as NSString).length
            output += text
            rewrite = output
            segments.append(HumanizeSegment(
                source: part.range, rewrite: NSRange(location: start, length: (text as NSString).length)))
            if let audit = result.audit { combined.add(audit) } else { audited = false }
            totals.add(result)
            stats = totals
            truncated = truncated || result.stopReason == .maxTokens
            stopReason = truncated ? .maxTokens : result.stopReason
        }
        checks = audited ? combined : nil
        if finished { ruleFindings = rules.check(draft: draft, rewrite: output) }
    }

    private func ensureEngine() async throws -> HumanizeEngine {
        try Task.checkCancellation()
        guard let path = operationPath, installedPath() == path else {
            throw HumanizeError.notInstalled
        }
        if let engine, enginePath == path { return engine }
        phase = .loadingModel
        let opened = try await openEngine(path)
        // Native opening can finish after Stop or Unload. Never republish it.
        try Task.checkCancellation()
        guard installedPath() == path else { throw HumanizeError.notInstalled }
        engine = opened
        enginePath = path
        engineLoaded = true
        return opened
    }
}

enum HumanizeError: LocalizedError {
    case notInstalled
    case passageTooLong

    var errorDescription: String? {
        switch self {
        case .passageTooLong:
            return String(localized: "One passage is too long for the model. Add paragraph breaks and try again.", bundle: .module)
        case .notInstalled:
            return String(localized: "The Humanizer model is not installed.", bundle: .module)
        }
    }
}

/// Every part's audit read as one. Each part is checked against its own
/// draft text, so joining the findings is exact.
struct HumanizeChecks: Equatable, Sendable {
    var droppedNumbers: [String] = []
    var inventedNumbers: [String] = []
    var languageDrift = false
    var reuseFlagged = false

    var isClean: Bool {
        droppedNumbers.isEmpty && inventedNumbers.isEmpty && !languageDrift && !reuseFlagged
    }

    mutating func add(_ audit: HumanizeAudit) {
        droppedNumbers += audit.droppedNumbers
        inventedNumbers += audit.inventedNumbers
        languageDrift = languageDrift || audit.languageDrift
        reuseFlagged = reuseFlagged || audit.reuseFlagged
    }
}

/// Totals across a rewrite's parts.
struct HumanizeStats: Equatable, Sendable {
    var tokens = 0
    var decodeSeconds = 0.0
    var seconds = 0.0

    var tokensPerSecond: Double? { decodeSeconds > 0 ? Double(tokens) / decodeSeconds : nil }

    mutating func add(_ result: GenerationResult) {
        tokens += result.newTokens
        decodeSeconds += result.decodeSeconds
        seconds += result.prefillSeconds + result.decodeSeconds
    }
}
