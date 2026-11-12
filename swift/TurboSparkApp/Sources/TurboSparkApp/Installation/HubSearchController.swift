import Foundation
import SwiftUI
import TurboSpark

/// State for searching Hugging Face from Discover and checking one result.
///
/// Both FFI calls block on the network, so each runs on a detached task. The
/// two rules that make that safe are the ones `ModelProbeSheet` already keeps:
/// a generation counter so a slower earlier search cannot overwrite a newer
/// one, and results keyed by what they answer so a late check cannot land on a
/// different repository.
///
/// **Nothing here searches on its own.** A request goes out only when the
/// caller asks (Return in the search field, or the row that says so). Text
/// typed into a filter box is not a request to send it to a third party.
@MainActor
final class HubSearchController: ObservableObject {
    /// Shorter queries match most of the Hub and say nothing about intent.
    /// The ABI refuses them too.
    static let minimumQueryLength = 2

    enum Phase: Equatable {
        /// No search has run for the current query.
        case idle
        case searching
        /// The Hub answered. `rows` may be empty.
        case loaded(rows: [HubSearchResult], fromCache: Bool)
        /// The Hub did not answer: offline, rate limited, rejected or broken.
        case failed(status: HubSearchStatus, message: String, retryAfterSecs: UInt64?)
    }

    /// What `check` knows about one repository.
    enum CheckPhase: Equatable {
        case checking
        case done(HubCheck)
        /// The call itself threw (a malformed request), which is a bug to
        /// show rather than a verdict.
        case failed(String)
    }

    /// One check's identity. The file is part of it because the fit is per
    /// quantization, and the configuration because it changes the fit.
    struct CheckKey: Hashable {
        let repo: String
        let file: String?
        let configuration: String
    }

    typealias Searcher = @Sendable (String, HubSearchSort, HubSearchFormat, Bool) throws -> HubSearchPage
    typealias Checker = @Sendable (HubCheckRequest) throws -> HubCheck

    /// Everything a check needs, as one value so a test can assert on it.
    struct HubCheckRequest: Sendable {
        let repo: String
        let sidecarHint: String?
        let file: String?
        let context: UInt32
        let slots: OpenOptions.Sizing?
        let loadGuard: OpenOptions.LoadGuard?
    }

    /// `owner/name` (optionally `@revision`) when `query` is shaped like a
    /// repository, or like a Hugging Face URL for one, so it can be checked
    /// directly. `nil` for anything with a space or a different shape: a search
    /// phrase is not a repository, and offering to "check" it would send a
    /// guess to the Hub.
    static func exactRepo(from query: String) -> String? {
        var cleaned = ModelProbeGating.sanitizeRepo(query)
        // A pasted page URL (`.../owner/name/tree/main`) names the repository
        // in its first two path components.
        let lowered = query.lowercased()
        if lowered.contains("huggingface.co/") || lowered.contains("hf.co/") {
            cleaned = cleaned.split(separator: "/").prefix(2).joined(separator: "/")
        }
        let pattern = #"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(@[A-Za-z0-9._-]+)?$"#
        return cleaned.range(of: pattern, options: .regularExpression) != nil ? cleaned : nil
    }

    @Published private(set) var phase: Phase = .idle
    /// The query the current `phase` answers.
    @Published private(set) var searchedQuery = ""
    @Published var format: HubSearchFormat = .any
    @Published var sort: HubSearchSort = .downloads
    @Published private(set) var checks: [CheckKey: CheckPhase] = [:]

    private var searchGeneration = 0
    private let searcher: Searcher
    private let checker: Checker

    init(
        searcher: @escaping Searcher = { query, sort, format, refresh in
            try TurboSparkCatalog.hubSearch(
                query: query, sort: sort, format: format, refresh: refresh)
        },
        checker: @escaping Checker = { request in
            try TurboSparkCatalog.hubCheck(
                repo: request.repo,
                sidecarHint: request.sidecarHint,
                file: request.file,
                context: request.context,
                expertCacheSlots: request.slots,
                loadGuard: request.loadGuard)
        }
    ) {
        self.searcher = searcher
        self.checker = checker
    }

    /// The live rows of the current answer.
    var rows: [HubSearchResult] {
        if case .loaded(let rows, _) = phase { return rows }
        return []
    }

    func row(for repo: String) -> HubSearchResult? {
        rows.first { $0.repoId.caseInsensitiveCompare(repo) == .orderedSame }
    }

    /// Forgets the current answer, e.g. because the query changed. Bumps the
    /// generation so a search still in flight cannot land afterwards.
    func reset() {
        searchGeneration += 1
        phase = .idle
        searchedQuery = ""
    }

    /// Runs one search with the current `format` and `sort`. Returns when the
    /// answer has been applied or discarded.
    func search(query: String, refresh: Bool = false) async {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count >= Self.minimumQueryLength else {
            reset()
            return
        }
        searchGeneration += 1
        let generation = searchGeneration
        phase = .searching
        searchedQuery = text

        let (sort, format, searcher) = (self.sort, self.format, self.searcher)
        let outcome = await Task.detached(priority: .userInitiated) {
            Result { try searcher(text, sort, format, refresh) }
        }.value
        guard generation == searchGeneration else { return }

        switch outcome {
        case .success(let page):
            if page.status == .ok {
                phase = .loaded(rows: page.results, fromCache: page.fromCache ?? false)
                // No count in the sentence: a count needs plural forms in every
                // language, and the list itself is what VoiceOver reads next.
                announce(
                    page.results.isEmpty
                        ? AppStrings.localized("No text models found")
                        : AppStrings.localized("Hugging Face search complete."))
            } else {
                let message = page.message ?? AppStrings.localized("Hugging Face did not answer.")
                phase = .failed(
                    status: page.status, message: message, retryAfterSecs: page.retryAfterSecs)
                announce(message)
            }
        case .failure(let error):
            // A thrown error is a malformed call, not the Hub's answer.
            phase = .failed(status: .error, message: error.localizedDescription, retryAfterSecs: nil)
            announce(error.localizedDescription)
        }
    }

    func checkPhase(_ key: CheckKey) -> CheckPhase? { checks[key] }

    /// Checks one repository unless an answer is already held or in flight.
    /// `force` re-asks after a failure the user chose to retry.
    func check(
        _ request: HubCheckRequest, configuration: String, force: Bool = false
    ) async {
        let key = CheckKey(repo: request.repo, file: request.file, configuration: configuration)
        if !force, let existing = checks[key] {
            switch existing {
            case .checking, .done: return
            case .failed: break
            }
        }
        // A needsAuth or unavailable answer is a verdict about the REQUEST,
        // so it is retried on demand and never silently cached forever.
        checks[key] = .checking
        let checker = self.checker
        let outcome = await Task.detached(priority: .userInitiated) {
            Result { try checker(request) }
        }.value
        switch outcome {
        case .success(let result):
            checks[key] = .done(result)
            announce(Self.announcement(for: result))
        case .failure(let error):
            checks[key] = .failed(error.localizedDescription)
            announce(error.localizedDescription)
        }
    }

    /// Spoken result of a check, so a VoiceOver user hears the outcome of a
    /// multi-second wait.
    static func announcement(for check: HubCheck) -> String {
        switch check.state {
        case .runnable: return AppStrings.localized("Compatibility check complete. This model would run here.")
        case .refused: return AppStrings.localized("Compatibility check complete. This model would not run here.")
        case .needsAuth: return AppStrings.localized("This repository needs a Hugging Face sign-in.")
        case .unavailable: return AppStrings.localized("Could not check this repository.")
        }
    }

    private func announce(_ message: String) {
        _ = AccessibilityNotification.Announcement.post(.init(message))
    }
}
