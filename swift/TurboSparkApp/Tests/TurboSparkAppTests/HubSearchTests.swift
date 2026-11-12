import XCTest
import TurboSpark

@testable import TurboSparkApp

/// Alias suggestion, the install gate for a Hugging Face check, and the search
/// controller's concurrency rules. The controller takes its two network calls
/// as closures, so none of this touches the network.
@MainActor
final class HubSearchTests: XCTestCase {

    // MARK: - Fixtures

    private func page(_ json: String) throws -> HubSearchPage {
        try JSONDecoder().decode(HubSearchPage.self, from: Data(json.utf8))
    }

    private func okPage(repos: [String]) throws -> HubSearchPage {
        let rows = repos.map { repo in
            """
            {"repoId":"\(repo)","revision":"\(String(repeating: "a", count: 40))","downloads":1,"likes":1,
             "hasGguf":true,"hasMlx":false,"gated":false,"baseModel":null,"architecture":"qwen3",
             "archHint":"recognized","archNeeds":null,"bundledAliases":[]}
            """
        }.joined(separator: ",")
        return try page(
            #"{"status":"ok","message":null,"retryAfterSecs":null,"fetchedAt":1,"fromCache":false,"rejected":0,"results":["# + rows + "]}")
    }

    private func check(
        state: String, reason: String? = nil, kind: String? = "gguf", file: String? = "m-Q4_K_M.gguf",
        variants: String = "[]", fit: String = "null", sidecar: String = "null"
    ) throws -> HubCheck {
        let reasonJSON = reason.map { "\"\($0)\"" } ?? "null"
        let kindJSON = kind.map { "\"\($0)\"" } ?? "null"
        let fileJSON = file.map { "\"\($0)\"" } ?? "null"
        let json = """
            {"repo":"o/m","revision":"main","state":"\(state)","reason":\(reasonJSON),
             "kind":\(kindJSON),"file":\(fileJSON),"architecture":"qwen3","family":null,
             "downloadBytes":4000000000,"trainedContext":null,"chatTemplate":null,
             "warnings":[],"sidecarsMissing":[],"sidecarRepo":\(sidecar),
             "variants":\(variants),"shardedSkipped":0,"fit":\(fit)}
            """
        return try JSONDecoder().decode(HubCheck.self, from: Data(json.utf8))
    }

    private func variant(_ file: String, executable: Bool, bytes: UInt64? = 4_000_000_000) throws -> RepoVariant {
        let bytesJSON = bytes.map(String.init) ?? "null"
        let json = """
            {"file":"\(file)","bytes":\(bytesJSON),"quantLabel":"Q4_K_M","ladderRank":1,"executable":\(executable)}
            """
        return try JSONDecoder().decode(RepoVariant.self, from: Data(json.utf8))
    }

    /// A destination with room to spare, so the disk rules are not the thing
    /// under test.
    private var roomy: InstallGateDestination {
        InstallGateDestination(libraryName: nil, notConnectedName: nil, freeBytes: 500_000_000_000)
    }

    // MARK: - ModelAlias

    func testAliasDropsTheOwnerAndTheRedundantGgufSuffix() {
        XCTAssertEqual(
            ModelAlias.suggest(repo: "unsloth/Qwen3-4B-GGUF", taken: []), "qwen3-4b")
        XCTAssertEqual(
            ModelAlias.suggest(repo: "Qwen/Qwen3-8B-GGUF@main", taken: []), "qwen3-8b")
        XCTAssertEqual(
            ModelAlias.suggest(repo: "mlx-community/Qwen3-4B-Instruct-2507-4bit", taken: []),
            "qwen3-4b-instruct-2507-4bit")
    }

    func testAliasAppendsTheQuantizationOnceAndOnlyWhenItAddsInformation() {
        XCTAssertEqual(
            ModelAlias.suggest(repo: "o/Model-GGUF", quantLabel: "Q4_K_M", taken: []),
            "model-q4_k_m")
        XCTAssertEqual(
            ModelAlias.suggest(repo: "o/Model-4bit", quantLabel: "4bit", taken: []),
            "model-4bit", "a tag the name already carries is not repeated")
    }

    func testAliasNumbersACollisionIgnoringCase() {
        XCTAssertEqual(
            ModelAlias.suggest(repo: "o/Model", taken: ["MODEL"]), "model-2")
        XCTAssertEqual(
            ModelAlias.suggest(repo: "o/Model", taken: ["model", "model-2", "Model-3"]), "model-4")
        XCTAssertTrue(ModelAlias.isTaken("Model", in: ["model"]))
        XCTAssertFalse(ModelAlias.isTaken("model-x", in: ["model"]))
    }

    func testEverySuggestedAliasIsSafeToInstallUnder() {
        // The install refuses anything but one plain path component, so the
        // suggestion must never produce what it would refuse, whatever the
        // repository is called.
        let hostile = [
            "", "/", "a/..", "owner/..", "../../etc/passwd", "o/.hidden", "o/-", "o/???",
            "o/Qwen 3 \u{1F680}", "o/\u{0645}\u{0648}\u{062F}\u{064A}\u{0644}", String(repeating: "x", count: 500),
            "o/a@../b", "o/.", "o/..",
        ]
        for repo in hostile {
            for quant in [nil, "Q4_K_M", "/../", ""] {
                let alias = ModelAlias.suggest(repo: repo, quantLabel: quant, taken: [])
                XCTAssertTrue(AppModel.isSafeInstallAlias(alias), "\(repo) + \(String(describing: quant)) -> \(alias)")
                XCTAssertLessThanOrEqual(alias.count, ModelAlias.maxLength + 4)
            }
        }
        XCTAssertEqual(ModelAlias.suggest(repo: "", taken: []), "model")
    }

    // MARK: - exactRepo

    func testExactRepoRecognizesRepositoriesAndUrlsButNotSearchPhrases() {
        XCTAssertEqual(HubSearchController.exactRepo(from: "Qwen/Qwen3-8B-GGUF"), "Qwen/Qwen3-8B-GGUF")
        XCTAssertEqual(HubSearchController.exactRepo(from: "o/m@abc123"), "o/m@abc123")
        XCTAssertEqual(
            HubSearchController.exactRepo(from: "https://huggingface.co/unsloth/Qwen3-4B-GGUF"),
            "unsloth/Qwen3-4B-GGUF")
        XCTAssertEqual(
            HubSearchController.exactRepo(from: "https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main"),
            "unsloth/Qwen3-4B-GGUF", "a pasted page URL names the repository in its first two parts")
        for phrase in ["qwen", "qwen 3 coder", "a/b/c", "/x", "owner/", "o/m extra", "../x"] {
            XCTAssertNil(HubSearchController.exactRepo(from: phrase), phrase)
        }
    }

    // MARK: - Install gate

    func testOnlyARunnableCheckCanBeInstalled() throws {
        for state in ["refused", "needsAuth", "unavailable"] {
            let decision = HubInstallGating.decision(
                check: try check(state: state, reason: "why"), selectedVariant: nil, destination: roomy)
            XCTAssertTrue(decision.isBlocked, state)
            XCTAssertNotNil(decision.reason, state)
        }
        let ok = HubInstallGating.decision(
            check: try check(state: "runnable"), selectedVariant: nil, destination: roomy)
        XCTAssertFalse(ok.isBlocked)
    }

    func testARefusalCarriesTheProbesOwnWords() throws {
        let decision = HubInstallGating.decision(
            check: try check(state: "refused", reason: "config.json declares model_type \\\"zz\\\""),
            selectedVariant: nil, destination: roomy)
        XCTAssertTrue(decision.reason?.contains("zz") ?? false, "\(String(describing: decision.reason))")
    }

    func testAQuantizationWithNoKernelsIsRefusedBeforeTheDownload() throws {
        let bad = try variant("m-weird.gguf", executable: false)
        let decision = HubInstallGating.decision(
            check: try check(state: "runnable"), selectedVariant: bad, destination: roomy)
        XCTAssertTrue(decision.isBlocked)
        XCTAssertFalse(
            HubInstallGating.decision(
                check: try check(state: "runnable"), selectedVariant: try variant("m.gguf", executable: true),
                destination: roomy
            ).isBlocked)
    }

    func testTheDiskRuleStillAppliesToARunnableCheck() throws {
        let tight = InstallGateDestination(libraryName: nil, notConnectedName: nil, freeBytes: 1_000)
        let decision = HubInstallGating.decision(
            check: try check(state: "runnable"), selectedVariant: nil, destination: tight)
        XCTAssertNotEqual(decision, .allowed, "4 GB into 1 KB of space must not be allowed")
    }

    // MARK: - Controller: search

    /// Releases a blocked call from the test, from any thread.
    private final class Gate: @unchecked Sendable {
        private let semaphore = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var calls: [String] = []
        func wait() { semaphore.wait() }
        func open() { semaphore.signal() }
        func record(_ call: String) { lock.lock(); calls.append(call); lock.unlock() }
        var recorded: [String] { lock.lock(); defer { lock.unlock() }; return calls }
    }

    func testASlowEarlierSearchCannotOverwriteANewerOne() async throws {
        let gate = Gate()
        let slow = try okPage(repos: ["o/slow"])
        let fast = try okPage(repos: ["o/fast"])
        let controller = HubSearchController(
            searcher: { query, _, _, _ in
                gate.record(query)
                if query == "slow query" {
                    gate.wait()
                    return slow
                }
                return fast
            },
            checker: { _ in throw TurboSparkError(code: .json, message: "unused") })

        let first = Task { await controller.search(query: "slow query") }
        // Let the first search reach its blocked call before the second starts.
        while gate.recorded.isEmpty { await Task.yield() }
        await controller.search(query: "fast query")
        XCTAssertEqual(controller.rows.map(\.repoId), ["o/fast"])

        gate.open()
        await first.value
        XCTAssertEqual(
            controller.rows.map(\.repoId), ["o/fast"],
            "the slower, earlier search finished last and must not win")
        XCTAssertEqual(controller.searchedQuery, "fast query")
    }

    func testAResetDiscardsASearchStillInFlight() async throws {
        let gate = Gate()
        let page = try okPage(repos: ["o/late"])
        let controller = HubSearchController(
            searcher: { query, _, _, _ in
                gate.record(query)
                gate.wait()
                return page
            },
            checker: { _ in throw TurboSparkError(code: .json, message: "unused") })
        let running = Task { await controller.search(query: "qwen") }
        while gate.recorded.isEmpty { await Task.yield() }
        controller.reset()
        gate.open()
        await running.value
        XCTAssertEqual(controller.phase, .idle, "typing a new query must drop the old answer")
        XCTAssertTrue(controller.rows.isEmpty)
    }

    func testAShortQueryNeverReachesTheNetwork() async {
        let gate = Gate()
        let controller = HubSearchController(
            searcher: { query, _, _, _ in
                gate.record(query)
                throw TurboSparkError(code: .json, message: "should not be called")
            },
            checker: { _ in throw TurboSparkError(code: .json, message: "unused") })
        for query in ["", " ", "a", "  a "] { await controller.search(query: query) }
        XCTAssertTrue(gate.recorded.isEmpty)
        XCTAssertEqual(controller.phase, .idle)
    }

    func testFailureStatusesBecomeAFailedPhaseWithTheirStatus() async throws {
        let limited = try page(
            #"{"status":"rateLimited","message":"slow down","retryAfterSecs":30,"results":[]}"#)
        let controller = HubSearchController(
            searcher: { _, _, _, _ in limited },
            checker: { _ in throw TurboSparkError(code: .json, message: "unused") })
        await controller.search(query: "qwen")
        XCTAssertEqual(
            controller.phase,
            .failed(status: .rateLimited, message: "slow down", retryAfterSecs: 30))
    }

    func testAThrownErrorIsAFailedPhaseNotAnEmptyResultList() async {
        let controller = HubSearchController(
            searcher: { _, _, _, _ in throw TurboSparkError(code: .json, message: "bad call") },
            checker: { _ in throw TurboSparkError(code: .json, message: "unused") })
        await controller.search(query: "qwen")
        guard case .failed(let status, _, _) = controller.phase else {
            return XCTFail("a malformed call must not read as 'no models found': \(controller.phase)")
        }
        XCTAssertEqual(status, .error)
    }

    func testTheFormatAndSortAreSentWithTheQuery() async throws {
        final class Seen: @unchecked Sendable {
            let lock = NSLock()
            var value: (HubSearchSort, HubSearchFormat, Bool)?
        }
        let seen = Seen()
        let empty = try okPage(repos: [])
        let controller = HubSearchController(
            searcher: { _, sort, format, refresh in
                seen.lock.lock(); seen.value = (sort, format, refresh); seen.lock.unlock()
                return empty
            },
            checker: { _ in throw TurboSparkError(code: .json, message: "unused") })
        controller.format = .mlx
        controller.sort = .likes
        await controller.search(query: "qwen", refresh: true)
        seen.lock.lock(); defer { seen.lock.unlock() }
        XCTAssertEqual(seen.value?.0, .likes)
        XCTAssertEqual(seen.value?.1, .mlx)
        XCTAssertEqual(seen.value?.2, true)
    }

    // MARK: - Controller: check

    private func request(file: String? = nil) -> HubSearchController.HubCheckRequest {
        .init(repo: "o/m", sidecarHint: "base/m", file: file, context: 4096, slots: nil, loadGuard: nil)
    }

    func testACheckIsHeldPerFileAndNotRepeated() async throws {
        let gate = Gate()
        let runnable = try check(state: "runnable")
        let none = try okPage(repos: [])
        let controller = HubSearchController(
            searcher: { _, _, _, _ in none },
            checker: { request in
                gate.record(request.file ?? "default")
                return runnable
            })
        await controller.check(request(), configuration: "c1")
        await controller.check(request(), configuration: "c1")
        await controller.check(request(file: "m-Q2_K.gguf"), configuration: "c1")
        XCTAssertEqual(gate.recorded, ["default", "m-Q2_K.gguf"],
                       "the same file is asked once; another quantization is its own question")
        // A different fit configuration changes the answer, so it re-asks.
        await controller.check(request(), configuration: "c2")
        XCTAssertEqual(gate.recorded.count, 3)
    }

    func testAFailedCheckIsRetriedAndASuccessfulOneIsNot() async throws {
        let gate = Gate()
        let unavailable = try check(state: "unavailable", reason: "offline", kind: nil, file: nil)
        let runnable = try check(state: "runnable")
        let none = try okPage(repos: [])
        let controller = HubSearchController(
            searcher: { _, _, _, _ in none },
            checker: { _ in
                gate.record("call")
                return gate.recorded.count == 1 ? unavailable : runnable
            })
        await controller.check(request(), configuration: "c")
        let key = HubSearchController.CheckKey(repo: "o/m", file: nil, configuration: "c")
        XCTAssertEqual(controller.checkPhase(key), .done(unavailable))

        // "Check again" forces: a not-a-verdict answer must not stick forever.
        await controller.check(request(), configuration: "c", force: true)
        XCTAssertEqual(controller.checkPhase(key), .done(runnable))
        await controller.check(request(), configuration: "c")
        XCTAssertEqual(gate.recorded.count, 2, "a held verdict is not re-requested")
    }

    func testAThrownCheckIsAFailureNotAVerdict() async throws {
        let none = try okPage(repos: [])
        let controller = HubSearchController(
            searcher: { _, _, _, _ in none },
            checker: { _ in throw TurboSparkError(code: .json, message: "malformed") })
        await controller.check(request(), configuration: "c")
        let key = HubSearchController.CheckKey(repo: "o/m", file: nil, configuration: "c")
        guard case .failed = controller.checkPhase(key) else {
            return XCTFail("\(String(describing: controller.checkPhase(key)))")
        }
    }
}
