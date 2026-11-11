import XCTest

@testable import TurboSpark

/// The shapes `ts_hub_search_json` and `ts_hub_check_json` produce. The
/// fixtures carry exactly the keys crates/ffi/src/models/hub.rs emits, and
/// that file asserts the same key set, so a renamed field fails one side's
/// tests rather than decoding to nothing in the app.
final class HubSearchWireTests: XCTestCase {
    private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    func testSearchPageDecodesRowsAndTheirAdvisoryMetadata() throws {
        let page = try decode(
            HubSearchPage.self,
            #"""
            {"status":"ok","message":null,"retryAfterSecs":null,
             "fetchedAt":1700000000,"fromCache":true,"rejected":0,
             "results":[
              {"repoId":"o/gguf","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
               "downloads":2444691,"likes":1970,"hasGguf":true,"hasMlx":false,
               "gated":false,"baseModel":"Qwen/Qwen3.8-27B","architecture":"qwen35",
               "archHint":"recognized","archNeeds":null,"bundledAliases":["qwen3.8:27b"]},
              {"repoId":"o/mlx","revision":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
               "downloads":null,"likes":null,"hasGguf":false,"hasMlx":true,
               "gated":true,"baseModel":null,"architecture":"brand_new",
               "archHint":"unknown","archNeeds":null,"bundledAliases":[]}]}
            """#)
        XCTAssertEqual(page.status, .ok)
        XCTAssertEqual(page.fromCache, true)
        XCTAssertEqual(page.results.map(\.id), ["o/gguf", "o/mlx"])
        let gguf = page.results[0]
        XCTAssertTrue(gguf.hasGguf && !gguf.hasMlx && !gguf.gated)
        XCTAssertEqual(gguf.baseModel, "Qwen/Qwen3.8-27B")
        XCTAssertEqual(gguf.archHint, .recognized)
        XCTAssertEqual(gguf.bundledAliases, ["qwen3.8:27b"])
        // Missing counts stay nil rather than becoming 0: a 0-download row
        // reads as unpopular, which is not what "the Hub did not say" means.
        XCTAssertNil(page.results[1].downloads)
        XCTAssertTrue(page.results[1].gated)
        XCTAssertEqual(page.results[1].archHint, .unknown)
    }

    func testAFailedSearchIsAStatusWithNoRowsAndNoSuccessOnlyFields() throws {
        let page = try decode(
            HubSearchPage.self,
            #"""
            {"status":"rateLimited","message":"hub rate limit reached; retry after 30s",
             "retryAfterSecs":30,"results":[]}
            """#)
        XCTAssertEqual(page.status, .rateLimited)
        XCTAssertEqual(page.retryAfterSecs, 30)
        XCTAssertNil(page.fetchedAt)
        XCTAssertNil(page.fromCache)
        XCTAssertTrue(page.results.isEmpty)
    }

    func testUnknownStatusAndHintSpellingsDegradeInsteadOfFailing() throws {
        let page = try decode(
            HubSearchPage.self,
            #"""
            {"status":"teapot","message":"newer library","retryAfterSecs":null,
             "results":[{"repoId":"o/x","revision":"cccccccccccccccccccccccccccccccccccccccc",
              "downloads":1,"likes":1,"hasGguf":true,"hasMlx":false,"gated":false,
              "baseModel":null,"architecture":null,"archHint":"maybe","archNeeds":null,
              "bundledAliases":[]}]}
            """#)
        XCTAssertEqual(page.status, .error, "an unknown status is a generic error")
        XCTAssertEqual(page.results[0].archHint, .unknown, "an unknown hint claims nothing")
        XCTAssertEqual(try decode(HubCheckState.self, "\"surprise\""), .unavailable)
    }

    func testCheckDecodesARunnableGgufWithItsSidecarAndVariants() throws {
        let check = try decode(
            HubCheck.self,
            #"""
            {"repo":"o/gguf","revision":"main","state":"runnable","reason":null,
             "kind":"gguf","file":"m-Q8_0.gguf","architecture":"qwen35","family":"qwen_gdn_dense",
             "downloadBytes":28000000000,"trainedContext":262144,"chatTemplate":"tokenizer_config.json:chat_template",
             "warnings":[],"sidecarsMissing":["tokenizer.json"],"sidecarRepo":"Qwen/Qwen3.8-27B",
             "variants":[
              {"file":"m-Q8_0.gguf","bytes":28000000000,"quantLabel":"Q8_0","ladderRank":0,"executable":true},
              {"file":"m-weird.gguf","bytes":null,"quantLabel":null,"ladderRank":null,"executable":false}],
             "shardedSkipped":0,"fit":null}
            """#)
        XCTAssertEqual(check.state, .runnable)
        XCTAssertEqual(check.sidecarRepo, "Qwen/Qwen3.8-27B")
        XCTAssertEqual(check.variants.count, 2)
        XCTAssertFalse(check.variants[1].executable)
        XCTAssertNil(check.variants[1].bytes, "unknown size is nil, never 0")
        XCTAssertNil(check.fit, "no shape read is nil, never a zeroed fit")
    }

    func testCheckStatesThatAreNotVerdictsCarryAReasonAndNoReport() throws {
        for state in ["needsAuth", "unavailable"] {
            let check = try decode(
                HubCheck.self,
                """
                {"repo":"o/m","revision":"main","state":"\(state)","reason":"GET x: HTTP 401",
                 "kind":null,"file":null,"architecture":null,"family":null,
                 "downloadBytes":null,"trainedContext":null,"chatTemplate":null,
                 "warnings":[],"sidecarsMissing":[],"sidecarRepo":null,
                 "variants":[],"shardedSkipped":0,"fit":null}
                """)
            XCTAssertEqual(check.state.rawValue, state)
            XCTAssertNotEqual(check.state, .refused, "\(state) is not a refusal")
            XCTAssertNil(check.kind)
            XCTAssertNotNil(check.reason)
        }
    }

    func testAMalformedCallThrowsInsteadOfReturningAnEmptyPage() {
        // No network: the ABI refuses a one-letter query before any request.
        XCTAssertThrowsError(try TurboSparkCatalog.hubSearch(query: "a")) { error in
            XCTAssertTrue("\(error)".contains("at least 2"), "\(error)")
        }
        XCTAssertThrowsError(try TurboSparkCatalog.hubCheck(repo: "nameonly")) { error in
            XCTAssertTrue("\(error)".contains("owner/name"), "\(error)")
        }
    }
}
