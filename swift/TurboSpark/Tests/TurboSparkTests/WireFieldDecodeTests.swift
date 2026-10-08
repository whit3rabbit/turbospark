import XCTest

@testable import TurboSpark

/// Fields the engine already sends that this binding used to drop. Fixtures
/// are hand-written in the shape crates/ffi/src/wire.rs and the catalog's own
/// serde derives produce (snake_case for catalog rows, camelCase elsewhere).
final class WireFieldDecodeTests: XCTestCase {

    func testToolCallsDecodeWithObjectAndStringArguments() throws {
        let json = #"""
        {"promptTokens":1,"newTokens":1,"prefillSeconds":0,"decodeSeconds":0,
         "stopReason":"toolCalls","tokensPerSecond":null,"content":"","reasoning":"",
         "toolCalls":[
           {"id":"c1","name":"read","arguments":{"path":"/a","n":2}},
           {"id":"c2","name":"echo","arguments":"{\"x\":1}"},
           {"id":"c3","name":"noargs"}]}
        """#
        let r = try JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
        XCTAssertEqual(r.stopReason, .toolCalls)
        XCTAssertEqual(r.toolCalls.map(\.name), ["read", "echo", "noargs"])
        XCTAssertEqual(r.toolCalls[0].argumentsJSON, #"{"n":2,"path":"/a"}"#)
        XCTAssertEqual(r.toolCalls[1].argumentsJSON, #"{"x":1}"#)
        XCTAssertEqual(r.toolCalls[2].argumentsJSON, "")
    }

    func testResultWithoutToolCallsKeyDecodesEmpty() throws {
        let json = #"""
        {"promptTokens":1,"newTokens":1,"prefillSeconds":0,"decodeSeconds":0,
         "stopReason":"eos","tokensPerSecond":null,"content":"x"}
        """#
        let r = try JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
        XCTAssertTrue(r.toolCalls.isEmpty)
    }

    func testCatalogEntryKeepsKindSourceGatesAndMeasurements() throws {
        let json = #"""
        {"alias":"a","name":"A","family":"gemma4","kind":"vision-tower",
         "include_vision":true,
         "source":{"kind":"gguf","repo":"o/r","revision":"abc","file":"m.gguf"},
         "download_bytes":10,"install_bytes":20,"status":"verified",
         "gates":["t1"],
         "measured":[{"chip":"Apple M4 Max","context":8192,"expert_cache_slots":16,
           "peak_footprint_mib":9000,"decode_tok_s_min":10.5,"decode_tok_s_max":12.5,
           "measured_on":"2026-01-01","source":"AC"}],
         "mtp":{"repo":"o/mtp","revision":"def"},
         "notes":null,"installed":false}
        """#
        let e = try JSONDecoder().decode(CatalogEntry.self, from: Data(json.utf8))
        XCTAssertTrue(e.isVisionTower)
        XCTAssertTrue(e.includeVision)
        XCTAssertEqual(e.source?.file, "m.gguf")
        XCTAssertEqual(e.gates, ["t1"])
        XCTAssertEqual(e.measured.first?.peakFootprintMib, 9000)
        XCTAssertEqual(e.mtp?.repo, "o/mtp")
    }

    func testCatalogEntryFromOlderEngineDefaultsToTrunkModel() throws {
        let json = #"""
        {"alias":"a","name":"A","family":"gemma4","download_bytes":1,"install_bytes":2,
         "status":"runs","installed":true}
        """#
        let e = try JSONDecoder().decode(CatalogEntry.self, from: Data(json.utf8))
        XCTAssertEqual(e.kind, "model")
        XCTAssertFalse(e.isVisionTower)
        XCTAssertTrue(e.gates.isEmpty && e.measured.isEmpty)
        XCTAssertNil(e.source)
    }

    func testInstalledModelCarriesStatusKindAndVariant() throws {
        let json = #"""
        {"alias":"a","repo":"o/r","revision":"main","path":"/p","family":"gemma4",
         "install_bytes":1,"installed_on":"2026-01-01",
         "status":"unlisted","kind":"vision-tower","variant":"Q4_K_M"}
        """#
        let m = try JSONDecoder().decode(InstalledModel.self, from: Data(json.utf8))
        XCTAssertEqual(m.status, "unlisted")
        XCTAssertEqual(m.kind, "vision-tower")
        XCTAssertEqual(m.variant, "Q4_K_M")
    }

    func testImageInstalledFamilyPrefersEngineValueThenFallsBack() throws {
        let base = #"""
        "alias":"i","modelID":"mlx-community/Qwen-Image-2.1-4bit","revision":"r",
        "path":"/p","width":1,"height":1,"schedulerSteps":9,"quantization":"4bit"
        """#
        let sent = try JSONDecoder().decode(
            ImageInstalledModel.self, from: Data("{\(base),\"family\":\"engine-says\"}".utf8))
        XCTAssertEqual(sent.family, "engine-says")
        let old = try JSONDecoder().decode(
            ImageInstalledModel.self, from: Data("{\(base)}".utf8))
        XCTAssertEqual(old.family, "qwen-image-2.1")
    }
}

final class DaemonStatusDecodeTests: XCTestCase {
    func testModelDecodesWhenPresentAndIsNilWhenNullOrAbsent() throws {
        let running = #"{"running":true,"pid":9,"port":8080,"endpoint":"http://127.0.0.1:8080/v1","logPath":"/l","model":"gemma4"}"#
        let a = try JSONDecoder().decode(TurboSparkDaemonStatus.self, from: Data(running.utf8))
        XCTAssertEqual(a.model, "gemma4")
        let nul = #"{"running":true,"pid":9,"port":1,"endpoint":"e","logPath":"l","model":null}"#
        XCTAssertNil(try JSONDecoder().decode(TurboSparkDaemonStatus.self, from: Data(nul.utf8)).model)
        let off = #"{"running":false}"#
        XCTAssertNil(try JSONDecoder().decode(TurboSparkDaemonStatus.self, from: Data(off.utf8)).model)
    }
}

final class RecommendationExtraFieldsTests: XCTestCase {
    func testEvidenceAndSuspiciousDecodeAndDefaultSafely() throws {
        let base = #"""
        "alias":"a","name":"A","family":null,"verdict":"resident","verdictSummary":"s","runs":true,
        "countedBytes":1,"countedSource":"measured","installBytes":2,"slotCacheSlots":16,
        "largestContext":8192,"notes":[],"toksPerSecondMin":null,"toksPerSecondMax":null,"throughput":null
        """#
        let withFields = try JSONDecoder().decode(
            ModelRecommendation.self,
            from: Data("{\(base),\"evidence\":\"verified\",\"suspicious\":true}".utf8))
        XCTAssertEqual(withFields.evidence, "verified")
        XCTAssertTrue(withFields.isSuspicious)
        let older = try JSONDecoder().decode(
            ModelRecommendation.self, from: Data("{\(base)}".utf8))
        XCTAssertNil(older.evidence)
        XCTAssertFalse(older.isSuspicious, "an engine that does not say is not accused")
    }
}
