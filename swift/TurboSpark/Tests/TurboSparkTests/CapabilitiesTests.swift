import XCTest

@testable import TurboSpark

/// These run through the real archive, so they also pin the header: a wrong
/// signature for either call shows up as a link error or a wrong answer.
final class CapabilitiesTests: XCTestCase {

    func testSteeringFollowsTheEnginesFamilyTable() {
        for family in ["gemma4", "llama", "qwen3moe", "qwen3", "qwen2", "gptOss", "museGlimmer"] {
            XCTAssertTrue(
                TurboSparkCapabilities.family(family).steeringSupported, "\(family) dispatches steering")
        }
        // The two families the app's old copy documented as deliberately absent.
        for family in ["deepseekV4Flash", "qwen4exp"] {
            let c = TurboSparkCapabilities.family(family)
            XCTAssertFalse(c.steeringSupported, family)
        }
    }

    func testAnUnknownFamilyIsUnknownAndUnsupported() {
        let c = TurboSparkCapabilities.family("definitely-not-a-family")
        XCTAssertFalse(c.known)
        XCTAssertFalse(c.steeringSupported)
        XCTAssertEqual(c.family, "definitely-not-a-family")
    }

    func testKvQuantMatchesTheEnginesRule() {
        XCTAssertTrue(TurboSparkCapabilities.kvQuantSupported(
            fullHeadDim: 128, layerMask: [1, 0, 1, 1], numLayers: 4))
        XCTAssertFalse(TurboSparkCapabilities.kvQuantSupported(
            fullHeadDim: 96, layerMask: [1, 1, 1, 1], numLayers: 4), "not a power of two")
        XCTAssertFalse(TurboSparkCapabilities.kvQuantSupported(
            fullHeadDim: 128, layerMask: [0, 0, 0, 1], numLayers: 4), "last full layer excluded")
        XCTAssertTrue(TurboSparkCapabilities.kvQuantSupported(
            fullHeadDim: 128, layerMask: [1, 1], numLayers: 2), "a two-layer stack counts every layer")
    }

    func testUnreadableFactsAnswerNo() {
        XCTAssertFalse(TurboSparkCapabilities.kvQuantSupported(fullHeadDim: 128, layerMask: [], numLayers: 4))
        XCTAssertFalse(TurboSparkCapabilities.kvQuantSupported(fullHeadDim: 128, layerMask: [1, 1], numLayers: 0))
        XCTAssertFalse(TurboSparkCapabilities.kvQuantSupported(fullHeadDim: 128, layerMask: [1, 300], numLayers: 2))
        XCTAssertFalse(TurboSparkCapabilities.kvQuantSupported(fullHeadDim: 128, layerMask: [-1, 1], numLayers: 2))
    }
}
