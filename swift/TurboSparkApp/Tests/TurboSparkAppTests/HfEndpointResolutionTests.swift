import XCTest

@testable import TurboSparkApp

/// `swift/docs/SWIFT_SETTINGS_AUDIT.md`'s HF endpoint item: the Server Advanced
/// field and `HfAuthTokenCardView`'s own editor each inlined their own
/// trim-and-compare for "what counts as the default endpoint," and the
/// Server Advanced one never called `TurboSparkCatalog.setHfEndpoint` at
/// all, so editing there left the catalog's install/probe/browse client on
/// the stale endpoint. `HfEndpointResolution.effectiveEndpoint(from:)` is
/// the one place both readers ask now; this test is what stops a future
/// edit to one from quietly disagreeing with the other again.
final class HfEndpointResolutionTests: XCTestCase {
    func testTheDefaultEndpointResolvesToNil() {
        XCTAssertNil(HfEndpointResolution.effectiveEndpoint(from: "https://huggingface.co"))
    }

    func testAnEmptyOrWhitespaceInputResolvesToNil() {
        XCTAssertNil(HfEndpointResolution.effectiveEndpoint(from: ""))
        XCTAssertNil(HfEndpointResolution.effectiveEndpoint(from: "   "))
    }

    func testAMirrorURLIsReturnedTrimmed() {
        XCTAssertEqual(
            HfEndpointResolution.effectiveEndpoint(from: "  https://hf-mirror.com  "),
            "https://hf-mirror.com")
    }

    func testTheDefaultConstantMatchesWhatTheEnginesConsiderDefault() {
        XCTAssertNil(HfEndpointResolution.effectiveEndpoint(from: HfEndpointResolution.defaultEndpoint))
    }
}
