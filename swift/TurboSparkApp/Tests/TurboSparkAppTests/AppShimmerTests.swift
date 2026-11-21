import XCTest
@testable import TurboSparkApp

/// The shimmer animates only when asked to, with motion allowed, in the key
/// window; every other combination draws static.
final class AppShimmerTests: XCTestCase {
    func testAnimatesOnlyWhenActiveMotionAllowedAndKey() {
        XCTAssertTrue(AppShimmerModifier.animates(active: true, reduceMotion: false, windowIsKey: true))
        XCTAssertFalse(AppShimmerModifier.animates(active: false, reduceMotion: false, windowIsKey: true))
        XCTAssertFalse(AppShimmerModifier.animates(active: true, reduceMotion: true, windowIsKey: true))
        XCTAssertFalse(AppShimmerModifier.animates(active: true, reduceMotion: false, windowIsKey: false))
    }
}
