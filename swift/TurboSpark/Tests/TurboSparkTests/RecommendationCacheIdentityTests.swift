import XCTest
@testable import TurboSpark

final class RecommendationCacheIdentityTests: XCTestCase {
    func testCatalogIdentityIgnoresInstallationAndObjectOrdering() throws {
        let first = Data(#"[{"alias":"a","revision":"123","installed":false},{"alias":"b","installed":true}]"#.utf8)
        let second = Data(#"[{"installed":false,"alias":"b"},{"revision":"123","installed":true,"alias":"a"}]"#.utf8)
        XCTAssertEqual(
            try TurboSparkCatalog.recommendationCatalogFingerprint(json: first),
            try TurboSparkCatalog.recommendationCatalogFingerprint(json: second))
    }

    func testCatalogIdentityIncludesFieldsOmittedByDisplayRows() throws {
        func fingerprint(revision: String, counted: Int) throws -> String {
            try TurboSparkCatalog.recommendationCatalogFingerprint(json: Data(
                """
                [{"alias":"a","revision":"\(revision)","measured":{"counted_bytes":\(counted)}}]
                """.utf8))
        }
        XCTAssertNotEqual(try fingerprint(revision: "123", counted: 42),
                          try fingerprint(revision: "456", counted: 42))
        XCTAssertNotEqual(try fingerprint(revision: "123", counted: 42),
                          try fingerprint(revision: "123", counted: 43))
        XCTAssertEqual(try TurboSparkCatalog.recommendationCatalogFingerprint().count, 64)
    }

    func testMalformedCatalogIsAnErrorInsteadOfACrash() {
        XCTAssertThrowsError(try TurboSparkCatalog.recommendationCatalogFingerprint(json: Data("{}".utf8)))
    }
}
