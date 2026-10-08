import Foundation
import XCTest
import ZImage

/// Regression tests for crash and escape hardening in the vendored ZImage
/// package (low-severity review round).
final class ZImageHardeningTests: XCTestCase {
    private func writeSafeTensors(header: String, headerLengthOverride: UInt64? = nil, body: Data = Data()) throws -> URL {
        let headerData = Data(header.utf8)
        var file = Data()
        var length = (headerLengthOverride ?? UInt64(headerData.count)).littleEndian
        withUnsafeBytes(of: &length) { file.append(contentsOf: $0) }
        file.append(headerData)
        file.append(body)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zimage-st-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }

    func testSpecMadeOnlyOfColonsIsNotAModelIdAndDoesNotTrap() {
        XCTAssertFalse(ModelResolution.isHuggingFaceModelId(":"))
        XCTAssertFalse(ModelResolution.isHuggingFaceModelId(":::"))
    }

    func testHeaderLengthWithTopBitSetThrowsInsteadOfTrapping() throws {
        let url = try writeSafeTensors(header: "{}", headerLengthOverride: UInt64.max)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try SafeTensorsReader(fileURL: url))
    }

    func testNegativeOffsetsAndOverflowingShapesAreRefused() throws {
        let headers = [
            #"{"t":{"dtype":"F32","shape":[1],"data_offsets":[-1000000,-999996]}}"#,
            #"{"t":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,4]}}"#,
            #"{"t":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
        ]
        for header in headers {
            let url = try writeSafeTensors(header: header, body: Data(count: 16))
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try SafeTensorsReader(fileURL: url), header)
        }
    }

    func testWellFormedHeaderStillParses() throws {
        let url = try writeSafeTensors(
            header: #"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, body: Data(count: 8))
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try SafeTensorsReader(fileURL: url)
        XCTAssertEqual(reader.metadata(for: "t")?.shape, [2])
    }
}
