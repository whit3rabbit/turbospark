import XCTest
@testable import TurboSparkApp

final class SkillMarketplaceConfinementTests: XCTestCase {
    private func makeCheckout() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("skill_confine_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("skills/ok"), withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    func testRelativeSubfolderAndNilPathAreAccepted() throws {
        let root = try makeCheckout()
        XCTAssertEqual(try SkillMarketplaceManager.confinedGitSkillSource(path: nil, checkout: root), root)
        let sub = try SkillMarketplaceManager.confinedGitSkillSource(path: "skills/ok", checkout: root)
        XCTAssertTrue(sub.path.hasSuffix("skills/ok"))
    }

    func testTraversalIsRefused() throws {
        let root = try makeCheckout()
        XCTAssertThrowsError(try SkillMarketplaceManager.confinedGitSkillSource(
            path: "../../../../Users/me/.claude/skills/private", checkout: root))
        // An absolute-looking path is appended as a relative component, so it
        // stays under the checkout rather than naming /etc.
        let abs = try SkillMarketplaceManager.confinedGitSkillSource(path: "/etc", checkout: root)
        XCTAssertTrue(PathContainment.isContained(abs, in: root))
    }

    func testSymlinkEscapeIsRefused() throws {
        let root = try makeCheckout()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("skill_outside_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: outside) }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("skills/link"), withDestinationURL: outside)
        XCTAssertThrowsError(try SkillMarketplaceManager.confinedGitSkillSource(
            path: "skills/link", checkout: root))
    }
}
