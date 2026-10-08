import Foundation
import XCTest

@testable import TurboSparkApp

/// Real-git tests against a local origin: a tag-pinned source must keep
/// refreshing, and a changed ref must re-clone instead of being ignored.
final class MarketplaceGitTests: XCTestCase {
    private func git(_ args: [String], in dir: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-c", "user.name=t", "-c", "user.email=t@example.com",
                             "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"] + args
        process.currentDirectoryURL = dir
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, "git \(args)")
    }

    func testTagPinnedSourceRefreshesAndRefChangeReclones() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mgit-\(UUID().uuidString)", isDirectory: true)
        let origin = root.appendingPathComponent("origin", isDirectory: true)
        let cache = root.appendingPathComponent("cache", isDirectory: true)
        try FileManager.default.createDirectory(at: origin, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try git(["init", "-q", "-b", "main"], in: origin)
        try "one".write(to: origin.appendingPathComponent("f.txt"), atomically: true, encoding: .utf8)
        try git(["add", "."], in: origin)
        try git(["commit", "-q", "-m", "one"], in: origin)
        try git(["tag", "v1"], in: origin)
        try "two".write(to: origin.appendingPathComponent("f.txt"), atomically: true, encoding: .utf8)
        try git(["commit", "-q", "-am", "two"], in: origin)
        try git(["tag", "v2"], in: origin)

        let url = "file://" + origin.path
        try await MarketplaceGit.cloneOrPull(url: url, targetDir: cache, ref: "v1", sparsePaths: nil)
        XCTAssertEqual(try String(contentsOf: cache.appendingPathComponent("f.txt")), "one")

        // Second refresh of a tag (detached HEAD): pull --ff-only used to fail here forever.
        try await MarketplaceGit.cloneOrPull(url: url, targetDir: cache, ref: "v1", sparsePaths: nil)
        XCTAssertEqual(try String(contentsOf: cache.appendingPathComponent("f.txt")), "one")

        // Changing the ref must not be ignored.
        try await MarketplaceGit.cloneOrPull(url: url, targetDir: cache, ref: "v2", sparsePaths: nil)
        XCTAssertEqual(try String(contentsOf: cache.appendingPathComponent("f.txt")), "two")
    }
}
