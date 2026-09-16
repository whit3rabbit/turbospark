import Foundation
import XCTest

@testable import TurboSparkApp

final class ContextReferenceSecurityTests: XCTestCase {
    private func git(_ arguments: [String], in directory: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.currentDirectoryURL = directory
        process.arguments = arguments
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
    }

    private func repository(for reference: String, driver: String) throws -> (URL, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-context-security-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try git(["init", "-q"], in: root)
        try git(["config", "user.email", "t@example.com"], in: root)
        try git(["config", "user.name", "T"], in: root)
        let tracked = root.appendingPathComponent("tracked.txt")
        try "base\n".write(to: tracked, atomically: true, encoding: .utf8)
        try git(["add", "."], in: root)
        try git(["commit", "-qm", "base"], in: root)

        if reference == "@git:1" {
            try "committed change\n".write(to: tracked, atomically: true, encoding: .utf8)
            try git(["add", "."], in: root)
            try git(["commit", "-qm", "second"], in: root)
        } else {
            try "working change\n".write(to: tracked, atomically: true, encoding: .utf8)
            if reference == "@staged" {
                try git(["add", "."], in: root)
            }
        }

        let marker = root.appendingPathComponent("helper-ran")
        let helper = root.appendingPathComponent("malicious-diff")
        try "#!/bin/sh\ntouch '\(marker.path)'\n".write(
            to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helper.path)
        if driver == "external" {
            try git(["config", "diff.external", helper.path], in: root)
        } else {
            try "* diff=malicious\n".write(
                to: root.appendingPathComponent(".gitattributes"), atomically: true, encoding: .utf8)
            try git(["config", "diff.malicious.textconv", helper.path], in: root)
        }
        return (root, marker)
    }

    @MainActor
    func testGitReferencesDoNotExecuteConfiguredDiffHelpers() async throws {
        for reference in ["@diff", "@staged", "@git:1"] {
            for driver in ["external", "textconv"] {
                let (root, marker) = try repository(for: reference, driver: driver)
                defer { try? FileManager.default.removeItem(at: root) }

                let model = AppModel()
                let chatID = UUID()
                _ = await MentionResolver.resolveMentions(
                    in: reference, projectRoot: root, chatID: chatID, into: model)

                XCTAssertFalse(
                    FileManager.default.fileExists(atPath: marker.path),
                    "\(reference) executed the configured \(driver) helper")
                XCTAssertEqual(
                    model.chats.first(where: { $0.id == chatID })?.draftAttachments.count, 1,
                    "\(reference) should still produce an attachment")
            }
        }
    }
}
