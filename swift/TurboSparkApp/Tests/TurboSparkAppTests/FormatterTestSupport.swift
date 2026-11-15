import Foundation
@testable import TurboSparkApp

/// OC-11 test double: a bash script standing in for a formatter. No real
/// formatter binary is ever run by these tests. The script lives in its own
/// temporary folder OUTSIDE the project, like a user-installed binary.
final class FormatterFixture {
    let project: URL
    let tools: URL
    let log: URL

    init(files: [String: String] = ["Sources/A.swift": "let a = 1\n"]) throws {
        let fm = FileManager.default
        let base = fm.temporaryDirectory.appendingPathComponent("fmt-test-\(UUID().uuidString)")
        let projectDir = base.appendingPathComponent("project")
        let toolDir = base.appendingPathComponent("tools")
        try fm.createDirectory(at: projectDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: toolDir, withIntermediateDirectories: true)
        project = PathContainment.canonical(projectDir)
        tools = PathContainment.canonical(toolDir)
        log = tools.appendingPathComponent("log.txt")
        for (path, text) in files {
            let url = project.appendingPathComponent(path)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try text.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    deinit { try? FileManager.default.removeItem(at: tools.deletingLastPathComponent()) }

    /// Writes an executable script and returns a row that runs it on `.swift` files.
    func row(_ body: String, name: String = "fixture-fmt", extensions: [String] = ["swift"]) throws -> FormatterRow {
        let script = tools.appendingPathComponent(name)
        try ("#!/bin/bash\n" + body + "\n").write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return FormatterRow(
            id: "custom-test", name: name, extensions: extensions, command: script.path,
            args: [FormatterRow.filePlaceholder], enabled: true, isBuiltin: false)
    }

    func read(_ path: String) throws -> String {
        try String(contentsOf: project.appendingPathComponent(path), encoding: .utf8)
    }

    var logText: String { (try? String(contentsOf: log, encoding: .utf8)) ?? "" }

    /// Names in the project directory tree, to catch a leftover temp copy.
    func leftoverTemps() -> [String] {
        let walker = FileManager.default.enumerator(atPath: project.path)
        return (walker?.allObjects as? [String] ?? []).filter { $0.contains(FormatterRunner.tempPrefix) }
    }

    /// Uppercases the file in place through a rename, like a real formatter.
    static let upcase = #"tr a-z A-Z < "$1" > "$1.out" && mv "$1.out" "$1""#
}
