import Foundation

/// Pass, fail, and skip counts for a shell command that ran `cargo test` or
/// `swift test` (XCTest and swift-testing), with each failing test's name
/// and first failure line (component audit, test-results).
///
/// Read back from the printed output, so `parse` checks every count it
/// reports against a second source in the same output (cargo's "running N
/// tests" header, XCTest's per-case lines, swift-testing's per-test lines)
/// and returns nil on any disagreement, a run that never printed its final
/// summary (a crash), or output the runner compacted. No summary is better
/// than a wrong one: the raw output still shows.
struct TestRunSummary: Equatable {
    struct Failure: Equatable {
        let name: String
        /// The first line that says why, when the output had one.
        let message: String?
    }

    let passed: Int
    let failed: Int
    let skipped: Int
    /// Summed from the runners' own timings. Nil when a section lacked one.
    let durationSeconds: Double?
    let failures: [Failure]

    var allPassed: Bool { failed == 0 }

    static func parse(_ output: String) -> TestRunSummary? {
        let lines = TerminalResultSummary.normalizedLines(output)
        // The runner cut the middle out: some summary lines may be gone.
        if lines.contains(where: { $0.hasPrefix("... [") && $0.hasSuffix(" chars truncated] ...") }) {
            return nil
        }
        var parts: [TestRunSummary] = []
        for section in [cargo(lines), xctest(lines), swiftTesting(lines)] {
            switch section {
            case .absent: continue
            case .unsure: return nil
            case .parsed(let part): parts.append(part)
            }
        }
        guard !parts.isEmpty else { return nil }
        let passed = parts.reduce(0) { $0 + $1.passed }
        let failed = parts.reduce(0) { $0 + $1.failed }
        let skipped = parts.reduce(0) { $0 + $1.skipped }
        // An empty run would read as a green one that tested nothing.
        guard passed + failed + skipped > 0 else { return nil }
        let durations = parts.compactMap(\.durationSeconds)
        return TestRunSummary(
            passed: passed, failed: failed, skipped: skipped,
            durationSeconds: durations.count == parts.count ? durations.reduce(0, +) : nil,
            failures: parts.flatMap(\.failures))
    }

    private enum Section {
        /// The runner printed nothing here.
        case absent
        /// The runner printed something this parser cannot vouch for.
        case unsure
        case parsed(TestRunSummary)
    }

    // MARK: - cargo test

    private static let cargoRunning = regex(#"^running (\d+) tests?$"#)
    private static let cargoResult = regex(
        #"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; \d+ filtered out; finished in ([0-9.]+)s$"#)

    /// One "running N tests" header and one "test result:" line per test
    /// binary (doc-tests included), summed. The failing names come from the
    /// "failures:" list every binary prints, which `-q` keeps too.
    private static func cargo(_ lines: [String]) -> Section {
        var sawAny = false
        var pendingHeader: Int?
        var passed = 0, failed = 0, ignored = 0
        var duration = 0.0
        var listed: [String] = []
        var messages: [String: String] = [:]

        var index = 0
        while index < lines.count {
            let line = lines[index].trimmingCharacters(in: .whitespaces)
            if let groups = captures(cargoRunning, line) {
                sawAny = true
                // A header with no result before the next: that binary died.
                guard pendingHeader == nil, let count = Int(groups[0]) else { return .unsure }
                pendingHeader = count
            } else if line.hasPrefix("test result: ") {
                sawAny = true
                guard let header = pendingHeader, let groups = captures(cargoResult, line),
                      let p = Int(groups[1]), let f = Int(groups[2]), let i = Int(groups[3]),
                      let m = Int(groups[4]), let seconds = Double(groups[5]),
                      p + f + i + m == header, (groups[0] == "FAILED") == (f > 0)
                else { return .unsure }
                pendingHeader = nil
                // Measured (benchmark) entries are neither pass nor fail.
                passed += p
                failed += f
                ignored += i
                duration += seconds
            } else if lines[index] == "failures:" {
                // The name list is indented four spaces. The first
                // "failures:" of a binary is followed by a blank line and the
                // stdout sections, so it lists nothing.
                var next = index + 1
                while next < lines.count, lines[next].hasPrefix("    "),
                      !lines[next].trimmingCharacters(in: .whitespaces).isEmpty {
                    listed.append(lines[next].trimmingCharacters(in: .whitespaces))
                    next += 1
                }
                index = next
                continue
            } else if let name = cargoStdoutHeader(lines[index]) {
                var next = index + 1
                var section: [String] = []
                while next < lines.count, cargoStdoutHeader(lines[next]) == nil,
                      lines[next] != "failures:", !lines[next].hasPrefix("test result: ") {
                    section.append(lines[next])
                    next += 1
                }
                if messages[name] == nil, let message = cargoMessage(section) {
                    messages[name] = message
                }
                index = next
                continue
            }
            index += 1
        }

        guard sawAny else { return .absent }
        guard pendingHeader == nil, listed.count == failed, Set(listed).count == listed.count
        else { return .unsure }
        return .parsed(TestRunSummary(
            passed: passed, failed: failed, skipped: ignored, durationSeconds: duration,
            failures: listed.map { Failure(name: $0, message: messages[$0]) }))
    }

    private static func cargoStdoutHeader(_ line: String) -> String? {
        guard line.hasPrefix("---- "), line.hasSuffix(" stdout ----") else { return nil }
        let name = line.dropFirst("---- ".count).dropLast(" stdout ----".count)
        return name.isEmpty ? nil : String(name)
    }

    /// The panic message when the section has one (Rust 1.73+ prints it on
    /// the line after "panicked at <location>:", older toolchains quote it
    /// inline), else the section's first line, as for a doc-test that
    /// failed to compile.
    private static func cargoMessage(_ section: [String]) -> String? {
        let nonBlank = section
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        if let at = nonBlank.firstIndex(where: { $0.hasPrefix("thread '") && $0.contains(" panicked at ") }) {
            let line = nonBlank[at]
            if let open = line.range(of: " panicked at '"), let close = line.range(of: "', ", options: .backwards),
               open.upperBound <= close.lowerBound {
                return String(line[open.upperBound..<close.lowerBound])
            }
            if line.hasSuffix(":"), at + 1 < nonBlank.count {
                return nonBlank[at + 1]
            }
        }
        return nonBlank.first
    }

    // MARK: - XCTest

    private static let xcCase = regex(#"^Test Case '(.+)' (passed|failed|skipped) \([0-9.]+ seconds\)\.$"#)
    private static let xcTopEnd = regex(#"^Test Suite '(?:All tests|Selected tests)' (?:passed|failed) at "#)
    private static let xcExecuted = regex(
        #"^Executed (\d+) tests?, with (?:(\d+) tests? skipped and )?(\d+) failures? \(\d+ unexpected\) in ([0-9.]+) \([0-9.]+\) seconds$"#)
    private static let xcError = regex(#"^.+?:\d+: error: (.+?) : (.*)$"#)

    /// The total comes from the top suite's "Executed" line ('All tests', or
    /// 'Selected tests' under --filter). Its failure count is assertions, not
    /// tests, so failed tests are the "Test Case ... failed" lines, and the
    /// per-case lines must add up to the total.
    private static func xctest(_ lines: [String]) -> Section {
        var sawAny = false
        var casePassed = 0, caseFailed = 0, caseSkipped = 0
        var failedIDs: [String] = []
        var messages: [String: String] = [:]
        var total = 0, totalSkipped = 0, totalFailures = 0
        var duration = 0.0
        var sawTop = false

        for (index, line) in lines.enumerated() {
            if line.hasPrefix("Test Case '") || line.hasPrefix("Test Suite '") { sawAny = true }
            if let groups = captures(xcCase, line) {
                switch groups[1] {
                case "passed": casePassed += 1
                case "failed":
                    caseFailed += 1
                    if !failedIDs.contains(groups[0]) { failedIDs.append(groups[0]) }
                default: caseSkipped += 1
                }
            } else if captures(xcTopEnd, line) != nil {
                let next = lines[(index + 1)...].first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                guard let executed = next.map({ $0.trimmingCharacters(in: .whitespaces) }),
                      let groups = captures(xcExecuted, executed),
                      let count = Int(groups[0]), let failures = Int(groups[2]),
                      let seconds = Double(groups[3])
                else { return .unsure }
                sawTop = true
                total += count
                totalSkipped += groups[1].isEmpty ? 0 : (Int(groups[1]) ?? 0)
                totalFailures += failures
                duration += seconds
            } else if let groups = captures(xcError, line), messages[groups[0]] == nil {
                messages[groups[0]] = groups[1]
            }
        }

        guard sawAny else { return .absent }
        guard sawTop, casePassed + caseFailed + caseSkipped == total, caseSkipped == totalSkipped,
              (totalFailures > 0) == (caseFailed > 0)
        else { return .unsure }
        return .parsed(TestRunSummary(
            passed: casePassed, failed: caseFailed, skipped: caseSkipped, durationSeconds: duration,
            failures: failedIDs.map { Failure(name: xcDisplayName($0), message: messages[$0]) }))
    }

    /// "-[Module.Class testName]" reads as "Class.testName".
    private static func xcDisplayName(_ id: String) -> String {
        guard id.hasPrefix("-["), id.hasSuffix("]") else { return id }
        let parts = id.dropFirst(2).dropLast().split(separator: " ")
        guard parts.count == 2 else { return id }
        let type = parts[0].split(separator: ".").last.map(String.init) ?? String(parts[0])
        return "\(type).\(parts[1])"
    }

    // MARK: - swift-testing

    private static let stRun = regex(
        #"^Test run with (\d+) tests?(?: in \d+ suites?)? (passed|failed) after ([0-9.]+) seconds"#)
    private static let stPassed = regex(#"^Test (.+) passed after [0-9.]+ seconds"#)
    private static let stFailed = regex(#"^Test (.+) failed after [0-9.]+ seconds"#)
    private static let stSkipped = regex(#"^Test (.+) skipped(?:\.|: .*)$"#)
    private static let stIssue = regex(#"^Test (.+?) recorded an issue.*? at [^ ]+:\d+:\d+: (.*)$"#)

    /// The total comes from "Test run with N tests ... passed|failed", which
    /// counts skipped tests too, and must equal the per-test result lines.
    private static func swiftTesting(_ lines: [String]) -> Section {
        var sawAny = false
        var passed = 0, failed = 0, skipped = 0
        var failedNames: [String] = []
        var messages: [String: String] = [:]
        var total = 0
        var runFailed = false
        var duration = 0.0
        var sawRun = false

        for line in lines {
            guard let body = swiftTestingBody(line) else { continue }
            if body.hasPrefix("Test run started.") {
                sawAny = true
            } else if let groups = captures(stRun, body) {
                guard let count = Int(groups[0]), let seconds = Double(groups[2]) else { return .unsure }
                sawAny = true
                sawRun = true
                total += count
                runFailed = runFailed || groups[1] == "failed"
                duration += seconds
            } else if body.hasPrefix("Test case ") {
                // A parameterized test's per-argument line; the test's own
                // result line follows.
                continue
            } else if captures(stPassed, body) != nil {
                sawAny = true
                passed += 1
            } else if let groups = captures(stFailed, body) {
                sawAny = true
                failed += 1
                if !failedNames.contains(groups[0]) { failedNames.append(groups[0]) }
            } else if captures(stSkipped, body) != nil {
                sawAny = true
                skipped += 1
            } else if let groups = captures(stIssue, body), messages[groups[0]] == nil {
                messages[groups[0]] = groups[1]
            }
        }

        guard sawAny else { return .absent }
        guard sawRun, passed + failed + skipped == total, runFailed == (failed > 0) else { return .unsure }
        return .parsed(TestRunSummary(
            passed: passed, failed: failed, skipped: skipped, durationSeconds: duration,
            failures: failedNames.map { Failure(name: unquoted($0), message: messages[$0]) }))
    }

    /// The line without its leading status symbol (SF Symbols in a
    /// terminal, other glyphs in a pipe), when what follows is a "Test "
    /// line. XCTest's "Test Case '" and "Test Suite '" lines are not ours.
    private static func swiftTestingBody(_ line: String) -> String? {
        var body = Substring(line.trimmingCharacters(in: .whitespaces))
        if !body.hasPrefix("Test "), let space = body.firstIndex(of: " "),
           !body[..<space].contains(where: { $0.isASCII && ($0.isLetter || $0.isNumber) }) {
            body = body[body.index(after: space)...]
        }
        guard body.hasPrefix("Test "), !body.hasPrefix("Test Case '"), !body.hasPrefix("Test Suite '")
        else { return nil }
        return String(body)
    }

    private static func unquoted(_ name: String) -> String {
        guard name.count >= 2, name.hasPrefix("\""), name.hasSuffix("\"") else { return name }
        return String(name.dropFirst().dropLast())
    }

    // MARK: - Helpers

    private static func regex(_ pattern: String) -> NSRegularExpression {
        // The patterns are literals, so a failure is a programming error.
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: pattern)
    }

    /// The capture groups of a whole-line match, "" for a group that did
    /// not take part.
    private static func captures(_ regex: NSRegularExpression, _ line: String) -> [String]? {
        let range = NSRange(line.startIndex..., in: line)
        guard let match = regex.firstMatch(in: line, range: range) else { return nil }
        return (1..<match.numberOfRanges).map { group in
            Range(match.range(at: group), in: line).map { String(line[$0]) } ?? ""
        }
    }
}
