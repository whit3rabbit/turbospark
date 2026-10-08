import Foundation

enum WorkflowGitReadOp: String, Equatable, Sendable {
    case status
    case diff
    case log
    case changedFiles
}

enum WorkflowWorldRead: Equatable, Sendable {
    case glob(pattern: String)
    case read(path: String, maxBytes: Int)
    case grep(pattern: String, pathHint: String?)
    case git(op: WorkflowGitReadOp)
}

struct WorkflowWorldObservation: Equatable, Sendable {
    var argv: [String]
    var exitStatus: Int
    var outputText: String
    var truncated: Bool

    var canonicalValue: WorkflowCanonicalValue {
        .object([
            "argv": .array(argv.map(WorkflowCanonicalValue.string)),
            "exitStatus": .integer(Int64(exitStatus)),
            "outputText": .string(outputText),
            "truncated": .boolean(truncated),
        ])
    }
}

struct WorkflowWorldLimits: Equatable, Sendable {
    var maximumReadBytes: Int
    var maximumOutputBytes: Int
    var maximumDirectoryEntries: Int
    var maximumScanBytes: Int
    var maximumPatternBytes: Int
    var gitTimeoutSeconds: TimeInterval

    init(
        maximumReadBytes: Int = 1_000_000,
        maximumOutputBytes: Int = 1_000_000,
        maximumDirectoryEntries: Int = 10_000,
        maximumScanBytes: Int = 8 * 1_024 * 1_024,
        maximumPatternBytes: Int = 4_096,
        gitTimeoutSeconds: TimeInterval = 30
    ) {
        self.maximumReadBytes = maximumReadBytes
        self.maximumOutputBytes = maximumOutputBytes
        self.maximumDirectoryEntries = maximumDirectoryEntries
        self.maximumScanBytes = maximumScanBytes
        self.maximumPatternBytes = maximumPatternBytes
        self.gitTimeoutSeconds = gitTimeoutSeconds
    }
}

struct WorkflowWorldProcessInvocation: Equatable, Sendable {
    var executableURL: URL
    var arguments: [String]
    var workingDirectoryURL: URL
    var environment: [String: String]
    var timeoutSeconds: TimeInterval
    var outputCapBytes: Int
    /// Parse stdout only. NUL-delimited records (`status -z`) are corrupted
    /// by a stderr warning interleaved into the merged stream.
    var stdoutOnly = false

    var argv: [String] {
        ["git"] + arguments
    }
}

protocol WorkflowWorldExecutorPort: Sendable {
    func execute(
        _ invocation: WorkflowWorldProcessInvocation
    ) async throws -> WorkflowWorldObservation
}

struct WorkflowWorldSystemExecutor: WorkflowWorldExecutorPort {
    func execute(
        _ invocation: WorkflowWorldProcessInvocation
    ) async throws -> WorkflowWorldObservation {
        let output = try await ProcessExecutor.run(
            executableURL: invocation.executableURL,
            arguments: invocation.arguments,
            currentDirectoryURL: invocation.workingDirectoryURL,
            environment: invocation.environment,
            timeoutSeconds: invocation.timeoutSeconds,
            outputCapBytes: invocation.outputCapBytes,
            mergeStreams: !invocation.stdoutOnly)
        guard !output.timedOut else {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "The bounded workflow git read timed out.",
                site: nil)
        }
        return WorkflowWorldObservation(
            argv: invocation.argv,
            exitStatus: Int(output.exitCode),
            outputText: invocation.stdoutOnly ? output.stdout : output.combinedText,
            truncated: output.outputTruncated)
    }
}

/// Read-only workspace capabilities. The run engine owns journal identity and
/// replay, so a recorded read never reaches this boundary a second time.
final class WorkflowWorld: WorkflowCapabilities, Sendable {
    private struct DirectoryEnumeration {
        var entries: [URL]
        var truncated: Bool
    }

    private struct BoundedText {
        private(set) var data = Data()
        private(set) var truncated = false
        let maximumBytes: Int

        mutating func append(_ value: String, separator: Bool = false) -> Bool {
            let prefix = separator && !data.isEmpty ? "\n" : ""
            let bytes = Data((prefix + value).utf8)
            let remaining = max(0, maximumBytes - data.count)
            if bytes.count <= remaining {
                data.append(bytes)
                return true
            }

            if remaining > 0 {
                var length = remaining
                while length > 0,
                      length < bytes.count,
                      bytes[length] & 0b1100_0000 == 0b1000_0000
                {
                    length -= 1
                }
                data.append(bytes.prefix(length))
            }
            truncated = true
            return false
        }

        var string: String {
            String(decoding: data, as: UTF8.self)
        }
    }

    private let workspaceRoot: URL
    private let executor: any WorkflowWorldExecutorPort
    private let limits: WorkflowWorldLimits

    init(
        workspaceRoot: URL,
        executor: any WorkflowWorldExecutorPort = WorkflowWorldSystemExecutor(),
        limits: WorkflowWorldLimits = WorkflowWorldLimits()
    ) {
        self.workspaceRoot = workspaceRoot
        self.executor = executor
        self.limits = limits
    }

    func read(
        _ operation: WorkflowWorldRead,
        identity: WorkflowRequestIdentity
    ) async throws -> WorkflowWorldObservation {
        do {
            switch operation {
            case .glob(let pattern):
                return try glob(pattern, site: identity.site)
            case .read(let path, let maxBytes):
                return try readFile(path, maxBytes: maxBytes, site: identity.site)
            case .grep(let pattern, let pathHint):
                return try grep(pattern, pathHint: pathHint, site: identity.site)
            case .git(let op):
                return try await git(op, site: identity.site)
            }
        } catch let error as WorkflowError {
            throw WorkflowError(kind: error.kind, message: error.message, site: identity.site)
        } catch is CancellationError {
            throw WorkflowError(
                kind: .cancelled,
                message: "The workflow workspace read was cancelled.",
                site: identity.site)
        } catch {
            throw refusal("the workspace read could not be completed safely", site: identity.site)
        }
    }

    func perform(
        _ operation: WorkflowInterpreterOperation,
        at site: WorkflowSiteKey?,
        context: WorkflowAttemptContext
    ) async throws -> WorkflowCapabilityResult {
        guard case .worldRead(let canonicalOperation) = operation,
              let site,
              site.lane == "main"
        else {
            throw refusal("the operation is outside the read-only world facade", site: site)
        }
        guard !Task.isCancelled, !context.cancellation.isCancelled else {
            throw WorkflowError(
                kind: .cancelled,
                message: "The workflow workspace read was cancelled.",
                site: site)
        }
        if let deadline = context.deadline, ContinuousClock.now >= deadline {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "The workflow workspace read exceeded its deadline.",
                site: site)
        }

        let read = try decodeRead(canonicalOperation, site: site)
        let input = WorkflowCanonicalValue.object([
            "kind": .string("worldRead"),
            "operation": canonicalOperation,
        ])
        let identity = try WorkflowRequestIdentity.make(site: site, input: input)
        let observation = try await self.read(read, identity: identity)

        guard !Task.isCancelled, !context.cancellation.isCancelled else {
            throw WorkflowError(
                kind: .cancelled,
                message: "The workflow workspace read was cancelled.",
                site: site)
        }
        if let deadline = context.deadline, ContinuousClock.now >= deadline {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "The workflow workspace read exceeded its deadline.",
                site: site)
        }
        return WorkflowCapabilityResult(value: observation.canonicalValue)
    }

    private func decodeRead(
        _ value: WorkflowCanonicalValue,
        site: WorkflowSiteKey
    ) throws -> WorkflowWorldRead {
        guard case .object(let fields) = value,
              Set(fields.keys) == Set(["kind", "arguments"]),
              case .string(let kind)? = fields["kind"],
              case .array(let arguments)? = fields["arguments"]
        else {
            throw refusal("the world read does not match a fixed facade operation", site: site)
        }

        switch kind {
        case "glob":
            guard arguments.count == 1,
                  case .string(let pattern) = arguments[0]
            else {
                throw refusal("glob requires one fixed string pattern", site: site)
            }
            return .glob(pattern: pattern)
        case "read":
            guard arguments.count == 2,
                  case .string(let path) = arguments[0]
            else {
                throw refusal("read requires a fixed path and byte limit", site: site)
            }
            // The interpreter emits every script numeral as `.number`, so
            // only accepting `.integer` refused every read. A whole,
            // finite double is the same request; the byte-limit bounds are
            // enforced where the read executes.
            let requestedBytes: Int?
            switch arguments[1] {
            case .integer(let value):
                requestedBytes = Int(exactly: value)
            case .number(let value):
                requestedBytes = value.isFinite && value.rounded(.towardZero) == value
                    ? Int(exactly: value) : nil
            default:
                requestedBytes = nil
            }
            guard let maxBytes = requestedBytes else {
                throw refusal("read requires a fixed path and byte limit", site: site)
            }
            return .read(path: path, maxBytes: maxBytes)
        case "grep":
            guard (1...2).contains(arguments.count),
                  case .string(let pattern) = arguments[0]
            else {
                throw refusal("grep requires a fixed pattern and optional path hint", site: site)
            }
            let pathHint: String?
            if arguments.count == 1 {
                pathHint = nil
            } else {
                switch arguments[1] {
                case .null:
                    pathHint = nil
                case .string(let path):
                    pathHint = path
                default:
                    throw refusal("grep path hints must be fixed workspace-relative paths", site: site)
                }
            }
            return .grep(pattern: pattern, pathHint: pathHint)
        case "git":
            guard arguments.count == 1,
                  case .string(let rawOperation) = arguments[0],
                  let operation = WorkflowGitReadOp(rawValue: rawOperation)
            else {
                throw refusal("the git read operation is unsupported", site: site)
            }
            return .git(op: operation)
        default:
            throw refusal("the world read operation is unsupported", site: site)
        }
    }

    private func glob(_ pattern: String, site: WorkflowSiteKey) throws -> WorkflowWorldObservation {
        let root = try canonicalRoot(site: site)
        try validateRelativePattern(pattern, site: site)
        let expression = try globExpression(pattern, site: site)
        let enumeration = try entries(under: root, site: site)
        var matches: [String] = []

        // Both sides standardized the same way: the enumerator may hand back
        // /var/... for a root spelled /private/var/...
        let standardRoot = root.standardizedFileURL.path
        let rootPrefix = standardRoot.hasSuffix("/") ? standardRoot : standardRoot + "/"
        for candidate in enumeration.entries {
            // Match the pattern on the candidate's own workspace-relative name
            // BEFORE checking containment: an unrelated symlink that points
            // outside the tree (a venv's python3 -> /opt/homebrew/...) used to
            // abort the whole glob although nothing it names was a match. A
            // candidate that matches and escapes is still refused.
            let candidatePath = candidate.standardizedFileURL.path
            if candidatePath.hasPrefix(rootPrefix) {
                let named = String(candidatePath.dropFirst(rootPrefix.count))
                guard expression.firstMatch(
                    in: named,
                    range: NSRange(named.startIndex..., in: named)) != nil
                else {
                    continue
                }
            }
            let resolved = PathContainment.canonical(candidate)
            guard PathContainment.isContained(resolved, in: root) else {
                throw refusal("a glob match resolves outside the workspace", site: site)
            }
            let relative = try relativePath(resolved, from: root, site: site)
            guard expression.firstMatch(
                in: relative,
                range: NSRange(relative.startIndex..., in: relative)) != nil
            else {
                continue
            }
            let values = try resolved.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                matches.append(relative)
            }
        }

        var output = BoundedText(maximumBytes: limits.maximumOutputBytes)
        for match in matches.sorted() {
            if !output.append(match, separator: true) { break }
        }
        return WorkflowWorldObservation(
            argv: ["glob", pattern],
            exitStatus: 0,
            outputText: output.string,
            truncated: enumeration.truncated || output.truncated)
    }

    private func readFile(
        _ path: String,
        maxBytes: Int,
        site: WorkflowSiteKey
    ) throws -> WorkflowWorldObservation {
        guard limits.maximumReadBytes > 0,
              maxBytes > 0,
              maxBytes <= limits.maximumReadBytes
        else {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "The requested workflow file read exceeds its byte limit.",
                site: site)
        }
        let root = try canonicalRoot(site: site)
        let resolved = try resolve(path, under: root, allowsDirectory: false, site: site)
        let handle = try FileHandle(forReadingFrom: resolved)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: maxBytes + 1) ?? Data()
        let wasTruncated = data.count > maxBytes
        let visible = Data(data.prefix(maxBytes))
        guard let text = Self.utf8Text(visible, allowingPartialTail: wasTruncated) else {
            throw WorkflowError(
                kind: .validation,
                message: "Workflow file reads require UTF-8 text.",
                site: site)
        }
        return WorkflowWorldObservation(
            argv: ["read", path, String(maxBytes)],
            exitStatus: 0,
            outputText: text,
            truncated: wasTruncated)
    }

    private func grep(
        _ pattern: String,
        pathHint: String?,
        site: WorkflowSiteKey
    ) throws -> WorkflowWorldObservation {
        guard pattern.utf8.count <= limits.maximumPatternBytes,
              !pattern.utf8.contains(0)
        else {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "The workflow grep pattern exceeds its byte limit.",
                site: site)
        }
        guard limits.maximumScanBytes > 0,
              limits.maximumDirectoryEntries > 0,
              limits.maximumOutputBytes > 0
        else {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "Workflow grep limits must be positive.",
                site: site)
        }

        let root = try canonicalRoot(site: site)
        let searchRoot = try pathHint.map {
            try resolve($0, under: root, allowsDirectory: true, site: site)
        } ?? root
        let values = try searchRoot.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
        let candidates: [URL]
        let enumerationTruncated: Bool
        if values.isDirectory == true {
            let enumeration = try entries(under: searchRoot, site: site, skippingMetadata: true)
            candidates = enumeration.entries
            enumerationTruncated = enumeration.truncated
        } else if values.isRegularFile == true {
            candidates = [searchRoot]
            enumerationTruncated = false
        } else {
            throw refusal("grep path hints must name a file or directory", site: site)
        }

        var output = BoundedText(maximumBytes: limits.maximumOutputBytes)
        var scannedBytes = 0
        var scanTruncated = false
        let sortedCandidates = candidates.sorted { $0.path < $1.path }
        for candidate in sortedCandidates {
            let resolved = PathContainment.canonical(candidate)
            guard PathContainment.isContained(resolved, in: root) else {
                // A stray symlink met while walking a directory is skipped
                // (nothing is read through it); only an explicitly named
                // file that escapes is refused.
                if candidate == searchRoot {
                    throw refusal("a grep candidate resolves outside the workspace", site: site)
                }
                continue
            }
            guard try resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                continue
            }
            let remaining = limits.maximumScanBytes - scannedBytes
            guard remaining > 0 else {
                scanTruncated = true
                break
            }

            let handle = try FileHandle(forReadingFrom: resolved)
            let data: Data
            do {
                data = try handle.read(upToCount: remaining + 1) ?? Data()
                try handle.close()
            } catch {
                try? handle.close()
                throw error
            }
            let fileTruncated = data.count > remaining
            let visibleData = Data(data.prefix(remaining))
            // A binary file is not searchable text: skip it WITHOUT charging
            // the scan budget, or compressed objects and build products
            // exhaust it before any source file is read.
            if visibleData.contains(0) { continue }
            scannedBytes += visibleData.count
            if fileTruncated {
                scanTruncated = true
            }
            guard let text = Self.utf8Text(visibleData, allowingPartialTail: fileTruncated)
            else {
                if fileTruncated { break }
                continue
            }

            let relative = try relativePath(resolved, from: root, site: site)
            let lines = text.components(separatedBy: "\n")
            for (index, line) in lines.enumerated() where line.contains(pattern) {
                let cleanLine = line.hasSuffix("\r") ? String(line.dropLast()) : line
                if !output.append("\(relative):\(index + 1):\(cleanLine)", separator: true) {
                    break
                }
            }
            if output.truncated || fileTruncated {
                break
            }
        }

        return WorkflowWorldObservation(
            argv: pathHint.map { ["grep", pattern, $0] } ?? ["grep", pattern],
            exitStatus: 0,
            outputText: output.string,
            truncated: enumerationTruncated || scanTruncated || output.truncated)
    }

    private static let gitEnvironment: [String: String] = [
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/var/empty",
        "LC_ALL": "C",
        "PAGER": "cat",
        "PATH": "/usr/bin:/bin",
    ]

    /// `-c filter.<name>.{clean,smudge,process}=` for every filter driver the
    /// repository defines. A workspace from an untrusted archive can ship
    /// `.git/config` plus `.gitattributes` that run a clean filter during a
    /// "read-only" status or diff. The enumeration (`git config --get-regexp`)
    /// executes nothing. An empty command runs nothing, and `-c` wins over the
    /// file config.
    private func filterNeutralizingOverrides(root: URL) async throws -> [String] {
        let probe = WorkflowWorldProcessInvocation(
            executableURL: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: ["config", "--get-regexp", "^filter\\."],
            workingDirectoryURL: root,
            environment: Self.gitEnvironment,
            timeoutSeconds: limits.gitTimeoutSeconds,
            outputCapBytes: limits.maximumOutputBytes)
        let observation = try await executor.execute(probe)
        return Self.filterNeutralizingOverrides(configOutput: observation.outputText)
    }

    static func filterNeutralizingOverrides(configOutput: String) -> [String] {
        var names: [String] = []
        for line in configOutput.split(separator: "\n") {
            guard line.hasPrefix("filter.") else { continue }
            let key = line.split(separator: " ", maxSplits: 1)[0]
            let afterPrefix = key.dropFirst("filter.".count)
            guard let lastDot = afterPrefix.lastIndex(of: "."), lastDot != afterPrefix.startIndex
            else { continue }
            let name = String(afterPrefix[afterPrefix.startIndex..<lastDot])
            if !names.contains(name) { names.append(name) }
        }
        return names.flatMap { name in
            ["clean", "smudge", "process"].flatMap { ["-c", "filter.\(name).\($0)="] }
        }
    }

    private func git(
        _ operation: WorkflowGitReadOp,
        site: WorkflowSiteKey
    ) async throws -> WorkflowWorldObservation {
        let root = try canonicalRoot(site: site)
        guard limits.maximumOutputBytes > 0,
              limits.gitTimeoutSeconds > 0
        else {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "Workflow git limits must be positive.",
                site: site)
        }
        let arguments: [String]
        switch operation {
        case .status:
            arguments = [
                "--no-pager", "-c", "core.fsmonitor=false",
                "status", "--short", "--untracked-files=all", "--no-renames",
            ]
        case .diff:
            arguments = [
                "--no-pager", "-c", "core.fsmonitor=false",
                "diff", "--no-ext-diff", "--no-textconv", "--no-color",
            ]
        case .log:
            arguments = [
                "--no-pager", "-c", "core.fsmonitor=false",
                "-c", "log.showSignature=false",
                "log", "--no-decorate", "--no-color", "-n", "20", "--format=%h %s", "--",
            ]
        case .changedFiles:
            arguments = [
                "--no-pager", "-c", "core.fsmonitor=false",
                "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames",
            ]
        }
        // Status and diff run clean filters on stale-stat files, and log may
        // run gpg.program; both are repo-configured programs (see
        // filterNeutralizingOverrides). Neutralize them before the read.
        let overrides = try await filterNeutralizingOverrides(root: root)
        let invocation = WorkflowWorldProcessInvocation(
            executableURL: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: overrides + arguments,
            workingDirectoryURL: root,
            environment: Self.gitEnvironment,
            timeoutSeconds: limits.gitTimeoutSeconds,
            outputCapBytes: limits.maximumOutputBytes,
            stdoutOnly: operation == .changedFiles)
        let observation = try await executor.execute(invocation)
        guard operation == .changedFiles else { return observation }
        return try changedFilesObservation(from: observation, site: site)
    }

    private static func utf8Text(_ data: Data, allowingPartialTail: Bool) -> String? {
        if let text = String(data: data, encoding: .utf8) { return text }
        guard allowingPartialTail, let last = data.last else { return nil }
        // Only a valid multibyte lead followed by too few continuation bytes may be trimmed.
        var leadIndex = data.count - 1
        while leadIndex > 0, data[leadIndex] & 0xC0 == 0x80 { leadIndex -= 1 }
        let lead = data[leadIndex]
        let expectedLength: Int
        switch lead {
        case 0xC2...0xDF: expectedLength = 2
        case 0xE0...0xEF: expectedLength = 3
        case 0xF0...0xF4: expectedLength = 4
        default: return nil
        }
        guard data.count - leadIndex < expectedLength,
              last == lead || last & 0xC0 == 0x80 else { return nil }
        if data.count - leadIndex > 1 {
            let second = data[leadIndex + 1]
            // These leads constrain the next byte even before the scalar is complete.
            switch lead {
            case 0xE0 where second < 0xA0,
                 0xED where second > 0x9F,
                 0xF0 where second < 0x90,
                 0xF4 where second > 0x8F:
                return nil
            default:
                break
            }
        }
        return String(data: data.prefix(leadIndex), encoding: .utf8)
    }

    private func changedFilesObservation(
        from observation: WorkflowWorldObservation,
        site: WorkflowSiteKey
    ) throws -> WorkflowWorldObservation {
        let data = Data(observation.outputText.utf8)
        let records = data.split(separator: 0, omittingEmptySubsequences: true)
        let hasPartialRecord = !data.isEmpty && data.last != 0
        var output = BoundedText(maximumBytes: limits.maximumOutputBytes)
        let completeRecords = hasPartialRecord ? Array(records.dropLast()) : records
        // A truncated status output may end mid-path, so only complete NUL records are parsed.
        for record in completeRecords {
            guard record.count >= 4 else {
                throw refusal("git returned a malformed changed-file record", site: site)
            }
            let path = String(decoding: record.dropFirst(3), as: UTF8.self)
            if !output.append(path, separator: true) { break }
        }
        return WorkflowWorldObservation(
            argv: observation.argv,
            exitStatus: observation.exitStatus,
            outputText: output.string,
            truncated: observation.truncated || output.truncated)
    }

    /// Directory names a root walk never descends into: VCS and build
    /// metadata spend the entry cap and scan budget before any source file.
    private static let skippedDescendantNames: Set<String> = [".git", ".build", "node_modules"]

    private func entries(
        under directory: URL,
        site: WorkflowSiteKey,
        skippingMetadata: Bool = false
    ) throws -> DirectoryEnumeration {
        guard limits.maximumDirectoryEntries > 0 else {
            throw WorkflowError(
                kind: .resourceLimit,
                message: "Workflow directory reads require a positive entry limit.",
                site: site)
        }
        let keys: [URLResourceKey] = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [])
        else {
            throw refusal("the workspace directory could not be enumerated", site: site)
        }
        var result: [URL] = []
        var truncated = false
        while let candidate = enumerator.nextObject() as? URL {
            if skippingMetadata,
               Self.skippedDescendantNames.contains(candidate.lastPathComponent) {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir),
                   isDir.boolValue {
                    enumerator.skipDescendants()
                    continue
                }
            }
            if result.count >= limits.maximumDirectoryEntries {
                truncated = true
                break
            }
            result.append(candidate)
        }
        return DirectoryEnumeration(entries: result, truncated: truncated)
    }

    private func canonicalRoot(site: WorkflowSiteKey) throws -> URL {
        let path = PathContainment.canonical(workspaceRoot).path
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue,
              let resolvedPath = path.withCString({ realpath($0, nil) })
        else {
            throw refusal("the declared workspace root is unavailable", site: site)
        }
        defer { free(resolvedPath) }
        return PathContainment.canonical(
            URL(fileURLWithPath: String(cString: resolvedPath), isDirectory: true))
    }

    private func resolve(
        _ path: String,
        under root: URL,
        allowsDirectory: Bool,
        site: WorkflowSiteKey
    ) throws -> URL {
        try validateRelativePath(path, site: site)
        let candidate = URL(fileURLWithPath: path, relativeTo: root).standardizedFileURL
        let resolved = PathContainment.canonical(candidate)
        guard PathContainment.isContained(resolved, in: root) else {
            throw refusal("the requested path resolves outside the workspace", site: site)
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDirectory) else {
            throw refusal("the requested workspace path does not exist", site: site)
        }
        let isRegularFile = try resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
        guard (allowsDirectory || !isDirectory.boolValue),
              (isDirectory.boolValue || isRegularFile)
        else {
            throw refusal("the requested path is not a supported workspace file", site: site)
        }
        return resolved
    }

    private func validateRelativePath(_ path: String, site: WorkflowSiteKey) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.utf8.contains(0),
              !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else {
            throw refusal("workspace paths must be non-empty, relative, and traversal-free", site: site)
        }
    }

    private func validateRelativePattern(_ pattern: String, site: WorkflowSiteKey) throws {
        let components = pattern.split(separator: "/", omittingEmptySubsequences: false)
        guard pattern.utf8.count <= limits.maximumPatternBytes,
              !pattern.isEmpty,
              !pattern.hasPrefix("/"),
              !pattern.utf8.contains(0),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else {
            throw refusal("glob patterns must remain relative to the workspace", site: site)
        }
    }

    private func globExpression(
        _ pattern: String,
        site: WorkflowSiteKey
    ) throws -> NSRegularExpression {
        let characters = Array(pattern)
        var expression = "^"
        var index = 0
        while index < characters.count {
            switch characters[index] {
            case "*":
                if index + 1 < characters.count, characters[index + 1] == "*" {
                    if index + 2 < characters.count, characters[index + 2] == "/" {
                        expression += "(?:.*/)?"
                        index += 3
                    } else {
                        expression += ".*"
                        index += 2
                    }
                } else {
                    expression += "[^/]*"
                    index += 1
                }
            case "?":
                expression += "[^/]"
                index += 1
            case "[", "]", "\\":
                throw WorkflowError(
                    kind: .validation,
                    message: "Glob reads support only the *, **, and ? pattern forms.",
                    site: site)
            default:
                expression += NSRegularExpression.escapedPattern(for: String(characters[index]))
                index += 1
            }
        }
        expression += "$"
        do {
            return try NSRegularExpression(pattern: expression)
        } catch {
            throw WorkflowError(kind: .validation, message: "The glob pattern is invalid.", site: site)
        }
    }

    private func relativePath(
        _ candidate: URL,
        from root: URL,
        site: WorkflowSiteKey
    ) throws -> String {
        if candidate.path == root.path { return "" }
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path.hasPrefix(rootPath) else {
            throw refusal("a workspace result is not beneath the declared root", site: site)
        }
        return String(candidate.path.dropFirst(rootPath.count))
    }

    private func refusal(_ reason: String, site: WorkflowSiteKey?) -> WorkflowError {
        WorkflowError(
            kind: .sandboxRefusal,
            message: "Workflow world read refused: \(reason).",
            site: site)
    }
}
