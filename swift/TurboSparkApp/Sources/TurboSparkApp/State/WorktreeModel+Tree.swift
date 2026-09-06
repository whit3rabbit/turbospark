import Foundation

/// Node representing either a directory folder or a leaf file in the changed files tree.
public final class WorktreeTreeNode: Identifiable, ObservableObject {
    public let id: String
    public let name: String
    public let relativePath: String
    public let isDirectory: Bool
    public var children: [WorktreeTreeNode]
    public var fileChange: WorktreeFileChange?
    public var additions: Int
    public var deletions: Int

    public init(
        id: String,
        name: String,
        relativePath: String,
        isDirectory: Bool,
        children: [WorktreeTreeNode] = [],
        fileChange: WorktreeFileChange? = nil,
        additions: Int = 0,
        deletions: Int = 0
    ) {
        self.id = id
        self.name = name
        self.relativePath = relativePath
        self.isDirectory = isDirectory
        self.children = children
        self.fileChange = fileChange
        self.additions = additions
        self.deletions = deletions
    }
}

extension WorktreeModel {
    /// Builds a directory tree hierarchy from a flat list of file changes.
    nonisolated static func buildTree(from changes: [WorktreeFileChange]) -> [WorktreeTreeNode] {
        class TempDir {
            let name: String
            let path: String
            var subdirs: [String: TempDir] = [:]
            var files: [WorktreeFileChange] = []

            init(name: String, path: String) {
                self.name = name
                self.path = path
            }
        }

        let root = TempDir(name: "", path: "")

        for file in changes {
            let parts = file.relativePath.split(separator: "/").map(String.init)
            if parts.count <= 1 {
                root.files.append(file)
            } else {
                var current = root
                var currentPath = ""
                for dirPart in parts.dropLast() {
                    currentPath = currentPath.isEmpty ? dirPart : "\(currentPath)/\(dirPart)"
                    if let existing = current.subdirs[dirPart] {
                        current = existing
                    } else {
                        let newDir = TempDir(name: dirPart, path: currentPath)
                        current.subdirs[dirPart] = newDir
                        current = newDir
                    }
                }
                current.files.append(file)
            }
        }

        func convert(temp: TempDir) -> [WorktreeTreeNode] {
            var result: [WorktreeTreeNode] = []

            let sortedSubdirs = temp.subdirs.values.sorted { $0.name < $1.name }
            for subdir in sortedSubdirs {
                let children = convert(temp: subdir)
                let adds = children.reduce(0) { $0 + $1.additions }
                let dels = children.reduce(0) { $0 + $1.deletions }
                let dirNode = WorktreeTreeNode(
                    id: "dir:\(subdir.path)",
                    name: subdir.name,
                    relativePath: subdir.path,
                    isDirectory: true,
                    children: children,
                    fileChange: nil,
                    additions: adds,
                    deletions: dels
                )
                result.append(dirNode)
            }

            let sortedFiles = temp.files.sorted { $0.fileName < $1.fileName }
            for file in sortedFiles {
                let fileNode = WorktreeTreeNode(
                    id: "file:\(file.relativePath)",
                    name: file.fileName,
                    relativePath: file.relativePath,
                    isDirectory: false,
                    children: [],
                    fileChange: file,
                    additions: file.additions,
                    deletions: file.deletions
                )
                result.append(fileNode)
            }

            return result
        }

        return convert(temp: root)
    }

    /// Pure parser for git worktree list --porcelain output.
    nonisolated static func parseWorktreeListOutput(_ stdout: String, currentRoot: String) -> [GitWorktreeInfo] {
        var worktrees: [GitWorktreeInfo] = []
        let blocks = stdout.components(separatedBy: "\n\n")
        let canonicalCurrent = URL(fileURLWithPath: currentRoot).resolvingSymlinksInPath().path

        for block in blocks {
            var path = ""
            var head = ""
            var branch = ""
            for line in block.components(separatedBy: .newlines) {
                if line.hasPrefix("worktree ") {
                    path = String(line.dropFirst(9)).trimmingCharacters(in: .whitespaces)
                } else if line.hasPrefix("HEAD ") {
                    head = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
                } else if line.hasPrefix("branch ") {
                    let fullBranch = String(line.dropFirst(7)).trimmingCharacters(in: .whitespaces)
                    branch = fullBranch.replacingOccurrences(of: "refs/heads/", with: "")
                }
            }
            if !path.isEmpty {
                let canonicalPath = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
                worktrees.append(
                    GitWorktreeInfo(
                        path: path,
                        head: String(head.prefix(7)),
                        branch: branch.isEmpty ? "HEAD" : branch,
                        isCurrent: canonicalPath == canonicalCurrent
                    )
                )
            }
        }
        return worktrees
    }

    /// Parses combined numstat and name-status outputs into a sorted list of WorktreeFileChange.
    nonisolated static func parseNameStatusAndNumstat(
        numstatOutput: String,
        nameStatusOutput: String
    ) -> [WorktreeFileChange] {
        var statMap: [String: (adds: Int, dels: Int)] = [:]
        for line in numstatOutput.components(separatedBy: .newlines) {
            let parts = line.split(separator: "\t")
            if parts.count >= 3 {
                let adds = Int(parts[0]) ?? 0
                let dels = Int(parts[1]) ?? 0
                statMap[Self.numstatPath(String(parts[2]))] = (adds, dels)
            }
        }

        var changes: [WorktreeFileChange] = []
        for line in nameStatusOutput.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let parts = trimmed.split(separator: "\t")
            guard parts.count >= 2 else { continue }
            let statusCode = String(parts[0])
            let rawPath = String(parts.last ?? "")
            let filePath = Self.porcelainPath(rawPath)
            guard !filePath.isEmpty else { continue }

            let status: WorktreeFileChange.Status
            if statusCode.hasPrefix("A") {
                status = .added
            } else if statusCode.hasPrefix("D") {
                status = .deleted
            } else if statusCode.hasPrefix("R") {
                status = .renamed
            } else if statusCode.hasPrefix("M") {
                status = .modified
            } else {
                status = .unknown
            }

            let stats = statMap[filePath] ?? (0, 0)
            changes.append(
                WorktreeFileChange(
                    relativePath: filePath,
                    status: status,
                    additions: stats.adds,
                    deletions: stats.dels
                )
            )
        }

        changes.sort { lhs, rhs in
            if lhs.directoryPath != rhs.directoryPath {
                return lhs.directoryPath < rhs.directoryPath
            }
            return lhs.fileName < rhs.fileName
        }
        return changes
    }
}
