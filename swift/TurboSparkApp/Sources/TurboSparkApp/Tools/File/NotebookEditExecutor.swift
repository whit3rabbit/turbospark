import Foundation

/// Executor for editing Jupyter notebooks (.ipynb JSON format).
public enum NotebookEditExecutor {
    public static func execute(arguments: [String: String], rootURL: URL) async throws -> String {
        guard let relPath = arguments["notebook_path"] ?? arguments["notebookPath"] ?? arguments["path"] ?? arguments["file_path"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 40,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'notebook_path' argument for NotebookEdit."]
            )
        }
        let cellId = arguments["cell_id"] ?? arguments["cellId"]
        let cellIndexStr = arguments["cell_index"] ?? arguments["cellIndex"] ?? arguments["index"]
        let cellIndex = cellIndexStr.flatMap { Int($0) }
        let newSource = arguments["new_source"] ?? arguments["newSource"] ?? arguments["content"] ?? arguments["source"] ?? ""
        let cellType = arguments["cell_type"] ?? arguments["cellType"] ?? "code"
        let editMode = arguments["edit_mode"] ?? arguments["editMode"] ?? arguments["action"] ?? "replace"

        let targetURL = try AppToolRegistry.resolveSecurePath(relPath: relPath, rootURL: rootURL)
        try AppToolSandbox.validateWritePath(targetURL, rootURL: rootURL)

        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 40,
                userInfo: [NSLocalizedDescriptionKey: "Notebook file not found at '\(relPath)'."]
            )
        }

        let data = try Data(contentsOf: targetURL)
        guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              var cells = json["cells"] as? [[String: Any]] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 40,
                userInfo: [NSLocalizedDescriptionKey: "Invalid Jupyter notebook format in '\(relPath)'."]
            )
        }

        let sourceLines = newSource.components(separatedBy: "\n").enumerated().map { (i, line) in
            i == newSource.components(separatedBy: "\n").count - 1 ? line : line + "\n"
        }

        var modified = false
        var oldContent: String?

        if editMode.lowercased() == "insert" {
            var newCell: [String: Any] = [
                "cell_type": cellType,
                "metadata": [String: Any](),
                "source": sourceLines
            ]
            if let cellId {
                newCell["id"] = cellId
            } else {
                newCell["id"] = UUID().uuidString.prefix(8).lowercased()
            }
            if cellType == "code" {
                newCell["execution_count"] = NSNull()
                newCell["outputs"] = [Any]()
            }
            if let idx = cellIndex, idx >= 0 && idx <= cells.count {
                cells.insert(newCell, at: idx)
            } else {
                cells.append(newCell)
            }
            modified = true
        } else if editMode.lowercased() == "delete" {
            if let cellId {
                if let idx = cells.firstIndex(where: { ($0["id"] as? String) == cellId }) {
                    cells.remove(at: idx)
                    modified = true
                }
            } else if let idx = cellIndex {
                guard idx >= 0 && idx < cells.count else {
                    throw NSError(
                        domain: "TurboSparkTool",
                        code: 40,
                        userInfo: [NSLocalizedDescriptionKey: "Cell index \(idx) is out of bounds (total cells: \(cells.count))."]
                    )
                }
                cells.remove(at: idx)
                modified = true
            } else if !cells.isEmpty {
                cells.removeLast()
                modified = true
            }
        } else {
            // Replace mode
            var targetIndex: Int?
            if let cellId {
                targetIndex = cells.firstIndex(where: { ($0["id"] as? String) == cellId })
            } else if let idx = cellIndex {
                guard idx >= 0 && idx < cells.count else {
                    throw NSError(
                        domain: "TurboSparkTool",
                        code: 40,
                        userInfo: [NSLocalizedDescriptionKey: "Cell index \(idx) is out of bounds (total cells: \(cells.count))."]
                    )
                }
                targetIndex = idx
            } else if !cells.isEmpty {
                targetIndex = 0
            }

            if let idx = targetIndex, idx < cells.count {
                var cell = cells[idx]
                if let prevSource = cell["source"] as? [String] {
                    oldContent = prevSource.joined()
                } else if let prevSourceStr = cell["source"] as? String {
                    oldContent = prevSourceStr
                }
                cell["source"] = sourceLines
                cell["cell_type"] = cellType
                if cellType == "code" {
                    cell["outputs"] = [Any]()
                    cell["execution_count"] = NSNull()
                }
                cells[idx] = cell
                modified = true
            } else {
                // If not found, append cell
                var newCell: [String: Any] = [
                    "cell_type": cellType,
                    "metadata": [String: Any](),
                    "source": sourceLines,
                    "id": cellId ?? UUID().uuidString.prefix(8).lowercased()
                ]
                if cellType == "code" {
                    newCell["execution_count"] = NSNull()
                    newCell["outputs"] = [Any]()
                }
                cells.append(newCell)
                modified = true
            }
        }

        guard modified else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 40,
                userInfo: [NSLocalizedDescriptionKey: "Failed to apply edit mode '\(editMode)' to notebook '\(relPath)'."]
            )
        }

        json["cells"] = cells
        let outData = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try outData.write(to: targetURL, options: .atomic)

        await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: String(decoding: outData, as: UTF8.self))

        var summary = "Successfully updated notebook '\(relPath)' (mode: \(editMode), total cells: \(cells.count))."
        if let old = oldContent {
            summary += "\nReplaced cell content of \(old.count) bytes with \(newSource.count) bytes."
        }
        return summary
    }
}
