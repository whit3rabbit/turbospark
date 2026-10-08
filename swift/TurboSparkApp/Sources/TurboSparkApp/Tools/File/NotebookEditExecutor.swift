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

        func fail(_ message: String) -> NSError {
            NSError(domain: "TurboSparkTool", code: 40, userInfo: [NSLocalizedDescriptionKey: message])
        }
        // Resolves a cell by id, accepting the `cell-N` index spelling for
        // notebooks (nbformat < 4.5) whose cells carry no ids.
        func indexOfCell(id: String) -> Int? {
            if let idx = cells.firstIndex(where: { ($0["id"] as? String) == id }) { return idx }
            if id.hasPrefix("cell-"), let n = Int(id.dropFirst(5)), n >= 0, n < cells.count { return n }
            return nil
        }
        func freshCellID() -> String {
            let existing = Set(cells.compactMap { $0["id"] as? String })
            while true {
                let candidate = String(UUID().uuidString.prefix(8)).lowercased()
                if !existing.contains(candidate) { return candidate }
            }
        }
        // Replace and delete must be told WHICH cell. Defaulting to the first
        // (replace) or last (delete) cell silently destroyed work, and an id
        // that matched nothing used to append a duplicate cell and report
        // success.
        func requireTargetIndex() throws -> Int {
            if let cellId {
                guard let idx = indexOfCell(id: cellId) else {
                    throw fail("No cell with id '\(cellId)' in '\(relPath)' (total cells: \(cells.count)).")
                }
                return idx
            }
            if let idx = cellIndex {
                guard idx >= 0 && idx < cells.count else {
                    throw fail("Cell index \(idx) is out of bounds (total cells: \(cells.count)).")
                }
                return idx
            }
            throw fail("\(editMode) needs `cell_id` or `cell_index` to say which cell to change.")
        }

        if editMode.lowercased() == "insert" {
            var newCell: [String: Any] = [
                "cell_type": cellType,
                "metadata": [String: Any](),
                "source": sourceLines,
                // Never reuse the anchor's id: duplicate ids made later edits
                // by id hit the wrong cell.
                "id": freshCellID()
            ]
            if cellType == "code" {
                newCell["execution_count"] = NSNull()
                newCell["outputs"] = [Any]()
            }
            if let cellId {
                // Claude Code meaning: insert AFTER the cell with this id.
                guard let anchor = indexOfCell(id: cellId) else {
                    throw fail("No cell with id '\(cellId)' in '\(relPath)' to insert after (total cells: \(cells.count)).")
                }
                cells.insert(newCell, at: anchor + 1)
            } else if let idx = cellIndex, idx >= 0 && idx <= cells.count {
                cells.insert(newCell, at: idx)
            } else {
                cells.append(newCell)
            }
            modified = true
        } else if editMode.lowercased() == "delete" {
            cells.remove(at: try requireTargetIndex())
            modified = true
        } else {
            // Replace mode
            let idx = try requireTargetIndex()
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
