import Foundation

/// Schema for the `document_write` tool (ZP-09).
public enum DocumentWriteToolDefinitions {
    public static let documentWrite = OpenAITool.function(
        name: "document_write",
        description: """
            Create a new .docx, .xlsx, .pptx or .pdf file in the project from a JSON spec. \
            Native writer: no Word, Excel, PowerPoint, Python or Node is involved. It never \
            overwrites a file and never writes macros, formulas, charts or external links. The \
            file is checked after writing (valid package, expected parts, readable text) and is \
            only kept if the check passes. Load the matching bundled skill (docx, xlsx, pptx, \
            pdf) first: it lists the spec shape and limits.
            """,
        parameters: .object(
            properties: [
                "path": .string(description: "Project-relative path for the new file. The extension picks the format: .docx, .xlsx, .pptx or .pdf. Must not exist."),
                "spec": .string(description: """
                    JSON object. docx and pdf: {"title","page":"letter"|"a4","blocks":[{"type":"heading","level":1,"text"} | {"type":"paragraph","text"|"runs":[{"text","bold","italic","underline","color":"RRGGBB","size"}],"align"} | {"type":"list","ordered":false,"items":["..."]} | {"type":"table","header":["..."],"rows":[["..."]]} | {"type":"image","path":"assets/a.png","width_pt":300,"alt"} | {"type":"page_break"}]}. \
                    xlsx: {"sheets":[{"name","columns":[widths],"freeze_rows":1,"rows":[[cell,...]]}]} where a cell is a string, number, boolean, null, or {"v","type":"date","bold","italic","color","fill","format","align"}. \
                    pptx: {"slides":[{"layout":"title"|"content","title","subtitle","bullets":["..." | {"text","level":0-2}],"paragraphs":["..."]}]}.
                    """),
            ],
            required: ["path", "spec"]
        )
    )

    public static let all: [OpenAITool] = [documentWrite]
}

/// The `document_write` executor: validate the path and spec, build the file
/// in memory, write it to a private temp directory, run the postcheck there,
/// and only then move it into the project.
///
/// **Ladder position:** category `.fileWrite`, workspace-rooted (needs a
/// project), so it answers to the project's File write permission and the
/// sensitive-path hard gate exactly as `write_file` does (`AppToolCatalog`,
/// `ToolRiskClassifier`). It never overwrites: a binary file the agent cannot
/// have "read" must not be silently replaced, and the ordinary
/// read-before-write gate has no meaning for it. Choose a new path.
public enum DocumentWriteExecutor {
    public struct Result: Sendable {
        public let report: String
        public let url: URL
    }

    private static func error(_ code: Int, _ message: String) -> NSError {
        NSError(domain: "TurboSparkDocumentWrite", code: code, userInfo: [NSLocalizedDescriptionKey: message])
    }

    /// The destination, validated before anything is built.
    static func resolveDestination(path: String, rootURL: URL) throws -> (url: URL, format: DocumentFormat) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw error(1, "document_write needs a 'path' for the new file.") }
        let target = try AppToolRegistry.resolveSecurePath(relPath: trimmed, rootURL: rootURL)
        try AppToolSandbox.validateWritePath(target, rootURL: rootURL)
        guard target.path != rootURL.standardizedFileURL.resolvingSymlinksInPath().path else {
            throw error(2, "The destination must be a file path inside the project.")
        }
        if ToolRiskClassifier.isSensitivePath(trimmed) || ToolRiskClassifier.isSensitivePath(target.path) {
            throw error(3, "Refusing to write a document to a sensitive path: '\(trimmed)'.")
        }
        let ext = target.pathExtension.lowercased()
        if ["docm", "xlsm", "pptm", "dotm", "xltm", "potm"].contains(ext) {
            throw error(4, "Macro-enabled formats are never written. Use .docx, .xlsx or .pptx.")
        }
        guard let format = DocumentFormat.forExtension(ext) else {
            throw error(5, "The path must end in .docx, .xlsx, .pptx or .pdf: '\(trimmed)'.")
        }
        if FileManager.default.fileExists(atPath: target.path) {
            throw error(6, "'\(trimmed)' already exists. document_write never overwrites; choose a new path.")
        }
        return (target, format)
    }

    /// Reads one image from the project: the path is resolved (symlinks
    /// included) and must stay inside the root, the target must be a regular
    /// file, not a sensitive path, PNG or JPEG by content, bounded.
    static func loadImage(
        path: String, rootURL: URL, limits: DocumentWriteLimits
    ) throws -> DocumentImage {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        let url = try AppToolRegistry.resolveSecurePath(relPath: trimmed, rootURL: rootURL)
        if ToolRiskClassifier.isSensitivePath(trimmed) || ToolRiskClassifier.isSensitivePath(url.path) {
            throw error(7, "Refusing to read a sensitive path as an image: '\(trimmed)'.")
        }
        var info = stat()
        guard lstat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
            throw error(8, "Image '\(trimmed)' is not a regular file in the project.")
        }
        guard Int64(info.st_size) <= Int64(limits.maximumImageBytes) else {
            throw error(9, "Image '\(trimmed)' is over \(limits.maximumImageBytes / (1_024 * 1_024)) MiB.")
        }
        guard let data = try? Data(contentsOf: url) else {
            throw error(8, "Image '\(trimmed)' could not be read.")
        }
        return try DocumentImage.make(data: data, limits: limits)
    }

    private static func imagePaths(in spec: DocumentSpec) -> [String] {
        var seen: [String] = []
        for block in spec.blocks {
            if case .image(let reference) = block, !seen.contains(reference.path) { seen.append(reference.path) }
        }
        return seen
    }

    static func build(
        spec: DocumentSpec, format: DocumentFormat, images: [String: DocumentImage],
        limits: DocumentWriteLimits
    ) throws -> Data {
        switch format {
        case .docx: return try DocxWriter.write(spec: spec, images: images, limits: limits)
        case .xlsx: return try XlsxWriter.write(spec: spec, limits: limits)
        case .pptx: return try PptxWriter.write(spec: spec, limits: limits)
        case .pdf: return try PdfWriter.write(spec: spec, images: images, limits: limits)
        }
    }

    static func execute(
        arguments: [String: String], rootURL: URL, limits: DocumentWriteLimits = .default,
        postcheck: (URL, DocumentExpectations) throws -> DocumentPostcheckReport = DocumentPostcheck.run
    ) throws -> Result {
        guard let path = arguments["path"] ?? arguments["file_path"] ?? arguments["filePath"] else {
            throw error(1, "Missing 'path' argument for document_write.")
        }
        guard let rawSpec = arguments["spec"] ?? arguments["spec_json"] ?? arguments["content"] else {
            throw error(1, "Missing 'spec' argument for document_write (a JSON object as a string).")
        }
        let destination = try resolveDestination(path: path, rootURL: rootURL)
        let spec = try DocumentSpec.decode(rawSpec, limits: limits)
        try DocumentSpecValidation.validate(spec, format: destination.format, limits: limits)

        var images: [String: DocumentImage] = [:]
        for imagePath in imagePaths(in: spec) {
            images[imagePath] = try loadImage(path: imagePath, rootURL: rootURL, limits: limits)
        }
        let bytes = try build(spec: spec, format: destination.format, images: images, limits: limits)

        // The file is built and checked in a private temp directory, so a
        // failed postcheck leaves nothing in the project.
        let work = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-docwrite-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: work, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: work) }
        let staged = work.appendingPathComponent("out." + destination.format.rawValue)
        try bytes.write(to: staged, options: [.atomic])

        let report = try postcheck(
            staged, DocumentExpectations(spec: spec, format: destination.format))

        try FileManager.default.createDirectory(
            at: destination.url.deletingLastPathComponent(), withIntermediateDirectories: true)
        // `moveItem` fails when the destination exists, which closes the race
        // between the existence check above and this move.
        do {
            try FileManager.default.moveItem(at: staged, to: destination.url)
        } catch {
            throw self.error(10, "Could not place the file at '\(path)': it may have been created meanwhile.")
        }

        let shown = path.trimmingCharacters(in: .whitespacesAndNewlines)
        var lines = [
            "Wrote \(shown) (\(bytes.count) bytes) and its postcheck passed: \(report.passed.joined(separator: ", ")).",
        ]
        if destination.format == .pdf {
            lines.append("\(report.pageCount) page(s), \(report.characterCount) extractable characters.")
        } else {
            lines.append("\(report.partCount) package parts, \(report.characterCount) readable characters.")
        }
        lines.append("The postcheck proves the file is a well-formed package the app's own viewer reads; it does not prove Word, Excel or PowerPoint opens it. Tell the user it is ready to review in the file viewer.")
        return Result(report: lines.joined(separator: "\n"), url: destination.url)
    }
}
