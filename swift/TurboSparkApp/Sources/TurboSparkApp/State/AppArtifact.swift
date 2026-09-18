import Foundation
import TurboSpark

/// The small request envelope retained with a saved generated image.
public struct AppImageRequest: Codable, Equatable, Sendable {
    public let prompt: String
    public let seed: UInt64
    public let width: UInt32
    public let height: UInt32
    public let steps: UInt32

    public init(options: ImageGenerateOptions) {
        prompt = options.prompt
        seed = options.seed
        width = options.width
        height = options.height
        steps = options.steps
    }

    public var options: ImageGenerateOptions {
        ImageGenerateOptions(
            prompt: prompt, seed: seed, width: width, height: height, steps: steps)
    }
}

/// How the artifact panel should render a file.
///
/// A NEW enum rather than a reuse of `AppPromptAttachment.PreviewKind`, and
/// the difference is not cosmetic. That one answers "how do we show the user
/// what the MODEL sees", where `.text` means render `extractedText` -- a
/// field an artifact has no analogue of, because nothing extracted anything.
/// It also folds "the file is missing" into `.text`, which is exactly the
/// conflation this panel must not make: a missing file has to SAY so, not
/// render as an empty text view (`swift/CLAUDE.md` Gotcha 23).
public enum AppArtifactRenderKind: String, Codable, Sendable {
    case markdown
    /// Rendered by the sandboxed web panel (`ArtifactWebView`), never by
    /// loading the file into anything with broader reach.
    case html
    case text
    case image
    case pdf
    /// Renderable by nothing here: docx, xlsx, zip. The panel offers the
    /// external opener instead, which is the whole point of the case.
    case opaque

    /// The extensions this app renders as markdown.
    ///
    /// ONE list with ONE reader, for `AppFileSymbol`'s reason. It is also the
    /// artifact policy's own markdown test (`ArtifactRegistrar`), so a format
    /// added here becomes registrable and renderable in the same edit.
    public static let markdownExtensions: Set<String> = ["md", "markdown", "mdx"]

    /// The extensions the web panel renders. Read by the artifact policy the
    /// same way `markdownExtensions` is: registrable and renderable is one
    /// edit, never two.
    public static let htmlExtensions: Set<String> = ["html", "htm"]

    /// Extensions shown as monospaced source rather than as prose.
    public static let plainTextExtensions: Set<String> = [
        "txt", "log", "json", "yaml", "yml", "toml", "csv", "xml", "ini",
        "swift", "rs", "py", "c", "cpp", "h", "hpp", "js", "ts", "tsx",
        "go", "rb", "sh", "zsh", "sql", "css", "metal",
    ]

    /// Derived from the file name, never restated per call site (Gotcha 22).
    ///
    /// The image set comes from `AppPromptAttachment.imageFileExtensions`
    /// rather than a second list: two answers to "is this a picture" is how
    /// the attachment chip and the artifact card come to disagree.
    public static func forFileName(_ fileName: String) -> AppArtifactRenderKind {
        let ext = (fileName as NSString).pathExtension.lowercased()
        if markdownExtensions.contains(ext) { return .markdown }
        if htmlExtensions.contains(ext) { return .html }
        if ext == "pdf" { return .pdf }
        if AppPromptAttachment.imageFileExtensions.contains(ext) { return .image }
        if plainTextExtensions.contains(ext) { return .text }
        return .opaque
    }
}

/// One file a chat produced, and the panel that renders it.
///
/// **Paths and metadata, never bytes.** `persistChats()` re-encodes the whole
/// archive on every keystroke of the draft, so a 20 KB plan stored inline
/// would be paid per character typed (`swift/CLAUDE.md` Gotcha 13, which
/// `AppChatMessage.imagePaths` already states for pictures).
///
/// `path` is nil for exactly one case: a ghost chat's plan, whose text is
/// sealed in `GhostChatVault` and never written to disk. That is the user's
/// stated requirement rather than an accident of the model, and it is why
/// every reader here treats a nil path as "text-backed" and not as "broken".
public struct AppArtifact: Identifiable, Codable, Equatable, Sendable {
    /// What produced the file. Drives the card's wording and, for `.plan`,
    /// whether the bytes may be synthesized at all.
    public enum Origin: String, Codable, Sendable {
        /// write_file, edit_file, apply_patch, notebook_edit.
        case fileWrite
        /// The model deliberately presented it through SendUserFile.
        case sentToUser
        /// ExitPlanMode finalized a plan.
        case plan
        /// Native image generation wrote the PNG.
        case imageGeneration
    }

    public var id: UUID
    /// The chat that produced it. Captured from the TURN, never from the
    /// selection: a call can be approved after the user switched chats
    /// (state#9, state#30).
    public var chatID: UUID
    /// Absolute and `.standardizedFileURL.path`, or nil when text-backed.
    public var path: String?
    public var title: String
    public var origin: Origin
    /// The transcript anchor. The inline card renders beside the tool call
    /// that produced the file, and `appendToolExecutionTurn` replaces the
    /// entry for the same `call.id` rather than minting a new one, so this
    /// survives the turn it was recorded in.
    public var toolCallID: UUID?
    public var createdAt: Date
    public var updatedAt: Date
    /// Bumped when the same path is written again. Also the panel's reload
    /// key: it moves once per tool call and never per streamed token, which
    /// is what keeps the markdown parse off the per-token path (Gotcha 16).
    public var revision: Int
    /// Last measured size. OPTIONAL because an absent measurement must not
    /// be presentable as a zero (Gotcha 23).
    public var lastKnownByteSize: Int?
    public var lastKnownModified: Date?
    /// The request that produced an image-generation artifact, when known.
    public var imageRequest: AppImageRequest?

    public init(
        id: UUID = UUID(),
        chatID: UUID,
        path: String?,
        title: String,
        origin: Origin,
        toolCallID: UUID? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        revision: Int = 1,
        lastKnownByteSize: Int? = nil,
        lastKnownModified: Date? = nil,
        imageRequest: AppImageRequest? = nil
    ) {
        self.id = id
        self.chatID = chatID
        self.path = path.map(AppArtifact.standardize)
        self.title = title
        self.origin = origin
        self.toolCallID = toolCallID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.revision = revision
        self.lastKnownByteSize = lastKnownByteSize
        self.lastKnownModified = lastKnownModified
        self.imageRequest = imageRequest
    }

    // MARK: - Identity

    /// The dedupe key: one artifact per (chat, standardized path).
    ///
    /// Standardizing is what stops `./docs/plan.md` and `docs/../docs/plan.md`
    /// becoming two rows for one file. No `try?` on the way: that combination
    /// is a compiler warning, since none of it throws (Gotcha 27).
    public static func standardize(_ path: String) -> String {
        URL(fileURLWithPath: path).standardizedFileURL.path
    }

    /// A text-backed artifact has no file and can never gain one.
    public var isTextBacked: Bool { path == nil }

    public var url: URL? { path.map { URL(fileURLWithPath: $0) } }

    public var fileName: String {
        guard let path else { return title }
        return (path as NSString).lastPathComponent
    }

    public var fileExtension: String {
        (fileName as NSString).pathExtension.lowercased()
    }

    public var renderKind: AppArtifactRenderKind {
        // A text-backed artifact is always the ghost plan, which is markdown.
        guard path != nil else { return .markdown }
        return AppArtifactRenderKind.forFileName(fileName)
    }

    public var symbolName: String {
        AppFileSymbol.name(
            forExtension: fileExtension,
            isImage: renderKind == .image)
    }

    /// Whether the file is still where it was written.
    ///
    /// DERIVED rather than stored, for the reason `AppPromptAttachment.isImage`
    /// carries: a stored flag is a new non-optional key on a `Codable` the
    /// archive already holds, and it would be wrong the moment anything moved
    /// the file from outside this app.
    public var existsOnDisk: Bool {
        guard let path else { return false }
        return FileManager.default.fileExists(atPath: path)
    }

    /// Whether the panel can show contents at all.
    public var hasReadableContent: Bool { isTextBacked || existsOnDisk }

    /// The panel and card can act on the file system only for a real file.
    public var canOpenExternally: Bool { existsOnDisk }

    // MARK: - Presentation

    /// The format word shown beside the name.
    public var formatLabel: String {
        switch renderKind {
        case .markdown: return "Markdown"
        case .html: return "HTML"
        case .pdf: return "PDF"
        case .image: return "Image"
        case .text: return fileExtension.isEmpty ? "Text" : fileExtension.uppercased()
        case .opaque: return fileExtension.isEmpty ? "File" : fileExtension.uppercased()
        }
    }

    /// The subtitle a card or panel header shows under the file name.
    ///
    /// A VALUE rather than inline view code, which is the only reason it can
    /// be tested at all (Gotcha 26's `ServerStatusRows` lesson, and
    /// `AppPromptAttachment.detailText` for the same reason one type over).
    ///
    /// **A missing file quotes no size.** `lastKnownByteSize` is a real
    /// measurement of a file that is no longer there, and printing it beside
    /// the name reads as a fact about what is on disk right now, which is
    /// precisely the zero-as-a-fact failure of Gotcha 23.
    public var detailText: String {
        if isTextBacked {
            return "\(formatLabel) • in memory"
        }
        guard existsOnDisk else {
            return "\(formatLabel) • file missing"
        }
        let size = MetricFormat.fileSize(lastKnownByteSize).map { " • \($0)" } ?? ""
        let revised = revision > 1 ? " • v\(revision)" : ""
        return "\(formatLabel)\(size)\(revised)"
    }

    /// The panel's reload key. `.task(id:)` on this reloads from disk exactly
    /// when the bytes could have changed, and never per streamed token.
    public var contentKey: String { "\(id.uuidString)-\(revision)" }

    // MARK: - Upsert

    /// Registers `incoming`, merging it into an existing row for the same
    /// file rather than appending a second one.
    ///
    /// **The dedupe key is (chatID, standardized path), and the surviving
    /// `id` is the EXISTING one.** That stability is load-bearing twice. It
    /// is what makes auto-open-once work at all -- the "already opened" set
    /// is keyed on the id, so a fresh id per write would re-pop the panel the
    /// user just closed on every rewrite, which is the exact failure the
    /// reference implementation documents. And it is what keeps the panel
    /// from resetting its own `@State` when the file it is showing is
    /// rewritten underneath it.
    ///
    /// A TEXT-BACKED artifact (a ghost chat's plan) has no path to key on and
    /// is always appended: two plans in one ghost chat are two plans.
    ///
    /// Returns the row as it now stands, which is what the caller needs for
    /// the auto-open decision.
    @discardableResult
    public static func upsert(
        _ incoming: AppArtifact, into artifacts: inout [AppArtifact]
    ) -> AppArtifact {
        guard let path = incoming.path,
              let index = artifacts.firstIndex(where: {
                  $0.chatID == incoming.chatID && $0.path == path
              })
        else {
            artifacts.append(incoming)
            return incoming
        }

        var merged = artifacts[index]
        merged.revision += 1
        merged.updatedAt = incoming.updatedAt
        merged.lastKnownByteSize = incoming.lastKnownByteSize
        merged.lastKnownModified = incoming.lastKnownModified
        merged.imageRequest = incoming.imageRequest ?? merged.imageRequest
        // The card follows the LATEST production point, so a file rewritten
        // three turns later is reachable from the turn that rewrote it.
        merged.toolCallID = incoming.toolCallID ?? merged.toolCallID
        if !incoming.title.isEmpty { merged.title = incoming.title }
        // `origin` is deliberately NOT overwritten: a file first presented
        // with SendUserFile and later edited is still the file the model
        // presented, and demoting it to `.fileWrite` would change the card's
        // wording under the user for no new information.
        artifacts[index] = merged
        return merged
    }

    // MARK: - Tolerant decoding

    private enum CodingKeys: String, CodingKey {
        case id, chatID, path, title, origin, toolCallID
        case createdAt, updatedAt, revision, lastKnownByteSize, lastKnownModified, imageRequest
    }

    /// Hand-written for Gotcha 13 and state#45.
    ///
    /// `origin` goes through `decodeTolerant`: a value a newer build wrote
    /// throws `dataCorrupted` on the synthesized decoder, which throws for
    /// the array, which throws for the chat, which quarantines every chat the
    /// user has. `.fileWrite` is the fallback because it is the only origin
    /// that claims nothing extra -- reading an unknown origin as `.plan`
    /// would put a file the app did not synthesize under plan-mode wording.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(UUID.self, forKey: .id)) ?? UUID()
        chatID = try container.decode(UUID.self, forKey: .chatID)
        path = try? container.decodeIfPresent(String.self, forKey: .path)
        title = (try? container.decodeIfPresent(String.self, forKey: .title)) ?? ""
        origin = container.decodeTolerant(Origin.self, forKey: .origin, fallback: .fileWrite)
        toolCallID = try? container.decodeIfPresent(UUID.self, forKey: .toolCallID)
        createdAt = (try? container.decodeIfPresent(Date.self, forKey: .createdAt)) ?? Date()
        updatedAt = (try? container.decodeIfPresent(Date.self, forKey: .updatedAt)) ?? createdAt
        revision = (try? container.decodeIfPresent(Int.self, forKey: .revision)) ?? 1
        lastKnownByteSize = try? container.decodeIfPresent(Int.self, forKey: .lastKnownByteSize)
        lastKnownModified = try? container.decodeIfPresent(Date.self, forKey: .lastKnownModified)
        imageRequest = try? container.decodeIfPresent(AppImageRequest.self, forKey: .imageRequest)
    }
}
