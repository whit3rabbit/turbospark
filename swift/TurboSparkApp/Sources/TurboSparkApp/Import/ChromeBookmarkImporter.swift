import Foundation

public struct ChromeBookmarkProfileCandidate: Equatable, Identifiable, Sendable {
    /// The single directory name under Chrome's profile root, not an arbitrary path.
    public let id: String
    public let displayLabel: String

    public init(id: String, displayLabel: String) {
        self.id = id
        self.displayLabel = displayLabel
    }
}

public enum ChromeBookmarkImportIssue: Equatable, Sendable {
    case profileRootUnavailable
    case noProfilesFound
    case bookmarksFileMissing(profileLabel: String)
    case bookmarksFileUnreadable(profileLabel: String)
    case bookmarksFileTooLarge(profileLabel: String)
    case malformedBookmarks(profileLabel: String)
    case profileUnavailable(profileLabel: String)
    case unknownSelection(profileLabel: String, count: Int)
}

public struct ChromeBookmarkProfileEnumeration: Equatable, Sendable {
    public let profiles: [ChromeBookmarkProfileCandidate]
    public let issues: [ChromeBookmarkImportIssue]

    public init(
        profiles: [ChromeBookmarkProfileCandidate],
        issues: [ChromeBookmarkImportIssue]
    ) {
        self.profiles = profiles
        self.issues = issues
    }
}

public struct ChromeBookmarkProfilePreview: Equatable, Sendable {
    public let profile: ChromeBookmarkProfileCandidate
    public let folders: [BrowserBookmarkFolder]
    public let selectableIdentifiers: Set<String>
    public let skippedItemCount: Int
    public let issue: ChromeBookmarkImportIssue?

    public init(
        profile: ChromeBookmarkProfileCandidate,
        folders: [BrowserBookmarkFolder],
        selectableIdentifiers: Set<String>,
        skippedItemCount: Int,
        issue: ChromeBookmarkImportIssue?
    ) {
        self.profile = profile
        self.folders = folders
        self.selectableIdentifiers = selectableIdentifiers
        self.skippedItemCount = skippedItemCount
        self.issue = issue
    }
}

public struct ChromeBookmarkImportResult: Equatable, Sendable {
    public let trees: [BrowserBookmarkTree]
    public let importedBookmarkCount: Int
    public let skippedItemCount: Int
    public let issues: [ChromeBookmarkImportIssue]

    public init(
        trees: [BrowserBookmarkTree],
        importedBookmarkCount: Int,
        skippedItemCount: Int,
        issues: [ChromeBookmarkImportIssue]
    ) {
        self.trees = trees
        self.importedBookmarkCount = importedBookmarkCount
        self.skippedItemCount = skippedItemCount
        self.issues = issues
    }
}

/// Reads only the exact Chrome `Bookmarks` file for profiles selected from the current scan.
public final class ChromeBookmarkImporter {
    public static let maximumBookmarkFileBytes = BrowserSettings.bookmarksSizeLimit
    public static let maximumProfileCount = 128
    public static let maximumTreeDepth = 32
    public static let maximumTreeNodeCount = 20_000
    public static let maximumBookmarkTitleLength = 1_024
    public static let maximumBookmarkURLLength = 8_192

    private static let bookmarksFileName = "Bookmarks"

    private let profileRootURL: URL
    private let fileManager: FileManager
    private let readFile: (URL) throws -> Data
    private var profileDirectories: [String: URL] = [:]
    private var cachedPreviews: [String: ChromeBookmarkProfilePreview] = [:]

    public init(
        profileRootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Google/Chrome", isDirectory: true),
        fileManager: FileManager = .default,
        readFile: @escaping (URL) throws -> Data = { try Data(contentsOf: $0) }
    ) {
        self.profileRootURL = profileRootURL.standardizedFileURL
        self.fileManager = fileManager
        self.readFile = readFile
    }

    /// Enumerates immediate profile directories with a regular Bookmarks file. Directory names
    /// and file metadata are inspected, but profile files are not opened during enumeration.
    public func enumerateProfiles() -> ChromeBookmarkProfileEnumeration {
        profileDirectories.removeAll(keepingCapacity: true)
        cachedPreviews.removeAll(keepingCapacity: true)

        guard isDirectory(profileRootURL),
              let children = try? fileManager.contentsOfDirectory(
                at: profileRootURL,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
                options: [])
        else {
            return ChromeBookmarkProfileEnumeration(
                profiles: [], issues: [.profileRootUnavailable])
        }

        var profiles: [ChromeBookmarkProfileCandidate] = []
        for directoryURL in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard let values = try? directoryURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]),
                  values.isDirectory == true,
                  values.isSymbolicLink != true
            else {
                continue
            }

            let directoryName = directoryURL.lastPathComponent
            let label = Self.sanitizedDirectoryLabel(directoryName)
            guard !label.isEmpty else { continue }

            let bookmarksURL = directoryURL.appendingPathComponent(Self.bookmarksFileName, isDirectory: false)
            guard Self.isAllowedProfileFile(bookmarksURL.lastPathComponent),
                  isRegularNonSymlinkFile(bookmarksURL)
            else {
                continue
            }

            let candidate = ChromeBookmarkProfileCandidate(id: directoryName, displayLabel: label)
            profileDirectories[candidate.id] = directoryURL.standardizedFileURL
            profiles.append(candidate)
            if profiles.count == Self.maximumProfileCount { break }
        }

        let issues: [ChromeBookmarkImportIssue] = profiles.isEmpty
            ? [.noProfilesFound]
            : []
        return ChromeBookmarkProfileEnumeration(profiles: profiles, issues: issues)
    }

    /// Previews only the selected profile IDs from the most recent enumeration.
    public func preview(profileIDs: Set<String>) -> [ChromeBookmarkProfilePreview] {
        cachedPreviews.removeAll(keepingCapacity: true)
        return profileIDs.sorted().map { profileID in
            let profile = ChromeBookmarkProfileCandidate(
                id: profileID,
                displayLabel: Self.sanitizedDirectoryLabel(profileID)
            )
            guard let directoryURL = profileDirectories[profileID] else {
                return store(ChromeBookmarkProfilePreview(
                    profile: profile,
                    folders: [],
                    selectableIdentifiers: [],
                    skippedItemCount: 0,
                    issue: .profileUnavailable(profileLabel: profile.displayLabel)
                ))
            }

            let bookmarksURL = directoryURL.appendingPathComponent(Self.bookmarksFileName, isDirectory: false)
            guard Self.isAllowedProfileFile(bookmarksURL.lastPathComponent) else {
                return store(failedPreview(profile, issue: .bookmarksFileUnreadable(profileLabel: profile.displayLabel)))
            }
            guard isRegularNonSymlinkFile(bookmarksURL) else {
                return store(failedPreview(profile, issue: .bookmarksFileMissing(profileLabel: profile.displayLabel)))
            }

            do {
                let fileValues = try bookmarksURL.resourceValues(forKeys: [.fileSizeKey])
                if let fileSize = fileValues.fileSize, fileSize > Self.maximumBookmarkFileBytes {
                    return store(failedPreview(profile, issue: .bookmarksFileTooLarge(profileLabel: profile.displayLabel)))
                }
                let data = try readFile(bookmarksURL)
                guard data.count <= Self.maximumBookmarkFileBytes else {
                    return store(failedPreview(profile, issue: .bookmarksFileTooLarge(profileLabel: profile.displayLabel)))
                }
                var parser = ChromeBookmarkTreeParser(data: data)
                guard let parsed = parser.parse() else {
                    return store(failedPreview(profile, issue: .malformedBookmarks(profileLabel: profile.displayLabel)))
                }
                return store(ChromeBookmarkProfilePreview(
                    profile: profile,
                    folders: parsed.folders,
                    selectableIdentifiers: parsed.identifiers,
                    skippedItemCount: parsed.skippedItemCount,
                    issue: nil
                ))
            } catch {
                return store(failedPreview(profile, issue: .bookmarksFileUnreadable(profileLabel: profile.displayLabel)))
            }
        }
    }

    /// Returns data to persist only after the user confirms exact bookmark or folder IDs.
    /// Folder selection includes that folder's descendants. No profile path is accepted here.
    public func importConfirmed(
        selectionsByProfileID: [String: Set<String>]
    ) -> ChromeBookmarkImportResult {
        var trees: [BrowserBookmarkTree] = []
        var importedCount = 0
        var skippedCount = 0
        var issues: [ChromeBookmarkImportIssue] = []

        for profileID in selectionsByProfileID.keys.sorted() {
            guard let preview = cachedPreviews[profileID] else {
                let label = Self.sanitizedDirectoryLabel(profileID)
                issues.append(.profileUnavailable(profileLabel: label))
                continue
            }
            if let issue = preview.issue {
                issues.append(issue)
                skippedCount += preview.skippedItemCount
                continue
            }

            let selected = selectionsByProfileID[profileID] ?? []
            let unknown = selected.subtracting(preview.selectableIdentifiers)
            if !unknown.isEmpty {
                issues.append(.unknownSelection(profileLabel: preview.profile.displayLabel, count: unknown.count))
                skippedCount += unknown.count
            }

            let folders = preview.folders.compactMap { filter($0, selected: selected) }
            let count = Self.bookmarkCount(in: folders)
            skippedCount += preview.skippedItemCount
            guard !folders.isEmpty else { continue }
            trees.append(BrowserBookmarkTree(
                sourceProfileDirectoryLabel: preview.profile.displayLabel,
                folders: folders
            ))
            importedCount += count
        }

        return ChromeBookmarkImportResult(
            trees: trees,
            importedBookmarkCount: importedCount,
            skippedItemCount: skippedCount,
            issues: issues
        )
    }

    static func isAllowedProfileFile(_ fileName: String) -> Bool {
        fileName == bookmarksFileName
    }

    private func store(_ preview: ChromeBookmarkProfilePreview) -> ChromeBookmarkProfilePreview {
        cachedPreviews[preview.profile.id] = preview
        return preview
    }

    private func failedPreview(
        _ profile: ChromeBookmarkProfileCandidate,
        issue: ChromeBookmarkImportIssue
    ) -> ChromeBookmarkProfilePreview {
        ChromeBookmarkProfilePreview(
            profile: profile,
            folders: [],
            selectableIdentifiers: [],
            skippedItemCount: 0,
            issue: issue
        )
    }

    private func isDirectory(_ url: URL) -> Bool {
        guard let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]) else {
            return false
        }
        return values.isDirectory == true && values.isSymbolicLink != true
    }

    private func isRegularNonSymlinkFile(_ url: URL) -> Bool {
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else {
            return false
        }
        return values.isRegularFile == true && values.isSymbolicLink != true
    }

    private static func sanitizedDirectoryLabel(_ value: String) -> String {
        let clean = value
            .components(separatedBy: .controlCharacters)
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "./\\")))
        return String(clean.prefix(64))
    }

    private func filter(
        _ folder: BrowserBookmarkFolder,
        selected: Set<String>
    ) -> BrowserBookmarkFolder? {
        let includesFolder = selected.contains(folder.id)
        if includesFolder { return folder }
        let bookmarks = folder.bookmarks.filter { selected.contains($0.id) }
        let folders = folder.folders.compactMap {
            filter($0, selected: selected)
        }
        guard includesFolder || !bookmarks.isEmpty || !folders.isEmpty else { return nil }
        return BrowserBookmarkFolder(
            id: folder.id,
            name: folder.name,
            folders: folders,
            bookmarks: bookmarks
        )
    }

    private static func bookmarkCount(in folders: [BrowserBookmarkFolder]) -> Int {
        folders.reduce(0) { subtotal, folder in
            subtotal + folder.bookmarks.count + bookmarkCount(in: folder.folders)
        }
    }
}

private struct ChromeBookmarkTreeParser {
    struct ParsedTree {
        let folders: [BrowserBookmarkFolder]
        let identifiers: Set<String>
        let skippedItemCount: Int
    }

    private let data: Data
    private var visitedNodeCount = 0
    private var skippedItemCount = 0
    private var identifiers = Set<String>()

    init(data: Data) {
        self.data = data
    }

    mutating func parse() -> ParsedTree? {
        guard let rootObject = try? JSONSerialization.jsonObject(with: data),
              let document = rootObject as? [String: Any],
              let roots = document["roots"] as? [String: Any]
        else {
            return nil
        }

        var folders: [BrowserBookmarkFolder] = []
        for key in ["bookmark_bar", "other", "synced"] {
            guard let root = roots[key] as? [String: Any] else { continue }
            let identifier = "root:\(key)"
            identifiers.insert(identifier)
            if let folder = parseFolder(root, identifier: identifier, depth: 0) {
                folders.append(folder)
            }
        }
        return ParsedTree(
            folders: folders,
            identifiers: identifiers,
            skippedItemCount: skippedItemCount
        )
    }

    private mutating func parseFolder(
        _ object: [String: Any],
        identifier: String,
        depth: Int
    ) -> BrowserBookmarkFolder? {
        guard depth <= ChromeBookmarkImporter.maximumTreeDepth,
              visitedNodeCount < ChromeBookmarkImporter.maximumTreeNodeCount
        else {
            skippedItemCount += 1
            return nil
        }
        visitedNodeCount += 1

        let name = Self.cleanTitle(object["name"] as? String) ?? "Bookmarks"
        var folders: [BrowserBookmarkFolder] = []
        var bookmarks: [BrowserBookmark] = []
        let children = object["children"] as? [Any] ?? []
        for (index, item) in children.enumerated() {
            guard visitedNodeCount < ChromeBookmarkImporter.maximumTreeNodeCount else {
                skippedItemCount += children.count - index
                break
            }
            guard let child = item as? [String: Any] else {
                skippedItemCount += 1
                continue
            }
            switch child["type"] as? String {
            case "folder":
                guard let chromeID = Self.validIdentifier(child["id"] as? String) else {
                    skippedItemCount += 1
                    continue
                }
                let childIdentifier = "folder:\(chromeID)"
                guard identifiers.insert(childIdentifier).inserted else {
                    skippedItemCount += 1
                    continue
                }
                if let folder = parseFolder(child, identifier: childIdentifier, depth: depth + 1) {
                    folders.append(folder)
                }
            case "url":
                guard let chromeID = Self.validIdentifier(child["id"] as? String),
                      let title = Self.cleanTitle(child["name"] as? String),
                      let url = Self.safeBookmarkURL(child["url"] as? String),
                      identifiers.insert("bookmark:\(chromeID)").inserted
                else {
                    skippedItemCount += 1
                    continue
                }
                visitedNodeCount += 1
                bookmarks.append(BrowserBookmark(id: "bookmark:\(chromeID)", title: title, url: url))
            default:
                skippedItemCount += 1
            }
        }

        return BrowserBookmarkFolder(
            id: identifier,
            name: name,
            folders: folders,
            bookmarks: bookmarks
        )
    }

    private static func validIdentifier(_ value: String?) -> String? {
        guard let value, !value.isEmpty, value.count <= 128,
              value.rangeOfCharacter(from: .controlCharacters) == nil
        else {
            return nil
        }
        return value
    }

    private static func cleanTitle(_ value: String?) -> String? {
        guard let value else { return nil }
        let clean = value
            .components(separatedBy: .controlCharacters)
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return nil }
        return String(clean.prefix(ChromeBookmarkImporter.maximumBookmarkTitleLength))
    }

    private static func safeBookmarkURL(_ value: String?) -> String? {
        guard let value, value.count <= ChromeBookmarkImporter.maximumBookmarkURLLength,
              let url = URL(string: value), BrowserOrigin(url: url) != nil
        else {
            return nil
        }
        return url.absoluteString
    }
}
