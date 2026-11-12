import Foundation

/// Policy used when a page presents a native browser dialog during automation.
public enum BrowserDialogPolicy: String, Codable, CaseIterable, Sendable {
    case autoAccept = "auto-accept"
    case autoDismiss = "auto-dismiss"
    case ask
}

/// A persisted CSS viewport. Invalid dimensions or zooms fall back to a safe standard size.
public struct BrowserViewportPreference: Codable, Equatable, Sendable {
    public static let standard = BrowserViewportPreference(width: 1280, height: 800, zoom: 1)

    public let width: Int
    public let height: Int
    public let zoom: Double

    public init(width: Int, height: Int, zoom: Double) {
        guard width > 0, height > 0, zoom.isFinite, zoom > 0 else {
            self.width = 1280
            self.height = 800
            self.zoom = 1
            return
        }

        self.width = width
        self.height = height
        self.zoom = zoom
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let width = container.decodeLenient(Int.self, forKey: .width, fallback: 0)
        let height = container.decodeLenient(Int.self, forKey: .height, fallback: 0)
        let zoom = container.decodeLenient(Double.self, forKey: .zoom, fallback: 0)
        guard width > 0, height > 0, zoom.isFinite, zoom > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .width,
                in: container,
                debugDescription: "Browser viewport dimensions and zoom must be positive."
            )
        }
        self.init(width: width, height: height, zoom: zoom)
    }

    private enum CodingKeys: String, CodingKey {
        case width
        case height
        case zoom
    }
}

/// One bookmark imported from a browser profile.
public struct BrowserBookmark: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var title: String
    public var url: String

    public init(id: String = UUID().uuidString, title: String, url: String) {
        self.id = id
        self.title = title
        self.url = url
    }
}

/// A named folder that can contain bookmarks and nested folders.
public struct BrowserBookmarkFolder: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var folders: [BrowserBookmarkFolder]
    public var bookmarks: [BrowserBookmark]

    public init(
        id: String = UUID().uuidString,
        name: String,
        folders: [BrowserBookmarkFolder] = [],
        bookmarks: [BrowserBookmark] = []
    ) {
        self.id = id
        self.name = name
        self.folders = folders
        self.bookmarks = bookmarks
    }
}

/// Bookmark folders imported from one Chrome profile.
public struct BrowserBookmarkTree: Codable, Equatable, Sendable {
    public var sourceProfileDirectoryLabel: String
    public var folders: [BrowserBookmarkFolder]

    public init(sourceProfileDirectoryLabel: String, folders: [BrowserBookmarkFolder] = []) {
        self.sourceProfileDirectoryLabel = sourceProfileDirectoryLabel
        self.folders = folders
    }
}

/// Global browser preferences stored under the `browser` object in settings.json.
public struct BrowserSettings: Codable, Equatable, Sendable {
    public static let bookmarksSizeLimit = 2 * 1024 * 1024

    public var enabled: Bool
    public var dialogPolicy: BrowserDialogPolicy
    public var defaultViewport: BrowserViewportPreference
    public var onboardingDone: Bool
    public private(set) var bookmarks: [BrowserBookmarkTree]

    public init(
        enabled: Bool = false,
        dialogPolicy: BrowserDialogPolicy = .ask,
        defaultViewport: BrowserViewportPreference = .standard,
        onboardingDone: Bool = false
    ) {
        self.enabled = enabled
        self.dialogPolicy = dialogPolicy
        self.defaultViewport = defaultViewport
        self.onboardingDone = onboardingDone
        self.bookmarks = []
    }

    /// Replaces the saved tree only when its encoded JSON value fits the storage cap.
    @discardableResult
    public mutating func replaceBookmarks(_ value: [BrowserBookmarkTree]) -> Bool {
        guard Self.bookmarkTreeFitsLimit(value) else { return false }
        bookmarks = value
        return true
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.enabled = container.decodeLenient(Bool.self, forKey: .enabled, fallback: false)
        self.dialogPolicy = container.decodeLenient(
            BrowserDialogPolicy.self, forKey: .dialogPolicy, fallback: .ask)
        self.defaultViewport = container.decodeLenient(
            BrowserViewportPreference.self, forKey: .defaultViewport, fallback: .standard)
        self.onboardingDone = container.decodeLenient(
            Bool.self, forKey: .onboardingDone, fallback: false)
        // A legacy `viewport_pref` (the popover's old saved copy) is ignored
        // and dropped by the next save: `default_viewport` is the only one.

        let decodedBookmarks = container.decodeLenientElements(
            BrowserBookmarkTree.self, forKey: .bookmarks)
        self.bookmarks = Self.bookmarkTreeFitsLimit(decodedBookmarks) ? decodedBookmarks : []
    }

    private static func bookmarkTreeFitsLimit(_ value: [BrowserBookmarkTree]) -> Bool {
        guard let data = try? JSONEncoder().encode(value) else { return false }
        return data.count <= bookmarksSizeLimit
    }

    private enum CodingKeys: String, CodingKey {
        case enabled
        case dialogPolicy = "dialog_policy"
        case defaultViewport = "default_viewport"
        case onboardingDone = "onboarding_done"
        case bookmarks
    }
}

