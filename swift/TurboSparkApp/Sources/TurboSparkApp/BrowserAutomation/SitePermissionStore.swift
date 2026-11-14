import Foundation

/// The only page permission classes the app governs. Everything else the
/// delegate could deliver is denied (see `SitePermissionClassifier`).
public enum SitePermissionClass: String, Codable, CaseIterable, Sendable {
    case camera
    case microphone
    case geolocation
}

public enum SitePermissionVerdict: String, Codable, Sendable {
    case allow
    case deny
}

public enum SitePermissionAuditDecision: String, Codable, Sendable {
    case allowOnce
    case allowSite
    /// A stored allow record answered without asking.
    case allowRecord
    case denyOnce
    case denySite
    /// A stored deny record answered without asking.
    case denyRecord
    /// Denied with no user answer; `reason` says why.
    case autoDeny
    case recordRemoved
}

public enum SitePermissionAutoDenyReason: String, Codable, Sendable {
    case agentTab
    case timeout
    case navigation
    case background
    case tabClosed
    case ownershipChanged
    case processTerminated
    case unclassified
    case unknownOrigin
    case tooManyRequests
    case otherClassDenied
}

/// One audit row. It carries a class, a decision, a canonical origin and a
/// time, never a page URL path or any page content.
public struct BrowserPermissionDiagnosticEvent: Codable, Equatable, Identifiable, Sendable {
    /// Stands in for an origin the app cannot name (a `file:` or `about:`
    /// frame, say). It is never a key for a record.
    public static let opaqueOrigin = "opaque"

    public let id: UUID
    /// Nil when the request was not a class the app governs.
    public let permissionClass: SitePermissionClass?
    public let decision: SitePermissionAuditDecision
    public let canonicalOrigin: String
    public let reason: SitePermissionAutoDenyReason?
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        permissionClass: SitePermissionClass?,
        decision: SitePermissionAuditDecision,
        origin: BrowserOrigin?,
        reason: SitePermissionAutoDenyReason? = nil,
        timestamp: Date
    ) {
        self.id = id
        self.permissionClass = permissionClass
        self.decision = decision
        self.canonicalOrigin = origin?.canonicalString ?? Self.opaqueOrigin
        self.reason = reason
        self.timestamp = timestamp
    }
}

/// A saved per-site decision, keyed by `BrowserOrigin.canonicalString` (exact
/// scheme, host and effective port) and permission class.
public struct SitePermissionRecord: Codable, Equatable, Identifiable, Sendable {
    public let origin: String
    public let permissionClass: SitePermissionClass
    public var verdict: SitePermissionVerdict
    public let createdAt: Date

    public var id: String { Self.key(origin: origin, permissionClass: permissionClass) }

    static func key(origin: String, permissionClass: SitePermissionClass) -> String {
        "\(origin)|\(permissionClass.rawValue)"
    }
}

private struct SitePermissionArchive: Codable {
    var version = 1
    var records: [SitePermissionRecord] = []
    var audit: [BrowserPermissionDiagnosticEvent] = []
}

/// Per-profile persistent site permission records plus a bounded audit log.
///
/// The file lives under `AppStorageRoot.directory`, which is already the
/// profile boundary, so a grant in one profile never applies in another.
/// Nothing here ever leaves the machine. Records never expire on their own;
/// the user removes them from Settings. An unreadable or corrupt file is
/// quarantined by `AppJSONStore` and the store starts empty, which means every
/// request is a miss: denied in agent tabs and asked in user tabs.
@MainActor
public final class SitePermissionStore: ObservableObject {
    public static let maximumRecords = 500
    public static let maximumAuditEntries = 200

    /// The profile-scoped store. Reads its file on first use, not at launch.
    public static let shared = SitePermissionStore(
        fileURL: AppStorageRoot.file("browser_site_permissions.json")
    )

    @Published public private(set) var records: [SitePermissionRecord] = []
    /// Newest first.
    @Published public private(set) var auditLog: [BrowserPermissionDiagnosticEvent] = []

    private let fileURL: URL?
    private let now: () -> Date
    private var loaded = false

    /// A nil `fileURL` keeps everything in memory (tests, previews).
    public init(fileURL: URL?, now: @escaping () -> Date = Date.init) {
        self.fileURL = fileURL
        self.now = now
        if fileURL == nil { loaded = true }
    }

    // MARK: - Lookup

    public func verdict(
        for origin: BrowserOrigin,
        permissionClass: SitePermissionClass
    ) -> SitePermissionVerdict? {
        loadIfNeeded()
        let key = SitePermissionRecord.key(
            origin: origin.canonicalString, permissionClass: permissionClass)
        return records.first { $0.id == key }?.verdict
    }

    // MARK: - Mutation

    /// Saves or replaces a record. Returns false, saving nothing, when a new
    /// record would exceed the cap: the caller then treats the answer as
    /// once-only, which is the safe direction for both verdicts.
    @discardableResult
    public func setVerdict(
        _ verdict: SitePermissionVerdict,
        origin: BrowserOrigin,
        permissionClass: SitePermissionClass
    ) -> Bool {
        loadIfNeeded()
        let key = SitePermissionRecord.key(
            origin: origin.canonicalString, permissionClass: permissionClass)
        if let index = records.firstIndex(where: { $0.id == key }) {
            records[index].verdict = verdict
        } else {
            guard records.count < Self.maximumRecords else { return false }
            records.append(SitePermissionRecord(
                origin: origin.canonicalString,
                permissionClass: permissionClass,
                verdict: verdict,
                createdAt: now()
            ))
            records.sort { $0.id < $1.id }
        }
        save()
        return true
    }

    /// Removes a record and writes an audit row. Entries are never edited, so
    /// "append-only" holds for the log while the records stay deletable.
    @discardableResult
    public func removeRecord(id: String) -> Bool {
        loadIfNeeded()
        guard let index = records.firstIndex(where: { $0.id == id }) else { return false }
        let removed = records.remove(at: index)
        appendAudit(BrowserPermissionDiagnosticEvent(
            permissionClass: removed.permissionClass,
            decision: .recordRemoved,
            origin: BrowserOrigin(origin: removed.origin),
            timestamp: now()
        ), saving: false)
        save()
        return true
    }

    public func recordAudit(
        permissionClass: SitePermissionClass?,
        decision: SitePermissionAuditDecision,
        origin: BrowserOrigin?,
        reason: SitePermissionAutoDenyReason? = nil
    ) {
        loadIfNeeded()
        appendAudit(BrowserPermissionDiagnosticEvent(
            permissionClass: permissionClass,
            decision: decision,
            origin: origin,
            reason: reason,
            timestamp: now()
        ), saving: true)
    }

    // MARK: - Search

    /// Case-insensitive substring match on the origin or the class name.
    public static func filter(
        _ records: [SitePermissionRecord],
        query: String
    ) -> [SitePermissionRecord] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty else { return records }
        return records.filter {
            $0.origin.lowercased().contains(needle)
                || $0.permissionClass.rawValue.contains(needle)
        }
    }

    // MARK: - Persistence

    private func appendAudit(_ event: BrowserPermissionDiagnosticEvent, saving: Bool) {
        auditLog.insert(event, at: 0)
        if auditLog.count > Self.maximumAuditEntries {
            auditLog.removeLast(auditLog.count - Self.maximumAuditEntries)
        }
        if saving { save() }
    }

    /// Reads the file once. Settings calls it on appear so its lists are not
    /// empty before the first lookup.
    public func loadIfNeeded() {
        guard !loaded, let fileURL else { return }
        loaded = true
        guard let archive = AppJSONStore.load(
            SitePermissionArchive.self, from: fileURL, label: "Site permissions")
        else { return }
        // A record that no longer parses to its own canonical origin is
        // dropped rather than trusted: a key that matches nothing is a miss.
        var seen = Set<String>()
        records = archive.records.filter { record in
            guard let origin = BrowserOrigin(origin: record.origin),
                  origin.canonicalString == record.origin,
                  seen.insert(record.id).inserted
            else { return false }
            return true
        }
        .prefix(Self.maximumRecords)
        .sorted { $0.id < $1.id }
        auditLog = Array(archive.audit.prefix(Self.maximumAuditEntries))
    }

    private func save() {
        guard let fileURL else { return }
        AppJSONStore.save(
            SitePermissionArchive(records: records, audit: auditLog),
            to: fileURL,
            label: "Site permissions"
        )
    }
}
