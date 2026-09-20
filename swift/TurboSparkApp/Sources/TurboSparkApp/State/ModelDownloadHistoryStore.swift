import Foundation

/// Download sources can contain private repository names, so history belongs
/// in the profile vault even though installed model files are machine-wide.
struct ModelDownloadHistoryStore {
    static let key = "downloads:history"
    static let limit = 12
    var repository: ProfileRepository = .shared

    func load() throws -> [ModelDownload] {
        let saved = try repository.load([ModelDownload].self, key: Self.key) ?? []
        return saved.prefix(Self.limit).map { row in
            var restored = row
            restored.status = row.status.afterRelaunch
            return restored
        }
    }

    func save(_ downloads: [ModelDownload]) throws {
        try repository.save(Array(downloads.prefix(Self.limit)), key: Self.key)
    }
}
