import Foundation
import SQLCipher

/// What the audio search index stores for one library item: its words,
/// with times, and its labels. Rebuilt whenever the hash changes.
struct AudioSearchProjection: Equatable, Sendable {
    struct Segment: Equatable, Sendable {
        let start: Double
        let end: Double
        let text: String
    }
    let itemID: UUID
    let hash: String
    let title: String
    let collection: String
    let tags: [String]
    let segments: [Segment]
}

struct AudioLexicalHit: Equatable, Sendable {
    let itemID: UUID
    /// nil when the match is in the title, collection or tags.
    let start: Double?
    let end: Double?
    let snippet: String
}

/// One embedded span: a transcript chunk (with its text) or an audio window.
struct AudioVectorRow: Equatable, Sendable {
    let itemID: UUID
    let index: Int
    let start: Double
    let end: Double
    let text: String?
    let vector: [Float]
}

enum AudioVectorKind: String, Sendable { case text, audio }

/// The audio library's search index: FTS5 over transcript segments and item
/// labels, and embedding vectors for meaning and sound. Everything here is
/// derived from `audio:item:` records and can be rebuilt from them, which is
/// why its schema is optional (`audioIndexAvailable`) and versioned apart.
extension ProfileDatabase {
    static let audioIndexSchemaVersion: Int64 = 1

    func migrateAudioIndexSchema() throws -> Bool {
        try transaction {
            try execute("""
                CREATE TABLE IF NOT EXISTS audio_index_schema_migrations(
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS audio_segments_fts USING fts5(
                    item_id UNINDEXED, start_seconds UNINDEXED, end_seconds UNINDEXED, text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS audio_items_fts USING fts5(
                    item_id UNINDEXED, title, collection, tags,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                CREATE TABLE IF NOT EXISTS audio_index_items(
                    item_id TEXT PRIMARY KEY,
                    projection_hash TEXT NOT NULL,
                    text_key TEXT,
                    audio_key TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audio_vectors(
                    item_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    model TEXT NOT NULL,
                    span_index INTEGER NOT NULL,
                    start_seconds REAL NOT NULL,
                    end_seconds REAL NOT NULL,
                    text TEXT,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(item_id, kind, model, span_index)
                );
                """)
            try withStatement("INSERT OR IGNORE INTO audio_index_schema_migrations(version, applied_at) VALUES (?1, ?2)") { s in
                try bind(Self.audioIndexSchemaVersion, at: 1, to: s)
                try bind(Date().timeIntervalSince1970, at: 2, to: s)
                try stepDone(s)
            }
        }
        return true
    }

    /// The hash the index holds for an item, to skip unchanged saves.
    func audioIndexHash(itemID: UUID) throws -> String? {
        try withStatement("SELECT projection_hash FROM audio_index_items WHERE item_id = ?1") { s in
            try bind(itemID.uuidString, at: 1, to: s)
            return sqlite3_step(s) == SQLITE_ROW ? columnText(s, index: 0) : nil
        }
    }

    /// Replaces an item's words and labels. Its vectors stay until the
    /// indexer sees the new hash and re-embeds; their key marks them stale.
    func replaceAudioSearchProjection(_ p: AudioSearchProjection) throws {
        try transaction {
            try deleteAudioLexicalRows(p.itemID)
            for segment in p.segments where !segment.text.isEmpty {
                try withStatement("INSERT INTO audio_segments_fts(item_id, start_seconds, end_seconds, text) VALUES (?1, ?2, ?3, ?4)") { s in
                    try bind(p.itemID.uuidString, at: 1, to: s)
                    try bind(segment.start, at: 2, to: s)
                    try bind(segment.end, at: 3, to: s)
                    try bind(segment.text, at: 4, to: s)
                    try stepDone(s)
                }
            }
            try withStatement("INSERT INTO audio_items_fts(item_id, title, collection, tags) VALUES (?1, ?2, ?3, ?4)") { s in
                try bind(p.itemID.uuidString, at: 1, to: s)
                try bind(p.title, at: 2, to: s)
                try bind(p.collection, at: 3, to: s)
                try bind(p.tags.joined(separator: " "), at: 4, to: s)
                try stepDone(s)
            }
            try withStatement("""
                INSERT INTO audio_index_items(item_id, projection_hash, updated_at) VALUES (?1, ?2, ?3)
                ON CONFLICT(item_id) DO UPDATE SET projection_hash = excluded.projection_hash, updated_at = excluded.updated_at
                """) { s in
                try bind(p.itemID.uuidString, at: 1, to: s)
                try bind(p.hash, at: 2, to: s)
                try bind(Date().timeIntervalSince1970, at: 3, to: s)
                try stepDone(s)
            }
        }
    }

    /// Removes every index row of a deleted item.
    func deleteAudioSearchRows(itemID: UUID) throws {
        try transaction {
            try deleteAudioLexicalRows(itemID)
            for table in ["audio_index_items", "audio_vectors"] {
                try withStatement("DELETE FROM \(table) WHERE item_id = ?1") { s in
                    try bind(itemID.uuidString, at: 1, to: s)
                    try stepDone(s)
                }
            }
        }
    }

    private func deleteAudioLexicalRows(_ itemID: UUID) throws {
        for table in ["audio_segments_fts", "audio_items_fts"] {
            try withStatement("DELETE FROM \(table) WHERE item_id = ?1") { s in
                try bind(itemID.uuidString, at: 1, to: s)
                try stepDone(s)
            }
        }
    }

    /// User text as an FTS5 expression of quoted terms only, so no operator,
    /// column filter or syntax error can come from what was typed. Every
    /// term must match; the last may be a prefix, for search as you type.
    static func audioMatchExpression(_ query: String) -> String? {
        let words = query.split { !$0.isLetter && !$0.isNumber }.prefix(12).map { String($0).lowercased() }
        guard let last = words.last else { return nil }
        return (words.dropLast().map { "\"\($0)\"" } + ["\"\(last)\"*"]).joined(separator: " ")
    }

    /// Transcript and label matches, best first. Segment hits carry times.
    func audioLexicalSearch(query: String, limit: Int = 50) throws -> [AudioLexicalHit] {
        guard let expression = Self.audioMatchExpression(query) else { return [] }
        var hits: [AudioLexicalHit] = []
        try withStatement("""
            SELECT item_id, start_seconds, end_seconds, snippet(audio_segments_fts, 3, '', '', '...', 12)
            FROM audio_segments_fts WHERE audio_segments_fts MATCH ?1 ORDER BY bm25(audio_segments_fts) LIMIT ?2
            """) { s in
            try bind(expression, at: 1, to: s)
            try bind(Int64(limit), at: 2, to: s)
            while sqlite3_step(s) == SQLITE_ROW {
                guard let raw = columnText(s, index: 0), let id = UUID(uuidString: raw) else { continue }
                hits.append(AudioLexicalHit(itemID: id, start: sqlite3_column_double(s, 1), end: sqlite3_column_double(s, 2),
                                            snippet: columnText(s, index: 3) ?? ""))
            }
        }
        try withStatement("""
            SELECT item_id, snippet(audio_items_fts, -1, '', '', '...', 8)
            FROM audio_items_fts WHERE audio_items_fts MATCH ?1 ORDER BY bm25(audio_items_fts) LIMIT ?2
            """) { s in
            try bind(expression, at: 1, to: s)
            try bind(Int64(limit), at: 2, to: s)
            while sqlite3_step(s) == SQLITE_ROW {
                guard let raw = columnText(s, index: 0), let id = UUID(uuidString: raw) else { continue }
                hits.append(AudioLexicalHit(itemID: id, start: nil, end: nil, snippet: columnText(s, index: 1) ?? ""))
            }
        }
        return hits
    }

    /// The key (model and projection hash) an item's vectors were built
    /// from, by kind.
    func audioVectorKey(itemID: UUID, kind: AudioVectorKind) throws -> String? {
        try withStatement("SELECT \(kind == .text ? "text_key" : "audio_key") FROM audio_index_items WHERE item_id = ?1") { s in
            try bind(itemID.uuidString, at: 1, to: s)
            return sqlite3_step(s) == SQLITE_ROW ? columnText(s, index: 0) : nil
        }
    }

    /// Replaces an item's vectors of one kind and records the key they were
    /// built from.
    func replaceAudioVectors(itemID: UUID, kind: AudioVectorKind, model: String, key: String, rows: [AudioVectorRow]) throws {
        try transaction {
            try withStatement("DELETE FROM audio_vectors WHERE item_id = ?1 AND kind = ?2") { s in
                try bind(itemID.uuidString, at: 1, to: s)
                try bind(kind.rawValue, at: 2, to: s)
                try stepDone(s)
            }
            for row in rows {
                try withStatement("""
                    INSERT INTO audio_vectors(item_id, kind, model, span_index, start_seconds, end_seconds, text, dimensions, vector)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                    """) { s in
                    try bind(itemID.uuidString, at: 1, to: s)
                    try bind(kind.rawValue, at: 2, to: s)
                    try bind(model, at: 3, to: s)
                    try bind(Int64(row.index), at: 4, to: s)
                    try bind(row.start, at: 5, to: s)
                    try bind(row.end, at: 6, to: s)
                    try bind(row.text, at: 7, to: s)
                    try bind(Int64(row.vector.count), at: 8, to: s)
                    try bind(Self.vectorBlob(row.vector), at: 9, to: s)
                    try stepDone(s)
                }
            }
            // An item with no words yet (a sound recording) has no row from
            // the projection; without one the key would never be recorded
            // and the indexer would embed it again on every pass.
            try withStatement("INSERT INTO audio_index_items(item_id, projection_hash, updated_at) VALUES (?1, '', ?2) ON CONFLICT(item_id) DO NOTHING") { s in
                try bind(itemID.uuidString, at: 1, to: s)
                try bind(Date().timeIntervalSince1970, at: 2, to: s)
                try stepDone(s)
            }
            try withStatement("UPDATE audio_index_items SET \(kind == .text ? "text_key" : "audio_key") = ?2 WHERE item_id = ?1") { s in
                try bind(itemID.uuidString, at: 1, to: s)
                try bind(key, at: 2, to: s)
                try stepDone(s)
            }
        }
    }

    /// Every vector of one kind and model. A row whose blob does not match
    /// its declared dimensions is skipped, not trusted.
    func loadAudioVectors(kind: AudioVectorKind, model: String) throws -> [AudioVectorRow] {
        try withStatement("""
            SELECT item_id, span_index, start_seconds, end_seconds, text, dimensions, vector
            FROM audio_vectors WHERE kind = ?1 AND model = ?2
            """) { s in
            try bind(kind.rawValue, at: 1, to: s)
            try bind(model, at: 2, to: s)
            var rows: [AudioVectorRow] = []
            while sqlite3_step(s) == SQLITE_ROW {
                guard let raw = columnText(s, index: 0), let id = UUID(uuidString: raw),
                      let vector = Self.vector(columnData(s, index: 6) ?? Data(), dimensions: Int(sqlite3_column_int64(s, 5)))
                else { continue }
                rows.append(AudioVectorRow(itemID: id, index: Int(sqlite3_column_int64(s, 1)), start: sqlite3_column_double(s, 2),
                                           end: sqlite3_column_double(s, 3), text: columnText(s, index: 4), vector: vector))
            }
            return rows
        }
    }

    /// Little-endian Float32.
    static func vectorBlob(_ vector: [Float]) -> Data {
        var data = Data(capacity: vector.count * 4)
        for value in vector { withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
        return data
    }

    static func vector(_ data: Data, dimensions: Int) -> [Float]? {
        guard dimensions > 0, dimensions <= 4_096, data.count == dimensions * 4 else { return nil }
        return (0..<dimensions).map { i in
            let bits = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self) }
            return Float(bitPattern: UInt32(littleEndian: bits))
        }
    }
}
