import SwiftUI

/// The qwen-code sidebar organization parity: sessions grouped under time
/// headers, and a deterministic accent color per project (their
/// `workspaceColor`).
///
/// The bucket function is DAY-BOUNDARY based, not 24-hour based: a chat
/// updated at 23:50 yesterday is "Yesterday" at 00:05, not "an hour ago".
enum ChatDateBucket: String, CaseIterable {
    case today
    case yesterday
    case previous7Days
    case older

    /// The bucket a date falls into. `now` is injectable for tests.
    static func bucket(for date: Date, now: Date = Date()) -> ChatDateBucket {
        let calendar = Calendar.current
        if calendar.isDate(date, inSameDayAs: now) { return .today }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now),
            calendar.isDate(date, inSameDayAs: yesterday)
        {
            return .yesterday
        }
        if let weekAgo = calendar.date(byAdding: .day, value: -7, to: now),
            date >= calendar.startOfDay(for: weekAgo)
        {
            return .previous7Days
        }
        return .older
    }

    /// The sidebar header for the bucket.
    var label: String {
        switch self {
        case .today: return "Today"
        case .yesterday: return "Yesterday"
        case .previous7Days: return "Previous 7 Days"
        case .older: return "Older"
        }
    }
}

/// The sidebar list, derived once from the already-sorted history. Pure so the
/// view can build it once per body and tests can pin the grouping.
struct ChatSidebarLayout {
    struct Group {
        let bucket: ChatDateBucket
        let chats: [AppChat]
    }

    /// True while a non-blank search is active: `matches` is the flat result
    /// and the pinned/date grouping is suppressed.
    let isSearching: Bool
    let matches: [AppChat]
    let pinned: [AppChat]
    let groups: [Group]

    var isEmpty: Bool { isSearching ? matches.isEmpty : (pinned.isEmpty && groups.isEmpty) }

    /// `sortedHistory` must already be in `AppChat.sortedForSidebar` order;
    /// that order is kept inside the pinned list and every bucket.
    static func make(sortedHistory: [AppChat], searchText: String, now: Date = Date()) -> ChatSidebarLayout {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            let hits = sortedHistory.filter {
                $0.title.localizedCaseInsensitiveContains(trimmed)
                    || $0.preview.localizedCaseInsensitiveContains(trimmed)
            }
            return ChatSidebarLayout(isSearching: true, matches: hits, pinned: [], groups: [])
        }
        var pinned: [AppChat] = []
        var byBucket: [ChatDateBucket: [AppChat]] = [:]
        for chat in sortedHistory {
            if chat.isPinned {
                pinned.append(chat)
            } else {
                byBucket[ChatDateBucket.bucket(for: chat.updatedAt, now: now), default: []].append(chat)
            }
        }
        let groups = ChatDateBucket.allCases.compactMap { bucket in
            byBucket[bucket].map { Group(bucket: bucket, chats: $0) }
        }
        return ChatSidebarLayout(isSearching: false, matches: [], pinned: pinned, groups: groups)
    }
}

/// The six-preset palette qwen-code's `workspaceColor` cycles through, in
/// its order. The pick is a hash of the project's stable id, not its name:
/// a rename must not repaint a project the user has already learned.
enum ProjectAccentColor: Int, CaseIterable {
    case blue = 0
    case purple
    case pink
    case red
    case orange
    case green

    static func forProject(id: UUID?) -> ProjectAccentColor {
        guard let id else { return .blue }
        // uuidString hashes stably across launches; `hashValue` does not
        // (it is per-process seeded).
        var hash = 5381
        for byte in id.uuidString.utf8 {
            hash = ((hash << 5) &+ hash) &+ Int(byte)
        }
        return ProjectAccentColor(rawValue: abs(hash) % ProjectAccentColor.allCases.count) ?? .blue
    }

    var color: Color {
        switch self {
        case .blue: return Color(red: 0.25, green: 0.55, blue: 0.95)
        case .purple: return Color(red: 0.65, green: 0.45, blue: 0.95)
        case .pink: return Color(red: 0.95, green: 0.45, blue: 0.70)
        case .red: return Color(red: 0.92, green: 0.38, blue: 0.38)
        case .orange: return Color(red: 0.95, green: 0.62, blue: 0.28)
        case .green: return Color(red: 0.35, green: 0.78, blue: 0.45)
        }
    }
}
