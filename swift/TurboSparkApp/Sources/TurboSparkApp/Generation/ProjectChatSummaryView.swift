import SwiftUI

struct ChatSource: Identifiable, Equatable {
    let url: URL
    var id: String { url.absoluteString }
    var title: String { url.host ?? url.absoluteString }
}

/// Derived from the selected conversation only. No fetching, model calls,
/// or second persistent task/artifact store is needed to refresh the panel.
enum ProjectChatSummary {
    static let width: CGFloat = 320
    static let previewLimit = 3

    static func isAvailable(projectID: UUID?, isChat: Bool) -> Bool { projectID != nil && isChat }

    static func canPin(availableWidth: CGFloat) -> Bool {
        availableWidth >= AppChromeLayout.primaryMinimumWidth + width + AppChromeLayout.dividerWidth
    }

    static func sources(messages: [AppChatMessage]) -> [ChatSource] {
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
        else { return [] }
        var seen = Set<String>()
        var sources: [ChatSource] = []
        func collect(_ text: String) {
            for match in detector.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
                guard let url = match.url,
                      ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
                      var components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                      let host = components.host else { continue }
                components.host = host.lowercased()
                components.scheme = components.scheme?.lowercased()
                if components.path.isEmpty { components.path = "/" }
                if (components.scheme == "https" && components.port == 443)
                    || (components.scheme == "http" && components.port == 80) { components.port = nil }
                guard let normalized = components.url,
                      seen.insert(normalized.absoluteString).inserted else { continue }
                sources.append(ChatSource(url: normalized))
            }
        }
        for message in messages {
            if message.role == .user || message.role == .assistant { collect(message.content) }
            for call in message.toolCalls {
                guard ["globe", "network", "search"].contains(ToolPresentation.resolve(call.name).icon)
                else { continue }
                if let url = ToolPresentation.webURL(for: call) { collect(url.absoluteString) }
                for result in message.toolResults where result.callID == call.id { collect(result.output) }
            }
        }
        return sources
    }

    static func orderedTasks(_ tasks: [TodoItem]) -> [TodoItem] {
        tasks.filter(\.isInProgress)
            + tasks.filter { !$0.isInProgress && !$0.isCompleted && !$0.isCancelled }
            + tasks.filter { $0.isCompleted || $0.isCancelled }
    }
}

@MainActor
struct ProjectChatSummaryView: View {
    @ObservedObject var model: AppModel
    @State private var expanded: Set<String> = []
    @State private var sources: [ChatSource] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Chat summary", bundle: .module)
                    .themedFont(.callout, weight: .semibold)
                tasks
                Divider()
                outputs
                Divider()
                sourceList
            }
            .padding(18)
        }
        .background(.appPage)
        // Root updates while tokens stream; scan links only when the stored
        // conversation changes, rather than on every view initialization.
        .onChange(of: model.selectedTurnMessages, initial: true) { _, messages in
            sources = ProjectChatSummary.sources(messages: messages)
        }
        .onChange(of: model.selectedChatID) { _, _ in expanded = [] }
    }

    private var tasks: some View {
        let items = ProjectChatSummary.orderedTasks(model.currentTodos)
        return VStack(alignment: .leading, spacing: 10) {
            sectionTitle("Tasks", count: items.count)
            if items.isEmpty {
                empty("No tasks yet")
            } else {
                Text(TodoChecklistSummary.text(items))
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                ForEach(visible(items, section: "tasks")) { TodoItemRow(item: $0) }
                more("tasks", count: items.count)
            }
        }
    }

    private var outputs: some View {
        let items = model.selectedChat.artifacts.filter { $0.chatID == model.selectedChatID }
        return VStack(alignment: .leading, spacing: 10) {
            sectionTitle("Outputs", count: items.count)
            if items.isEmpty { empty("No outputs yet") }
            ForEach(visible(items, section: "outputs")) { artifact in
                Button { model.openArtifact(id: artifact.id) } label: {
                    HStack(spacing: 8) {
                        BundledToolIcon(name: "file-output")
                        Text(artifact.title).lineLimit(2)
                        Spacer(minLength: 0)
                    }
                }
                .buttonStyle(.plain)
                .themedFont(.small)
                .help(artifact.path ?? artifact.title)
            }
            more("outputs", count: items.count)
        }
    }

    private var sourceList: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionTitle("Sources", count: sources.count)
            if sources.isEmpty { empty("No sources yet") }
            ForEach(visible(sources, section: "sources")) { source in
                Link(destination: source.url) {
                    HStack(alignment: .top, spacing: 8) {
                        OfflineSiteIcon(url: source.url)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(source.title).lineLimit(1)
                            Text(source.url.absoluteString).lineLimit(2)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                    .themedFont(.small)
                }
                .help(source.url.absoluteString)
            }
            more("sources", count: sources.count)
        }
    }

    private func sectionTitle(_ title: String, count: Int) -> some View {
        HStack {
            Text(String(localized: String.LocalizationValue(title), bundle: .module))
                .themedFont(.small, weight: .semibold)
            Spacer()
            Text(verbatim: "\(count)").themedFont(.tiny).foregroundStyle(.appSecondary)
        }
    }

    private func empty(_ text: String) -> some View {
        Text(String(localized: String.LocalizationValue(text), bundle: .module))
            .themedFont(.small).foregroundStyle(.appSecondary)
    }

    private func visible<T>(_ items: [T], section: String) -> [T] {
        expanded.contains(section) ? items : Array(items.prefix(ProjectChatSummary.previewLimit))
    }

    @ViewBuilder
    private func more(_ section: String, count: Int) -> some View {
        if count > ProjectChatSummary.previewLimit {
            Button {
                if expanded.contains(section) { expanded.remove(section) }
                else { expanded.insert(section) }
            } label: {
                if expanded.contains(section) { Text("Show less", bundle: .module) }
                else { Text("View all", bundle: .module) }
            }
            .buttonStyle(.plain).themedFont(.tiny).foregroundStyle(.appAccent)
        }
    }
}
