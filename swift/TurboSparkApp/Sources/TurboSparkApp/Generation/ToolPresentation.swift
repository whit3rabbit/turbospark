import Foundation
import SwiftUI

/// Exact aliases keep presentation independent of permission categories and
/// prevent a web read from accidentally borrowing a file-edit presentation.
struct ToolPresentation: Equatable {
    let label: String
    let icon: String
    let isBuiltIn: Bool

    static let registry: [String: ToolPresentation] = {
        var result: [String: ToolPresentation] = [:]
        func register(_ label: String, _ icon: String, _ names: String) {
            for name in names.split(separator: " ") {
                precondition(result[String(name)] == nil)
                result[String(name)] = ToolPresentation(label: label, icon: icon, isBuiltIn: true)
            }
        }
        register("List Directory", "folder", "list_directory list_dir ls")
        register("Find Files", "folder-search", "glob")
        register("Read File", "file-text", "read_file view_file cat fileread read")
        register("Write File", "file-plus", "write_file save_file filewrite write")
        register("Edit File", "file-pen", "edit_file fileedit edit editor")
        register("Apply Patch", "file-diff", "apply_patch applypatch")
        register("Recall Output", "archive-restore", "recall_tool_output")
        register("Search Code", "search-code", "search_code grep search grep_search")
        register("Run Command", "terminal", "run_command bash shell exec terminal")
        register("Read Command Output", "logs", "bashoutput bash_output")
        register("Stop Command", "square-terminal", "killshell kill_shell")
        register("Search Web", "search", "websearch web_search search_web")
        register("Read Webpage", "globe", "webfetch web_fetch fetch_url read_url_content")
        register("HTTP Request", "network", "http_request httprequest")
        register("Use Skill", "book-open", "skill")
        register("Update Tasks", "list-checks", "todowrite todo_write")
        register("Run Agent", "bot", "agent subagent task")
        register("Run Batch", "layers", "batch")
        register("Search Documentation", "book-search", "codesearch code_search")
        register("Edit Files", "files", "multiedit multi_edit")
        register("Stop Agent", "bot-off", "stop_agent agentstop kill_agent")
        register("Ask Question", "circle-help", "askuserquestion ask_user_question ask_question question")
        register("Enter Plan Mode", "clipboard-list", "enterplanmode enter_plan_mode plan_mode plan")
        register("Exit Plan Mode", "clipboard-check", "exitplanmode exit_plan_mode")
        register("Report Findings", "clipboard-search", "reportfindings report_findings findings")
        register("Propose Skills", "book-plus", "proposeskills propose_skills")
        register("Propose Goal", "target", "proposegoal propose_goal")
        register("Send Feedback", "message-square", "sendfeedback send_feedback")
        register("Edit Notebook", "notebook-pen", "notebookedit notebook_edit")
        register("Extract Snippet", "scissors", "snip extract_snippet")
        register("Send File", "file-output", "senduserfile send_user_file")
        register("Create Task", "list-plus", "taskcreate task_create task_add")
        register("Read Task", "list-filter", "taskget task_get")
        register("List Tasks", "list", "tasklist task_list")
        register("Update Task", "list-todo", "taskupdate task_update")
        register("Stop Task", "list-x", "taskstop task_stop task_cancel")
        register("Read Task Output", "text-search", "taskoutput task_output")
        register("Create Schedule", "calendar-plus", "croncreate cron_create")
        register("Delete Schedule", "calendar-x", "crondelete cron_delete")
        register("List Schedules", "calendar-days", "cronlist cron_list")
        register("Schedule Wakeup", "alarm-clock", "schedulewakeup schedule_wakeup")
        register("Wait", "timer", "sleep delay")
        register("Notify", "bell", "pushnotification push_notification notify")
        register("Configure", "settings", "config config_tool")
        register("Inspect Context", "scan-text", "ctxinspect ctx_inspect")
        register("Enter Worktree", "git-branch", "enterworktree enter_worktree")
        register("Exit Worktree", "git-merge", "exitworktree exit_worktree")
        register("Memory", "brain", "memory remember")
        register("Call Extension", "plug", "call_mcp_tool callmcptool mcp_tool")
        register("List Resources", "library", "listmcpresources list_mcp_resources list_resources")
        register("Read Resource", "book-open-text", "readmcpresource read_mcp_resource read_resource")
        return result
    }()

    static func resolve(_ name: String) -> ToolPresentation {
        if let known = registry[name.lowercased()] { return known }
        let readable = name.replacingOccurrences(of: "mcp__", with: "")
            .replacingOccurrences(of: "__", with: " / ")
            .replacingOccurrences(of: "_", with: " ")
        return ToolPresentation(label: readable, icon: "plug", isBuiltIn: false)
    }

    var localizedLabel: String {
        isBuiltIn ? String(localized: String.LocalizationValue(label), bundle: .module) : label
    }

    static func status(call: AppToolCall, result: AppToolResult?) -> AppToolCallStatus {
        result?.isError == true ? .failed : call.status
    }

    static func webURL(for call: AppToolCall) -> URL? {
        guard ["globe", "network", "search"].contains(resolve(call.name).icon),
              let value = call.arguments["url"] ?? call.arguments["uri"],
              let url = URL(string: value),
              ["http", "https"].contains(url.scheme?.lowercased() ?? ""), url.host != nil
        else { return nil }
        return url
    }
}

enum BundledToolArtwork {
    static let images: [String: NSImage] = {
        var images: [String: NSImage] = [:]
        for name in Set(ToolPresentation.registry.values.map(\.icon)) {
            if let url = Bundle.module.url(forResource: "tool-" + name, withExtension: "png"),
               let image = NSImage(contentsOf: url) { images[name] = image }
        }
        return images
    }()
}

struct BundledToolIcon: View {
    let name: String
    var body: some View {
        if let image = BundledToolArtwork.images[name] {
            Image(nsImage: image)
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 16, height: 16)
                .accessibilityHidden(true)
        }
    }
}

/// Only packaged resources are resolved here. An unknown site never triggers
/// a favicon request or a read of a model-supplied filesystem path.
enum OfflineSiteIcons {
    static let domains: [String: String] = {
        guard let url = Bundle.module.url(forResource: "sites", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let map = try? JSONDecoder().decode([String: String].self, from: data) else { return [:] }
        return map
    }()

    static let images: [String: NSImage] = {
        var images: [String: NSImage] = [:]
        for asset in Set(domains.values) {
            if let url = Bundle.module.url(forResource: asset, withExtension: "png"),
               let image = NSImage(contentsOf: url) { images[asset] = image }
        }
        return images
    }()

    static func asset(for url: URL) -> String? { domains[url.host?.lowercased() ?? ""] }
}

struct OfflineSiteIcon: View {
    let url: URL
    var localImage: NSImage? = nil
    var body: some View {
        Group {
            if let localImage {
                Image(nsImage: localImage).resizable().scaledToFit()
            } else if let asset = OfflineSiteIcons.asset(for: url), let image = OfflineSiteIcons.images[asset] {
                Image(nsImage: image).renderingMode(.template).resizable().scaledToFit()
            } else {
                Text(String((url.host ?? "?").replacingOccurrences(of: "www.", with: "").prefix(1)).uppercased())
                    .themedFont(.tiny, weight: .semibold)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.appBorder, in: RoundedRectangle(cornerRadius: 4))
            }
        }
        .frame(width: 18, height: 18)
        .accessibilityHidden(true)
    }
}
