import SwiftUI

/// Sheet for creating or editing lifecycle hooks.
public struct HookEditorSheet: View {
    public let existingHook: AppHookCommand?
    public let onSave: (AppHookCommand) -> Void
    public let onDismiss: () -> Void

    @State private var name: String = ""
    @State private var event: AppHookEvent = .preToolUse
    @State private var type: AppHookType = .command
    @State private var command: String = ""
    @State private var matcher: String = ""
    @State private var ifCondition: String = ""
    @State private var shell: AppHookShell = .zsh
    @State private var timeoutSeconds: Double = 30.0
    @State private var statusMessage: String = ""
    @State private var isAsync: Bool = false
    @FocusState private var isNameFocused: Bool

    public init(
        existingHook: AppHookCommand? = nil,
        onSave: @escaping (AppHookCommand) -> Void,
        onDismiss: @escaping () -> Void
    ) {
        self.existingHook = existingHook
        self.onSave = onSave
        self.onDismiss = onDismiss
    }

    /// The execution engine never evaluates `.prompt` hooks, so a NEW one
    /// would silently gate nothing. It stays selectable only when editing a
    /// discovered hook that already has that type, so the editor can show it.
    static func selectableTypes(existing: AppHookType?) -> [AppHookType] {
        AppHookType.allCases.filter { $0 != .prompt || existing == .prompt }
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Group {
                        if existingHook == nil {
                            Text("New Lifecycle Hook", bundle: .module)
                        } else {
                            Text("Edit Hook", bundle: .module)
                        }
                    }
                    .themedFont(.base, weight: .semibold)
                    Text("Configure deterministic actions, guardrails, and lifecycle event handlers.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                AppIconButton("xmark", label: Text("Close", bundle: .module)) {
                    onDismiss()
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(.appPage)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Hook Name
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Hook Name", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                        TextField(text: $name, prompt: Text("e.g. Lint before tool use, Auto-format on edit...", bundle: .module)) {
                            Text("Hook Name", bundle: .module)
                        }
                        .focused($isNameFocused)
                        .labelsHidden()
                        .textFieldStyle(.roundedBorder)
                    }

                    // Event & Type
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Lifecycle Event", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            // Event names are the identifiers hook configs
                            // use (PreToolUse, Stop, ...), shown verbatim.
                            Picker(selection: $event) {
                                ForEach(AppHookEvent.allCases) { ev in
                                    Text(ev.displayName).tag(ev)
                                }
                            } label: {
                                Text("Lifecycle Event", bundle: .module)
                            }
                            .labelsHidden()
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Hook Type", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            Picker(selection: $type) {
                                ForEach(Self.selectableTypes(existing: existingHook?.type)) { t in
                                    // `title` is fixed English from the model
                                    // layer; each value is a catalog key.
                                    Text(LocalizedStringKey(t.title), bundle: .module).tag(t)
                                }
                            } label: {
                                Text("Hook Type", bundle: .module)
                            }
                            .labelsHidden()
                        }
                    }

                    // Command / URL / Prompt Text
                    VStack(alignment: .leading, spacing: 6) {
                        commandTitle
                            .themedFont(.small, weight: .semibold)
                        TextEditor(text: $command)
                            .themedCode(.base)
                            .appEditorSurface(minHeight: 90)
                            .accessibilityLabel(commandTitle)
                        Text("Payload is supplied via JSON over stdin. Exit code 2 blocks PreToolUse execution.", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                    }

                    // Matchers & Filters
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Tool Matcher (Optional)", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField(text: $matcher, prompt: Text(verbatim: "Write|Edit, Bash, *")) {
                                Text("Tool Matcher (Optional)", bundle: .module)
                            }
                            .labelsHidden()
                            .textFieldStyle(.roundedBorder)
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("If Condition (Optional)", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField(text: $ifCondition, prompt: Text(verbatim: "Bash(git *), Read(*.ts)")) {
                                Text("If Condition (Optional)", bundle: .module)
                            }
                            .labelsHidden()
                            .textFieldStyle(.roundedBorder)
                        }
                    }

                    // Shell, Timeout, Status Message
                    HStack(spacing: 12) {
                        if type == .command {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Shell", bundle: .module)
                                    .themedFont(.small, weight: .semibold)
                                Picker(selection: $shell) {
                                    ForEach(AppHookShell.allCases) { sh in
                                        Text(verbatim: sh.rawValue).tag(sh)
                                    }
                                } label: {
                                    Text("Shell", bundle: .module)
                                }
                                .labelsHidden()
                            }
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Timeout (Seconds)", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField(value: $timeoutSeconds, format: .number, prompt: Text(verbatim: "30")) {
                                Text("Timeout (Seconds)", bundle: .module)
                            }
                            .labelsHidden()
                            .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                            Text("Hooks discovered from Claude Code config default to 600s when unset.", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Status Message", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField(text: $statusMessage, prompt: Text("e.g. Running pre-check...", bundle: .module)) {
                                Text("Status Message", bundle: .module)
                            }
                            .labelsHidden()
                            .textFieldStyle(.roundedBorder)
                        }
                    }

                    // Async Toggle
                    Toggle(isOn: $isAsync) {
                        Text("Run in background asynchronously (non-blocking)", bundle: .module)
                    }
                    .themedFont(.small)
                    .padding(.top, 4)
                    // The engine ignores async for these events (the result gates the
                    // action), so the switch would promise something it cannot deliver.
                    .disabled(AppHookExecutionEngine.blockingEvents.contains(event))
                    .help(AppHookExecutionEngine.blockingEvents.contains(event)
                        ? Text("Blocking events cannot be run asynchronously.", bundle: .module)
                        : Text("Run in background asynchronously (non-blocking)", bundle: .module))
                }
                .padding(20)
            }

            Divider()

            // Footer
            HStack {
                Button {
                    onDismiss()
                } label: { Text("Cancel", bundle: .module) }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button {
                    save()
                } label: { Text("Save Hook", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
                .disabled(
                    command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        || (type == .prompt && existingHook?.type != .prompt))
                .help(command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? Text("Enter a command script to save this hook.", bundle: .module)
                    : Text("Save Hook", bundle: .module))
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(.appPage)
        }
        .frame(minWidth: 540, minHeight: 480)
        .onAppear {
            if let hook = existingHook {
                name = hook.name
                event = hook.event
                type = hook.type
                command = hook.command
                matcher = hook.matcher ?? ""
                ifCondition = hook.ifCondition ?? ""
                shell = hook.shell
                timeoutSeconds = hook.timeoutSeconds
                statusMessage = hook.statusMessage ?? ""
                isAsync = hook.isAsync
            }
            isNameFocused = true
        }
    }

    /// The command field's title, which is also its accessible name.
    private var commandTitle: Text {
        switch type {
        case .http: return Text("Webhook URL", bundle: .module)
        case .prompt: return Text("Evaluator Prompt", bundle: .module)
        case .command: return Text("Shell Command or Script", bundle: .module)
        }
    }

    private func save() {
        let hookName = name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "\(event.rawValue) Hook"
            : name.trimmingCharacters(in: .whitespacesAndNewlines)

        let hook = AppHookCommand(
            id: existingHook?.id ?? UUID(),
            name: hookName,
            event: event,
            type: type,
            command: command.trimmingCharacters(in: .whitespacesAndNewlines),
            ifCondition: ifCondition.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : ifCondition.trimmingCharacters(in: .whitespacesAndNewlines),
            matcher: matcher.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : matcher.trimmingCharacters(in: .whitespacesAndNewlines),
            shell: shell,
            timeoutSeconds: timeoutSeconds > 0 ? timeoutSeconds : 30.0,
            statusMessage: statusMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : statusMessage.trimmingCharacters(in: .whitespacesAndNewlines),
            isAsync: isAsync,
            isEnabled: existingHook?.isEnabled ?? true,
            sourceType: existingHook?.sourceType ?? .custom,
            sourcePath: existingHook?.sourcePath,
            pluginName: existingHook?.pluginName,
            options: existingHook?.options ?? [:]
        )

        onSave(hook)
    }
}
