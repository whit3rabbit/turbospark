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

    public init(
        existingHook: AppHookCommand? = nil,
        onSave: @escaping (AppHookCommand) -> Void,
        onDismiss: @escaping () -> Void
    ) {
        self.existingHook = existingHook
        self.onSave = onSave
        self.onDismiss = onDismiss
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(existingHook == nil ? "New Lifecycle Hook" : "Edit Hook")
                        .font(.headline)
                    Text("Configure deterministic actions, guardrails, and lifecycle event handlers.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Hook Name
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Hook Name")
                            .font(.caption.weight(.semibold))
                        TextField("e.g. Lint before tool use, Auto-format on edit...", text: $name)
                            .textFieldStyle(.roundedBorder)
                    }

                    // Event & Type
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Lifecycle Event")
                                .font(.caption.weight(.semibold))
                            Picker("", selection: $event) {
                                ForEach(AppHookEvent.allCases) { ev in
                                    Text(ev.displayName).tag(ev)
                                }
                            }
                            .labelsHidden()
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Hook Type")
                                .font(.caption.weight(.semibold))
                            Picker("", selection: $type) {
                                ForEach(AppHookType.allCases) { t in
                                    Text(t.title).tag(t)
                                }
                            }
                            .labelsHidden()
                        }
                    }

                    // Command / URL / Prompt Text
                    VStack(alignment: .leading, spacing: 6) {
                        Text(type == .http ? "Webhook URL" : (type == .prompt ? "Evaluator Prompt" : "Shell Command or Script"))
                            .font(.caption.weight(.semibold))
                        TextEditor(text: $command)
                            .font(.system(.body, design: .monospaced))
                            .frame(minHeight: 90)
                            .padding(4)
                            .background(Color(nsColor: .controlBackgroundColor))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
                            )
                        Text("Payload is supplied via JSON over stdin. Exit code 2 blocks PreToolUse execution.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    // Matchers & Filters
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Tool Matcher (Optional)")
                                .font(.caption.weight(.semibold))
                            TextField("e.g. Write|Edit, Bash, *", text: $matcher)
                                .textFieldStyle(.roundedBorder)
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("If Condition (Optional)")
                                .font(.caption.weight(.semibold))
                            TextField("e.g. Bash(git *), Read(*.ts)", text: $ifCondition)
                                .textFieldStyle(.roundedBorder)
                        }
                    }

                    // Shell, Timeout, Status Message
                    HStack(spacing: 12) {
                        if type == .command {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Shell")
                                    .font(.caption.weight(.semibold))
                                Picker("", selection: $shell) {
                                    ForEach(AppHookShell.allCases) { sh in
                                        Text(sh.rawValue).tag(sh)
                                    }
                                }
                                .labelsHidden()
                            }
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Timeout (Seconds)")
                                .font(.caption.weight(.semibold))
                            TextField("30", value: $timeoutSeconds, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                            Text("Hooks discovered from Claude Code config default to 600s when unset.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Status Message")
                                .font(.caption.weight(.semibold))
                            TextField("e.g. Running pre-check...", text: $statusMessage)
                                .textFieldStyle(.roundedBorder)
                        }
                    }

                    // Async Toggle
                    Toggle("Run in background asynchronously (non-blocking)", isOn: $isAsync)
                        .font(.subheadline)
                        .padding(.top, 4)
                }
                .padding(20)
            }

            Divider()

            // Footer
            HStack {
                Button("Cancel") {
                    onDismiss()
                }
                .controlSize(.regular)

                Spacer()

                Button("Save Hook") {
                    save()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(nsColor: .windowBackgroundColor))
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
