import SwiftUI

/// Modal detail sheet matching the Codex modal for inspecting and managing hooks grouped by lifecycle events.
public struct HookSourceDetailSheet: View {
    public let groupID: String
    @ObservedObject var hookStore: AppHookStore = .shared

    @State private var expandedHookIDs: Set<UUID> = []
    @State private var editingHook: AppHookCommand? = nil
    @State private var showingOptionsSheet = false
    @Environment(\.dismiss) private var dismiss

    public init(groupID: String) {
        self.groupID = groupID
    }

    private var currentGroup: AppHookSourceGroup? {
        hookStore.sourceGroups.first(where: { $0.id == groupID })
    }

    public var body: some View {
        VStack(spacing: 0) {
            if let group = currentGroup {
                // Modal Header
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Image(systemName: "gearshape")
                                .font(.title3)
                                .foregroundStyle(.secondary)
                            Text(group.title)
                                .font(.title2.weight(.bold))
                        }
                        Text(group.subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if !group.optionSpecs.isEmpty {
                        Button {
                            showingOptionsSheet = true
                        } label: {
                            Label("Options", systemImage: "slider.horizontal.3")
                        }
                        .controlSize(.small)
                    }

                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 24, height: 24)
                            .background(Color(nsColor: .controlBackgroundColor))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .appPointerCursor()
                }
                .padding(.horizontal, 24)
                .padding(.top, 20)
                .padding(.bottom, 14)
                .background(Color(nsColor: .windowBackgroundColor))

                Divider()

                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        // Review banner if untrusted hooks exist
                        if group.unreviewedCount > 0 {
                            reviewBanner(group: group)
                        }

                        // Event Groups
                        let eventsPresent = Array(Set(group.hooks.map { $0.event })).sorted { $0.rawValue < $1.rawValue }

                        if eventsPresent.isEmpty {
                            VStack(spacing: 12) {
                                Image(systemName: "link.badge.plus")
                                    .font(.system(size: 36))
                                    .foregroundStyle(.secondary)
                                Text("No hooks registered in \(group.title).")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 40)
                        } else {
                            ForEach(eventsPresent) { event in
                                eventSection(event: event, hooks: group.hooks.filter { $0.event == event })
                            }
                        }
                    }
                    .padding(24)
                }
            } else {
                VStack(spacing: 12) {
                    Text("Source group not found.")
                        .foregroundStyle(.secondary)
                    Button("Close") { dismiss() }
                }
                .padding(40)
            }
        }
        .frame(minWidth: 620, minHeight: 520)
        .sheet(item: $editingHook) { hook in
            HookEditorSheet(
                existingHook: hook,
                onSave: { updated in
                    if updated.sourceType == .custom {
                        hookStore.updateCustomHook(updated)
                    }
                    editingHook = nil
                },
                onDismiss: {
                    editingHook = nil
                }
            )
        }
        .sheet(isPresented: $showingOptionsSheet) {
            if let group = currentGroup {
                HookOptionsConfigSheet(group: group)
            }
        }
    }

    // MARK: - Review Warning Banner

    private func reviewBanner(group: AppHookSourceGroup) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(.orange)
                .font(.title3)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 4) {
                Text("Hooks can run outside of the sandbox so we ask you to review any recently installed or modified hooks")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)

                if group.unreviewedCount > 0 {
                    HStack(spacing: 12) {
                        Text("\(group.unreviewedCount) hook\(group.unreviewedCount == 1 ? "" : "s") pending trust review")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Button {
                            hookStore.trustAllInGroup(group.id)
                        } label: {
                            Label("Trust All in Group", systemImage: "checkmark.shield.fill")
                                .font(.caption.weight(.semibold))
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .tint(.orange)
                    }
                    .padding(.top, 4)
                }
            }

            Spacer()
        }
        .padding(14)
        .background(Color.orange.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.orange.opacity(0.35), lineWidth: 1)
        )
    }

    // MARK: - Event Section

    private func eventSection(event: AppHookEvent, hooks: [AppHookCommand]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: event.systemImage)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 1) {
                    Text(event.displayName)
                        .font(.headline)
                    Text(event.eventDescription)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                let unreviewedInEvent = hooks.filter { !hookStore.isHookTrusted($0) }.count
                if unreviewedInEvent > 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                        Text("\(unreviewedInEvent) needs review")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.orange)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.15))
                    .clipShape(Capsule())
                }
            }

            VStack(spacing: 1) {
                ForEach(hooks) { hook in
                    hookRow(hook: hook)
                }
            }
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.35), lineWidth: 1)
            )
        }
    }

    // MARK: - Hook Row & Expandable Card

    private func hookRow(hook: AppHookCommand) -> some View {
        let isExpanded = expandedHookIDs.contains(hook.id)
        let isTrusted = hookStore.isHookTrusted(hook)

        return VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                // Hook Name / Title
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(hook.name)
                            .font(.subheadline.weight(.medium))

                        if let matcher = hook.matcher, !matcher.isEmpty {
                            Text("for \(matcher)")
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.15))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }

                        if hook.isAsync {
                            Text("async")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color(nsColor: .separatorColor).opacity(0.3))
                                .clipShape(RoundedRectangle(cornerRadius: 3))
                        }
                    }
                }

                Spacer()

                // Source reveal link
                if let sourcePath = hook.sourcePath {
                    Button {
                        NSWorkspace.shared.selectFile(sourcePath, inFileViewerRootedAtPath: "")
                    } label: {
                        Image(systemName: "arrow.up.right.square")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Reveal source file: \(sourcePath)")
                }

                // Expand/Collapse Chevron
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        if isExpanded {
                            expandedHookIDs.remove(hook.id)
                        } else {
                            expandedHookIDs.insert(hook.id)
                        }
                    }
                } label: {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)

                // Trust Button (if unreviewed)
                if !isTrusted {
                    Button {
                        hookStore.trustHook(hook)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "checkmark.shield")
                            Text("Trust")
                        }
                        .font(.caption.weight(.semibold))
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .tint(.orange)
                }

                // Toggle Switch
                Toggle("", isOn: Binding(
                    get: { hook.isEnabled },
                    set: { _ in hookStore.toggleHookEnabled(id: hook.id) }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
                .controlSize(.small)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)

            // Expanded detail view
            if isExpanded {
                Divider()
                VStack(alignment: .leading, spacing: 10) {
                    // Command box
                    VStack(alignment: .leading, spacing: 4) {
                        Text(hook.type == .command ? "Command Script" : (hook.type == .http ? "Webhook URL" : "Prompt"))
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)

                        Text(hook.command)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(nsColor: .textBackgroundColor).opacity(0.7))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }

                    // Metadata row
                    HStack(spacing: 16) {
                        if let ifCond = hook.ifCondition, !ifCond.isEmpty {
                            Label("If: \(ifCond)", systemImage: "line.3.horizontal.decrease.circle")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Label("Shell: \(hook.shell.rawValue)", systemImage: "terminal")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Label("Timeout: \(Int(hook.timeoutSeconds))s", systemImage: "clock")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Spacer()

                        if hook.sourceType == .custom {
                            Button("Edit...") {
                                editingHook = hook
                            }
                            .controlSize(.small)

                            Button("Delete", role: .destructive) {
                                hookStore.deleteCustomHook(id: hook.id)
                            }
                            .controlSize(.small)
                        }
                    }
                }
                .padding(14)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
            }
        }
    }
}
