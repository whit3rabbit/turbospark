import SwiftUI

/// The Profiles settings pane: create, rename, delete, and switch between
/// the users of this installation. Switching is a save-and-relaunch
/// (`AppModel.switchToProfile`); this pane says so in as many words rather
/// than surprising someone with an app restart, and both destructive doors
/// (delete and switch) confirm before acting.
struct ProfilesSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var newProfileName: String = ""
    @State private var renameTarget: UserProfile?
    @State private var renameText: String = ""
    @State private var deleteTarget: UserProfile?
    @State private var switchTarget: UserProfile?

    var body: some View {
        Form {
            usersSection
            addSection
            notesSection
        }
        .formStyle(.grouped)
        .padding(16)
        .sheet(item: $renameTarget) { profile in
            renameSheet(profile)
        }
        .confirmationDialog(
            "Delete Profile",
            isPresented: Binding(
                get: { deleteTarget != nil },
                set: { if !$0 { deleteTarget = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete \"\(deleteTarget?.name ?? "")\"", role: .destructive) {
                if let target = deleteTarget {
                    model.deleteProfile(target)
                }
                deleteTarget = nil
            }
            Button("Cancel", role: .cancel) {
                deleteTarget = nil
            }
        } message: {
            Text("This removes the user's settings, chat history, projects, agents, global MCP servers and marketplaces, skills, plugins, custom tools, hooks, memory, and model favorites. The folder moves to the Trash, but restoring it does not bring the profile back; it is only for recovering files by hand. Shared and untouched: downloaded models, the install registry, the Keychain server API key, and the UI language.", bundle: .module)
        }
        .confirmationDialog(
            Text("Switch Profile", bundle: .module),
            isPresented: Binding(
                get: { switchTarget != nil },
                set: { if !$0 { switchTarget = nil } }),
            titleVisibility: .visible
        ) {
            Button("Switch and Relaunch") {
                if let target = switchTarget {
                    model.switchToProfile(target)
                }
                switchTarget = nil
            }
            Button("Cancel", role: .cancel) {
                switchTarget = nil
            }
        } message: {
            Text("The app saves everything and relaunches as this user. Each user has separate settings, chat history, skills, MCP servers, agents, and plugins; work stays with the user who created it.", bundle: .module)
        }
    }

    // MARK: - Sections (separate properties: a Form holding every section
    // inline is one type-checker expression and walks into the solver
    // budget, the reason `AppSettingsView.engineSettingsTab` is split too.)

    private var usersSection: some View {
        Section("Users on this Mac") {
            defaultProfileRow
            ForEach(model.profiles) { profile in
                additionalProfileRow(profile)
            }
            if model.profiles.isEmpty {
                Text("Only the Default user exists. Add one below to give it its own settings, chats, and skills.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var defaultProfileRow: some View {
        profileRow(
            name: UserProfileStore.defaultProfile.name,
            isCurrent: model.isDefaultProfileActive,
            subtitle: "Built in. Shares the ~/.turbospark skills, agents, and tools with other apps.",
            profile: nil)
    }

    private func additionalProfileRow(_ profile: UserProfile) -> some View {
        profileRow(
            name: profile.name,
            isCurrent: profile.id == model.currentProfile.id,
            subtitle: "Self-contained settings, chats, skills, plugins, and MCP servers.",
            profile: profile)
        .contextMenu {
            profileActions(profile)
        }
    }

    /// One row for either kind of user. `profile` is nil for the built-in
    /// Default user, which cannot be renamed or deleted and switches through
    /// the fixed `defaultProfile` row rather than a registry lookup.
    private func profileRow(
        name: String,
        isCurrent: Bool,
        subtitle: String,
        profile: UserProfile?
    ) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "person.crop.circle")
                .font(theme.ui(.title3))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(name)
                        .font(theme.ui(.base, weight: .medium))
                    if isCurrent {
                        Text("Current", bundle: .module)
                            .font(theme.ui(.tiny, weight: .semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.15), in: Capsule())
                            .foregroundStyle(Color.accentColor)
                    }
                }
                Text(subtitle)
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
            }
            // Combining only the text keeps the Switch button and the actions
            // menu individually reachable to VoiceOver; combining the whole
            // row flattened them into the label.
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(name)\(isCurrent ? ", current profile" : "")")
            Spacer()
            if !isCurrent {
                Button("Switch") {
                    switchTarget = profile ?? UserProfileStore.defaultProfile
                }
                .disabled(!model.canSwitchProfile)
                .help("Saves everything and relaunches the app as this user")
            }
            if let profile {
                Menu {
                    profileActions(profile)
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(theme.ui(.base))
                        .foregroundStyle(.secondary)
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .accessibilityLabel("More actions for \(name)")
            }
        }
    }

    /// The rename/delete pair, shared by the context menu and the row's
    /// ellipsis menu so neither surface can drift from the other.
    @ViewBuilder
    private func profileActions(_ profile: UserProfile) -> some View {
        Button("Rename...") {
            renameTarget = profile
            renameText = profile.name
        }
        if profile.id != model.currentProfile.id {
            Button("Delete...", role: .destructive) {
                deleteTarget = profile
            }
        }
    }

    private var addSection: some View {
        Section("Add a User") {
            HStack {
                TextField("Profile name", text: $newProfileName)
                    .onSubmit(addProfile)
                Button("Add Profile", action: addProfile)
                    .disabled(newProfileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            Text("New users start empty: their own settings, chats, skills, and MCP servers, isolated from the Default user and the shared ~/.turbospark folders.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.secondary)
        }
    }

    private func addProfile() {
        let name = newProfileName
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        model.createProfile(named: name)
        newProfileName = ""
    }

    private var notesSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                Text(
                    "Each profile is a folder under Application Support. Switching saves all "
                        + "work and relaunches the app; the Default user keeps using the shared "
                        + "~/.turbospark folders, and every other profile is self-contained. "
                        + "Downloaded models are shared by all users.")
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
                Button("Reveal Profiles Folder in Finder") {
                    model.revealProfilesInFinder()
                }
            }
        }
    }

    // MARK: - Rename sheet

    private func renameSheet(_ profile: UserProfile) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Rename Profile", bundle: .module)
                .font(theme.ui(.title3, weight: .semibold))
            TextField("Profile name", text: $renameText)
                .textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button("Cancel") {
                    renameTarget = nil
                }
                .keyboardShortcut(.cancelAction)
                Button("Rename") {
                    model.renameProfile(profile, to: renameText)
                    renameTarget = nil
                }
                .keyboardShortcut(.defaultAction)
                .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 340)
    }
}
