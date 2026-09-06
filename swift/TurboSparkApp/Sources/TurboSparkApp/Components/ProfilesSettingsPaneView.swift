import SwiftUI

/// The Profiles settings pane: create, rename, delete, and switch between
/// the users of this installation. Switching is a save-and-relaunch
/// (`AppModel.switchToProfile`); this pane says so in as many words rather
/// than surprising someone with an app restart.
struct ProfilesSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var newProfileName: String = ""
    @State private var renameTarget: UserProfile?
    @State private var renameText: String = ""
    @State private var deleteTarget: UserProfile?

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
            Text(
                "The profile's settings, chats, and skills folder move to the Trash. "
                    + "Downloaded models stay shared and are not deleted.")
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
                Text("Only the Default user exists. Add one below to give it its own settings, chats, and skills.")
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var defaultProfileRow: some View {
        profileRowHeader(
            name: UserProfileStore.defaultProfile.name,
            isCurrent: model.isDefaultProfileActive,
            subtitle: "Built in. Shares the ~/.turbospark skills, agents, and tools with other apps.")
    }

    private func additionalProfileRow(_ profile: UserProfile) -> some View {
        let isCurrent = profile.id == model.currentProfile.id
        return profileRowHeader(
            name: profile.name,
            isCurrent: isCurrent,
            subtitle: "Self-contained settings, chats, skills, plugins, and MCP servers.")
        .contextMenu {
            Button("Rename...") {
                renameTarget = profile
                renameText = profile.name
            }
            if !isCurrent {
                Button("Delete...", role: .destructive) {
                    deleteTarget = profile
                }
            }
        }
    }

    private func profileRowHeader(name: String, isCurrent: Bool, subtitle: String) -> some View {
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
                        Text("Current")
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
            Spacer()
            if !isCurrent {
                Button("Switch") {
                    if name == UserProfileStore.defaultProfile.name {
                        model.switchToProfile(UserProfileStore.defaultProfile)
                    } else if let profile = model.profiles.first(where: { $0.name == name }) {
                        model.switchToProfile(profile)
                    }
                }
                .disabled(!model.canSwitchProfile)
                .help("Saves everything and relaunches the app as this user")
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(name)\(isCurrent ? ", current profile" : "")")
    }

    private var addSection: some View {
        Section("Add a User") {
            HStack {
                TextField("Profile name", text: $newProfileName)
                    .onSubmit(addProfile)
                Button("Add Profile", action: addProfile)
                    .disabled(newProfileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
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
            Text("Rename Profile")
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
