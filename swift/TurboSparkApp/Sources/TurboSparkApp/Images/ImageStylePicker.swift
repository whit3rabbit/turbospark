import SwiftUI

/// Compact image style selector: favorites and category submenus in a
/// dropdown, with the searchable browser one click deeper.
struct ImageStyleMenu: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var favorites = ImageStyleFavorites.shared
    @State private var browsing = false

    private var selectedName: String? {
        model.imageStyleID.flatMap { AppImageStyleCatalog.style(id: $0)?.name }
    }

    var body: some View {
        Menu {
            Toggle(isOn: noneBinding) { Text("None", bundle: .module) }
            let starred = favorites.styles
            if !starred.isEmpty {
                Divider()
                ForEach(starred) { style in
                    styleToggle(style)
                }
            }
            Divider()
            ForEach(AppImageStyleCatalog.groups) { group in
                Menu {
                    ForEach(group.styles) { style in
                        styleToggle(style)
                    }
                } label: {
                    Text(verbatim: "\(group.name) (\(group.styles.count))")
                }
            }
            Divider()
            Button { browsing = true } label: {
                Label {
                    Text("Browse All Styles...", bundle: .module)
                } icon: {
                    Image(systemName: "magnifyingglass")
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "paintbrush")
                selectedName.map { Text(verbatim: $0) } ?? Text("None", bundle: .module)
            }
        }
        .fixedSize()
        .help(Text("Image style", bundle: .module))
        .accessibilityLabel(Text("Image style", bundle: .module))
        .sheet(isPresented: $browsing) {
            ImageStyleBrowserSheet(model: model)
        }
    }

    private var noneBinding: Binding<Bool> {
        Binding(
            get: { model.imageStyleID == nil },
            set: { if $0 { model.setImageStyle(id: nil) } })
    }

    private func styleToggle(_ style: AppImageStyle) -> some View {
        Toggle(
            isOn: Binding(
                get: { model.imageStyleID == style.id },
                set: { if $0 { model.setImageStyle(id: style.id) } })
        ) {
            Text(verbatim: style.name)
        }
    }
}

/// Searchable, star-able catalog browser opened from the style menu.
struct ImageStyleBrowserSheet: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var favorites = ImageStyleFavorites.shared
    @Environment(\.dismiss) private var dismiss
    @State private var search = ""
    /// Empty string means every category.
    @State private var categoryID = ""
    @FocusState private var searchFocused: Bool

    private var results: [AppImageStyle] {
        AppImageStyleCatalog.search(search)
            .filter { categoryID.isEmpty || $0.categoryID == categoryID }
    }

    private var favoriteResults: [AppImageStyle] {
        results.filter { favorites.contains($0.id) }
    }

    /// Non-favorite results grouped in catalog order. Favorites live in their
    /// own pinned section instead.
    private var categorySections: [(name: String, styles: [AppImageStyle])] {
        var order: [String] = []
        var buckets: [String: (name: String, styles: [AppImageStyle])] = [:]
        for style in results where !favorites.contains(style.id) {
            if buckets[style.categoryID] == nil {
                buckets[style.categoryID] = (style.categoryName, [])
                order.append(style.categoryID)
            }
            buckets[style.categoryID]?.styles.append(style)
        }
        return order.compactMap { buckets[$0] }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if results.isEmpty {
                emptyState
            } else {
                list
            }
            Divider()
            footer
        }
        .frame(minWidth: 560, idealWidth: 680, minHeight: 440, idealHeight: 540)
        .themedFont(.base)
        .onAppear { searchFocused = true }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Text("Style", bundle: .module)
                .themedFont(.title3, weight: .semibold)
            Spacer(minLength: 0)
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(.appSecondary)
                TextField(text: $search) { Text("Search styles", bundle: .module) }
                    .textFieldStyle(.plain)
                    .focused($searchFocused)
            }
            .padding(8)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
            .frame(width: 230)
            Picker(selection: $categoryID) {
                Text("All Categories", bundle: .module).tag("")
                ForEach(AppImageStyleCatalog.groups) { group in
                    Text(verbatim: group.name).tag(group.id)
                }
            } label: {
                Text("Category", bundle: .module)
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .fixedSize()
        }
        .padding(16)
    }

    private var list: some View {
        List {
            if !favoriteResults.isEmpty {
                Section {
                    ForEach(favoriteResults) { style in
                        row(style)
                    }
                } header: {
                    Text("Favorites", bundle: .module)
                }
            }
            ForEach(categorySections, id: \.name) { section in
                Section {
                    ForEach(section.styles) { style in
                        row(style)
                    }
                } header: {
                    Text(verbatim: section.name)
                }
            }
        }
        .listStyle(.inset)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "paintbrush")
                .themedFont(.hero)
                .foregroundStyle(.appSecondary)
            Text("No styles match your search.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var footer: some View {
        HStack {
            Button {
                model.setImageStyle(id: nil)
                dismiss()
            } label: {
                Text("None", bundle: .module)
            }
            Spacer(minLength: 0)
            Button { dismiss() } label: { Text("Done", bundle: .module) }
                .buttonStyle(.borderedProminent)
        }
        .padding(16)
    }

    private func row(_ style: AppImageStyle) -> some View {
        let isSelected = model.imageStyleID == style.id
        return Button {
            model.setImageStyle(id: style.id)
            dismiss()
        } label: {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? .appAccent : .appSecondary)
                    .padding(.top, 2)
                VStack(alignment: .leading, spacing: 4) {
                    Text(verbatim: style.name)
                        .themedFont(.base, weight: .medium)
                        .foregroundStyle(isSelected ? .appAccent : .appText)
                    Text(verbatim: style.prompt)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(2)
                        .truncationMode(.tail)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                favoriteButton(style)
            }
            .contentShape(Rectangle())
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
    }

    private func favoriteButton(_ style: AppImageStyle) -> some View {
        let starred = favorites.contains(style.id)
        return Button {
            favorites.toggle(style.id)
        } label: {
            Image(systemName: starred ? "star.fill" : "star")
                .foregroundStyle(starred ? .appAccent : .appSecondary)
                .padding(4)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            starred
                ? Text("Remove from favorites", bundle: .module)
                : Text("Add to favorites", bundle: .module)
        )
    }
}
