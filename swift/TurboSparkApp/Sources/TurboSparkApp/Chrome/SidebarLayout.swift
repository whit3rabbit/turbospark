import SwiftUI

/// The user's arrangement of the sidebar: which destinations are pinned to
/// the top group, in what order each group lists them, and which are hidden.
///
/// Pure value logic, so every rule is testable without a view. The default
/// (an empty layout) is `SidebarNavigationGroup` as shipped.
///
/// Rules that keep navigation from stranding anyone:
/// - Chat is owned by the Chat/Projects switcher and Models holds the model
///   manager, so neither can be hidden.
/// - A hidden destination stays reachable from the View menu and its shortcut,
///   and the section being shown is always listed even if hidden.
struct SidebarLayout: Codable, Equatable {
    typealias Section = AppModel.AppNavigationSection

    /// Section raw values in the user's order. Only the relative order of
    /// sections that share a group matters, so a partial list is fine.
    var order: [String] = []
    /// Section raw values pinned to the top group, in pin order.
    var pinned: [String] = []
    /// Section raw values removed from the sidebar.
    var hidden: [String] = []

    static let `default` = SidebarLayout()

    var isDefault: Bool { self == Self.default }

    // MARK: Rules

    static func isHideable(_ section: Section) -> Bool {
        section != .chat && section != .modelManager
    }

    func isHidden(_ section: Section) -> Bool { hidden.contains(section.rawValue) }
    func isPinned(_ section: Section) -> Bool { pinned.contains(section.rawValue) }

    /// Only destinations that do not already sit in the top group can be pinned.
    func canPin(_ section: Section) -> Bool {
        !isPinned(section) && !SidebarNavigationGroup.main.sections.contains(section)
    }

    var hiddenSections: [Section] {
        hidden.compactMap { Section(rawValue: $0) }.filter { Self.isHideable($0) }
    }

    // MARK: Layout

    /// The groups to draw, top group first, empty groups dropped.
    /// - Parameters:
    ///   - includesChat: false when the Chat/Projects switcher already owns Chat.
    ///   - keeping: a section that must stay listed even if hidden, so the
    ///     place the user is in is always visible.
    func groups(includesChat: Bool, keeping: Section? = nil) -> [[Section]] {
        let base = SidebarNavigationGroup.allCases.map { $0.sections }
        let pinnedSections = pinned.compactMap { Section(rawValue: $0) }
            .filter { section in
                !SidebarNavigationGroup.main.sections.contains(section)
                    && isVisible(section, keeping: keeping)
            }
        var result: [[Section]] = []
        for (index, sections) in base.enumerated() {
            var rows = sections.filter { section in
                guard isVisible(section, keeping: keeping) else { return false }
                // A pinned section moves to the top group.
                return index == 0 || !pinnedSections.contains(section)
            }
            rows = rows.enumerated()
                .sorted { lhs, rhs in
                    let l = rank(lhs.element), r = rank(rhs.element)
                    return l == r ? lhs.offset < rhs.offset : l < r
                }
                .map { $0.element }
            if index == 0 {
                // Chat first, then the pins, then the rest of the group.
                let chat = rows.filter { $0 == .chat }
                let rest = rows.filter { $0 != .chat }
                rows = chat + pinnedSections + rest
            }
            if !includesChat { rows.removeAll { $0 == .chat } }
            if !rows.isEmpty { result.append(rows) }
        }
        return result
    }

    private func isVisible(_ section: Section, keeping: Section?) -> Bool {
        !isHidden(section) || !Self.isHideable(section) || section == keeping
    }

    private func rank(_ section: Section) -> Int {
        order.firstIndex(of: section.rawValue) ?? Int.max
    }

    // MARK: Editing

    mutating func pin(_ section: Section) {
        guard canPin(section) else { return }
        pinned.append(section.rawValue)
    }

    mutating func unpin(_ section: Section) {
        pinned.removeAll { $0 == section.rawValue }
    }

    mutating func hide(_ section: Section) {
        guard Self.isHideable(section), !isHidden(section) else { return }
        hidden.append(section.rawValue)
        pinned.removeAll { $0 == section.rawValue }
    }

    mutating func show(_ section: Section) {
        hidden.removeAll { $0 == section.rawValue }
    }

    mutating func reset() { self = .default }

    /// The sections that may swap places with `section`: pins reorder among
    /// pins, everything else among the rest of its group. Chat never moves.
    func movableNeighbours(of section: Section) -> [Section] {
        guard section != .chat else { return [] }
        let all = groups(includesChat: true, keeping: section)
        guard let group = all.first(where: { $0.contains(section) }) else { return [] }
        if isPinned(section) { return group.filter { isPinned($0) } }
        return group.filter { $0 != .chat && !isPinned($0) }
    }

    func canMove(_ section: Section, by step: Int) -> Bool {
        let region = movableNeighbours(of: section)
        guard let index = region.firstIndex(of: section) else { return false }
        return region.indices.contains(index + step)
    }

    /// Move one place up (-1) or down (+1) within its region.
    mutating func move(_ section: Section, by step: Int) {
        var region = movableNeighbours(of: section)
        guard let index = region.firstIndex(of: section),
              region.indices.contains(index + step) else { return }
        region.swapAt(index, index + step)
        write(region)
    }

    /// Drop `section` before `target`, if they share a region.
    mutating func move(_ section: Section, before target: Section) {
        var region = movableNeighbours(of: section)
        guard section != target,
              let from = region.firstIndex(of: section),
              let to = region.firstIndex(of: target) else { return }
        region.remove(at: from)
        region.insert(section, at: from < to ? to - 1 : to)
        write(region)
    }

    private mutating func write(_ region: [Section]) {
        let ids = region.map(\.rawValue)
        if let first = region.first, isPinned(first) {
            // Keep the pins in the new order; other pins (none) keep their slots.
            pinned = ids + pinned.filter { !ids.contains($0) }
        } else {
            order.removeAll { ids.contains($0) }
            order.append(contentsOf: ids)
        }
    }
}

/// Persists `SidebarLayout` in UserDefaults and publishes changes, so the
/// sidebar and the icon rail redraw together. A value that no longer decodes
/// reads as the default layout rather than failing.
@MainActor
final class SidebarLayoutStore: ObservableObject {
    static let shared = SidebarLayoutStore()
    static let defaultsKey = "TurboSpark.sidebarLayout"

    @Published private(set) var layout: SidebarLayout
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.defaultsKey),
           let stored = try? JSONDecoder().decode(SidebarLayout.self, from: data) {
            layout = stored
        } else {
            layout = .default
        }
    }

    func update(_ change: (inout SidebarLayout) -> Void) {
        var next = layout
        change(&next)
        guard next != layout else { return }
        layout = next
        if next.isDefault {
            defaults.removeObject(forKey: Self.defaultsKey)
        } else if let data = try? JSONEncoder().encode(next) {
            defaults.set(data, forKey: Self.defaultsKey)
        }
    }
}

/// The context menu shared by sidebar rows and rail icons. Every action also
/// exists as an accessibility action on the row, so none needs a pointer.
@MainActor
struct SidebarRowCustomizationMenu: View {
    @ObservedObject var store: SidebarLayoutStore
    let section: AppModel.AppNavigationSection

    var body: some View {
        let layout = store.layout
        if layout.isPinned(section) {
            Button { store.update { $0.unpin(section) } } label: { Text("Unpin", bundle: .module) }
        } else if layout.canPin(section) {
            Button { store.update { $0.pin(section) } } label: { Text("Pin", bundle: .module) }
        }
        Button { store.update { $0.move(section, by: -1) } } label: { Text("Move Up", bundle: .module) }
            .disabled(!layout.canMove(section, by: -1))
        Button { store.update { $0.move(section, by: 1) } } label: { Text("Move Down", bundle: .module) }
            .disabled(!layout.canMove(section, by: 1))
        if SidebarLayout.isHideable(section) {
            Button { store.update { $0.hide(section) } } label: {
                Text("Hide from Sidebar", bundle: .module)
            }
        }
        if !layout.hiddenSections.isEmpty || !layout.isDefault {
            Divider()
        }
        if !layout.hiddenSections.isEmpty {
            Menu {
                ForEach(layout.hiddenSections) { hidden in
                    Button { store.update { $0.show(hidden) } } label: {
                        Text(LocalizedStringKey(hidden.title), bundle: .module)
                    }
                }
            } label: { Text("Show in Sidebar", bundle: .module) }
        }
        if !layout.isDefault {
            Button(role: .destructive) { store.update { $0.reset() } } label: {
                Text("Reset Sidebar Layout", bundle: .module)
            }
        }
    }
}

extension View {
    /// Context menu, drag-to-reorder and accessibility actions for one
    /// customizable sidebar destination.
    @MainActor
    func sidebarCustomizable(
        _ section: AppModel.AppNavigationSection, store: SidebarLayoutStore
    ) -> some View {
        self
            .contextMenu { SidebarRowCustomizationMenu(store: store, section: section) }
            .draggable(section.rawValue)
            .dropDestination(for: String.self) { items, _ in
                guard let id = items.first, let dragged = AppModel.AppNavigationSection(rawValue: id)
                else { return false }
                store.update { $0.move(dragged, before: section) }
                return true
            }
            .accessibilityAction(named: Text("Move Up", bundle: .module)) {
                store.update { $0.move(section, by: -1) }
            }
            .accessibilityAction(named: Text("Move Down", bundle: .module)) {
                store.update { $0.move(section, by: 1) }
            }
    }
}

/// Applies `sidebarCustomizable` to every destination except Chat, which the
/// Chat/Projects switcher owns and which cannot move, pin or hide.
@MainActor
struct CustomizableRow: ViewModifier {
    let section: AppModel.AppNavigationSection
    let store: SidebarLayoutStore

    @ViewBuilder
    func body(content: Content) -> some View {
        if section == .chat {
            content
        } else {
            content.sidebarCustomizable(section, store: store)
        }
    }
}
