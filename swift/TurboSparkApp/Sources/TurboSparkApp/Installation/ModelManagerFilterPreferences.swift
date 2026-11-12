import Combine
import Foundation

/// The two Installed Models list choices that outlive the window: how the list
/// is grouped and whether it shows only what is available now (requirements 3.5
/// and 3.6, task 6.5).
///
/// **WHY THESE TWO PERSIST AND THE OTHER FILTERS DO NOT.** Search text, the
/// architecture, drafter and source filters, the tag and Favorites Only are
/// plain `@State` of the view and start fresh each launch: they answer "what am
/// I looking for right now". Grouping and "Available now" describe HOW the user
/// wants the list kept, and a user with a drive that is often unplugged sets
/// them once. Stored in `UserDefaults` under `TurboSpark.` keys like the other
/// per-window view choices (sidebar and inspector visibility).
///
/// **A VALUE THAT IS NOT ONE OF THE MODES READS AS "NONE".** An older or newer
/// build may have written another spelling; the list is never left grouped by
/// something this build cannot draw.
@MainActor
final class ModelManagerFilterPreferences: ObservableObject {
    static let groupOptionKey = "TurboSpark.modelManager.groupOption"
    static let availableOnlyKey = "TurboSpark.modelManager.availableOnly"

    @Published var groupOption: ModelManagerView.GroupOption {
        didSet {
            guard groupOption != oldValue else { return }
            defaults.set(groupOption.rawValue, forKey: Self.groupOptionKey)
        }
    }

    @Published var availableOnly: Bool {
        didSet {
            guard availableOnly != oldValue else { return }
            defaults.set(availableOnly, forKey: Self.availableOnlyKey)
        }
    }

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        groupOption = defaults.string(forKey: Self.groupOptionKey)
            .flatMap(ModelManagerView.GroupOption.init(rawValue:)) ?? .none
        availableOnly = defaults.bool(forKey: Self.availableOnlyKey)
    }
}
