import Foundation
import TurboSpark

extension AudioWorkspaceController {
    /// The models a page needs one of before it can be used, shown over the page
    /// until one is on this Mac.
    struct ModelGate: Equatable {
        let page: AudioWorkspacePage
        /// The page's curated models, in listing order. Never empty.
        let entries: [AudioProfile]
    }

    /// Non-nil while Music or Voice Cloning has none of its curated models on this
    /// Mac. It lifts by itself when an install lands, so nothing remembers a
    /// dismissal.
    ///
    /// **ANY SIGN OF AN INSTALL KEEPS THE PAGE OPEN.** A model on a drive that is
    /// away, one on disk the receipt store does not know yet, and one that needs
    /// attention each have their own statement and action in the model card, and a
    /// gate with only a Download button would hide them. The gate also waits for
    /// the catalog: before the first read there are no entries, and during a
    /// refresh a gate would flash over a model that is installed.
    var modelGate: ModelGate? {
        guard page == .music || page == .voiceClone, !isRefreshingModels, modelCatalogError == nil else { return nil }
        let entries = availableProfiles
        guard !entries.isEmpty,
              entries.allSatisfy({ installStatus(for: $0).offersDownload && modelState($0, download: nil) == .notInstalled })
        else { return nil }
        return ModelGate(page: page, entries: entries)
    }
}
