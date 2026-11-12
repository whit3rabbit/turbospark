import CTurboSpark
import Foundation

extension TurboSparkCatalog {
    /// Searches Hugging Face for TEXT models. One list request: no repository
    /// is probed, so a row carries advisory metadata and never claims a model
    /// runs. Call `hubCheck(repo:)` on the row the user selects.
    ///
    /// Blocks on the network (seconds, cached for 15 minutes by query), so run
    /// it off the main actor. A failed request returns a page whose `status`
    /// is not `.ok`; only a malformed call throws.
    ///
    /// - Parameters:
    ///   - query: At least two characters.
    ///   - limit: Results to request, 1 through 50. Default 20.
    ///   - refresh: Skip the response cache.
    public static func hubSearch(
        query: String,
        limit: Int? = nil,
        sort: HubSearchSort = .downloads,
        format: HubSearchFormat = .any,
        refresh: Bool = false
    ) throws -> HubSearchPage {
        let json = try encodeHubOptions(
            HubSearchOptions(
                limit: limit,
                sort: sort,
                format: format,
                refresh: refresh ? true : nil))
        return try decode(
            HubSearchPage.self,
            from: try takeString { out in
                query.withCString { q in
                    withOptionalCString(json) { o in
                        ts_hub_search_json(q, o, out)
                    }
                }
            })
    }

    /// The compatibility verdict for ONE repository, as a state to render.
    /// Branch on `HubCheck.state`: `refused` carries the probe's reason, and
    /// `needsAuth` and `unavailable` are not verdicts on the checkpoint.
    ///
    /// Seconds, not milliseconds: the file list, one header or `config.json`,
    /// and for safetensors a `HEAD` per shard. Run it off the main actor, for
    /// the repository the user picked and not for every search row.
    ///
    /// **`context`, `expertCacheSlots` and `loadGuard` MUST be what your
    /// sessions will OPEN with**, for `probe`'s reason.
    ///
    /// - Parameter sidecarHint: `owner/name`, a search row's `baseModel`. Used
    ///   only when a GGUF repository has no `tokenizer.json`; the result's
    ///   `sidecarRepo` says whether it was needed.
    /// - Parameter file: the `.gguf` to read instead of the best quantization.
    ///   The fit is per file, so a picker re-checks with the user's choice.
    public static func hubCheck(
        repo: String,
        sidecarHint: String? = nil,
        file: String? = nil,
        context: UInt32 = 4096,
        expertCacheSlots: OpenOptions.Sizing? = nil,
        loadGuard: OpenOptions.LoadGuard? = nil
    ) throws -> HubCheck {
        let json = try encodeHubOptions(
            HubCheckOptions(
                contextWindow: context == 0 ? nil : context,
                loadGuard: loadGuard,
                expertCacheSlots: expertCacheSlots,
                sidecarHint: sidecarHint,
                file: file))
        return try decode(
            HubCheck.self,
            from: try takeString { out in
                repo.withCString { r in
                    withOptionalCString(json) { o in
                        ts_hub_check_json(r, o, out)
                    }
                }
            })
    }

    /// Mirrors `encodeOptions`, which is private to `Catalog.swift`: NULL and
    /// `{}` mean the same thing to the ABI, so send nothing when there is
    /// nothing to send.
    private static func encodeHubOptions<T: Encodable>(_ options: T) throws -> String? {
        let data = try JSONEncoder().encode(options)
        let text = String(decoding: data, as: UTF8.self)
        return text == "{}" ? nil : text
    }

    /// `ts_hub_search_json`'s options bag. Absent fields take the ABI's
    /// defaults, so a default call sends the same JSON it always did.
    private struct HubSearchOptions: Encodable {
        let limit: Int?
        let sort: HubSearchSort
        let format: HubSearchFormat
        let refresh: Bool?
    }

    /// `ts_hub_check_json`'s options bag: `ProbeOptions`' three knobs plus the
    /// base-model hint.
    private struct HubCheckOptions: Encodable {
        let contextWindow: UInt32?
        let loadGuard: OpenOptions.LoadGuard?
        let expertCacheSlots: OpenOptions.Sizing?
        let sidecarHint: String?
        let file: String?
    }
}
