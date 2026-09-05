import Foundation
import SwiftUI
import TurboSpark

/// Deep architectural and runtime feature descriptor for installed models and catalog entries.
public struct ModelFeatureDescriptor: Sendable, Equatable {
    public enum RoutingType: String, Sendable, Equatable {
        case moe = "MoE"
        case dense = "Dense"

        public var label: String { rawValue }
        public var iconSystemName: String {
            switch self {
            case .moe: return "circle.grid.cross.fill"
            case .dense: return "cube.fill"
            }
        }
    }

    public enum SpeculativeDrafterType: String, Sendable, Equatable {
        case none = "None"
        case dynamicSloth = "Dynamic Sloth"
        case mtp = "MTP Head"

        public var label: String { rawValue }
        public var iconSystemName: String {
            switch self {
            case .none: return "bolt"
            case .dynamicSloth: return "bolt.badge.clock.fill"
            case .mtp: return "bolt.fill"
            }
        }
    }

    public enum WeightContainerFormat: String, Sendable, Equatable {
        case mlx = "MLX (.gturbo)"
        case gguf = "GGUF"

        public var shortLabel: String {
            switch self {
            case .mlx: return "MLX"
            case .gguf: return "GGUF"
            }
        }

        public var iconSystemName: String {
            switch self {
            case .mlx: return "shippingbox.fill"
            case .gguf: return "doc.zipper"
            }
        }
    }

    public enum StorageSource: String, Sendable, Equatable {
        case turboSpark = "TurboSpark Store"
        case lmStudio = "LM Studio"
        case custom = "Custom Folder"

        public var shortLabel: String {
            switch self {
            case .turboSpark: return "TurboSpark"
            case .lmStudio: return "LM Studio"
            case .custom: return "Custom"
            }
        }

        public var iconSystemName: String {
            switch self {
            case .turboSpark: return "sparkles"
            case .lmStudio: return "desktopcomputer"
            case .custom: return "folder.fill"
            }
        }
    }

    public let alias: String
    public let name: String
    public let family: String
    public let routingType: RoutingType
    public let routingDetails: String
    public let speculativeDrafter: SpeculativeDrafterType
    public let format: WeightContainerFormat
    public let quantFormat: String
    public let storageSource: StorageSource
    /// Whether this family's decode flow dispatches the steering edit.
    ///
    /// **DERIVED FROM THE INSTALL'S OWN `family`, against the exact set
    /// `crates/runtime/src/steering.rs`'s `family_dispatches_steering`
    /// answers true for.** It used to match alias SUBSTRINGS ("gemma",
    /// "qwen38", "mixtral", ...), which is the U2 mistake this file's own
    /// comments already record for routing: a side-loaded model whose alias
    /// happens to contain "gemma" is not thereby a steerable family, and the
    /// chain MISSED `gptOss` and `museGlimmer`, both of which steer and are
    /// measured doing so on real installs in `docs/OBLITERATION.md`.
    ///
    /// A pre-open ESTIMATE from an exact table. When a session for this model
    /// is open, `resolve` takes `info.steering.supported` instead, which is
    /// the engine answering rather than this app agreeing with it.
    public let isSteeringReady: Bool
    public let hasLinearAttention: Bool
    public let supportsChunkedPrefill: Bool
    public let supportsReasoning: Bool
    /// Whether this checkpoint's own markup frames tool calls the engine
    /// parses, or `nil` when that is not yet known.
    ///
    /// **`nil` UNTIL A SESSION IS OPEN, AND RENDERED AS "unknown".** This was
    /// a hardcoded `true`, i.e. a badge that cannot fail (`swift/CLAUDE.md`
    /// Gotcha 22), and it drove a "Tool Guardrails" chip on the detail pane.
    /// The fact is a property of the DIALECT, which is resolved from the
    /// tokenizer at load and appears in neither `installed.json` nor the
    /// manifest's `arch` block -- so there is genuinely no pre-open answer,
    /// and inventing one is Gotcha 23's "a zero from an absent measurement is
    /// not a measurement of zero".
    public let supportsToolCalls: Bool?
    public let supportsVision: Bool
    /// Rough estimate of the resident working set, in bytes. `nil` when
    /// there is no real measurement or on-disk size to derive it from --
    /// never a placeholder byte count presented as if it were one (U3;
    /// `swift/CLAUDE.md` Gotcha 23 fixed the same shape once already, on
    /// the sibling numeric fields on this same pane).
    public let estimatedWorkingSetRAM: UInt64?
    public let isSlotCacheStreaming: Bool
    public let contextLimit: Int?
    public let layerCount: Int?
    public let hiddenSize: Int?
    public let vocabSize: Int?
    public let expertCount: Int?
    public let topKExperts: Int?

    /// Family strings this port has an actual measured/documented MoE
    /// architecture for (`AGENTS.md` Gotcha 36, `CLAUDE.local.md`'s oracle
    /// ceilings). Matched against the INSTALL's own `family` field, which
    /// comes from the scanner/installer rather than a user-editable alias,
    /// unlike the free-text `alias`/`name` substrings this used to key off
    /// of (U2): a side-loaded model whose alias happens to contain "gemma"
    /// or "deepseek" is not thereby a Gemma-4-shaped MoE, and `qwen38` is
    /// this repo's dense MTP/DFlash2 family, not MoE at all -- the old
    /// alias-substring chain wrongly flagged it as MoE.
    private static let knownMoEFamilies: Set<String> = [
        "gemma4", "qwen36", "qwen3moe", "qwen35moe", "gptoss", "mixtral"
    ]

    // MARK: - Resolution

    /// Family strings whose decode flow dispatches the steering edit.
    ///
    /// **THE EXACT SET FROM `crates/runtime/src/steering.rs:family_dispatches_steering`,
    /// spelled with `ModelFamily::as_str`'s own strings** (note the camelCase
    /// in `gptOss` and `museGlimmer`, which is what `installed.json` really
    /// carries). Matched against the scanner-assigned `family`, never against
    /// the free-text alias.
    ///
    /// The two families deliberately absent: `deepseekV4Flash` has no decode
    /// flow at all, and `qwen4exp`'s residual is several streams wide, so the
    /// boundary the edit would sit on is a different shape and has not been
    /// decided. Requesting steering on either is refused at open BY NAME.
    private static let steeringFamilies: Set<String> = [
        "gemma4", "qwen36", "qwen35", "llama", "qwen3moe", "gptOss", "museGlimmer",
    ]

    /// - Parameter sessionInfo: the OPEN session's own report, when this
    ///   model is the one loaded. Authoritative for the two capability fields
    ///   that the engine can answer and this app can only estimate.
    public static func resolve(
        installedModel: InstalledModel?,
        catalogEntry: CatalogEntry? = nil,
        sessionInfo: SessionInfo? = nil
    ) -> ModelFeatureDescriptor {
        let alias = installedModel?.alias ?? catalogEntry?.alias ?? ""
        let family = installedModel?.family ?? catalogEntry?.family ?? ""
        let path = installedModel?.path ?? ""
        let repo = installedModel?.repo ?? ""

        // U1: join to the curated catalog by alias when the caller did not
        // already resolve one. An installed row's alias identifies a real
        // catalog row far more often than any caller actually threads a
        // `CatalogEntry` through -- the escape hatch existed but nothing
        // used it, so every catalog-backed GGUF install (Q4_K_M, Q6_K,
        // MXFP4, ...) fell through to the text-guessing below and rendered
        // as generic MLX/INT4.
        let resolvedCatalogEntry: CatalogEntry? = catalogEntry ?? installedModel.flatMap { model in
            (try? TurboSparkCatalog.available())?.first(where: { $0.alias == model.alias })
        }
        let name = resolvedCatalogEntry?.name ?? catalogEntry?.name ?? alias

        let lAlias = alias.lowercased()
        let lFamily = family.lowercased()
        let lName = name.lowercased()
        let lPath = path.lowercased()
        let lRepo = repo.lowercased()
        let combined = "\(lAlias) \(lFamily) \(lName) \(lPath) \(lRepo)"

        // 1. Storage Source (U4): the hardcoded ".turbospark" substring
        // check ignored `TURBOSPARK_HOME`, so any install under a
        // configured custom store root -- which contains no ".turbospark"
        // path component at all -- mistagged every one of its rows as
        // "Custom Folder", and the "TurboSpark" source filter returned
        // zero results for exactly the installs it should have matched.
        let turboSparkHomeOverride = ProcessInfo.processInfo.environment["TURBOSPARK_HOME"]
        let isUnderTurboSparkStore: Bool = {
            if let override = turboSparkHomeOverride, !override.isEmpty {
                let normalized = ModelStorageManager.expandPath(override).lowercased()
                if !normalized.isEmpty && lPath.hasPrefix(normalized) { return true }
            }
            return lPath.contains(".turbospark")
        }()
        let source: StorageSource
        if lRepo.hasPrefix("lm studio/") || lPath.contains(".lmstudio") {
            source = .lmStudio
        } else if lRepo.hasPrefix("custom/") || (!lPath.isEmpty && !isUnderTurboSparkStore && !lPath.contains(".lmstudio")) {
            source = .custom
        } else {
            source = .turboSpark
        }

        // 2. Manifest inspection if path exists -- the one source of
        // REAL, per-install measurements available without a live session.
        var manifestContext: Int? = nil
        var manifestLayers: Int? = nil
        var manifestHidden: Int? = nil
        var manifestVocab: Int? = nil
        var manifestExperts: Int? = nil
        var manifestTopK: Int? = nil

        if !path.isEmpty {
            let manifestURL = URL(fileURLWithPath: (path as NSString).expandingTildeInPath).appendingPathComponent("manifest.json")
            if let data = try? Data(contentsOf: manifestURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let arch = json["arch"] as? [String: Any] {
                    manifestContext = (arch["trainedContext"] as? Int) ?? (arch["slidingWindow"] as? Int)
                    manifestLayers = arch["numLayers"] as? Int
                    manifestHidden = arch["hiddenSize"] as? Int
                    manifestVocab = arch["vocabSize"] as? Int
                    manifestExperts = arch["numExperts"] as? Int
                    manifestTopK = arch["topKExperts"] as? Int
                }
            }
        }

        // 3. Format & quantization (U1): a resolved catalog entry's own
        // `name` states this explicitly (every GGUF row's name leads with
        // "GGUF", e.g. "(GGUF Q4_K_M)"; every MLX row leads with "MLX",
        // e.g. "(MLX INT4, group 64)") via the SAME derivation the model
        // hub already uses (`ModelFamilyVisuals.formatLabel`,
        // `swift/CLAUDE.md` Gotcha 22). Restating it per-branch here,
        // independent of that fix, was the bug. Only a side-loaded file
        // with no catalog row at all falls back to guessing from text.
        let isGGUFPathHint = lPath.hasSuffix(".gguf") || lAlias.contains("gguf") || lName.contains("gguf") || lRepo.contains("gguf")
        let catalogFormatLabel = resolvedCatalogEntry.map { ModelFamilyVisuals.formatLabel(alias: $0.alias, name: $0.name) }

        let format: WeightContainerFormat
        let quant: String
        if let catalogFormatLabel {
            format = catalogFormatLabel.lowercased().contains("gguf") || isGGUFPathHint ? .gguf : .mlx
            quant = catalogFormatLabel
        } else {
            format = isGGUFPathHint ? .gguf : .mlx
            // No catalog row to read a stated format from (a side-loaded
            // LM Studio/Custom file): the best information actually
            // available is text association, same as before.
            if combined.contains("ternary") || combined.contains("1.58") {
                quant = "Ternary (1.58-bit)"
            } else if combined.contains("1-bit") || combined.contains("1bit") || combined.contains("bonsai") {
                quant = "1-Bit Affine"
            } else if combined.contains("iq3_xxs") || combined.contains("iq3") {
                quant = "IQ3_XXS (3-bit)"
            } else if combined.contains("iq4_nl") || combined.contains("iq4_xs") || combined.contains("iq4") {
                quant = "IQ4_NL (4-bit)"
            } else if combined.contains("q4_k_m") || combined.contains("q4_k") {
                quant = "Q4_K_M"
            } else if combined.contains("q5_k") {
                quant = "Q5_K"
            } else if combined.contains("q6_k") {
                quant = "Q6_K"
            } else if combined.contains("q8_0") || combined.contains("q8") {
                quant = "Q8_0"
            } else if combined.contains("mxfp4") {
                quant = "MXFP4"
            } else if combined.contains("int8") {
                quant = "INT8"
            } else if combined.contains("fp16") || combined.contains("f16") {
                quant = "FP16"
            } else if combined.contains("int4") {
                quant = "INT4 (Group 64)"
            } else if isGGUFPathHint {
                quant = "GGUF Quantized"
            } else {
                quant = "MLX (format unconfirmed)"
            }
        }

        // 4. Routing Type & Details (U2): the install's own manifest is
        // authoritative when present; the alias/name-substring fallback is
        // now scoped to a small set of family strings this port actually
        // has a documented MoE architecture for, matched EXACTLY against
        // the scanner-assigned `family` rather than the free-text alias a
        // side-loaded file can carry anything in.
        let isMoE = (manifestExperts ?? 0) > 1 || Self.knownMoEFamilies.contains(lFamily)

        let routingType: RoutingType
        let routingDetails: String
        if isMoE {
            routingType = .moe
            if let exp = manifestExperts, let topK = manifestTopK {
                routingDetails = "\(topK)/\(exp) Experts"
            } else {
                // No per-family expert/top-k counts guessed here: getting
                // even one wrong (this file's own prior "4/128" for Gemma
                // 4, which is actually top-8, is exactly that mistake)
                // restates a fact this port already measures elsewhere
                // instead of reading it.
                routingDetails = "Routed Slot Cache"
            }
        } else {
            routingType = .dense
            routingDetails = "Full Resident Weights"
        }

        // 5. Speculative Drafter
        let drafter: SpeculativeDrafterType
        if combined.contains("dflash") || combined.contains("sloth") {
            drafter = .dynamicSloth
        } else if combined.contains("mtp") || lAlias.contains("qwen38") {
            drafter = .mtp
        } else {
            drafter = .none
        }

        // 6. Special capabilities
        //
        // The engine's answer wins whenever there is one. Before that, the
        // family table below is the estimate.
        let isSteeringReady =
            sessionInfo?.steering.supported ?? Self.steeringFamilies.contains(family)

        let hasLinearAttention = lAlias.contains("qwen36") || lFamily == "qwen36"
        let supportsChunkedPrefill = lAlias.contains("gemma") || lFamily.contains("gemma") || lFamily.contains("llama") || lAlias.contains("mistral")
        let supportsReasoning = combined.contains("think") || combined.contains("reason") || combined.contains("harmony") || lAlias.contains("gptoss") || lAlias.contains("museglimmer") || lAlias.contains("deepseek") || lAlias.contains("qwen")
        let supportsToolCalls: Bool? = sessionInfo?.toolCalling.native
        let supportsVision = combined.contains("vision") || combined.contains("vlm") || combined.contains("mrope")

        // 7. Working Set RAM estimation (U3): a dense model's on-disk size
        // is a genuine (if approximate) proxy for its resident weight size,
        // so that is reported when known; when it is not, this is `nil`
        // rather than a flat guessed byte count. For MoE, only the two
        // families this port has an actual measured oracle ceiling for
        // (`CLAUDE.local.md`) get a specific figure; every other MoE
        // family is `nil` rather than an unfounded average.
        let isSlotCache = (routingType == .moe)
        let estimatedRAM: UInt64?
        if isSlotCache {
            if lFamily == "gemma4" {
                estimatedRAM = 2_200_000_000 // ~2.2 GiB, measured oracle ceiling
            } else if lFamily == "qwen36" {
                estimatedRAM = 1_700_000_000 // ~1.7 GiB, measured oracle ceiling
            } else {
                estimatedRAM = nil
            }
        } else {
            let bytes = installedModel?.installBytes ?? catalogEntry?.installBytes ?? resolvedCatalogEntry?.installBytes ?? 0
            estimatedRAM = bytes > 0 ? bytes : nil
        }

        return ModelFeatureDescriptor(
            alias: alias,
            name: name,
            family: family,
            routingType: routingType,
            routingDetails: routingDetails,
            speculativeDrafter: drafter,
            format: format,
            quantFormat: quant,
            storageSource: source,
            isSteeringReady: isSteeringReady,
            hasLinearAttention: hasLinearAttention,
            supportsChunkedPrefill: supportsChunkedPrefill,
            supportsReasoning: supportsReasoning,
            supportsToolCalls: supportsToolCalls,
            supportsVision: supportsVision,
            estimatedWorkingSetRAM: estimatedRAM,
            isSlotCacheStreaming: isSlotCache,
            // No per-alias guessed default (U3): a trained context this
            // port did not read off the install's own manifest is unknown,
            // not "probably 4096".
            contextLimit: manifestContext,
            layerCount: manifestLayers,
            hiddenSize: manifestHidden,
            vocabSize: manifestVocab,
            expertCount: manifestExperts,
            topKExperts: manifestTopK
        )
    }
}
