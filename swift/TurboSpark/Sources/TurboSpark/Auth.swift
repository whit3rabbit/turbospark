import CTurboSpark
import Foundation

/// Hugging Face token validation status returned by the whoami API.
public enum HfTokenValidationStatus: Decodable, Sendable, Equatable {
    case missing
    case valid(name: String?, fullname: String?, email: String?)
    case invalid(message: String?)
    case rateLimited(retryAfterSeconds: UInt64?)
    case unavailable(message: String)

    enum CodingKeys: String, CodingKey {
        case status, name, fullname, email, message
        case retryAfterSeconds = "retry_after_seconds"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let status = try container.decode(String.self, forKey: .status)
        switch status {
        case "missing":
            self = .missing
        case "valid":
            let name = try container.decodeIfPresent(String.self, forKey: .name)
            let fullname = try container.decodeIfPresent(String.self, forKey: .fullname)
            let email = try container.decodeIfPresent(String.self, forKey: .email)
            self = .valid(name: name, fullname: fullname, email: email)
        case "invalid":
            let msg = try container.decodeIfPresent(String.self, forKey: .message)
            self = .invalid(message: msg)
        case "rate_limited":
            let retry = try container.decodeIfPresent(UInt64.self, forKey: .retryAfterSeconds)
            self = .rateLimited(retryAfterSeconds: retry)
        case "unavailable":
            let msg = try container.decode(String.self, forKey: .message)
            self = .unavailable(message: msg)
        default:
            self = .unavailable(message: "Unknown status: \(status)")
        }
    }
}

/// Resolved Hugging Face token and its source origin.
public struct HfTokenInfo: Decodable, Sendable, Equatable {
    public let token: String
    public let source: String

    public init(token: String, source: String) {
        self.token = token
        self.source = source
    }
}

extension TurboSparkCatalog {
    /// Reads the currently resolved Hugging Face token, if any.
    public static func getHfToken() throws -> String? {
        try takeOptionalString { out in
            ts_hf_token_get(out)
        }
    }

    /// Reads the currently resolved Hugging Face token and its source origin.
    public static func getHfTokenInfo() throws -> HfTokenInfo? {
        guard let json = try takeOptionalString({ out in
            ts_hf_token_info_json(out)
        }) else {
            return nil
        }
        return try decode(HfTokenInfo.self, from: json)
    }

    /// Saves a Hugging Face token to the local store (~/.turbospark/hf_token).
    public static func setHfToken(_ token: String) throws {
        try check(token.withCString { ts_hf_token_set($0) })
    }

    /// Clears the Hugging Face token from the local store.
    public static func clearHfToken() throws {
        try check(ts_hf_token_clear())
    }

    /// Validates a Hugging Face token against the whoami-v2 API.
    public static func validateHfToken(_ token: String) throws -> HfTokenValidationStatus {
        let json = try takeString { out in
            token.withCString { ts_hf_token_validate_json($0, out) }
        }
        return try decode(HfTokenValidationStatus.self, from: json)
    }

    /// Reads the current Hugging Face mirror base URL ($HF_ENDPOINT).
    public static func getHfEndpoint() throws -> String {
        try takeString { out in
            ts_hf_endpoint_get(out)
        }
    }

    /// Sets or clears the Hugging Face mirror base URL ($HF_ENDPOINT).
    /// Pass nil or an empty string to remove the override and reset to default.
    public static func setHfEndpoint(_ endpoint: String?) throws {
        if let endpoint, !endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try check(endpoint.withCString { ts_hf_endpoint_set($0) })
        } else {
            try check(ts_hf_endpoint_set(nil))
        }
    }

    /// Checks whether a Hugging Face API token is currently configured or ambiently available.
    public static func hasHfToken() throws -> Bool {
        try getHfToken() != nil
    }
}
