import Foundation

extension AppModel {
    /// Applies `change` to the network policy and saves it. Every edit from
    /// the Network pane and the approval card goes through here, so the
    /// mirror in `AppNetworkPolicyStore` and the settings file stay in step.
    func updateNetworkPolicy(_ change: (inout AppNetworkPolicy) -> Void) {
        var policy = networkPolicy
        change(&policy)
        guard policy != networkPolicy else { return }
        networkPolicy = policy
        persistSettingsDebounced()
    }

    /// Adds `entry` to the allowed or blocked list, removing it from the
    /// other one so a host is never on both. Returns false for an entry that
    /// is not a bare host (`AppNetworkPolicy.normalizedDomainEntry`).
    @discardableResult
    func addNetworkDomain(_ entry: String, allow: Bool) -> Bool {
        guard let domain = AppNetworkPolicy.normalizedDomainEntry(entry) else { return false }
        updateNetworkPolicy { policy in
            if allow {
                policy.deniedDomains.removeAll { $0 == domain }
                if !policy.allowedDomains.contains(domain) { policy.allowedDomains.append(domain) }
            } else {
                policy.allowedDomains.removeAll { $0 == domain }
                if !policy.deniedDomains.contains(domain) { policy.deniedDomains.append(domain) }
            }
        }
        return true
    }

    func removeNetworkDomain(_ domain: String, allow: Bool) {
        updateNetworkPolicy { policy in
            if allow {
                policy.allowedDomains.removeAll { $0 == domain }
            } else {
                policy.deniedDomains.removeAll { $0 == domain }
            }
        }
    }
}
