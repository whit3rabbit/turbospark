import Foundation

/// Whether a file a project points at actually lives inside that project
/// (state#39).
///
/// **THREE READERS ANSWERED THIS QUESTION AND ONLY ONE OF THEM ASKED IT.**
/// `ProjectRuleDetector.readText` grew the check when a rules symlink was
/// found reaching `~/.aws/credentials` and putting it in every turn's system
/// prompt (state#23). `SkillParser` and `AgentParser` read the same class of
/// file out of the same untrusted clone and had no check at all: a
/// `.claude/skills/x/SKILL.md -> ~/.ssh/id_ed25519` becomes `skill.content`,
/// which the `skill` tool returns and `ToolRiskClassifier` rates always-safe,
/// and a `.claude/agents/y.md` symlink becomes a subagent's SYSTEM PROMPT
/// with no gate on the path at all.
///
/// One helper rather than three spellings, because the two that were missing
/// were missing precisely because nobody could see the one that existed.
public enum PathContainment {
    /// The canonical form of `url`, symlinks resolved.
    ///
    /// The ROOT is resolved as well as the target everywhere below, or a
    /// project under a symlinked path fails its own containment test --
    /// `/tmp` is a symlink to `/private/tmp` on macOS, so this is the common
    /// case rather than an exotic one.
    public static func canonical(_ url: URL) -> URL {
        url.resolvingSymlinksInPath().standardizedFileURL
    }

    /// Whether `url` resolves to `root` or to something beneath it.
    ///
    /// A symlink INSIDE the project still resolves and is allowed: this
    /// repository's own `CLAUDE.md` is a symlink to `AGENTS.md`, and refusing
    /// that would break the common case while catching nothing.
    public static func isContained(_ url: URL, in root: URL) -> Bool {
        let target = canonical(url).path
        let base = canonical(root).path
        let prefix = base.hasSuffix("/") ? base : base + "/"
        return target == base || target.hasPrefix(prefix)
    }

    /// The canonical `url` when it is contained in `root`, otherwise nil.
    ///
    /// `root` nil means no containment is claimed (a USER-scoped file, which
    /// the user placed under their own home directory on purpose), and the
    /// url is returned canonicalized. Passing nil is a decision, so make it
    /// at the call site rather than defaulting it here.
    public static func resolvedIfContained(_ url: URL, in root: URL?) -> URL? {
        guard let root else { return canonical(url) }
        return isContained(url, in: root) ? canonical(url) : nil
    }
}
