import XCTest

@testable import TurboSparkApp

/// S5: `isAutoApprovable` admitted an allowlisted head word regardless of the
/// flags or subcommand that turn it into a writer or an executor. The
/// `ToolRiskClassifier` denylist caught some of these by regex, but the
/// allowlist is the gate that must hold on its own (the subagent gate calls it
/// directly, and a denylist can only name what someone wrote down).
final class TerminalFlagHardeningTests: XCTestCase {

    /// Commands that execute, write or rewrite history even though their head
    /// word (or git subcommand) is on the allowlist. Each must be refused by
    /// the allowlist itself and be `.high` end to end.
    static let mustNotAutoApprove: [(command: String, why: String)] = [
        ("find . -delete", "find -delete"),
        ("find . -name '*.o' -exec rm {} +", "find -exec"),
        ("find . -fprint out.txt", "find -fprint writes a file"),
        ("find . -fls out.txt", "find -fls writes a file"),
        ("awk -f prog.awk data.txt", "awk program file can call system"),
        ("awk -i inplace 1 data.txt", "gawk in-place edit"),
        ("awk BEGIN system", "awk system keyword"),
        ("sort -o out.txt in.txt", "sort -o writes a file"),
        ("sort --output=out.txt in.txt", "sort --output writes a file"),
        ("tree -o out.txt", "tree -o writes a file"),
        ("rg --pre=./evil.sh foo", "rg --pre runs a command per file"),
        ("rg --pre ./evil.sh foo", "rg --pre runs a command per file"),
        ("git push origin main --force", "force flag after the refspec"),
        ("git push --force-with-lease origin main", "force with lease"),
        ("git push -f", "short force"),
        ("git push origin +main", "force refspec"),
        ("git push origin :main", "delete refspec"),
        ("git push --delete origin main", "delete remote branch"),
        ("git push --mirror", "mirror overwrites remote refs"),
        ("git push --prune origin", "prune remote refs"),
        ("git push origin main --receive-pack=evil", "receive-pack runs a command"),
        ("git fetch --upload-pack=evil origin", "upload-pack runs a command"),
        ("git pull --upload-pack=evil", "upload-pack runs a command"),
        ("git branch -D feature", "branch delete"),
        ("git branch --delete feature", "branch delete"),
        ("git branch -m old new", "branch rename"),
        ("git branch newbranch", "branch create via positional"),
        ("git tag -d v1", "tag delete"),
        ("git tag v1", "tag create"),
        ("git remote add evil https://example.com/x.git", "remote add"),
        ("git remote set-url origin https://example.com/x.git", "remote set-url"),
        ("git remote remove origin", "remote remove"),
        ("git diff --output=out.patch", "diff --output writes a file"),
        ("git log --output=out.txt", "log --output writes a file"),
        ("git show --output out.txt HEAD", "show --output writes a file"),
        ("git checkout .", "discards working tree changes"),
        ("git checkout -f main", "force checkout"),
        ("git checkout -- src/lib.rs", "restores a path"),
        ("git checkout -B main origin/main", "resets a branch"),
        ("git switch --discard-changes main", "discards changes"),
        ("git stash drop", "drops a stash"),
        ("git stash clear", "clears stashes"),
        ("python3 -cprint", "attached inline code flag"),
        ("python3 -ec x", "clustered inline code flag"),
        ("node -p process.env", "node --print evaluates code"),
        ("node --print=process.env", "node --print= evaluates code"),
        ("node --eval=x", "attached --eval"),
        ("python3 --command=x", "attached --command"),
    ]

    func testDangerousFlagFormsAreNotAutoApprovable() {
        for (command, why) in Self.mustNotAutoApprove {
            XCTAssertFalse(
                TerminalCommandClassifier.isAutoApprovable(command),
                "'\(command)' (\(why)) must not be auto-approvable")
        }
    }

    func testDangerousFlagFormsAskEndToEnd() {
        for (command, why) in Self.mustNotAutoApprove {
            let risk = ToolRiskClassifier.assessTerminalCommand(command)
            XCTAssertTrue(
                risk.isHighRisk, "'\(command)' (\(why)) must be .high, got \(risk.level)")
        }
    }

    /// The hardening must not turn `.auto` into a prompt on every build.
    func testOrdinaryFormsStillAutoApprove() {
        let ok: [String] = [
            "find . -name '*.swift'",
            "find src -type f -newer Cargo.toml",
            "awk '/foo/' data.txt",
            "sort -u names.txt",
            "sort -c data.txt",
            "tree -L 2",
            "rg -n 'struct' src/",
            "git status",
            "git diff HEAD~1",
            "git diff --stat",
            "git log --oneline -n 5",
            "git show HEAD",
            "git branch",
            "git branch -a",
            "git branch --list 'feat*'",
            "git branch --show-current",
            "git tag",
            "git tag --list",
            "git remote",
            "git remote -v",
            "git add -A",
            "git commit -m 'msg'",
            "git fetch origin",
            "git pull --rebase",
            "git push origin feature",
            "git push -u origin feature",
            "git checkout main",
            "git checkout -b topic",
            "git switch main",
            "git stash",
            "git stash list",
            "git stash push -m wip",
            "git stash pop",
            "python3 -m pytest tests/ -q",
            "node build.js",
            "mkdir -p build/artifacts",
        ]
        for command in ok {
            XCTAssertTrue(
                TerminalCommandClassifier.isAutoApprovable(command),
                "'\(command)' must stay auto-approvable")
        }
    }

    /// Core EGM: a sensitive path in an MCP argument is a deterministic guard
    /// and must keep its human, like the file tools do.
    func testMcpSensitivePathArgumentIsHardGated() {
        let risk = ToolRiskClassifier.assessMcpTool(
            name: "fs__read_file", arguments: ["path": "/etc/shadow"])
        XCTAssertTrue(risk.isHighRisk)
        XCTAssertTrue(risk.isHardGated)
    }

    /// Name-based MCP heuristics stay soft: the classifier exists to judge them.
    func testMcpNameHeuristicStaysSoft() {
        let risk = ToolRiskClassifier.assessMcpTool(name: "github__delete_repo", arguments: [:])
        XCTAssertTrue(risk.isHighRisk)
        XCTAssertFalse(risk.isHardGated)
    }
}
