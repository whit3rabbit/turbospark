# Projects

## Overview
Tool schemas and payload types for three project-adjacent capabilities (Artifact publishing, git worktrees, and the Projects knowledge-base tool), plus the one executor that has real behavior here: `WorktreeExecutor`. The Artifact and Projects entries are Codable input/output structs and `OpenAITool` schemas only; no executor for them lives in this directory.

## Interface
- Artifact: `ArtifactInput`, `ArtifactItemSummary`, `ArtifactOutput` [ArtifactWorktreeTools.swift].
- Worktree payloads: `EnterWorktreeInput/Output`, `ExitWorktreeInput/Output` [ArtifactWorktreeTools.swift].
- Schemas: `ArtifactWorktreeDefinitions` (`artifact`, `enterWorktree`, `exitWorktree`, `all`) [ArtifactWorktreeTools.swift:ArtifactWorktreeDefinitions]; `ProjectDocDefinitions` (`projects`, `all`) [ProjectDocTools.swift:ProjectDocDefinitions].
- Projects payloads: `ProjectsInput`, `ProjectDocInfo`, `ProjectSearchHit`, `ProjectsOutput` [ProjectDocTools.swift].
- Executor: `WorktreeExecutor.enter(arguments:rootURL:)` and `.exit(arguments:rootURL:)` [WorktreeExecutor.swift:WorktreeExecutor].
- Child directories: none.

## Invariants
- `EnterWorktree` does not move the tool root. File and shell tools keep acting on the main checkout, and the result text says so explicitly. Do not reword it to claim the session "entered" the worktree, since that led the model to run destructive experiments believing it was isolated [WorktreeExecutor.swift:enter].
- Worktree paths go through `AppToolRegistry.resolveSecurePath(relPath:rootURL:)`, so they cannot escape the project root. The default location is `.turbospark/worktrees/<branch>` [WorktreeExecutor.swift].
- The branch name is validated with `git check-ref-format --branch` before use (error 72). Creation tries `worktree add -b <branch> <path> HEAD` first and falls back to checking out an existing branch. Only a zero exit counts as success; an existing directory proves nothing (error 70, with a timeout note) [WorktreeExecutor.swift:enter].
- `ExitWorktree` with `action: remove` requires a `path` (or `worktree_path`), because nothing tracks which worktree was created (error 73, nothing removed). `discard_changes` ("true" or "1") adds `--force`. Any other action leaves the worktree in place and says so. Removal failure is error 71 [WorktreeExecutor.swift:exit].
- All git runs go through `HardenedGit.run` with a 15 s timeout, because repository config is attacker-influenceable. Do not call `git` through a plain shell here [WorktreeExecutor.swift:runGit].
- Error codes under domain `TurboSparkTool`: 70 create failed, 71 remove failed, 72 invalid branch name, 73 remove without path.
- `ExitWorktreeInput.path` matches the advertised `path` (the executor also accepts an unadvertised `worktree_path` alias). The `EnterWorktree` `path` is where the worktree is created (it must not exist), not an existing worktree; `ToolSchemaCodeAgreementTests` pins both [ArtifactWorktreeTools.swift, WorktreeExecutor.swift].

## Side Effects
- Process spawn: git subprocesses via `HardenedGit.run` (`check-ref-format`, `worktree add`, `worktree remove`) in the project root [WorktreeExecutor.swift].
- Disk: creates the parent directory of the worktree and a new git worktree and branch in the repository; `worktree remove` deletes a worktree directory [WorktreeExecutor.swift].
- None for the Artifact and Projects types (data and schemas only), and none for network, Keychain, UserDefaults, or NotificationCenter.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
