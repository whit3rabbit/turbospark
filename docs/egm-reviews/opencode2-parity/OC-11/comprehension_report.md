# Stage 6: Comprehension Test report (OC-11)

Target: docs/designs/opencode2-parity/OC-11-formatter-pass.md (full file). Reviewer egm-comprehension read the document alone. Completed.

## What this proposes

TurboSpark would add an opt-in, per-project step that runs the user's code formatter (swift-format, gofmt, clang-format, or user-configured ones like prettier, ruff, rustfmt) on a file right after the agent's edit tool writes it successfully. The formatter must be found on the login-shell PATH, otherwise the step is silently skipped. It runs without a shell, with a time limit. On success the tool output gets a one-line "formatted by X" note. On failure the file keeps the agent's content and the tool output gets the formatter's error tail. It only touches files the agent wrote that turn, inside the project root. It is off by default, controlled by a toggle in the project permissions UI.

## How the existing system works, per this document

There is no formatter pipeline. The only formatter-related code is output display helpers (ShellOutputFormatting.swift) and a terminal command classifier that recognizes prettier and rustfmt as known commands. The model may run a formatter itself, but nothing guarantees it. There is an existing "hardened process path" (argument vectors, no shell, bounded time) and existing "edit tools", "tool-output announcements", and per-project permissions and persistence. The document names these but does not describe them. OpenCode is said to run configured formatters after every agent write.

## Findings

CO-1 (Major, lines 48-50, 121-124). "The existing hardened process path" and "existing process hardening rules" are never defined, linked, or located. The reader cannot tell the time limit value, environment, or working directory. Fix: name the component or file and its guarantees in one sentence, or link its documentation.

CO-2 (Major, lines 42-44, 108-109). The login-shell PATH is the availability rule, but the document does not say how a GUI app obtains it, or how "swift-format (toolchain)" resolves when a toolchain binary is often not on PATH. Fix: state how PATH is obtained and how toolchain-bundled tools are found.

CO-3 (Major, lines 38-41, 70-73). The registry is "extension or language to formatter command template", but the template format is never shown: path substitution, in-place versus stdout, what "explicit commands" means, where users enter them. The architecture box says "rewrite file in place", which is only partly consistent. Fix: give one example template and say where user rows are configured.

CO-4 (Major, lines 48, 74-76, 134-141). "The edit tool", "file-write and edit tools", and "written-path result" are used without naming the tools. The Open Question hedges between two scopes while the Goals and CUJs already assume the narrow one. Fix: list the concrete tool names that trigger the pass and mark the Open Question resolved or deferred.

CO-5 (Major, lines 55, 140-141). "Files the agent did not write in that turn" has no mechanism: how written files are tracked, and what happens if the user also modified the file. Fix: define the tracking and the changed-on-disk case.

CO-6 (Major, lines 51-54, 85-87). "The file content shown in the transcript stays the post-format content" does not say whether the model is told the file changed, so its next edit may anchor on stale content. Fix: say whether post-format content or a diff returns to the model and how stale edits are avoided.

CO-7 (Major, lines 88-93 vs 42-44, 51-54). CUJ 2 says no error and no note when the formatter is missing, yet the Goals say the skip is "recorded in the pass log". The pass log is never defined. Fix: define the pass log (location, visibility) or drop the reference.

CO-8 (Nit, lines 45-47, 142-145). "Project" and "project root" are not defined, though the safety rule at line 56 depends on the root. Fix: define both in one line.

CO-9 (Nit, lines 121-124). "The project toggle plus the write permission cover the act" assumes the reader knows the permission model, and does not say who can author user rows or whether they need approval. Fix: state who authors rows and whether approval applies.

CO-10 (Nit, lines 3-8, 125, 155-157). Assumed references: "Section 8" of OPENCODE2_COMPARISON.md, "21 languages", "localization gates", `make design-check`, `make swift-lib`, "app scheme build". Fix: add a short note or link for each gate.

CO-11 (Nit, lines 26, 61-62, 148-151). The test file covers only the registry; nothing tests the runner, the failure note, or the project-root restriction. Fix: add runner and boundary tests or declare them out of scope.
