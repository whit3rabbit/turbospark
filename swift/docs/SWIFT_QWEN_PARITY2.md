# SWIFT_QWEN_PARITY2.md

The second web-shell parity pass: git commands (/diff, /log, /prs), bang (!cmd)
execution, split panes, sidebar organization, echarts code blocks, interactive
AskUserQuestion, and QR auth descoping.

Keep all code, comments, and docs ASCII: no emojis and no em dashes (project rule).

## Overview

This pass closes remaining parity gaps with modern AI web-shell interfaces,
focusing on developer workflows, workspace visibility, and interactive question
handling.

## Features

### 1. Git Parity Commands (/diff, /log, /prs)
- `/diff`: Shows staged, unstaged, or commit-range diffs directly in a modal sheet.
- `/log`: Displays recent commits with author, hash, relative date, and subject.
- `/prs`: Lists pull requests for the active project via GitHub CLI or API if configured.
- Implemented in `AppModel+GitParity.swift` and `ChatGitSheets.swift`.
- Git subprocess output uses the shared byte-capped executor, and diff bodies
  are additionally capped by line count before SwiftUI creates per-line rows.
  Both limits append a visible truncation marker.

### 2. Bang Command Execution (!cmd)
- Typing `!<command>` in the composer runs the command directly via the active shell
  without needing full agent iteration or natural language routing.
- Implemented in `AppModel+BangCommands.swift`.

### 3. Read-Only Split Panes
- Secondary view pane for comparing chat history, viewing side-by-side files, or
  monitoring ongoing generation while navigating past turns.
- Controlled via `AppModel+SplitView.swift` and rendered in `SplitChatPanes.swift`.

### 4. Sidebar Time Buckets and Project Accents
- Conversations grouped into recency buckets (Today, Yesterday, Previous 7 Days,
  Previous 30 Days, Older).
- Project accent colors allow visual grouping and quick switching across project contexts.
- Implemented in `ChatSidebarOrganizer.swift`, `ChatSidebarView.swift`, and
  `ChatSidebarChatRowView.swift`.

### 5. ECharts Code Block Previews
- Fenced code blocks with language `echarts` are recognized and rendered in preview cards.
- Integrated into `ChatMessageMarkdownView.swift`.

### 6. Interactive AskUserQuestion
- Static waiter pattern for tools requesting direct user choice or clarification.
- Rendered as an interactive banner/card (`InteractiveQuestionCardView.swift`) rather
  than plain text output.
- A multi-question set submits as ONE map from a footer button (the waiter is
  all-or-nothing, so a per-question early submit would strand questions 2..N);
  a single-question set answers on tap as before.
- Dismissing or closing the question acts as an answer or cancellation to unblock the agent loop.
- Implemented in `AppModel+Questions.swift` and `PlanningInteractiveExecutors.swift`.

### 7. QR Auth Descoping
- QR code login flows from mobile-centric web shells are explicitly descoped in favour
  of direct API key and local environment variable configuration.
