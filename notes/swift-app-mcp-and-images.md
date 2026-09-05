---
uuid: "448c3480-812a-4052-a9f9-6dd35d430f69"
title: "TurboSparkApp: MCP transports and image attachments"
summary: "McpClientEngine speaks JSON-RPC 2.0 over stdio or SSE. Images ride an [ChatImage] field beside content, prepended to text, and are lost silently in three specific places if you're not careful"
tags: ["swift", "app", "agent", "vision"]
source: "swift/CLAUDE.md Gotchas 20 and 33"
created: "2026-09-05"
updated: "2026-09-05"
depends_on: ["7b3eeef2-d664-48a7-9b54-9b250768029e"]
---

## How does the app talk to MCP servers, and how do images reach the model?

`McpClientEngine` handles Model Context Protocol JSON-RPC 2.0
initialization and tool calls over two transports: stdio subprocesses and
HTTP/SSE endpoints. Config values like `${HOME}` and `${workspaceFolder}`
expand before spawning, and a server's `autoApprove` flags skip the
interactive approval prompt unless the operation classifies as high-risk.

Images are a newer addition (2026-08-30). `ChatMessage` kept
`content: String` and gained `images: [ChatImage]` beside it rather than
turning `content` into an enum, since dozens of call sites read `content`
with nothing to do with vision. A message with no image still encodes
`"content": "hello"` byte for byte. Only a message carrying an image
switches to the parts array, which is what keeps this from being an ABI
break.

## Don't

- Don't append an image after the text when building the parts array.
  `apply_chat_template` builds `[image, text]`, so appending shifts every
  mRoPE position past the image and silently produces a different (wrong)
  prompt for the same request.
- Don't gate the image-attach control on "does this family have a vision
  tower." Gate it on `info.vision.active` instead. An install can carry a
  tower and still refuse every image, because the pixel budget comes from
  the checkpoint's own `preprocessor_config.json`, which has no safe
  default, so an install streamed without that sidecar reports inactive.
- Don't build a second picker list for attachment content types. Read
  `AppModel.visionIsActive` / `attachmentContentTypes`, the one accessor
  pair also used by the load-guard's refusal reason. A second, independently
  assembled list is how a UI ends up accepting a file the turn then refuses.
- Don't assume a picture attached mid-conversation survives to the next
  agent step. Store `imagePaths` on the MESSAGE, not just the composer
  draft. The prompt rebuilds from the transcript on every agent step, so an
  image held only in the draft sends once and silently vanishes on step two.
- Don't gate turn submission on non-empty text. An image-only turn has no
  text and is still a valid turn. A `guard !msg.content.isEmpty` written
  before images existed drops it whole.
- Don't route image files through the same importer as documents.
  `DocumentTextExtractor` throws `unsupportedFormat` on every image type,
  correctly for its own job (pixels go to the vision tower, not to text
  extraction) but that path alone means images can't attach at all.
- Don't read `updateTokenEstimate` as exact for a turn carrying images. It's
  a FLOOR: the template emits one marker per image regardless of size, and
  the real expansion to the merged-token count happens later in the
  engine's splice.
