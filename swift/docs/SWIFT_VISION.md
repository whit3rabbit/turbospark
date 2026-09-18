# Swift vision (image attachments)

Added 2026-08-30. How an image reaches a model turn: the wire shape, the
prompt-ordering rule, the gate on the attach control, and the four places a
picture used to be lost rather than errored.

Read this before touching `ChatMessage.images`, `AppChatMessage.imagePaths`,
`AttachmentImporter`, or `info.vision`.

## The wire shape stays backward compatible on purpose

Images travel as ORDERED content parts, and the one thing that keeps that
from being an ABI break is that an empty `images` array encodes as a bare
string. `ChatMessage` kept `content: String` and gained `images: [ChatImage]`
beside it rather than turning `content` into an enum: that field is read in
dozens of places with nothing to do with vision, and a message carrying no
picture still encodes `"content": "hello"` byte for byte, matching every
caller that predates this feature. Only a message with an image switches to
the parts array. `testAMessageWithoutImagesEncodesContentAsABareString` is
the guard.

## Images are prepended, matched to the reference rather than chosen

`apply_chat_template(processor, config, question, num_images=1)` builds
`[image, text]`. Appending moves every mRoPE position past the image and
produces a different prompt for the same request -- fluently, with no
error.

## Gate the attach control on `info.vision.active`

Not "does this family have a tower." An install can carry one and refuse
every image: the pixel budget comes from the checkpoint's own
`preprocessor_config.json` and has no default worth falling back to
(`crates/vision-io` Gotcha 6), so an install streamed without that sidecar
reports `active == false` with the reason. `AppModel`'s `visionIsActive` /
`attachmentContentTypes` are ONE accessor pair -- two views assembling
their own picker list would drift, and the one that drifted would accept a
file the turn then refuses. The same pattern as `activeLoadGuard`
(`swift/docs/SWIFT_MODEL_HUB.md`).

## Three places a picture was lost rather than errored

Each read as the model ignoring the image, all found by writing the tests.

- `AppChatMessage.imagePaths` had to be on the MESSAGE, not just the draft:
  the prompt is rebuilt from the transcript on every agent step, so a
  picture held only in the composer is sent on step one and silently
  dropped on step two. It decodes with `decodeIfPresent` and a default
  (`swift/docs/storage.md`'s rule).
- `executeGenerationTurn`'s `guard !msg.content.isEmpty` predates images and
  drops an image-only turn whole. An image-only turn has no text and is
  still a turn.
- `AttachmentImporter` sent every file through `DocumentTextExtractor`,
  which throws `unsupportedFormat` on every image type -- correct for its
  own job, and why images could not be attached at all: the pixels go to
  the tower and the prompt needs only the path.

## One place a zero was presented as a fact

An image extracts no text, so the chip's "0 chars" read as a failed import.
`AppPromptAttachment.detailText` is a VALUE rather than inline view code so
the branch can be tested at all -- the `ServerStatusRows` lesson
(`swift/docs/SWIFT_MODEL_HUB.md`) applied a second time.

## The live token estimate is a floor on an image turn

`updateTokenEstimate` (`swift/docs/SWIFT_CONTEXT_RING.md`) is a FLOOR on a
turn carrying images. `countTokens` renders the template, which emits one
marker per image whatever its size, and the expansion to that page's
merged-token count happens later in the engine's splice -- no client-side
count sees it.
