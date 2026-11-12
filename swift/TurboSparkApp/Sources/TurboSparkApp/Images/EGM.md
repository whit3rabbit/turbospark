# Images

## Overview
The Image Generation section of the app: a gallery of saved images with Create and Organize modes, a prompt composer with settings (model family, quantization, style, count, size, seed), a style catalog with favorites, a model download offer, and two native MLX session adapters that stream generation events from the vendored Z-Image and Qwen-Image pipelines. Views read and mutate `AppModel` (image state such as `imagesMode`, `imageModels`, `imageGenerationTask`, `imageJob`, `savedImageArtifacts` lives there, outside this directory). The session adapters conform to one protocol so `AppModel` can drive either pipeline the same way.

## Interface
- Section views
  - `ImagesSectionView`: header, organize toolbar, gallery grouped by day, progress overlay, unsaved-result card, recommendation sheet [ImagesSectionView.swift:ImagesSectionView]
  - `ImagesWorkspaceMode` (`create`, `organize`, a `SidebarPanelMode`) and `ImagesSidebarPanel` [ImagesSidebarPanel.swift:ImagesWorkspaceMode]
  - `ImageComposerView` (prompt, control row, Generate/Stop/Unload) [ImageComposerView.swift:ImageComposerView]
  - `ImageGenerationSettings`, `ImageModelControls`, `ImageModelPresentation` (`family(_:)`, `quantization(_:)`), `ImageModelDownloadOffer` [ImageGenerationSettings.swift:ImageGenerationSettings, ImageModelControls.swift:ImageModelPresentation, ImageModelDownloadOffer.swift:ImageModelDownloadOffer]
  - `ImageGalleryCard`, `ImageGalleryActions`, `ImageGalleryCarousel` [ImageGalleryCard.swift:ImageGalleryCard, ImageGalleryCarousel.swift:ImageGalleryCarousel]
  - `ImageResolutionPreset` (seven presets, `width`, `height`, `label`) [ImageResolutionPreset.swift:ImageResolutionPreset]
- Styles
  - `AppImageStyle`, `AppImageStyleCatalog` (`groups`, `all`, `style(id:)`, `search(_:)`) [AppImageStyle.swift:AppImageStyleCatalog]
  - `ImageStyleMenu`, `ImageStyleBrowserSheet` [ImageStylePicker.swift:ImageStyleMenu]
  - `ImageStyleFavoriteStore` (file-backed) and `ImageStyleFavorites` (`@MainActor` observable, `shared`) [ImageStyleFavoriteStore.swift:ImageStyleFavoriteStore]
- Generation sessions
  - `ImageGenerationSession` protocol (`cancel`, `generate(_:)`, `waitUntilIdle`, `unload`) with defaults [MLXImageGenerationSession.swift:ImageGenerationSession]
  - `ImageProducerSlot`: lock-guarded single running producer with identity token [MLXImageGenerationSession.swift:ImageProducerSlot]
  - `MLXImageGenerationSession` (Z-Image via `ZImagePipeline`) [MLXImageGenerationSession.swift:MLXImageGenerationSession]
  - `QwenImageGenerationSession` (Qwen-Image 2.1 via `QwenImage21Pipeline`) and `QwenImageSessionError` [QwenImageGenerationSession.swift:QwenImageGenerationSession]
  - `MLXKernelLibrary` (`locate`, `locateForCurrentProcess`, `requireForCurrentProcess`) and `MLXKernelLibraryError` [MLXKernelLibrary.swift:MLXKernelLibrary]
- Krea 2 manifest validation: `Krea2RAMQuantization.load(modelRoot:)`, `width(for:)`, `Krea2RAMQuantizationError`. Nothing inside this directory calls it, so its callers are elsewhere [Krea2RAMQuantization.swift:Krea2RAMQuantization].
- Subdirectories: none.

## Invariants
- A session's `generate` runs its producer in a detached task and must hold a `NativeHeavyWorkPermit` (polled every 50 ms) for the whole producer, including the cancellation drain, so served audio or text cannot overlap MLX work. The permit is released by `defer` only when the producer ends [MLXImageGenerationSession.swift:generate, QwenImageGenerationSession.swift:generate].
- Cancelling the stream finishes the consumer at once but the producer keeps running to its next cancellation point (model load, text encode and VAE decode have none). The caller must `await waitUntilIdle()` before releasing the job permit [MLXImageGenerationSession.swift:ImageGenerationSession].
- `ImageProducerSlot.clear(token:)` clears only if the token still owns the slot, so a late clear from an earlier run cannot null the next run's task (which would make it uncancellable); `waitUntilIdle` loops until the slot is empty or holds a finished task [MLXImageGenerationSession.swift:ImageProducerSlot].
- `unload()` is cancel, wait idle, then `pipeline.unloadModel()`. Merely dropping the session deallocates weights into the MLX buffer cache where 10 to 20 GB stays resident, so `unloadModel()` also runs in `deinit` as a backstop. The protocol default `unload()` only cancels [MLXImageGenerationSession.swift:unload, deinit].
- `MLXKernelLibrary.requireForCurrentProcess()` must run before any MLX work in both adapters: with no kernels MLX calls `exit(-1)` and the app quits without a crash report. A native SwiftPM build ships no `mlx-swift_Cmlx.bundle`. The search order mirrors mlx-swift's `load_default_library` [MLXKernelLibrary.swift].
- Z-Image: a root `quantization.json` short-circuits staging. Otherwise bits and group size are read from `quantize_config.json` or component configs, must agree across them, bits must be 2...8 and group size positive, and the four components (transformer, text_encoder, vae, tokenizer) must exist as directories. The snapshot is staged under the user Caches `TurboSpark/ImageMLX/<model>-<revision>` using real directories and symlinked files (the upstream loader enumerates directories), serialized by a static `NSLock`, and writes a synthetic `quantization.json`. A cache path occupied by a non-symlink fails closed rather than being overwritten. Without a manifest the vendor loader would apply packed tensors to dense layers and crash [MLXImageGenerationSession.swift:MLXImageModelSnapshot].
- With no retained source path, Z-Image falls back to the `modelID:revision` spec so the upstream resolver fetches that exact revision; Qwen refuses outright (`missingRetainedSource`) because an install predating source retention cannot run [MLXImageGenerationSession.swift:init, QwenImageGenerationSession.swift:init].
- Qwen width and height must be multiples of `QwenImageModelMetadata.dimensionMultiple`; otherwise it throws instead of returning an image smaller than the metadata claims. Z-Image uses `guidanceScale: 0` and Qwen `trueCfgScale: 1`. Qwen decodes the VAE in bfloat16 and defaults to `.staged` residency (documented measurement in `docs/QWEN_IMAGE_21_TURBO.md`); Z-Image uses `.warm` [QwenImageGenerationSession.swift:init, generate].
- Pipeline stages are mapped to the app's stage strings (`downloading_model`, `loading_model`, `loading_tokenizer`, `loading_text_encoder`, `loading_transformer`, `loading_vae`, `text_encoder`, `transformer`, `vae_decoder`, `png_encode`), which `ImagesSectionView.progressStageLabel` switches on. A new stage must be added to both ends, else it displays as "Loading model" [ImagesSectionView.swift:progressStageLabel].
- Cancellation yields `.cancelled` then finishes; other errors finish the stream throwing. `continuation.onTermination` with `.cancelled` calls `cancel()` [MLXImageGenerationSession.swift:generate].
- `Krea2RAMQuantization.load` fails closed: required files present, group size exactly 64, widths in {4,5,6,8}, module names restricted to `[A-Za-z0-9_.]` without `..` or `/`, shard names must be safe relative paths, every assignment's weight, scales and biases tensors must exist in one shard, and the packed geometry must satisfy `weight.shape[1] == scales.shape[1] * groupSize * width / 32` with overflow checks [Krea2RAMQuantization.swift].
- Style catalog is read from the bundle once (`Resources/image-styles.json`); a failed read gives an empty catalog and pickers degrade to None plus browse. Entries with an empty id, name or prompt are dropped, and empty categories are skipped [AppImageStyle.swift:groups].
- Style favorites live under `AppStorageRoot` (not `UserDefaults.standard`) so tests do not write developer preferences and so the data does not follow the bundle identity. The `cache` is guarded by an `NSLock` because the store is not actor-isolated; `setIDs` stores sorted ids; persistence goes through `AppJSONStore` [ImageStyleFavoriteStore.swift].
- UI rules: generation controls are disabled while `imageGenerationTask` is non-nil; model controls are disabled while `model.isRunning`; generation is blocked in a ghost (temporary) chat; Generate is the single primary action (Download in settings is secondary). Starting a generation forces Create mode and closes the preview. Selection is pruned when images disappear. Move to Trash always asks to confirm in the carousel, card and organize toolbar; the Regenerate action is disabled without a supported, selected model and an `imageRequest`. The first-run model recommendation shows once (`TurboSpark.imageModelRecommendationSeen`) [ImagesSectionView.swift, ImageGalleryCard.swift, ImageComposerView.swift].
- The thumbnail is decoded once per `artifact.contentKey` at a 600 px max, not the full PNG per progress tick [ImageGalleryCard.swift:body].

## Side Effects
- Disk reads: model `config.json`, `quantization.json`, `model_index.json`, safetensors index and tensor metadata (Krea 2), image files via `NSImage(contentsOf:)` and `CGImageSource` [MLXImageGenerationSession.swift, Krea2RAMQuantization.swift, ImageGalleryCarousel.swift, ImageGalleryCard.swift].
- Disk writes: staged snapshot directory, symlinks and `quantization.json` under the user Caches directory [MLXImageGenerationSession.swift:MLXImageModelSnapshot]; `image_style_favorites.json` under `AppStorageRoot` [ImageStyleFavoriteStore.swift]; user-chosen PNG export from the Save As panel (`NSSavePanel`, `Data.write(.atomic)`) [ImageGalleryCard.swift:export].
- UserDefaults: `@AppStorage("TurboSpark.imageModelRecommendationSeen")` [ImagesSectionView.swift].
- Workspace and panels: reveal in Finder via `NSWorkspace.activateFileViewerSelecting`; `fileImporter` for a model folder (sets `model.imageModelPathText`) [ImageGalleryCard.swift:ImageGalleryActions, ImagesSectionView.swift:body].
- Native GPU and memory: MLX pipelines load multi-GB weights, run generation, and are unloaded on request or deinit; acquires and releases `NativeHeavyWorkPermit` [MLXImageGenerationSession.swift, QwenImageGenerationSession.swift].
- Network: the Z-Image fallback spec lets the upstream resolver download the model revision into the Hub cache when no retained source exists [MLXImageGenerationSession.swift:init]. Downloads from the UI go through `AppModel.installImageModel`, outside this directory.
- Mutates `AppModel`: `trashGeneratedImages`, `saveImage`, `discardUnsavedImage`, `regenerateImage`, `generateImage`, `cancelImageGeneration`, `unloadImageModel`, `selectImageModel`, `setImageStyle`, `writePromptTextDirectly`, `showToast` [ImagesSectionView.swift, ImageGalleryCard.swift, ImageStylePicker.swift].
- No Keychain access and no NotificationCenter use in this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
