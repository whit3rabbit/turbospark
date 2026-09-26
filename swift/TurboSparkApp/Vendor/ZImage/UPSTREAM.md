# Z-Image.swift source

This directory vendors the `ZImage` library from
[`zhutao100/Z-Image.swift`](https://github.com/zhutao100/Z-Image.swift) at
revision `28bfcf3148c041a554629247170eb54d9ac46830`. The upstream repository
describes the project as MIT licensed. See `LICENSE.txt` and `UPSTREAM_README.md`.

TurboSpark carries one runtime patch in `Sources/ZImage/Weights/WeightsMapping.swift`:

- Promote FP16 checkpoint tensors to BF16, matching the supported runtime dtype
  and avoiding FP16 overflow in the transformer activations.
- Dequantize quantized pad tokens and auxiliary projections before the upstream
  code loads them as dense weights.

Keep unrelated upstream changes out of this copy. To refresh it, copy the
upstream `Sources/ZImage` tree at a pinned revision, then reapply and benchmark
the patch before changing the pin.
