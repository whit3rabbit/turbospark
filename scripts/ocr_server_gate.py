#!/usr/bin/env python3
"""Drive a running `turbospark-server` with an image on BOTH endpoints and write
each reply's text to a file for `scripts/check_ocr_page_output.py`
(`docs/DEEPSEEK_OCR_VISION_PHASE0.md`).

    python3 -I scripts/ocr_server_gate.py IMAGE.png OUT_PREFIX [--port 18081] [--model dsocr]

Writes OUT_PREFIX.openai.txt, OUT_PREFIX.anthropic.txt and the no-image
control OUT_PREFIX.control.txt. Standard library only. The prompt is the
model's own `Free OCR.`; the server's template turns the image part into the
`<image>\\n` the processor writes.
"""
import base64
import json
import sys
import urllib.request


def post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    image, prefix = sys.argv[1], sys.argv[2]
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "18081"
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "dsocr"
    base = f"http://127.0.0.1:{port}"
    b64 = base64.b64encode(open(image, "rb").read()).decode()
    media = "image/png" if image.lower().endswith(".png") else "image/jpeg"

    openai = post(
        base + "/v1/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 700,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                        {"type": "text", "text": "Free OCR."},
                    ],
                }
            ],
        },
    )
    text = openai["choices"][0]["message"]["content"]
    open(prefix + ".openai.txt", "w").write(text)
    print("openai usage:", openai.get("usage"), "finish:", openai["choices"][0].get("finish_reason"))

    anthropic = post(
        base + "/v1/messages",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 700,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media, "data": b64},
                        },
                        {"type": "text", "text": "Free OCR."},
                    ],
                }
            ],
        },
    )
    text = "".join(b.get("text", "") for b in anthropic["content"])
    open(prefix + ".anthropic.txt", "w").write(text)
    print("anthropic usage:", anthropic.get("usage"), "stop:", anthropic.get("stop_reason"))

    control = post(
        base + "/v1/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 700,
            "messages": [{"role": "user", "content": "Free OCR."}],
        },
    )
    open(prefix + ".control.txt", "w").write(control["choices"][0]["message"]["content"])
    print("control usage:", control.get("usage"))


if __name__ == "__main__":
    main()
