---
description: Code examples for calling TurboSpark using Python, TypeScript, and curl.
icon: code
---

# Calling the Local API

`turbospark-server` provides a drop-in local replacement for cloud AI APIs. You can point existing tools, SDKs, and scripts directly to your local Apple Silicon Mac.

Start the server:

```sh
turbospark-server --model gemma4 --port 8080 --api-key "test-key"
```

---

## 1. Python (Official OpenAI SDK)

Configure the standard `openai` Python package to use your local TurboSpark listener:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.environ.get("TURBOSPARK_API_KEY", "test-key")
)

# Non-streaming chat completion
response = client.chat.completions.create(
    model="gemma4",
    messages=[
        {"role": "system", "content": "You are a concise engineering assistant."},
        {"role": "user", "content": "What is the benefit of streaming MoE weights?"}
    ],
    temperature=0.7,
    max_tokens=150
)
print(response.choices[0].message.content)

# Streaming chat completion (tokens printed as generated)
stream = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "Write a short poem about compilers."}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## 2. TypeScript / Node.js (OpenAI SDK)

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: process.env.TURBOSPARK_API_KEY || "test-key",
});

async function main() {
  const completion = await openai.chat.completions.create({
    model: "gemma4",
    messages: [{ role: "user", content: "Explain KV cache compression." }],
    stream: true,
  });

  for await (const chunk of completion) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
  process.stdout.write("\n");
}

main().catch(console.error);
```

---

## 3. Anthropic Messages API (Python or curl)

TurboSpark natively translates Anthropic requests:

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gemma4",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Hello local TurboSpark!"}
    ]
  }'
```

---

## 4. Local Vector Embeddings (Python)

```python
response = client.embeddings.create(
    model="bge-small-en-v1.5",
    input=["First sentence to embed", "Second sentence for vector search"],
    encoding_format="float"
)
vector_0 = response.data[0].embedding
print(f"Dimension: {len(vector_0)}, First 3 values: {vector_0[:3]}")
```

---

## 5. Local Image Generation (curl)

```sh
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "sunset over Tokyo skyline in anime aesthetic",
    "size": "1024x1024"
  }' | jq -r '.data[0].b64_json' | base64 -d > tokyo.png
```

---

## 6. Local Speech-to-Text (curl)

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer test-key" \
  -F file=@audio.wav \
  -F model=whisper-base-en
```

For the complete API route and schema index, see the [HTTP API Reference](../../api-reference/http.md) and [OpenAPI Specifications](../../api-reference/http/schemas.md).
