# Qwen3.5-0.8B — GGUF Q4_K_M (iOS-ready)

## What's in this directory

| File | Description |
|------|-------------|
| `qwen3.5-0.8b-Q4_K_M.gguf` | The model weights, quantized to 4-bit (Q4_K_M). ~505 MB. |
| `tokenizer.json` | HuggingFace fast tokenizer (primary tokenizer file). |
| `tokenizer_config.json` | Tokenizer settings (special tokens, chat template reference). |
| `vocab.json` | Token vocabulary. |
| `merges.txt` | BPE merge rules. |
| `chat_template.jinja` | Jinja2 chat template for formatting prompts. |

## Format

- **GGUF** (llama.cpp format), quantization type **Q4_K_M** (4-bit, mixed precision).
- Source: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (SafeTensors, ~1.75 GB original).
- Converted via `llama.cpp` `convert_hf_to_gguf.py` -> `llama-quantize Q4_K_M`.

## How to use on iOS

This GGUF file is designed for use with **llama.cpp** on iOS. Common integration paths:

1. **llama.cpp C library** — link `libllama.a` (built for iOS/arm64) into your Xcode project and load the GGUF file directly.
2. **LLMFarm** or **llama.swift** — Swift wrappers around llama.cpp that accept GGUF files.

### Tokenization

The GGUF file **embeds the tokenizer vocabulary**, so llama.cpp handles tokenization internally — you do **not** need to ship the separate tokenizer files for basic inference. They are included here as a reference or if your app needs to do custom tokenization outside of llama.cpp.

### Loading the model (llama.cpp C API sketch)

```c
struct llama_model_params model_params = llama_model_default_params();
struct llama_model * model = llama_model_load_from_file("qwen3.5-0.8b-Q4_K_M.gguf", model_params);

struct llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048; // context window — adjust as needed
struct llama_context * ctx = llama_init_from_model(model, ctx_params);
```

## Notes

- **Qwen3.5-0.8B** is a multimodal-capable model (text + vision). This GGUF conversion covers the **text/language** component. Vision encoder weights are not included.
- Q4_K_M offers a good balance of quality and size for on-device use. If you need smaller, Q4_K_S or IQ4_XS are options; if you need better quality, Q5_K_M or Q6_K are available (at larger file size).
