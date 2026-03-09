
# llama-server Command Chart

This file lists commonly used `llama-server` commands for local models. Copy any command and run it in a terminal. Update `-m`/`--model` paths if your models live elsewhere.

Binary path used in examples:

`.llama.cpp/build/bin/llama-server`

Common flags (quick reference):

- **`-m` / `--model`:** path to GGUF model file
- **`-hf`:** HuggingFace path for GGUF model file
- **`-c` / `--context`:** context size (tokens)
- **`--n-gpu-layers`:** number of layers to store in VRAM
- **`--threads`:** number of CPU threads to use during generation
- **`--temp`:** sampling temperature
- **`--top-p`:** top-p sampling
- **`--top-k`:** top-k sampling
- **`--repeat-penalty`:** penalize repeat sequence of tokens 
- **`--jinja`:** enable Jinja templating
- **`--no-context-shift`:** not to use context shift on infinite text generation

-------------------------

## Model command examples

### google/gemma-3-27b-it-qat-q4_0-gguf
Description: Gemma 27B

```bash
./llama.cpp/build/bin/llama-server \
	-c 128000 \
	--n-gpu-layers 99 \
	--threads 8 \
	--temp 0.7 \
	--top-p 0.95 \
    -m ./models/gemma-3-27b-it/gemma-3-27b-it-q4_0.gguf \
```

### unsloth/medgemma-27b-text-it-GGUF
Description: MedGemma 27B (medical fine-tuned)

```bash
./llama.cpp/build/bin/llama-server \
	-c 128000 \
	--n-gpu-layers 99 \
	--threads 8 \
	--temp 0.7 \
	--top-p 0.95 \
    -m ./models/medgemma-27b-text-it-UD-Q4_K_XL.gguf \
```

### hiratagoh/SIP-jmed-llm-3-8x13b-AC-32k-instruct-GGUF
Reference: https://bone.jp/articles/2025/251031_jmed_llm3_long_context

```bash
./llama.cpp/build/bin/llama-server \
	-c 12800 \
	--n-gpu-layers 99 \
	--threads 8 \
	--temp 0.7 \
	--top-p 0.95 \
	--repeat-penalty 1.05 \
	-m ./models/SIP-jmed/SIP-jmed-llm-3-8x13b-AC-32k-instruct-Q8_0.gguf
```

### Qwen/Qwen3-Next-80B-A3B (Instruct / Thinking)
Instruct variant:

```bash
./llama.cpp/build/bin/llama-server \
    -fa on \
    -sm row \
	--jinja \
	--n-gpu-layers 99 \
	--threads 8 \
	--temp 0.6 \
	--top-k 20 \
	--top-p 0.95 \
	--min-p 0 \
	--presence-penalty 1.5 \
	-c 262144 \
	-n 256000 \
	--no-context-shift \
	-m ./models/Qwen3-Next-80B-A3B/Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf
```
Thinking variant:

```bash
./llama.cpp/build/bin/llama-server \
    -fa on \
    -sm row \
	--jinja \
	--n-gpu-layers 99 \
	--threads 8 \
	--temp 0.6 \
	--top-k 20 \
	--top-p 0.95 \
	--min-p 0 \
	--presence-penalty 1.5 \
	-c 262144 \
	-n 256000 \
	--no-context-shift \
	-m ./models/Qwen3-Next-80B-A3B/Qwen3-Next-80B-A3B-Thinking-Q8_0.gguf
```

### ggml-org/gpt-oss-20b-GGUF
Example using `-hf` option shown in original file; keep or remove depending on your binary.

```bash
./llama.cpp/build/bin/llama-server \
	-hf ggml-org/gpt-oss-20b-GGUF \
	-c 0 \
	--jinja \
	--reasoning-format none \
	--n-gpu-layers 99 \
	--threads 8 \
	--temp 1.0 \
	--top-p 1.0 \
	--top-k 0 \
	--min-p 0 \
	--chat-template-kwargs '{"reasoning_effort": "medium"}'
```

### ggml-org/gpt-oss-120b-GGUF

```bash
./llm/llama.cpp/build/bin/llama-server \
	-hf ggml-org/gpt-oss-120b-GGUF \
	-c 0 \
	--jinja \
	--reasoning-format none \
	--n-gpu-layers 99 \
	--threads 8 \
	--temp 1.0 \
	--top-p 1.0 \
	--top-k 0 \
	--min-p 0 \
	--chat-template-kwargs '{"reasoning_effort": "medium"}'
```


