# Fine-Tuning Qwen2.5-7B for Code Generation

Fine-tuned [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) on 122K Python code instructions using LoRA adapters, achieving **82.7% token accuracy** on held-out evaluation data. The model produces clean, functional code for common programming tasks after 4.5 hours of training on a single GPU.

---

## Highlights

| Metric | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-7B (7.6B params) |
| Method | LoRA (rank 128, alpha 64) |
| Trainable Parameters | 323M / 7.9B (4.07%) |
| Dataset | 122K code instructions (Alpaca format) |
| Training Time | ~4 hrs 26 min |
| Best Eval Loss | **0.7324** (step 600) |
| Eval Token Accuracy | **82.7%** |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |

---

## Training Details

### Model & Data

- **Base model**: `Qwen/Qwen2.5-7B` loaded in full bf16 precision (14.19 GB footprint) with Flash Attention 2.
- **Dataset**: [`TokenBender/code_instructions_122k_alpaca_style`](https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style) — 121,959 instruction-output pairs covering Python, C++, Java, and more.
- **Split**: 115,861 train / 6,098 eval (95/5).
- **Formatting**: Each example is converted to Qwen's chat template with a system prompt ("You are a highly skilled Python programmer. Write clean, efficient, and correct code."), then packed into 2048-token sequences.

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 128 |
| Alpha | 64 |
| Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Bias | None |
| Task Type | CAUSAL_LM |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch Size | 16 per device |
| Gradient Accumulation | 2 (effective batch = 32) |
| Learning Rate | 2e-4 |
| Scheduler | Cosine |
| Warmup Ratio | 3% |
| Optimizer | AdamW |
| Max Sequence Length | 2048 |
| Packing | Enabled |
| Precision | bf16 |
| Gradient Checkpointing | Enabled |

---

## Training Curves

### Loss

| Step | Train Loss | Eval Loss |
|---|---|---|
| 100 | 0.7584 | 0.7410 |
| 200 | 0.7485 | 0.7366 |
| 300 | 0.7394 | 0.7343 |
| 400 | 0.7309 | 0.7341 |
| 500 | 0.7205 | **0.7332** |
| 600 | 0.7177 | **0.7324** |
| 700 | 0.7247 | 0.7340 |
| 800 | 0.7101 | 0.7342 |
| 900 | 0.7392 | 0.7338 |
| 933 | 0.7211 | 0.7339 |

Training loss dropped from **1.75** (step 10) to **0.71** by the final steps. The best checkpoint (lowest eval loss of **0.7324**) was saved at step 600 (~epoch 1.9), with `load_best_model_at_end` restoring it automatically.

### Token Accuracy

| Step | Train Token Acc | Eval Token Acc |
|---|---|---|
| 100 | 82.2% | 82.6% |
| 300 | 82.4% | 82.7% |
| 600 | 83.0% | **82.7%** |
| 933 | 83.0% | 82.7% |

---

## Inference Examples

After merging the adapter back into the base model, the fine-tuned model generates clean, correct Python code:

### Palindrome Check

**Prompt**: *Write a Python function that checks if a given string is a palindrome.*

```python
def is_palindrome(s):
    return s == s[::-1]

print(is_palindrome("racecar"))
```

### Binary Search

**Prompt**: *Implement a binary search algorithm in Python that returns the index of the target element.*

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

arr = [1, 3, 5, 7, 9]
target = 5
index = binary_search(arr, target)
print(index)  # Output: 2
```

### Stack Data Structure

**Prompt**: *Write a Python class for a stack data structure with push, pop, peek, and is_empty methods.*

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0
```

---

## Repository Structure

```
.
├── finetune.ipynb           # Full training notebook (end-to-end)
├── README.md
└── results/
    ├── lora-adapter/        # Saved LoRA adapter weights (~1.2 GB)
    │   ├── adapter_model.safetensors
    │   └── adapter_config.json
    ├── merged-model/        # Full merged model (adapter + base)
    ├── checkpoint-500/
    └── checkpoint-933/
```

---

## Reproduce

### Requirements

- Python 3.10+
- CUDA 12.x compatible GPU (tested on 96 GB VRAM; lower VRAM possible with quantization)

### Setup

```bash
pip install transformers peft trl bitsandbytes accelerate datasets
pip install flash-attn --no-build-isolation --no-cache-dir
```

### Train

Open `finetune.ipynb` and run all cells. The only cell you need to edit is **Cell 4 (Configuration)** to adjust hyperparameters.

### Inference with the Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "./results/lora-adapter")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("./results/lora-adapter")

messages = [
    {"role": "system", "content": "You are a highly skilled Python programmer."},
    {"role": "user", "content": "Write a function to flatten a nested list."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)

print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Tech Stack

- [Transformers](https://github.com/huggingface/transformers) — model loading & tokenization
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter training
- [TRL](https://github.com/huggingface/trl) — `SFTTrainer` with sequence packing
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) — efficient attention on Blackwell GPU
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — optimizer support

---

## Key Takeaways

1. **LoRA rank 128 is generous but effective** — training only 4% of parameters still drives eval loss from 0.74 down to 0.73 with strong code generation quality.
2. **Packing matters** — combining short examples into full 2048-token sequences reduced total steps from ~10K+ to 933, cutting wall-clock time dramatically.
3. **Early stopping is important** — best eval loss was at step 600 (epoch ~1.9); the final epoch showed signs of mild overfitting on train loss while eval plateaued.
4. **Full bf16 on 96 GB VRAM** — no quantization needed. The 14 GB base model + LoRA adapters + optimizer states fit comfortably, leaving ~80 GB free for activations and batch size.

---

## License

This project fine-tunes Qwen2.5-7B under its original [Apache 2.0 license](https://huggingface.co/Qwen/Qwen2.5-7B). The training dataset is subject to its own license terms.
