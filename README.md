# open_qwen_3_5

This repository provides a non-official, research-oriented implementation of Qwen 3.5. At the time of writing, no peer-reviewed paper or official reference implementation appears to be publicly available. The architecture and hyperparameter details were inferred from publicly available model metadata, and the codebase reflects that specification as closely as possible.

## Model Overview

The core architecture is a hybrid transformer that interleaves two attention mechanisms within each layer:

1. **Gated DeltaNet (linear attention)**: uses a delta-rule recurrence to update a per-head state, enabling linear-time sequence processing.
2. **Gated Attention (GQA)**: grouped-query attention with separate Q and KV head counts, plus a learned output gate.

Each layer follows the pattern:

`[Gated DeltaNet + FFN] × 3  →  [Gated Attention + FFN] × 1`

The model is multimodal: text tokens are optionally fused with a ViT-style vision encoder output. A Multi-Token Prediction (MTP) head predicts multiple future tokens to accelerate inference.

## How It Works (High-Level)

- **Input embeddings**: token IDs are embedded and passed through stacked hybrid layers.
- **DeltaNet blocks**: maintain a recurrent state per QK head; update uses a delta-rule with a learned gate β.
- **Gated Attention block**: standard causal attention with GQA, rotary position embeddings, and output gating.
- **FFN**: SwiGLU feed-forward network after each block.
- **Outputs**: autoregressive next-token logits plus MTP logits for multi-token prediction.

## Architecture Details

Default configuration values are defined in `open_qwen/main.py` via `ModelConfig` and `VisionConfig`. The table below summarizes the primary architectural parameters.

### Core Model (ModelConfig)

| Component | Parameter | Default |
| --- | --- | --- |
| Vocabulary | `vocab_size` | 248,320 |
| Hidden size | `hidden_size` | 5,120 |
| Layers | `num_layers` | 64 |
| Linear blocks per layer | `num_linear_blocks_per_layer` | 3 |
| DeltaNet QK heads | `delta_qk_heads` | 16 |
| DeltaNet V heads | `delta_v_heads` | 48 |
| DeltaNet head dim | `delta_head_dim` | 128 |
| GQA Q heads | `attn_q_heads` | 24 |
| GQA KV heads | `attn_kv_heads` | 4 |
| GQA head dim | `attn_head_dim` | 256 |
| FFN hidden dim | `ffn_hidden_dim` | 17,408 |
| Native context | `max_seq_len` | 262,144 |
| RoPE base | `rope_base` | 10,000 |
| YaRN scale | `rope_scaling_factor` | 1.0 |
| YaRN β_fast | `yarn_beta_fast` | 32.0 |
| YaRN β_slow | `yarn_beta_slow` | 1.0 |
| YaRN mscale | `yarn_mscale` | 0.1 |
| MTP heads | `mtp_num_heads` | 4 |
| Dropout | `dropout` | 0.0 |
| RMSNorm eps | `rms_norm_eps` | 1e-6 |

### Vision Encoder (VisionConfig)

| Component | Parameter | Default |
| --- | --- | --- |
| Image size | `image_size` | 448 |
| Patch size | `patch_size` | 14 |
| Channels | `num_channels` | 3 |
| Hidden size | `hidden_size` | 1,152 |
| Layers | `num_layers` | 27 |
| Heads | `num_heads` | 16 |
| MLP ratio | `mlp_ratio` | 4.0 |
| Dropout | `dropout` | 0.0 |

## Usage

### Quick sanity-check (CPU/GPU)

Run the included example with a small configuration:

```bash
python3 example.py
```

This:
- Instantiates a small model.
- Runs text-only and multimodal forward passes.
- Computes LM loss and MTP loss.

### Programmatic usage

```python
import torch
from open_qwen.main import ModelConfig, Qwen35, VisionConfig

cfg = ModelConfig(
    vocab_size=1024,
    hidden_size=256,
    num_layers=2,
    delta_qk_heads=4,
    delta_v_heads=8,
    delta_head_dim=32,
    attn_q_heads=4,
    attn_kv_heads=2,
    attn_head_dim=64,
    ffn_hidden_dim=512,
    max_seq_len=512,
    mtp_num_heads=2,
    vision=VisionConfig(image_size=56, patch_size=14, hidden_size=128, num_layers=2, num_heads=4),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen35(cfg).to(device)

ids = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
logits, mtp = model(ids)
```

## Notes and Limitations

- This is a research implementation based on public metadata, not an official release.
- Performance characteristics may differ from the reference model due to implementation details.
- No pretrained weights are provided in this repository.

## References

[1] K. Gomez. *open_qwen_3_5: A non-official implementation of Qwen 3.5*. GitHub repository.

[2] Qwen Team. *Qwen3.5-27B* model card. Hugging Face.

URLs (for reference):
```text
https://github.com/kyegomez/open_qwen_3_5
https://huggingface.co/Qwen/Qwen3.5-27B
```
