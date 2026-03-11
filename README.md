# open_qwen

`open_qwen_3_5` is a non-official, research-oriented PyTorch implementation of a Qwen 3.5–style hybrid multimodal language model. In the absence of a publicly available peer-reviewed paper or official reference code (at the time of writing), this repository reconstructs the architecture and hyperparameters from publicly available model metadata. The implementation is intended for inspection, ablations, and educational use rather than as a drop-in reproduction of an official release.

## Scope and Non-Goals

- This is not an official Qwen implementation.
- No pretrained weights are shipped with this repository.
- Exact parity with any proprietary training recipe, tokenizer, data pipeline, or kernel-level optimizations is out of scope.

## Model at a Glance

The backbone is a *hybrid transformer* that alternates linear-time and quadratic-time attention mechanisms within each layer, and optionally prepends visual tokens from a ViT-style vision encoder.

Block layout per layer (see `open_qwen/main.py`):

`[Gated DeltaNet + SwiGLU FFN] × 3  →  [Gated Attention (GQA) + SwiGLU FFN] × 1`

Forward pass overview:

```text
input_ids ──► token embedding ─┐
                              ├─► hybrid layers ─► RMSNorm ─► LM head ─► logits
pixel_values ─► vision encoder ┘                         └─► MTP head ─► mtp_logits
```

Outputs:

- `logits`: next-token logits of shape `(B, T_out, V)`
- `mtp_logits`: multi-token logits of shape `(B, T_out, N, V)` where `N = mtp_num_heads`

`T_out = T` for text-only inputs and `T_out = T + N_v` when visual tokens are prepended.

## How the Model Works

### Shape Conventions

- `B`: batch size
- `T`: sequence length (text tokens)
- `N_v`: number of visual tokens (patch tokens, implementation-defined)
- `d`: model width (`hidden_size`)
- `H`: number of QK heads in DeltaNet (`delta_qk_heads`)
- `H_v`: number of value heads in DeltaNet (`delta_v_heads`)
- `d_k`: per-head dimension for DeltaNet (`delta_head_dim`)
- `H_q`: number of query heads in GQA (`attn_q_heads`)
- `H_kv`: number of key/value heads in GQA (`attn_kv_heads`)
- `d_h`: per-head dimension for GQA (`attn_head_dim`)

### (A) Gated DeltaNet (Linear-Time Attention)

The DeltaNet block implements a delta-rule recurrence with a learned gate `β_t ∈ (0, 1)^H` and a per-head state matrix `S_t ∈ R^{H×d_k×d_k}`.

With projections:

- `q_t, k_t ∈ R^{H×d_k}`, `v_t ∈ R^{H_v×d_k}`

The implementation maintains `S_t` and updates it sequentially over time (see `GatedDeltaNet._delta_recurrence` in `open_qwen/main.py`). Conceptually:

```text
v̂_t = S_{t-1} k_t
δ_t  = v_t − v̂_t
S_t  = (1 − β_t) ⊙ S_{t-1} + β_t ⊙ (δ_t ⊗ k_t)
o_t  = S_t q_t
```

Notes:

- Keys are RMS-normalized for numerical stability.
- The implementation supports asymmetric head counts (`H_v != H`) by grouping value heads over QK heads (`H_v // H`).
- A learned output gate `g_t = σ(W_g x_t)` modulates the block output.

### (B) Gated Attention (Quadratic, Grouped-Query Attention)

The attention block uses grouped-query attention (GQA): `H_q` query heads attend to `H_kv` KV heads, expanded by repetition to match `H_q`. Rotary positional embeddings are applied to `Q` and `K`, and attention is computed via PyTorch scaled dot-product attention with a causal mask (see `GatedAttention` in `open_qwen/main.py`).

Key properties:

- Causal attention (`is_causal=True`)
- RoPE / YaRN scaling parameters are configurable (`rope_scaling_factor`, `yarn_beta_fast`, `yarn_beta_slow`, `yarn_mscale`)
- Learned output gating analogous to the DeltaNet block

### (C) SwiGLU Feed-Forward Network

Each sub-block (DeltaNet or GQA) is followed by a SwiGLU FFN with inner dimension `ffn_hidden_dim`.

### (D) Multimodal Fusion (Vision Encoder)

When `pixel_values` are provided, a ViT-style encoder produces a sequence of visual tokens in the model width `d`, which are prepended to the text sequence before entering the hybrid backbone (see `VisionEncoder` and `Qwen35.forward` in `open_qwen/main.py`).

### (E) Multi-Token Prediction (MTP) Head

In addition to standard next-token logits, the model exposes an auxiliary head that predicts `N` future tokens per position (see `MultiTokenPredictionHead` in `open_qwen/main.py`). Head 0 predicts directly from the final hidden state; subsequent heads iteratively refine the hidden state and predict again. This is useful for speculative decoding or draft generation experiments.

## Architecture Details (Default Config)

Default configuration values are defined in `open_qwen/main.py` via `ModelConfig` and `VisionConfig`.

### Core Model (`ModelConfig`)

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

### Vision Encoder (`VisionConfig`)

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

## Installation

This repository is intentionally lightweight. The only required runtime dependency is PyTorch.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch
```

If you plan to run on GPU, install a CUDA-enabled build of PyTorch appropriate for your system.

Recommended environment: Python 3.10+ and PyTorch 2.0+.

## Usage

### Quick Sanity Check (CPU/GPU)

```bash
python3 example.py
```

The example instantiates a small configuration, runs text-only and multimodal forward passes, and computes both LM and MTP losses.

### Minimal Programmatic Usage

```python
import torch
from open_qwen.main import ModelConfig, Qwen35

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
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen35(cfg).to(device)

input_ids = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
logits, mtp_logits = model(input_ids)
```

### Multimodal Usage (Vision + Text)

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
model = Qwen35(cfg)

input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
pixel_values = torch.randn(2, 3, cfg.vision.image_size, cfg.vision.image_size)

logits, mtp_logits = model(input_ids, pixel_values=pixel_values)
```

## Code Organization

Primary implementation lives in `open_qwen/main.py`:

| Symbol | Description |
| --- | --- |
| `ModelConfig`, `VisionConfig` | Configuration dataclasses |
| `VisionEncoder` | ViT-style vision tower producing visual tokens |
| `GatedDeltaNet` | Linear-time delta-rule recurrence block |
| `GatedAttention` | Causal grouped-query attention (GQA) block |
| `HybridLayer` | Per-layer composition: 3×(DeltaNet+FFN) + 1×(GQA+FFN) |
| `MultiTokenPredictionHead` | Auxiliary head predicting multiple future tokens |
| `Qwen35` | End-to-end multimodal backbone and heads |
| `compute_lm_loss`, `compute_mtp_loss` | Reference loss helpers |

## Reproducibility Notes

- The DeltaNet recurrence is implemented as a simple sequential scan for clarity; optimized chunked/parallel kernels are not included.
- PyTorch attention uses `scaled_dot_product_attention` with causal masking; numeric differences are expected vs. fused kernels.

## Citation

If you use this repository in academic work, cite the implementation and the public model card that informed the configuration:

```bibtex
@misc{gomez_open_qwen_3_5_2026,
  title        = {open\_qwen\_3\_5: A non-official implementation of Qwen 3.5},
  author       = {Gomez, Kye},
  howpublished = {GitHub repository},
  year         = {2026},
  note         = {Accessed: 2026-03-11},
  url          = {https://github.com/kyegomez/open_qwen_3_5}
}

@misc{qwen_qwen3_5_27b_modelcard,
  title        = {Qwen3.5-27B Model Card},
  author       = {Qwen Team},
  howpublished = {Hugging Face},
  note         = {Accessed: 2026-03-11},
  url          = {https://huggingface.co/Qwen/Qwen3.5-27B}
}
```

## References

[1] K. Gomez. *open_qwen_3_5: A non-official implementation of Qwen 3.5*. GitHub repository.

[2] Qwen Team. *Qwen3.5-27B* model card. Hugging Face.

URLs (for reference):
```text
https://github.com/kyegomez/open_qwen_3_5
https://huggingface.co/Qwen/Qwen3.5-27B
```
