import torch

from open_qwen.main import (
    ModelConfig,
    Qwen35,
    VisionConfig,
    compute_lm_loss,
    compute_mtp_loss,
)

# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Use a tiny config so it can run on a laptop.
    mini = ModelConfig(
        vocab_size=1_024,
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
        vision=VisionConfig(
            image_size=56,
            patch_size=14,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen35(mini).to(device)

    total_params = model.num_parameters()
    print(f"Mini model params : {total_params:,}")

    B, T = 2, 32
    ids = torch.randint(0, mini.vocab_size, (B, T), device=device)
    imgs = torch.randn(B, 3, 56, 56, device=device)

    # Text-only pass.
    logits, mtp = model(ids)
    print(f"[text-only]  logits={tuple(logits.shape)}  mtp={tuple(mtp.shape)}")

    # Multimodal pass.
    logits_mm, mtp_mm = model(ids, pixel_values=imgs)
    print(f"[multimodal] logits={tuple(logits_mm.shape)}  mtp={tuple(mtp_mm.shape)}")

    # Loss.
    lm_loss = compute_lm_loss(logits, ids)
    mtp_loss = compute_mtp_loss(mtp, ids)
    print(f"LM loss: {lm_loss.item():.4f}   MTP loss: {mtp_loss.item():.4f}")

    print("All checks passed ✓")
