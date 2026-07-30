"""
HOME AI – Training Pipeline
Trains the HFM‑2 foundation model using the TransformerBlock architecture.
"""

import argparse
import pathlib
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

# Import the real transformer block from home-transformer
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "home-transformer" / "src"))
from transformer import TransformerBlock


class HFM2Model(nn.Module):
    """HOME Foundation Model 2 – a decoder‑only language model."""

    def __init__(self, vocab_size: int, dim: int, num_layers: int,
                 num_heads: int, hidden_dim: int, max_seq_len: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_embed.weight
        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # nn.Embedding defaults to N(0, 1). Because lm_head is tied to the
        # embedding, that default puts the logits on a scale ~10x too large,
        # saturating the softmax and starting the loss well above the ln(vocab)
        # random baseline. Normal(0, 0.02) is the usual choice for a tied head.
        # RMSNorm/LayerNorm weights are left alone — they must start at 1.
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.tok_embed(input_ids)
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x, _ = block(x, attn_mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # reshape, not view: targets are usually a slice (input_ids[:, 1:])
            # and so are not contiguous.
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_config(cfg_path: pathlib.Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="HFM‑2 training script")
    parser.add_argument("--config", type=pathlib.Path, required=True,
                        help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = HFM2Model(
        vocab_size=cfg["model"]["vocab_size"],
        dim=cfg["model"]["dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        hidden_dim=cfg["model"]["hidden_dim"],
        max_seq_len=cfg["data"]["seq_len"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    print(f"Model parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["scheduler"]["t_max"]
    )
    writer = SummaryWriter(log_dir=cfg.get("log_dir", "runs"))

    # Training loop (synthetic data for now – replace with real DataLoader)
    global_step = 0
    batch_size = cfg["data"]["batch_size"]
    seq_len = cfg["data"]["seq_len"]
    vocab_size = cfg["model"]["vocab_size"]

    for epoch in range(cfg["training"]["epochs"]):
        # Generate synthetic batch. seq_len + 1 tokens, so that after the
        # next-token shift below we still train on seq_len positions.
        batch = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
        # Next-token prediction: position i predicts token i+1. Predicting the
        # *input* token instead is trivially solvable — the residual stream
        # carries the token embedding to the output and lm_head is tied to it,
        # so the loss collapses to ~0 without the model learning anything.
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits, loss = model(input_ids, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
        optimizer.step()
        scheduler.step()

        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1
        print(f"[Epoch {epoch + 1}/{cfg['training']['epochs']}] "
              f"Loss: {loss.item():.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

    # Save checkpoint
    ckpt_dir = pathlib.Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "hfm2_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "global_step": global_step,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
