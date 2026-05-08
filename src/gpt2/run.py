"""Entry point for fine-tuning GPT-2 on Reuters descriptions.

CLI overrides any config field, e.g.::

    python -m src.gpt2.run --epochs 1 --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.common import resolve_device, seed_everything
from src.gpt2.config import CONFIG, GPT2Config
from src.gpt2.preprocess import load_data
from src.gpt2.trainer import train, validate
from src.gpt2.utils import build_dataloaders, split_data


def parse_args(argv: list[str] | None = None) -> GPT2Config:
    """Parse CLI overrides on top of the default ``GPT2Config``."""
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 on Reuters descriptions.")
    p.add_argument("--model-name", default=CONFIG.model_name)
    p.add_argument("--data-path", default=CONFIG.data_path)
    p.add_argument("--output-dir", default=CONFIG.output_dir)
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--device", default=CONFIG.device)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    args = p.parse_args(argv)
    return dataclasses.replace(
        CONFIG,
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


def main(cfg: GPT2Config | None = None) -> None:
    """Run the GPT-2 fine-tuning pipeline end-to-end."""
    cfg = cfg if cfg is not None else parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    description_df = load_data(file_path=cfg.data_path)
    train_description, val_description = split_data(
        description_df, test_size=cfg.test_size, random_state=cfg.seed
    )

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": cfg.pad_token,
            "bos_token": cfg.bos_token,
            "eos_token": cfg.eos_token,
        }
    )

    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_loader, val_loader = build_dataloaders(
        train_description, val_description, tokenizer, batch_size=cfg.batch_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, len(train_loader) * cfg.epochs), eta_min=0
    )

    print("\nTraining:")
    train_loss = train_perplexity = 0.0
    for epoch in range(cfg.epochs):
        train_loss, train_perplexity = train(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=cfg.epochs,
            scheduler=scheduler,
        )

    print("\nValidation:")
    val_loss, val_perplexity = validate(model, val_loader, device=device)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_dir = os.path.join(cfg.output_dir, "model_files")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    metrics_path = os.path.join(cfg.output_dir, "metrics.csv")
    pd.DataFrame(
        [
            {
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
            }
        ]
    ).to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
