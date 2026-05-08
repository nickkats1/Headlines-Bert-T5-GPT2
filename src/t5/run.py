"""Entry point for fine-tuning T5 on Reuters Description -> Headlines.

CLI overrides any config field, e.g.::

    python -m src.t5.run --epochs 1 --device cpu --model-name t5-small
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.common import resolve_device, seed_everything
from src.t5.config import CONFIG, T5Config
from src.t5.dataset import CustomDataset
from src.t5.metrics import compute_rouge, save_predictions
from src.t5.preprocess import load_data
from src.t5.trainer import train, validate


def parse_args(argv: list[str] | None = None) -> T5Config:
    """Parse CLI overrides on top of the default ``T5Config``."""
    p = argparse.ArgumentParser(description="Fine-tune T5 on description→headline summarization.")
    p.add_argument("--model-name", default=CONFIG.model_name)
    p.add_argument("--data-path", default=CONFIG.data_path)
    p.add_argument("--output-dir", default=CONFIG.output_dir)
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--source-length", type=int, default=CONFIG.source_length)
    p.add_argument("--target-length", type=int, default=CONFIG.target_length)
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
        source_length=args.source_length,
        target_length=args.target_length,
        device=args.device,
        seed=args.seed,
    )


def main(cfg: T5Config | None = None) -> None:
    """Run the T5 fine-tuning + evaluation pipeline end-to-end."""
    cfg = cfg if cfg is not None else parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    df = load_data(file_path=cfg.data_path)
    df["Description"] = cfg.source_prefix + df["Description"].astype(str)

    df_train, df_val = train_test_split(df, test_size=cfg.test_size, random_state=cfg.seed)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    tokenizer = T5Tokenizer.from_pretrained(cfg.model_name)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model_name).to(device)

    train_set = CustomDataset(
        df_train,
        tokenizer,
        source_len=cfg.source_length,
        target_len=cfg.target_length,
        source_col="Description",
        target_col="Headlines",
    )
    val_set = CustomDataset(
        df_val,
        tokenizer,
        source_len=cfg.source_length,
        target_len=cfg.target_length,
        source_col="Description",
        target_col="Headlines",
    )

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        train(
            model,
            train_loader,
            optimizer,
            device,
            tokenizer=tokenizer,
            epoch=epoch,
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_dir = os.path.join(cfg.output_dir, "model_files")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    predictions, actuals = validate(model, val_loader, device=device, tokenizer=tokenizer)
    save_predictions(predictions, actuals, path=os.path.join(cfg.output_dir, "predictions.csv"))
    scores = compute_rouge(predictions, actuals)
    print(f"ROUGE Scores: {scores}")


if __name__ == "__main__":
    main()
