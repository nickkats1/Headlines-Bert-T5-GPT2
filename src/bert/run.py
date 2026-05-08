"""Entry point for fine-tuning BERT on Guardian headline sentiment.

CLI overrides any config field, e.g.::

    python -m src.bert.run --epochs 1 --device cpu --batch-size 4
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import defaultdict

import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from src.bert.config import CONFIG, BertConfig
from src.bert.dataset import CustomDataset
from src.bert.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_f1_score,
    get_predictions,
)
from src.bert.model import BertClassifier
from src.bert.preprocess import clean_data, load_data
from src.bert.trainer import train, validate
from src.bert.utils import label_encode_sentiments, polarity, sentiment
from src.common import resolve_device, seed_everything


def parse_args(argv: list[str] | None = None) -> BertConfig:
    """Parse CLI overrides on top of the default ``BertConfig``."""
    p = argparse.ArgumentParser(description="Fine-tune BERT for sentiment classification.")
    p.add_argument("--model-name", default=CONFIG.model_name)
    p.add_argument("--data-path", default=CONFIG.data_path)
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    p.add_argument("--max-length", type=int, default=CONFIG.max_length)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--device", default=CONFIG.device)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    args = p.parse_args(argv)
    return dataclasses.replace(
        CONFIG,
        model_name=args.model_name,
        data_path=args.data_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


def _build_loader(df, tokenizer, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a DataLoader from a sentiment-labeled DataFrame."""
    dataset = CustomDataset(
        Headlines=df["Headlines"].to_numpy(),
        targets=df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main(cfg: BertConfig | None = None) -> None:
    """Run the BERT sentiment-classification pipeline end-to-end."""
    cfg = cfg if cfg is not None else parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    df = clean_data(load_data(file_path=cfg.data_path))
    df["polarity"] = df["Headlines"].apply(polarity)
    df["sentiment"] = df["polarity"].apply(sentiment)
    df = label_encode_sentiments(df)

    df_train, df_holdout = train_test_split(df, test_size=cfg.holdout_size, random_state=cfg.seed)
    df_val, df_test = train_test_split(
        df_holdout, test_size=cfg.test_size_from_holdout, random_state=cfg.seed
    )

    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    train_loader = _build_loader(df_train, tokenizer, cfg.max_length, cfg.batch_size, True)
    val_loader = _build_loader(df_val, tokenizer, cfg.max_length, cfg.batch_size, False)
    test_loader = _build_loader(df_test, tokenizer, cfg.max_length, cfg.batch_size, False)

    model = BertClassifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    history = defaultdict(list)
    for epoch in range(cfg.epochs):
        train_acc, train_loss = train(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            n_examples=len(df_train),
            scheduler=scheduler,
        )
        val_acc, val_loss = validate(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            n_examples=len(df_val),
        )

        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        print(f"  Train Accuracy: {train_acc * 100:.2f}% | Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy:   {val_acc * 100:.2f}% | Val Loss:   {val_loss:.4f}")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

    test_acc, test_loss = validate(
        model,
        test_loader,
        device=device,
        loss_fn=loss_fn,
        n_examples=len(df_test),
    )
    print(f"\nTest Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    _, y_pred, y_true = get_predictions(model, test_loader, device)
    print(f"F1 Score:  {compute_f1_score(y_true, y_pred):.4f}")
    print(f"Accuracy:  {compute_accuracy(y_true, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{compute_confusion_matrix(y_true, y_pred)}")


if __name__ == "__main__":
    main()
