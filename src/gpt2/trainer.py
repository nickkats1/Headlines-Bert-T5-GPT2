import torch
import torch.nn as nn
import numpy as np


def train_epoch(epochs, model, device, loader, optimizer, scheduler=None):
    """Runs training across all epochs.

    Args:
        epochs: Number of epochs to train.
        model: GPT2LMHeadModel.
        device: Torch device.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler. Defaults to None.

    Returns:
        Average training loss for the final epoch.
        Perplexity for the final epoch.
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            if i % 10 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"  Epoch [{epoch + 1}/{epochs}] | Step [{i}/{len(loader)}] | Loss: {avg_loss:.4f}")

        avg_loss = total_loss / len(loader)
        perplexity = np.exp(avg_loss)
        print(f"  Epoch [{epoch + 1}/{epochs}] | Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")

    return avg_loss, perplexity


def eval_epoch(model, device, loader):
    """Evaluates the model on a validation or test set.

    Args:
        model: GPT2LMHeadModel.
        device: Torch device.
        loader: Validation or test DataLoader.

    Returns:
        Average evaluation loss.
        Perplexity score.
    """
    model.eval()
    eval_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            eval_loss += outputs.loss.item()

    avg_eval_loss = eval_loss / len(loader)
    perplexity = np.exp(avg_eval_loss)

    print(f"  Val Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")
    return avg_eval_loss, perplexity