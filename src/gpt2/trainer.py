"""Training and validation loops for GPT-2 fine-tuning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _safe_perplexity(loss: float) -> float:
    """Return ``exp(loss)`` while guarding against overflow."""
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def train(
    model: nn.Module,
    loader,
    optimizer,
    device,
    *,
    epoch: int = 0,
    total_epochs: int = 1,
    scheduler=None,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
) -> tuple[float, float]:
    """Run a single training epoch.

    Args:
        model: ``GPT2LMHeadModel``.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Torch device.
        epoch: Current epoch index (0-based) for logging.
        total_epochs: Total number of epochs (for log formatting).
        scheduler: Optional LR scheduler stepped per batch.
        max_grad_norm: Max gradient norm for clipping.
        log_every: Log progress every N steps; 0 disables.

    Returns:
        Tuple ``(avg_loss, perplexity)``.

    Raises:
        ValueError: If ``loader`` is empty.
    """
    n_batches = len(loader)
    if n_batches == 0:
        raise ValueError("Training loader is empty.")

    model.train()
    total_loss = 0.0

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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        if log_every and i % log_every == 0:
            running = total_loss / (i + 1)
            print(
                f"  Epoch [{epoch + 1}/{total_epochs}] | "
                f"Step [{i}/{n_batches}] | Loss: {running:.4f}"
            )

    avg_loss = total_loss / n_batches
    perplexity = _safe_perplexity(avg_loss)
    print(
        f"  Epoch [{epoch + 1}/{total_epochs}] | "
        f"Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}"
    )
    return avg_loss, perplexity


def validate(
    model: nn.Module,
    loader,
    optimizer=None,
    device=None,
) -> tuple[float, float]:
    """Evaluate the model on a validation/test loader.

    Args:
        model: ``GPT2LMHeadModel``.
        loader: Validation or test DataLoader.
        optimizer: Unused; accepted for trainer signature symmetry.
        device: Torch device.

    Returns:
        Tuple ``(avg_loss, perplexity)``.

    Raises:
        ValueError: If ``loader`` is empty.
    """
    del optimizer
    n_batches = len(loader)
    if n_batches == 0:
        raise ValueError("Validation loader is empty.")

    model.eval()
    eval_loss = 0.0
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

    avg_eval_loss = eval_loss / n_batches
    perplexity = _safe_perplexity(avg_eval_loss)
    print(f"  Val Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")
    return avg_eval_loss, perplexity
