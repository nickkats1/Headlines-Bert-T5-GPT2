"""Training and validation loops for the T5 summarization pipeline."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.t5.config import CONFIG


def train(
    model: nn.Module,
    loader,
    optimizer,
    device,
    *,
    tokenizer,
    epoch: int = 0,
    log_every: int = 50,
) -> float:
    """Run one training epoch for the T5 model.

    Shifts target IDs to create decoder inputs and LM labels, masking pad
    tokens with ``-100`` so they are ignored by the loss.

    Args:
        model: ``T5ForConditionalGeneration``.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Torch device.
        tokenizer: T5Tokenizer (used for the pad token ID).
        epoch: Current epoch index (0-based) for logging.
        log_every: Log progress every N steps; 0 disables logging.

    Returns:
        Mean training loss for the epoch.

    Raises:
        ValueError: If ``loader`` is empty.
    """
    if len(loader) == 0:
        raise ValueError("Training loader is empty.")

    model.train()
    losses: list[float] = []

    for step, batch in enumerate(loader):
        target_ids = batch["target_ids"].to(device, dtype=torch.long)
        decoder_input_ids = target_ids[:, :-1].contiguous()

        labels = target_ids[:, 1:].clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100

        input_ids = batch["source_ids"].to(device, dtype=torch.long)
        attention_mask = batch["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        loss = outputs.loss
        losses.append(loss.item())

        if log_every and step % log_every == 0:
            print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(np.mean(losses))


def validate(
    model: nn.Module,
    loader,
    optimizer=None,
    device=None,
    *,
    tokenizer,
    max_length: int | None = None,
    num_beams: int = CONFIG.generate_num_beams,
    log_every: int = 10,
) -> tuple[list[str], list[str]]:
    """Generate predictions on a validation/test loader.

    Args:
        model: ``T5ForConditionalGeneration``.
        loader: Validation or test DataLoader.
        optimizer: Unused; accepted for trainer signature symmetry.
        device: Torch device.
        tokenizer: T5Tokenizer for decoding.
        max_length: Optional override for generation max length.
        num_beams: Beam-search width.
        log_every: Log progress every N steps; 0 disables logging.

    Returns:
        Tuple of ``(predictions, actuals)`` as lists of decoded strings.
    """
    del optimizer
    model.eval()
    predictions: list[str] = []
    actuals: list[str] = []

    gen_max_length = max_length if max_length is not None else CONFIG.generate_max_length

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["source_ids"].to(device, dtype=torch.long)
            attention_mask = batch["source_mask"].to(device, dtype=torch.long)
            target_ids = batch["target_ids"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=gen_max_length,
                num_beams=num_beams,
                repetition_penalty=CONFIG.repetition_penalty,
                length_penalty=CONFIG.length_penalty,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

            if log_every and step % log_every == 0:
                print(f"Validation step: {step}")

            predictions.extend(preds)
            actuals.extend(targets)

    return predictions, actuals
