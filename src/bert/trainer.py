import torch
import torch.nn as nn
import numpy as np


def train(model, dataloader, loss_fn, optimizer, device, n_examples, scheduler=None):
    """Run a single training epoch over the entire dataloader.

    Args:
        model: The BERT classifier model.
        dataloader: PyTorch DataLoader containing training batches.
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer (e.g., Adam, AdamW).
        device: Device to run training on ('cuda' or 'cpu').
        n_examples: Total number of training examples (for accuracy calculation).

    Returns:
        A tuple of (accuracy, mean_loss) for the epoch.
    """
    model.train()

    losses = []
    predictions = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()



    accuracy = predictions.double() / n_examples
    mean_loss = np.mean(losses)

    return accuracy, mean_loss


def validate(model, dataloader, loss_fn, device, n_examples):
    """Evaluate the model on a validation or test set."""
    model.eval()

    losses = []
    predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())

    accuracy = predictions.double() / n_examples
    mean_loss = np.mean(losses)

    return accuracy, mean_loss