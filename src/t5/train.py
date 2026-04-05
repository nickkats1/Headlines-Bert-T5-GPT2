import torch


def train(epoch, tokenizer, model, device, loader, optimizer):
    """Runs one training epoch for the T5 model.

    Shifts target IDs to create decoder inputs and language model labels.
    Masks padding tokens with -100 so they are ignored by the loss function.

    Args:
        epoch: Current epoch number.
        tokenizer: T5Tokenizer, used to identify pad token ID.
        model: T5ForConditionalGeneration.
        device: Torch device.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
    """
    model.train()
    for step, batch in enumerate(loader):
        target_ids = batch["target_ids"].to(device, dtype=torch.long)
        y_ids = target_ids[:, :-1].contiguous()

        lm_labels = target_ids[:, 1:].clone().detach()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100

        input_ids = batch["source_ids"].to(device, dtype=torch.long)
        attention_mask = batch["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )

        loss = outputs.loss

        if step % 50 == 0:
            print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(tokenizer, model, device, loader):
    """Generates predictions on a validation or test set.

    Uses beam search to generate text from source inputs, then decodes
    both predictions and targets back to strings.

    Args:
        tokenizer: T5Tokenizer for decoding generated IDs.
        model: T5ForConditionalGeneration.
        device: Torch device.
        loader: Validation or test DataLoader.

    Returns:
        predictions: List of generated text strings.
        actuals: List of ground truth text strings.
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["source_ids"].to(device, dtype=torch.long)
            attention_mask = batch["source_mask"].to(device, dtype=torch.long)
            target_ids = batch["target_ids"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

            if step % 10 == 0:
                print(f"Validation step: {step}")

            predictions.extend(preds)
            actuals.extend(targets)

    return predictions, actuals