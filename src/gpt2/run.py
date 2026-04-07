import os
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.gpt2.preprocess import load_data
from src.gpt2.trainer import train, validate
from src.gpt2.utils import split_data, build_dataloaders
from src.gpt2.config import (
    MODEL_NAME,
    EPOCHS,
    LEARNING_RATE,
    DATA_PATH,
    DEVICE,
    OUTPUT_DIR,
)


def main():
    # runtime device depending on gpu or cpu
    runtime_device = DEVICE if torch.cuda.is_available() else "cpu"

    # Load and split data
    description_df = load_data(file_path=DATA_PATH)
    train_description, val_description = split_data(
        description_df,
        test_size=0.20,
        random_state=42,
    )

    # Tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<|pad|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
        }
    )

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(runtime_device)

    # Dataloaders
    train_loader, val_loader = build_dataloaders(
        train_description,
        val_description,
        tokenizer,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * EPOCHS, eta_min=0
    )

    # Train across all epochs
    print("\nTraining:")
    train_loss, train_perplexity = train(
        EPOCHS,
        model,
        runtime_device,
        train_loader,
        optimizer,
        scheduler,
    )

    # Validate once after training
    print("\nValidation:")
    val_loss, val_perplexity = validate(
        model,
        device=runtime_device,
        loader=val_loader,
    )

    # Save artifacts
    model_dir = os.path.join(OUTPUT_DIR, "model_files")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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


if __name__ == "__main__":
    main()
        
            
        
        
    
        
        
    




