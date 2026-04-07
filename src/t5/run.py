import os


import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split

from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.t5.preprocess import load_data
from src.t5.dataset import CustomDataset
from t5.metrics import compute_rouge
from src.t5.trainer import train, validate
from src.t5.config import (
    SOURCE_LENGTH,
    TARGET_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    MODEL_NAME,
    LEARNING_RATE,
    DATA_PATH,
    DEVICE,
    OUTPUT_DIR
)


def main():
    # load data
    df = load_data(file_path=DATA_PATH)

    # add "summarize: " to source column

    df['Description'] = "summarize: " + df['Description']

    # split data

    df_train, df_val = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )


    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)


    # load model and tokenizer

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

  
    model.to(DEVICE)


    # Custom Dataset for training data

    train_set = CustomDataset(
        df_train,
        tokenizer,
        source_len=SOURCE_LENGTH,
        target_len=TARGET_LENGTH,
        source_col="Description",
        target_col="Headlines"
    )

    # custom dataset for validation data

    val_set = CustomDataset(
        df_val,
        tokenizer,
        source_len=SOURCE_LENGTH,
        target_len=TARGET_LENGTH,
        source_col="Description",
        target_col="Headlines"
    )


    # train and validation loader

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        train(epoch, tokenizer, model, DEVICE, train_loader, optimizer)

    path = os.path.join(".","model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    for epoch in range(EPOCHS):
        predictions, actuals = validate(tokenizer, model, DEVICE, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))
        scores = compute_rouge(final_df['Generated Text'], final_df['Actual Text'])
        print(f"Rouge Score: {scores}")

if __name__ == "__main__":
    main()