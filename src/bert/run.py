
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from collections import defaultdict

from src.bert import config
from src.bert.dataset import CustomDataset
from src.bert.model import BertClassifier
from src.bert.preprocess import load_data
from src.bert.trainer import training_epoch, evaluate
from src.bert.utils import polarity, sentiment, label_encode_sentiments
from src.bert.metrics import (
    compute_accuracy,
    compute_f1_score,
    compute_confusion_matrix,
    get_predictions,
)


if __name__ == "__main__":
    # load data
    df = load_data(file_path=config.DATA_PATH)

    # apply sentiments and polarity
    df["polarity"] = df["Headlines"].apply(polarity)
    df["sentiment"] = df["polarity"].apply(sentiment)
    df = label_encode_sentiments(df)

    # split dataframe
    df_train, df_test = train_test_split(
        df,
        test_size=0.50,
        random_state=42,
    )

    df_val, df_test = train_test_split(
        df_test,
        test_size=0.20,
        random_state=42,
    )

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)

    # custom datasets
    train_set = CustomDataset(
        Headlines=df_train["Headlines"].to_numpy(),
        targets=df_train["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
    )

    test_set = CustomDataset(
        Headlines=df_test["Headlines"].to_numpy(),
        targets=df_test["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
    )

    val_set = CustomDataset(
        Headlines=df_val["Headlines"].to_numpy(),
        targets=df_val["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
    )

    # data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # model, loss, optimizer
    model = BertClassifier()
    model.to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    # training loop
    history = defaultdict(list)

    for epoch in range(config.EPOCHS):
        train_acc, train_loss = training_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer=optimizer,
            device=config.DEVICE,
            n_examples=len(df_train),
        )

        val_acc, val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device=config.DEVICE,
            n_examples=len(df_val),
        )

        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"  Train Accuracy: {train_acc * 100:.2f}% | Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy:   {val_acc * 100:.2f}% | Val Loss:   {val_loss:.4f}")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

    # test evaluation
    test_acc, test_loss = evaluate(
        model,
        test_loader,
        loss_fn,
        device=config.DEVICE,
        n_examples=len(df_test),
    )

    print(f"\nTest Accuracy: {test_acc.item() * 100:.2f}%")

    # metrics
    headlines, y_pred, y_true = get_predictions(model, test_loader, config.DEVICE)

    print(f"F1 Score:  {compute_f1_score(y_true, y_pred):.4f}")
    print(f"Accuracy:  {compute_accuracy(y_true, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{compute_confusion_matrix(y_true, y_pred)}")
            
        
        
    
    
    




