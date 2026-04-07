from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch


def compute_accuracy(y_true, y_pred):
    """Compute accuracy score for BERT."""
    return accuracy_score(y_true, y_pred)


def compute_f1_score(y_true, y_pred):
    """Compute weighted F1 score for BERT."""
    return f1_score(y_true, y_pred, average="weighted")


def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix for BERT."""
    return confusion_matrix(y_true, y_pred)


def get_predictions(model, dataloader, device):
    """Collect headlines, predicted labels, and true labels."""
    model.eval()

    headlines = []
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in dataloader:
            batch_headlines = batch["Headlines"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            headlines.extend(batch_headlines)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(targets.cpu().tolist())

    return headlines, torch.tensor(y_pred), torch.tensor(y_true)

            
            
    
    
