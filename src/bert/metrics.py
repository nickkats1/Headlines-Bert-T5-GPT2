from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch


# --- Accuracy Score ---

def compute_accuracy(y_true, y_pred):
    """Compute accuracy score for Bert"""
    return accuracy_score(y_true, y_pred)


# --- weighted f1 score ---

def compute_f1_score(y_true, y_pred):
    """Compute weight macro f1-score for BERT"""
    return f1_score(y_true, y_pred, average="weighted")

# --- Confusion matrix ---

def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix for BERT"""
    return confusion_matrix(y_true, y_pred)




# --- Get Predictions ---
def get_predictions(model , dataloader):
    """Get predictions once Bert is done training"""
    y_pred = []
    y_true = []
    Headline = []
    
    with torch.no_grad():
        device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        for d in dataloader:
            Headlines = d['Headlines']
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            
            Headline.extend(Headlines)
            
            y_pred.extend(preds)
            
            y_true.extend(targets)
            
        y_pred = torch.stack(y_pred).cpu()
        y_true = torch.stack(y_true).cpu()
        return Headline, y_pred, y_true

            
            
    
    
