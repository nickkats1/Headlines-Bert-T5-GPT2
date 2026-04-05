import pandas as pd
from rouge_score import rouge_scorer


def compute_rouge(predictions, actuals):
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: List of generated headline strings.
        actuals: List of actual headline strings.

    Returns:
        Dictionary with average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    n = len(predictions)
    if n == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge1, rouge2, rougeL = 0, 0, 0

    for pred, actual in zip(predictions, actuals):
        scores = scorer.score(actual, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    results = {
        "rouge1": rouge1 / n,
        "rouge2": rouge2 / n,
        "rougeL": rougeL / n,
    }

    print(f"  ROUGE-1: {results['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']:.4f}")
    print(f"  ROUGE-L: {results['rougeL']:.4f}")

    return results


def save_predictions(predictions, actuals, path="predictions.csv"):
    """Saves generated vs actual headlines to CSV.

    Args:
        predictions: List of generated headline strings.
        actuals: List of actual headline strings.
        path: Output CSV path.
    """
    df = pd.DataFrame(
        {
            "Generated Text": predictions,
            "Actual Text": actuals,
        }
    )
    df.to_csv(path, index=False)
    print(f"  Predictions saved to {path}")