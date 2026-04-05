import os

from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch

import pandas as pd




device = "cuda:0"



def T5Trainer(dataframe, source_text, target_text, model_params, output_dir="."):
    """Fine tune T5 model on source, target text pairs and generate predictions.

    Args:
        dataframe: pandas dataframe combining source and target columns.
        source_text: column name for source text (input text).
        target_text: Column name for target text (output texr).
        model_params: Dictionary of hyper-parameters with keys.
        output_dir: Directory to save model files and output predictions. Defaults to ".".

    Returns:
        dataframe with "Generated Text" and "Actual Text" columns as CSV file
    """

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    dataframe = dataframe[[source_text, target_text]]

    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    training_set = CustomDataset(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = CustomDataset(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    training_loader = torch.utils.data.DataLoader(training_set, **train_params)
    val_loader = torch.utils.data.DataLoader(val_set, **val_params)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    return final_df