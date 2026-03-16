# Rutgers-Bert-T5-GPT2

A collection of fine-tuning pipelines and evaluation tooling for NLP models (BERT for classification, T5 for seq2seq / summarization, GPT-2 for headline generation) on the Guardian headlines dataset.



## Table of contents

- [Rutgers-Bert-T5-GPT2](#rutgers-bert-t5-gpt2)
  - [Table of contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Project structure](#project-structure)
  - [Quickstart](#quickstart)
  - [How to run each model](#how-to-run-each-model)
    - [BERT (classification)](#bert-classification)
    - [T5 (seq2seq / summarization / headline generation)](#t5-seq2seq--summarization--headline-generation)
    - [GPT-2 (headline generation)](#gpt-2-headline-generation)
  - [Evaluation metrics and scripts](#evaluation-metrics-and-scripts)
    - [Perplexity / loss (GPT-2)](#perplexity--loss-gpt-2)
    - [ROUGE (T5)](#rouge-t5)
    - [ROUGE / BLEU (optional for GPT-2)](#rouge--bleu-optional-for-gpt-2)
  - [License](#license)

---

## Requirements

Recommended Python 3.9+ (your environment uses 3.13 in notebooks; adjust as needed).

Install base dependencies (example):
```bash
python -m pip install -r requirements.txt
```

If you don't have a requirements.txt, install the minimum set:
```bash
pip install torch transformers pandas scikit-learn rouge-score nltk textblob contractions
```

## Project structure

```txt

├── data
└── src
    ├── bert
    │   ├── config.py
    │   ├── dataset
    │   │   ├── custom_dataset.py
    │   │   └── __init__.py
    │   ├── eval
    │   │   ├── __init__.py
    │   │   └── train_eval_epoch.py
    │   ├── __init__.py
    │   ├── model
    │   │   ├── bert.py
    │   │   └── __init__.py
    │   ├── run.py
    │   └── utils
    │       ├── __init__.py
    │       └── processing.py
    ├── gpt2
    │   ├── config.py
    │   ├── dataset
    │   │   ├── custom_dataset.py
    │   │   └── __init__.py
    │   ├── eval
    │   │   ├── generate.py
    │   │   └── train_eval_epoch.py
    │   ├── helpers.py
    │   ├── __init__.py
    │   ├── models
    │   └── run.py
    └── t5
        ├── config.py
        ├── dataset
        │   ├── custom_dataset.py
        │   └── __init__.py
        ├── evaluate
        │   ├── eval.py
        │   └── __init__.py
        ├── __init__.py
        └── train
            ├── epochs.py
            ├── __init__.py
            └── trainer.py
```


## Quickstart

1. Create and activate a virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare dataset(s) in `data/` (by default many scripts expect `data/guardian_headlines.csv`).

3. Run the model you want to work with (see below).

---

## How to run each model

### BERT (classification)

Train / run:
```bash
python src/bert/run.py
```

What it does:
- Loads CSV (default path from `src/bert/config.py`)
- Computes polarity and maps to discrete labels
- Builds DataLoaders, trains the BERT classifier, prints train/val metrics, runs final test evaluation

Notes:
- Ensure `src/bert/config.py` keys line up with what `run.py` expects (e.g., `max_len` vs `max_length`, `batch_size`, `epochs`).
- If you want a scheduler, pass one or make the training helper accept optional scheduler.

### T5 (seq2seq / summarization / headline generation)

The T5 pipeline is in `src/t5/`. Example (if you built a script to run the T5 trainer):
```bash
# if you implemented a runnable entry
python -m src.t5.train.trainer  # or run your T5 training entrypoint
```

To run T5, make sure you use training like this

```py
T5Trainer(dataframe=df[:5000], source_text="Description", target_text="Headlines", model_params=**params, output_dir=".")
```

Do not use your entire DataFrame for training T5. It will take too long to train


What to check:
- `src/t5/config.py` contains model and hyperparams.
- `src/t5/train/trainer.py` expects `train()` and `validate()` functions; ensure these exist or are imported.
- Evaluation in `src/t5/evaluate/eval.py` computes ROUGE-1/2/L.

### GPT-2 (headline generation)


Important:
- Ensure tokenizer and model saved files exist in `src/gpt2/models/` (HuggingFace saves `config.json`, `pytorch_model.bin` or `model.safetensors`, and tokenizer files).
- GPT-2 needs a `pad_token` configured (common pattern: `tokenizer.pad_token = tokenizer.eos_token` and set `model.config.pad_token_id = tokenizer.pad_token_id`) — this is already used in helpers.

--- 

## Evaluation metrics and scripts

### Perplexity / loss (GPT-2)
Perplexity is computed via:
- Average validation loss → perplexity = exp(avg_loss)
See: `src/gpt2/eval/train_eval_epoch.py`

### ROUGE (T5)
ROUGE metrics are computed in:
- `src/t5/evaluate/eval.py` — use `rouge_score` package

### ROUGE / BLEU (optional for GPT-2)
If you fine-tune GPT-2 on a deterministic target (like full headlines) you can compute ROUGE/BLEU to measure overlap:
- Add `src/gpt2/eval/metrics.py` (sample code available):
  - `compute_rouge(predictions, references)`
  - `compute_bleu(predictions, references)`


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 nickkats1


