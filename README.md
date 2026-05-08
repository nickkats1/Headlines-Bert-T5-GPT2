# Headlines-Bert-T5-GPT2

NLP playground for fine-tuning three transformer architectures on news headline data:

| Model  | Task                                            | Data                              |
| ------ | ----------------------------------------------- | --------------------------------- |
| BERT   | Sentiment classification (3 classes)            | Guardian headlines                |
| GPT-2  | Causal language modeling                        | Reuters descriptions              |
| T5     | Headline summarization (Description → Headline) | Reuters headlines + descriptions  |

Each model lives in its own package under `src/` with a clean separation between
config, preprocessing, dataset, model, trainer, and a runnable entry point.

## Project layout

```
rutgers-nlp/
├── data/                      # Raw CSV inputs (Guardian + Reuters)
├── src/
│   ├── common.py              # seed_everything, resolve_device
│   ├── bert/                  # BERT sentiment classifier
│   ├── gpt2/                  # GPT-2 fine-tuning
│   └── t5/                    # T5 summarization
├── tests/                     # Pytest suite (84 tests)
├── pyproject.toml             # Packaging, pytest, ruff config
├── requirements.txt
└── README.md
```

Every model package follows the same structure:

```
src/<model>/
├── __init__.py
├── config.py        # Frozen @dataclass with sane defaults (CONFIG)
├── preprocess.py    # CSV → cleaned DataFrame
├── dataset.py       # PyTorch Dataset
├── model.py         # (BERT only — others use HuggingFace heads directly)
├── trainer.py       # train() / validate() loops
├── metrics.py       # Task-appropriate evaluation
├── utils.py         # Helpers (data splits, label encoding, ...)
└── run.py           # End-to-end entry point with argparse CLI
```

## Installation

```bash
git clone https://github.com/<your-user>/rutgers-nlp.git
cd rutgers-nlp

python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate

pip install -r requirements.txt
# or, for an editable install:
pip install -e ".[dev]"
```

The CSV inputs are tracked in `data/`. No external download is required.

## Quick start

Run any of the three pipelines from the repo root:

```bash
python -m src.bert.run        # BERT sentiment classifier
python -m src.gpt2.run        # GPT-2 fine-tuning
python -m src.t5.run          # T5 summarization
```

Or via the installed entry-points:

```bash
rutgers-bert
rutgers-gpt2
rutgers-t5
```

Each pipeline accepts CLI overrides for any hyperparameter:

```bash
python -m src.bert.run --epochs 1 --device cpu --batch-size 4
python -m src.t5.run --model-name t5-small --epochs 1
```

Use `--help` on any of them to see the full list. Artifacts (models,
predictions, metrics) land under `src/<model>/artifacts/`.

The pipelines auto-detect CUDA via `src.common.resolve_device` and silently
fall back to CPU when no GPU is available.

## Configuration

Each pipeline has a frozen `dataclass` with sane defaults:

```python
from src.bert.config import CONFIG, BertConfig

CONFIG.model_name        # "bert-base-uncased"
CONFIG.epochs            # 4

# Override anything you like:
custom = BertConfig(epochs=1, learning_rate=1e-5)
```

`run.py` accepts the same fields as CLI flags, so most experiments don't
require editing files at all.

## Testing & linting

```bash
pytest                          # full suite (84 tests)
pytest tests/bert -v            # one package
pytest -k "preprocess"          # by keyword

ruff check src/ tests/          # lint
ruff format src/ tests/         # auto-format
```

CI runs the same checks on every push / PR via `.github/workflows/ci.yaml`
(ruff lint + format check, then pytest on Python 3.11 and 3.12).

## License

[MIT License](LICENSE).
