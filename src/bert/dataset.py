import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom Dataset for fine-tuning BERT."""

    def __init__(self, Headlines, targets, tokenizer, max_length):
        self.Headlines = Headlines
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.Headlines)

    def __getitem__(self, index):
        Headlines = str(self.Headlines[index])
        target = self.targets[index]

        encoder = self.tokenizer(
            Headlines,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "Headlines": Headlines,
            "input_ids": encoder['input_ids'].flatten(),
            "attention_mask": encoder['attention_mask'].flatten(),
            "targets": torch.tensor(target, dtype=torch.long)
        }