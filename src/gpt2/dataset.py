from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom Dataset for fine-tuning gpt-2"""
    def __init__(self, Description, tokenizer, max_length=512):
        self.Description = Description
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "bos_token": "<|startoftext|>",
        })
            

    def __len__(self):
        return len(self.Description)

    def __getitem__(self, index):
        Description = self.tokenizer.bos_token + self.Description[index] + self.tokenizer.eos_token
        return Description

    def collate_fn(self, batch):
        inputs = self.tokenizer(
            batch,
            padding="longest",
            return_tensors="pt",
            truncation=True
        )
        return inputs
