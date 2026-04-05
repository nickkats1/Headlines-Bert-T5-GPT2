from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Dataset that tokenizes source-target text pairs for T5.

    Takes a dataframe with source and target text columns, tokenizes both
    using the provided T5 tokenizer, and returns padded/truncated tensors
    ready for encoder-decoder training.

    Attributes:
        tokenizer: HuggingFace T5 tokenizer.
        data: Pandas DataFrame containing source and target text.
        source_len: Maximum token length for source (encoder) sequences.
        target_len: Maximum token length for target (decoder) sequences.
        source_text: Series of source text strings.
        target_text: Series of target text strings.
    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_col, target_col):
        """Initialize the dataset with a dataframe and tokenizer.

        Args:
            dataframe: Pandas DataFrame containing the text data.
            tokenizer: HuggingFace T5 tokenizer for encoding text.
            source_len: Maximum token length for source sequences.
            target_len: Maximum token length for target sequences.
            source_col: Column name for source (input) text in the dataframe.
            target_col: Column name for target (output) text in the dataframe.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[source_col]
        self.target_text = self.data[target_col]

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.target_text)

    def __getitem__(self, index):
        """Tokenize and return a single source-target pair.

        Args:
            index: Index of the example to retrieve.

        Returns:
            A dictionary containing:
                - source_ids: Tokenized source input IDs of shape (source_len,).
                - source_mask: Source attention mask of shape (source_len,).
                - target_ids: Tokenized target input IDs of shape (target_len,).
        """
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        source = self.tokenizer(
            source_text,
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer(
            target_text,
            max_length=self.target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "source_ids": source["input_ids"].squeeze(),
            "source_mask": source["attention_mask"].squeeze(),
            "target_ids": target["input_ids"].squeeze(),
        }