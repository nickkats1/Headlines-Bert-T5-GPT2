import torch.nn as nn
from transformers import BertModel



class BertClassifier(nn.Module):
    """BERT model with a dropout and linear classification head.

    Architecture:
        BERT base (768 hidden dim) -> Dropout(0.3) -> Linear(768, 3)

    Attributes:
        bert: Pretrained BERT model loaded from config.
        drop: Dropout layer for regularization.
        out: Linear layer mapping BERT pooled output to 3 sentiment classes.
    """

    def __init__(self):
        """Initialize the BERT classifier with pretrained weights, dropout, and linear head."""
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        """Forward pass through BERT and classification head.

        Args:
            input_ids: Tokenized input IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, 3).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)