import torch.nn as nn
from transformers import BertModel

"""
Classifier class for emotion classification using BERT.
initializes the BERT model, dropout layer, and output layer.
Args:
    n_classes (int): Number of emotion classes for classification.
"""
class EmoClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmoClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)
