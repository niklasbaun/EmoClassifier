import torch.nn as nn
from transformers import BertModel

"""
Custom Class
Emotion Classifier using BERT.
Args:
    - n_classes (int): Number of emotion classes.
    - bert_model (str): Pretrained BERT model name or path.
    - dropout_rate (float): Dropout rate for regularization.
    - out_features (int): Number of output features for the final layer.
"""
class EmoClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmoClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
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
