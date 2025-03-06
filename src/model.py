import torch
from transformers import BertForTokenClassification

class BertPOSModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(BertPOSModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained("google-bert/bert-base-cased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids, attention_mask=attention_mask, labels=labels)