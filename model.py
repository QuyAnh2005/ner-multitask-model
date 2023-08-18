import torch.nn as nn
from transformers import BertModel, BertConfig

class NeurondBERTMultitask(nn.Module):
    def __init__(self, n_pos: int, n_ner: int, model_name: str = "bert-base-uncased"):
        super(NeurondBERTMultitask, self).__init__()
        self.base_model = BertModel.from_pretrained(model_name)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.n_pos = n_pos
        self.n_ner = n_ner
        self.head1 = nn.Linear(768, self.n_ner)
        self.head2 = nn.Linear(768, self.n_pos)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        # You write you new head here
        output_ner = self.dropout1(outputs[0])
        output_ner = self.head1(output_ner)
        output_pos = self.dropout2(outputs[0])
        output_pos = self.head2(output_pos)
        return output_ner, output_pos
