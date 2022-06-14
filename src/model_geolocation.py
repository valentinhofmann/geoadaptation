from torch import nn
from transformers import BertModel, ElectraModel


# https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/electra/modeling_electra.py
class ClassificationHead(nn.Module):
    def __init__(self, output_dim, config):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, output_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gelu = nn.GELU()

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertForGeoprediction(nn.Module):
    def __init__(self, model, geo_type, output_dim):
        super(BertForGeoprediction, self).__init__()
        self.geo_type = geo_type
        self.bert = BertModel.from_pretrained(model)
        self.classifier = ClassificationHead(output_dim, self.bert.config)

    # https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/bert/modeling_bert.py
    def forward(self, input_ids, attention_mask, token_type_ids, points, cluster_labels):

        # Forward pass through BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Loss
        logits = self.classifier(bert_outputs[0])
        if self.geo_type == 'points':
            loss_function = nn.L1Loss()
            loss = loss_function(logits, points)
            preds = logits.detach().cpu().tolist()
        elif self.geo_type == 'kmeans':
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits, cluster_labels)
            preds = logits.detach().cpu().max(1).indices.tolist()
        return loss, preds


class ElectraForGeoprediction(nn.Module):
    def __init__(self, model, geo_type, output_dim):
        super(ElectraForGeoprediction, self).__init__()
        self.geo_type = geo_type
        self.electra = ElectraModel.from_pretrained(model)
        self.classifier = ClassificationHead(output_dim, self.electra.config)

    # https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/electra/modeling_electra.py
    def forward(self, input_ids, attention_mask, token_type_ids, points, cluster_labels):

        # Forward pass through ELECTRA
        electra_outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Loss
        logits = self.classifier(electra_outputs[0])
        if self.geo_type == 'points':
            loss_function = nn.L1Loss()
            loss = loss_function(logits, points)
            preds = logits.detach().cpu().tolist()
        elif self.geo_type == 'kmeans':
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits, cluster_labels)
            preds = logits.detach().cpu().max(1).indices.tolist()
        return loss, preds
