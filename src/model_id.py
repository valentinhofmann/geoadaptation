from torch import nn
from transformers import BertModel, ElectraModel, XLMRobertaModel


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


class BertForID(nn.Module):
    def __init__(self, model, output_dim):
        super(BertForID, self).__init__()
        self.bert = BertModel.from_pretrained(model)
        self.classifier = ClassificationHead(output_dim, self.bert.config)

    # https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/bert/modeling_bert.py
    def forward(self, input_ids, attention_mask, token_type_ids, labels):

        # Forward pass through BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Loss
        logits = self.classifier(bert_outputs[0])
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, labels)
        preds = logits.detach().cpu().max(1).indices.tolist()
        return loss, preds


class ElectraForID(nn.Module):
    def __init__(self, model, output_dim):
        super(ElectraForID, self).__init__()
        self.electra = ElectraModel.from_pretrained(model)
        self.classifier = ClassificationHead(output_dim, self.electra.config)

    # https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/electra/modeling_electra.py
    def forward(self, input_ids, attention_mask, token_type_ids, labels):

        # Forward pass through ELECTRA
        electra_outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Loss
        logits = self.classifier(electra_outputs[0])
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, labels)
        preds = logits.detach().cpu().max(1).indices.tolist()
        return loss, preds


class RobertaForID(nn.Module):
    def __init__(self, model, output_dim):
        super(RobertaForID, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model, add_pooling_layer=False)
        self.classifier = ClassificationHead(output_dim, self.roberta.config)

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py
    def forward(self, input_ids, attention_mask, token_type_ids, labels):

        # Forward pass through BERT
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Loss
        logits = self.classifier(roberta_outputs[0])
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, labels)
        preds = logits.detach().cpu().max(1).indices.tolist()
        return loss, preds
