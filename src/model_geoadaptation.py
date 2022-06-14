import torch
from torch import nn
from transformers import BertForMaskedLM, ElectraForMaskedLM


# https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/electra/modeling_electra.py
class ClassificationHead(nn.Module):
    def __init__(self, head, output_dim, config):
        super(ClassificationHead, self).__init__()
        self.head = head
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, output_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.head == 'cls':
            x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GeoBertForMaskedLM(BertForMaskedLM):
    # TODO: implement output_dim as part of config file
    def __init__(self, config, mtl, head, output_dim):
        super(GeoBertForMaskedLM, self).__init__(config)
        self.mtl = mtl
        self.head = head
        if self.mtl:
            self.classifier = ClassificationHead(self.head, output_dim, config)
        if self.mtl == 'uncertainty':
            self.etas = nn.Parameter(torch.zeros(2))

    # https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/bert/modeling_bert.py
    def forward(self, input_ids, attention_mask, token_type_ids, mlm_labels, points, val):

        # Forward pass through BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # MLM loss
        prediction_scores = self.cls(bert_outputs[0])
        mlm_loss_function = nn.CrossEntropyLoss()
        mlm_loss = mlm_loss_function(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        if not self.mtl:
            return mlm_loss, None, None

        # Geo loss
        logits = self.classifier(bert_outputs[0])
        if self.head == 'masked':
            logits = logits[mlm_labels != -100]
        geo_loss_function = nn.L1Loss()
        geo_loss = geo_loss_function(logits, points)
        preds = logits.detach().cpu().tolist()

        # Uncertainty weighting
        if self.mtl == 'uncertainty' and not val:
            mlm_loss = torch.exp(-self.etas[0]) * mlm_loss + self.etas[0]
            geo_loss = torch.exp(-self.etas[1]) * geo_loss + self.etas[1]

        return mlm_loss, geo_loss, preds

    def get_preds(self, input_ids, attention_mask, token_type_ids, masked):

        # Forward pass through BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get prediction scores
        prediction_scores = self.cls(bert_outputs[0])[masked]
        return prediction_scores


class GeoElectraForMaskedLM(ElectraForMaskedLM):
    # TODO: implement output_dim as part of config file
    def __init__(self, config, mtl, head, output_dim):
        super(GeoElectraForMaskedLM, self).__init__(config)
        self.mtl = mtl
        self.head = head
        if self.mtl:
            self.classifier = ClassificationHead(self.head, output_dim, config)
        if self.mtl == 'uncertainty':
            self.etas = nn.Parameter(torch.zeros(2))

    # https://github.com/huggingface/transformers/blob/v4.14.1/src/transformers/models/electra/modeling_electra.py
    def forward(self, input_ids, attention_mask, token_type_ids, mlm_labels, points, val):

        # Forward pass through ELECTRA
        electra_outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # MLM loss
        prediction_scores = self.generator_predictions(electra_outputs[0])
        prediction_scores = self.generator_lm_head(prediction_scores)
        mlm_loss_function = nn.CrossEntropyLoss()
        mlm_loss = mlm_loss_function(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        if not self.mtl:
            return mlm_loss, None, None

        # Geo loss
        logits = self.classifier(electra_outputs[0])
        if self.head == 'masked':
            logits = logits[mlm_labels != -100]
        geo_loss_function = nn.L1Loss()
        geo_loss = geo_loss_function(logits, points)
        preds = logits.detach().cpu().tolist()

        # Uncertainty weighting
        if self.mtl == 'uncertainty' and not val:
            mlm_loss = torch.exp(-self.etas[0]) * mlm_loss + self.etas[0]
            geo_loss = torch.exp(-self.etas[1]) * geo_loss + self.etas[1]

        return mlm_loss, geo_loss, preds

    def get_preds(self, input_ids, attention_mask, token_type_ids, masked):

        # Forward pass through ELECTRA
        electra_outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get prediction scores
        prediction_scores = self.generator_predictions(electra_outputs[0])
        prediction_scores = self.generator_lm_head(prediction_scores)[masked]
        return prediction_scores
