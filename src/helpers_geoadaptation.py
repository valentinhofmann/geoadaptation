import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class DatasetForMaskedLM(Dataset):
    def __init__(self, data, scaler=None):
        self.texts = list(data.text)
        if scaler is None:
            self.scaler = StandardScaler()
            self.points = self.scaler.fit_transform(list(zip(data.longitude, data.latitude)))
        else:
            self.scaler = scaler
            self.points = self.scaler.transform(list(zip(data.longitude, data.latitude)))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        point = self.points[idx]
        return text, point


class CollatorForMaskedLM:
    def __init__(self, tok, head, mlm_probability=0.15):
        self.tok = tok
        self.head = head
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        texts = [self.tok.encode(t, padding=True, truncation=True) for t, _ in batch]
        points = torch.tensor([p for _, p in batch]).float()
        batch_size = len(texts)
        max_length = max(len(text) for text in texts)
        input_ids = torch.zeros((batch_size, max_length)).long()
        attention_mask = torch.zeros((batch_size, max_length)).long()
        token_type_ids = torch.zeros((batch_size, max_length)).long()
        for i, text in enumerate(texts):
            input_ids[i, :len(text)] = torch.tensor(text)
            attention_mask[i, :len(text)] = 1
        batch_tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        # https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py
        mlm_labels = batch_tensors['input_ids'].clone()
        probability_matrix = torch.full(mlm_labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tok.get_special_tokens_mask(val, already_has_special_tokens=True) for val in mlm_labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        batch_tensors['input_ids'][indices_replaced] = self.tok.convert_tokens_to_ids(self.tok.mask_token)
        indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tok), mlm_labels.shape, dtype=torch.long)
        batch_tensors['input_ids'][indices_random] = random_words[indices_random]

        # Repeat points for masked tokens
        if self.head == 'masked':
            n_masks = masked_indices.sum(axis=-1)
            points = torch.repeat_interleave(points, n_masks, dim=0)

        return batch_tensors, mlm_labels, points
