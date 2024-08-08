import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetForID(Dataset):
    def __init__(self, data):
        self.texts = list(data.text)
        self.labels = list(data.id)
        self.n_labels = len(set(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


class CollatorForID:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        texts = [self.tok.encode(t, padding=True, truncation=True) for t, _ in batch]
        labels = torch.tensor([l for _, l in batch]).long()
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
        return batch_tensors, labels


def get_best(file):
    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                l_strip = l.strip().split()
                results.append((*[float(v) for v in l_strip[:-1]], int(l_strip[-1])))
        return max(results)
    except (FileNotFoundError, ValueError):
        return None, None


def get_stats(file):
    results = list()
    with open(file, 'r') as f:
        for l in f:
            if l.strip() == '':
                continue
            l_strip = l.strip().split()
            results.append(float(l_strip[0]))
    return np.mean(results), np.std(results)
