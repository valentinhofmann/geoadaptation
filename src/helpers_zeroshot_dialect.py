import torch
from torch.utils.data import Dataset


class DatasetForZeroShot(Dataset):
    def __init__(self, data):
        self.texts = list(data.text)
        self.words = list(data.word)
        self.pairs = list(data.pair)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word = self.words[idx]
        pair = self.pairs[idx]
        return text, word, pair


class CollatorForZeroShot:
    def __init__(self, tok, data):
        self.tok = tok
        self.data = data

    def prompt_encode(self, batch):
        texts, idxes, words, pairs = list(), list(), list(), list()
        for t, w, p in batch:
            t_encoded = self.tok.encode(t, padding=True, truncation=True)
            texts.append(t_encoded)
            idxes.append(t_encoded.index(self.tok.convert_tokens_to_ids(w)))
            words.append(self.tok.convert_tokens_to_ids(w))
            pairs.append(sorted([self.tok.convert_tokens_to_ids(p[0]), self.tok.convert_tokens_to_ids(p[1])]))
        return texts, idxes, words, pairs

    def __call__(self, batch):
        texts, idxes, words, pairs = self.prompt_encode(batch)
        batch_size = len(texts)
        max_length = max(len(text) for text in texts)
        input_ids = torch.zeros((batch_size, max_length)).long()
        attention_mask = torch.zeros((batch_size, max_length)).long()
        token_type_ids = torch.zeros((batch_size, max_length)).long()
        masked = torch.zeros((batch_size, max_length)).bool()
        for i, (text, idx) in enumerate(zip(texts, idxes)):
            input_ids[i, :len(text)] = torch.tensor(text)
            input_ids[i, idx] = self.tok.mask_token_id
            attention_mask[i, :len(text)] = 1
            masked[i, idx] = True
        batch_tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        return batch_tensors, masked, words, pairs


def get_best(file, k=None):
    results = list()
    with open(file, 'r') as f:
        for l in f:
            if l.strip() == '':
                continue
            l_split = l.strip().split()
            results.append(float(l_split[0]))
    return max(results)
