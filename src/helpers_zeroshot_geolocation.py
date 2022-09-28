import torch
from torch.utils.data import Dataset


class DatasetForZeroShot(Dataset):
    def __init__(self, data):
        self.texts = list(data.text)
        self.locations = list(data.location)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        location = self.locations[idx]
        return text, location


class CollatorForZeroShot:
    def __init__(self, tok, data, locations, prompt):
        self.tok = tok
        self.data = data
        self.classes = sorted([self.tok.convert_tokens_to_ids(l) for l in set(locations)])
        self.prompt = prompt

    def prompt_encode(self, batch):
        texts, idxes, classes = list(), list(), list()
        for t, l in batch:
            t += '. ' + self.prompt + " " + l
            #if self.data == 'bcms':
            #    t += '. To je {}'.format(l)
            #else:
            #    t += '. Das ist {}'.format(l)
            t_encoded = self.tok.encode(t, padding=True, truncation=True)
            texts.append(t_encoded)
            idxes.append(len(t_encoded) - 2)
            classes.append(self.tok.convert_tokens_to_ids(l))
        return texts, idxes, classes

    def __call__(self, batch):
        texts, idxes, classes = self.prompt_encode(batch)
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
        return batch_tensors, masked, classes


def get_best(file, k=None):
    results = list()
    with open(file, 'r') as f:
        for l in f:
            if l.strip() == '':
                continue
            l_split = l.strip().split()
            results.append(float(l_split[0]))
    return max(results)
