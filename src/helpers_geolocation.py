import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class DatasetForGeoprediction(Dataset):
    def __init__(self, data, kmeans=None, scaler=None):
        self.texts = list(data.text)
        if kmeans is None:
            self.kmeans = KMeans(n_clusters=75, random_state=123)
            self.cluster_labels = self.kmeans.fit_predict(list(zip(data.longitude, data.latitude)))
        else:
            self.kmeans = kmeans
            self.cluster_labels = self.kmeans.predict(list(zip(data.longitude, data.latitude)))
        if scaler is None:
            self.scaler = StandardScaler()
            self.points = self.scaler.fit_transform(list(zip(data.longitude, data.latitude)))
        else:
            self.scaler = scaler
            self.points = self.scaler.transform(list(zip(data.longitude, data.latitude)))

    def cluster_labels2points(self, cluster_labels):
        points = [self.kmeans.cluster_centers_[l].tolist() for l in cluster_labels]
        return points

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        point = self.points[idx]
        cluster_label = self.cluster_labels[idx]
        return text, point, cluster_label


class CollatorForGeoprediction:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        texts = [self.tok.encode(t, padding=True, truncation=True) for t, *_ in batch]
        points = torch.tensor([p for _, p, _ in batch]).float()
        cluster_labels = torch.tensor([l for *_, l in batch]).long()
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
        return batch_tensors, points, cluster_labels


def get_best(file, k=None):
    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                l_strip = l.strip().split()
                results.append((*[float(v) for v in l_strip[:-1]], int(l_strip[-1])))
        if k:
            return min([r for r in results if r[-1] <= k])
        else:
            return min(results)
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
