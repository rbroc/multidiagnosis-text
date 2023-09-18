import torch
from pathlib import Path
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.utils.class_weight import compute_class_weight

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, ids, ldict):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids
        self.ldict = ldict

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.ldict[self.labels[idx]])
        item['id'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.labels)

    
class FasttextDataset(torch.utils.data.Dataset):
    def __init__(self, split, mtype, diagnosis):
        data_path = str(Path('data') / 'vecs' / f'{split}.csv')
        df = pd.read_pickle(data_path).sample(frac=1)
        if mtype == 'binary':
            df = df[df['ExampleID'].str.contains(diagnosis)]
            df['labels'] = (df.labels == 0).astype(int)
        self.embeddings = torch.tensor([v for v in df.vec], dtype=torch.half)
        self.labels = torch.tensor([l for l in df.labels.tolist()])
        self.diagnosis = diagnosis
        self.mtype = mtype
        n_classes = 2 if mtype is 'binary' else 4
        weights = compute_class_weight(class_weight='balanced', 
                                       classes=list(range(n_classes)), 
                                       y=[int(l) for l in self.labels])
        self.weights = torch.tensor(weights, dtype=torch.half)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)
    
    
class BowDataset(torch.utils.data.Dataset):
    def __init__(self, split, dimensionality, mtype, diagnosis):
        data_path = str(Path('data') / 'vecs' / f'{split}.csv')
        df = pd.read_pickle(data_path).sample(frac=1)
        if mtype == 'binary':
            df = df[df['ExampleID'].str.contains(diagnosis)]
            df['labels'] = (df.labels == 0).astype(int)
        tknzr = Tokenizer(num_words=dimensionality)
        tknzr.fit_on_texts(df['text'].tolist())
        tokenized = tknzr.texts_to_matrix(df['text'].tolist(), mode='tfidf')
        self.embeddings = torch.tensor([t for t in tokenized], dtype=torch.half)
        self.labels = torch.tensor([l for l in df.labels.tolist()])
        self.diagnosis = diagnosis
        self.mtype = mtype
        n_classes = 2 if mtype is 'binary' else 4
        weights = compute_class_weight(class_weight='balanced', 
                                       classes=list(range(n_classes)), 
                                       y=[int(l) for l in self.labels])
        self.weights = torch.tensor(weights, dtype=torch.half)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
        
    def __len__(self):
        return len(self.labels)