import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import collate_fn
import json


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return FloatTensor(self.x[index]), LongTensor(self.y[index])
    
    def __len__(self):
        return len(self.x)

class DataSet2(Dataset):
    def __init__(self, x, y, embed):
        self.x = x
        self.y = y
        self.embed = embed
    def __getitem__(self, index):

        return FloatTensor(self.x[index]), LongTensor(self.y[index]), torch.as_tensor(self.embed[index][0], device=device)
    
    def __len__(self):
        return len(self.x)

def get_dataloader(data_name, test_size=0.2, random_state=42, glove_encoding=True, batch_size=32, glove_size='100'):
    if glove_encoding:
        with open(f'./KeyPhrase_Extraction/data/{data_name}_glove_encoding_{glove_size}.json', 'r') as f:
            save = json.loads(f.read())
    else:
        with open(f'./KeyPhrase_Extraction/data/{data_name}_bert_encoding.json', 'r') as f:
            save = json.loads(f.read())
    file_names, encoded_texts, encoded_targets, targets = save['file_names'], save['encoded_texts'], save['encoded_targets'], save['targets']

    x_train, x_test, y_train, y_test = train_test_split(encoded_texts, encoded_targets, test_size=test_size, random_state=random_state)
    train_dataset = DataSet(x_train, y_train)
    test_dataset = DataSet(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader