import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
import os
import time
import logging
import json
from sklearn.metrics import f1_score, recall_score, precision_score

from model import BERT_CRF
from data import get_dataloader
from metric import get_f1_score
from sklearn.metrics import f1_score, recall_score, precision_score
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, required=True)
parser.add_argument("--data_name", type=str, default='Inspec')
parser.add_argument("--num_epoch", type=int, default=10)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--num_epoch_save", type=float, default=5)

args = parser.parse_args()

file_name = args.file_name



"""
-----------------------------------------
"""

exp_folder = f'KeyPhrase_Extraction/experiments/{file_name}'
cp_folder = f'KeyPhrase_Extraction/saved_model/{file_name}'

os.makedirs(exp_folder, exist_ok=True)
os.makedirs(cp_folder, exist_ok=True)

logging.basicConfig(filename=f'{exp_folder}/{file_name}.txt',
                    filemode='w',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)



"""
--------------Train----------------------
"""
train_loader, test_loader = get_dataloader(args.data_name, test_size=args.test_size, batch_size=args.batch_size, glove_encoding=False)
output_tag = {'B': 0, 'I': 1, 'O': 2}

model = BERT_CRF(dropout_rate=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(reduction='none')


os.makedirs(cp_folder, exist_ok=True)
logging.info(f'\n***************** {file_name.upper()} **************')
for k, v in args._get_kwargs():
    logging.info(f'{k}: {v}')
logging.info('--------------------------------')
logging.info(model)
logging.info('--------------------------------')


train_losses = []
train_precison = []
train_recall = []
train_f1 = []

test_losses = []
test_precison = []
test_recall = []
test_f1 = []

for epoch in range(30):
    epoch_loss = 0
    n_sentences = 0
    train_predict = []
    train_target = []
    test_predict = []
    test_target = []
    i = 0
    for x, y, t in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        mask = (x != 0).to(device)
        mask[:,0] = 1
        y = y.to(device)
        try:
            loss = model(x, y, mask)
        except:
            continue
        # loss = model(x, y, mask)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_sentences += x.shape[0]
        predict = model.decode(x, mask)
        for i in range(len(predict)):
            train_predict.extend(predict[i])
            train_target.extend(y[i].cpu().tolist()[:len(predict[i])])

    train_losses.append(epoch_loss/n_sentences)
    train_f1.append(f1_score(train_target, train_predict, average='macro'))
    train_precison.append(precision_score(train_target, train_predict, average='macro'))
    train_recall.append(recall_score(train_target, train_predict, average='macro'))
    i = 0
    epoch_loss = 0
    n_sentences = 0
    for x, y, t in tqdm(test_loader):
        i += 1
        x = x.to(device)
        mask = (x != 0).to(device)
        mask[:,0] = 1
        y = y.to(device)
        try:
            loss = model(x, y, mask)
        except:
            continue
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_sentences += x.shape[0]

        predict = model.decode(x, mask)
        for i in range(len(predict)):
            test_predict.extend(predict[i])
            test_target.extend(y[i].cpu().tolist()[:len(predict[i])])

    test_losses.append(epoch_loss/n_sentences)
    test_f1.append(f1_score(test_target, test_predict, average='macro'))
    test_precison.append(precision_score(test_target, test_predict, average='macro'))
    test_recall.append(recall_score(test_target, test_predict, average='macro'))
    logging.info(f'---------------------------------------Epoch {epoch+1}---------------------------------------')
    logging.info(f'Train: loss {train_losses[-1]}, precision_score {train_precison[-1]}, recall_score {train_recall[-1]}, f1_score {train_f1[-1]}')
    logging.info(f'Test: loss {test_losses[-1]}, precision_score {test_precison[-1]}, recall_score {test_recall[-1]}, f1_score {test_f1[-1]}')    
    if (epoch+1) % args.num_epoch_save == 0:
        checkpoint = {'model.hidden2tag': model.hidden2tag.state_dict(), 'model.crf': model.crf.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, f'{cp_folder}/{file_name}_epoch{epoch+1}.pt')
logging.info('--------------------------------------------------------------------------------------')
logging.info('                                       Result                                         ')
logging.info('--------------------------------------------------------------------------------------')

#Final evaluate
train_predict = []
train_target = []
test_predict = []
test_target = []
for x, y, t in tqdm(train_loader):
    x = x.to(device)
    mask = (x != 0).to(device)
    mask[:,0] = 1
    y = y.to(device)
    try:
        predict = model.decode(x, mask)
    except:
        continue
    for i in range(len(predict)):
       for i in range(len(predict)):
            train_predict.extend(predict[i])
            train_target.extend(y[i].cpu().tolist()[:len(predict[i])])
for x, y, t in tqdm(test_loader):
    x = x.to(device)
    mask = (x != 0).to(device)
    mask[:,0] = 1
    y = y.to(device)
    try:
        predict = model.decode(x, mask)
    except:
        continue
    for i in range(len(predict)):
        test_predict.extend(predict[i])
        test_target.extend(y[i].cpu().tolist()[:len(predict[i])])
    
logging.info('Train:')
logging.info(f'\n{get_f1_score(train_target, train_predict, output_tag)}')
logging.info('Test: ')
logging.info(f'\n{get_f1_score(test_target, test_predict, output_tag)}')