import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
import os
import time
import logging
import json

from model import Seq2Seq, Encoder, Decoder
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
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--use_pretrain", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--num_epoch_save", type=float, default=2)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--glove_size", choices=['50', '100', '200', '300'], default='100')
parser.add_argument("--decode_emb_dim", choices=['32', '64', '128'], default='64')
parser.add_argument("--loss_weight", type=str, default='1.,1.,1.')

args = parser.parse_args()

file_name = args.file_name

"""
--------------Get vocab and glove embedding----------------------
"""

with open(f'./KeyPhrase_Extraction/data/glove_embedding_{args.glove_size}.json', 'r') as f:
    save = json.loads(f.read())
vocab = save['vocab']
embedding = np.array(save['embedding'])
output_tag = {'B': 0, 'I': 1, 'O': 2}


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
train_loader, test_loader = get_dataloader(args.data_name, test_size=args.test_size, batch_size=args.batch_size, glove_size=args.glove_size)
INPUT_DIM = len(vocab.keys())
OUTPUT_DIM = len(output_tag.keys())
ENC_EMB_DIM = int(args.glove_size)
DEC_EMB_DIM = int(args.decode_emb_dim)
HID_DIM = args.hidden_size
N_LAYERS = args.num_layers
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, embedding=embedding)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

TRG_PAD_IDX = vocab['<pad>']
weights = args.loss_weight
weights = torch.tensor([float(w) for w in weights.split(',')]).to(device)
criterion = nn.CrossEntropyLoss(weight= weights, ignore_index=TRG_PAD_IDX)
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
for epoch in range(args.num_epoch):
    epoch_loss = 0
    train_predict = []
    train_target = []
    test_predict = []
    test_target = []
    n_sentences = 0
    for batch in tqdm(train_loader):
        n_sentences += batch[0].shape[0]
        src = batch[0].transpose(1,0).to(device)
        trg = batch[1].transpose(1,0).to(device)
        mask = src != 0
        optimizer.zero_grad()
        # trg = [sen_len, batch_size]
        # output = [trg_len, batch_size, output_dim]
        output = model(src, trg)
        # print((mask.transpose(1,0).unsqueeze(2)).shape)
        # print(output.shape)
        output = output*mask.unsqueeze(2)
        output_dim = output.shape[-1]

        m = torch.sum(mask, dim=1)
        out = torch.argmax(output.transpose(1,0), dim=2)
        for i in range(out.shape[0]):
            train_predict.extend(out[i].cpu().tolist()[:m[i]])
            train_target.extend(trg[i].cpu().tolist()[:m[i]])
        # transfrom our output : slice off the first column, and flatten the output into 2 dim.
        output = output[1:].view(-1, output_dim) 
        # print(trg[1:].shape)
        trg = trg[1:].reshape(-1,1)[:,0]
        # trg = [(trg_len-1) * batch_size]
        # output = [(trg_len-1) * batch_size, output_dim]
        # print(output.shape)
        # print(trg.shape)
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    train_losses.append(epoch_loss/n_sentences)
    train_f1.append(f1_score(train_target, train_predict, average='macro'))
    train_precison.append(precision_score(train_target, train_predict, average='macro'))
    train_recall.append(recall_score(train_target, train_predict, average='macro'))
    epoch_loss = 0
    n_sentences = 0
    for batch in tqdm(test_loader):
        n_sentences += batch[1].shape[0]
        src = batch[0].transpose(1,0).to(device)
        trg = batch[1].transpose(1,0).to(device)
        mask = src != 0
        output = model(src, trg, 0) # turn off teacher forcing.
        
        output = output*mask.unsqueeze(2)
        output_dim = output.shape[-1]

        m = torch.sum(mask, dim=1)
        out = torch.argmax(output.transpose(1,0), dim=2)
        for i in range(out.shape[0]):
            test_predict.extend(out[i].cpu().tolist()[:m[i]])
            test_target.extend(trg[i].cpu().tolist()[:m[i]])
        # transfrom our output : slice off the first column, and flatten the output into 2 dim.
        output = output[1:].view(-1, output_dim) 
        # print(trg[1:].shape)
        trg = trg[1:].reshape(-1,1)[:,0]
        # trg = [(trg_len-1) * batch_size]
        # output = [(trg_len-1) * batch_size, output_dim]
        # print(output.shape)
        # print(trg.shape)
        loss = criterion(output, trg)
        
        epoch_loss += loss.item()
    test_losses.append(epoch_loss/n_sentences)
    test_f1.append(f1_score(test_target, test_predict, average='macro'))
    test_precison.append(precision_score(test_target, test_predict, average='macro'))
    test_recall.append(recall_score(test_target, test_predict, average='macro'))
    logging.info(f'---------------------------------------Epoch {epoch+1}---------------------------------------')
    logging.info(f'Train: loss {train_losses[-1]}, precision_score {train_precison[-1]}, recall_score {train_recall[-1]}, f1_score {train_f1[-1]}')
    logging.info(f'Test: loss {test_losses[-1]}, precision_score {test_precison[-1]}, recall_score {test_recall[-1]}, f1_score {test_f1[-1]}')
    if (epoch+1) % args.num_epoch_save == 0:
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, f'{cp_folder}/{file_name}_epoch{epoch+1}.pt')
logging.info('--------------------------------------------------------------------------------------')
logging.info('                                       Result                                         ')
logging.info('--------------------------------------------------------------------------------------')

train_predict = []
train_target = []
test_predict = []
test_target = []
for batch in tqdm(train_loader):
    src = batch[0].transpose(1,0).to(device)
    trg = batch[1].transpose(1,0).to(device)
    mask = src != 0
    # trg = [sen_len, batch_size]
    # output = [trg_len, batch_size, output_dim]
    output = model(src, trg)
    # print((mask.transpose(1,0).unsqueeze(2)).shape)
    # print(output.shape)
    output = output*mask.unsqueeze(2)
    output_dim = output.shape[-1]

    m = torch.sum(mask, dim=1)
    out = torch.argmax(output.transpose(1,0), dim=2)
    for i in range(out.shape[0]):
        train_predict.extend(out[i].cpu().tolist()[:m[i]])
        train_target.extend(trg[i].cpu().tolist()[:m[i]])
    
for batch in tqdm(test_loader):
    n_sentences += batch[1].shape[0]
    src = batch[0].transpose(1,0).to(device)
    trg = batch[1].transpose(1,0).to(device)
    mask = src != 0
    output = model(src, trg, 0) # turn off teacher forcing.
    
    output = output*mask.unsqueeze(2)
    output_dim = output.shape[-1]

    m = torch.sum(mask, dim=1)
    out = torch.argmax(output.transpose(1,0), dim=2)
    for i in range(out.shape[0]):
        test_predict.extend(out[i].cpu().tolist()[:m[i]])
        test_target.extend(trg[i].cpu().tolist()[:m[i]])
logging.info('Train:')
logging.info(f'\n{get_f1_score(train_target, train_predict, output_tag)}')
logging.info('Test: ')
logging.info(f'\n{get_f1_score(test_target, test_predict, output_tag)}')