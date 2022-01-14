import numpy as np
import os
from os.path import join, isfile
import nltk
import argparse  
import json
from torch.nn.modules.module import T
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

import torch
nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='Inspec')
parser.add_argument("--get_embedding", type=bool, default=False)
parser.add_argument("--bert_encoding", type=bool, default=False)
parser.add_argument("--glove_encoding", type=bool, default=False)
parser.add_argument("--doc_embedding", type=bool, default=False)

args = parser.parse_args()

def encode(text, key, replaced_text, vocab, output_tag):
    file_names = []
    encoded_texts = []
    encoded_targets = []
    targets = []
    for file_name in tqdm(text.keys()):
        tokenized_x = nltk.word_tokenize(text[file_name])
        encoding = []
        for word in tokenized_x:
            if word in vocab.keys():
                encoding.append(vocab[word])
            else:
                encoding.append(vocab['<unk>'])
        encoded_texts.append(encoding)

        replaced_x = nltk.word_tokenize(replaced_text[file_name])
        target = []
        encoded_target = []
        for word in replaced_x:
            if word[:9] != 'xxxkeyxxx':
                target.append('O')
                encoded_target.append(output_tag['O'])
            else:
                try:
                    k = key[file_name][int(word[9:])]
                except:
                    print(word)
                tokenized_k = nltk.word_tokenize(k)
                target.append('B')
                encoded_target.append(output_tag['B'])
                for i in range(1, len(tokenized_k)):
                    target.append('I')
                    encoded_target.append(output_tag['I'])
        targets.append(target)
        encoded_targets.append(encoded_target)
        file_names.append(file_name)
    return file_names, encoded_texts, encoded_targets, targets

def encode_bert(text, key, replaced_text, output_tag):
    from tqdm import tqdm
    from transformers import RobertaTokenizer
    bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    file_names = []
    encoded_texts = []
    encoded_targets = []
    targets = []
    for file_name in tqdm(replaced_text.keys()):
        encoded_text = []
        encoded_target = []
        target = []
        words = replaced_text[file_name].split()
        for i in range(len(words)):
            word = words[i]
            if word[:9] != 'xxxkeyxxx':
                if i != 0:
                    word = ' ' + word
                enc = bert_tokenizer.encode(word)[1:-1]
                encoded_text.extend(enc)
                encoded_target.extend([output_tag['O']]*len(enc))
                target.extend(['O']*len(enc))
            else:
                k = key[file_name][int(word[9:])][1:-1]
                if i != 0:
                    k = ' ' + k
                enc = bert_tokenizer.encode(k)[1:-1]
                encoded_text.extend(enc)
                encoded_target.extend([output_tag['B']] + [output_tag['I']]*(len(enc)-1))
                target.extend(['B'] + ['I']*(len(enc)-1))
        file_names.append(file_name)
        encoded_texts.append([0]+encoded_text+[2])
        encoded_targets.append([output_tag['O']] + encoded_target + [output_tag['O']])
        targets.append(['O'] + target + ['O'])
    return file_names, encoded_texts, encoded_targets, targets

# Read data
data_name = args.data_name
text_folder = f'./KeyPhrase_Extraction/data/{data_name}/docsutf8'
key_folder = f'./KeyPhrase_Extraction/data/{data_name}/keys'
text_files = [file for file in  os.listdir(text_folder) if isfile(join(text_folder, file))]
key_files = [file for file in  os.listdir(key_folder) if isfile(join(key_folder, file))]

text = {}
for text_file in text_files:
    with open(join(text_folder, text_file), 'r', encoding='utf8') as f:
        text[text_file[:-4]] = ' '.join(nltk.word_tokenize(f.read().strip().replace('\n', ' ').replace('\t', '')))
key = {}
for key_file in key_files:
    with open(join(key_folder, key_file), 'r', encoding="utf8") as f:
        key[key_file[:-4]] = f.read().strip().split('\n')
    key[key_file[:-4]] = [(' ' + ' '.join(k.split()) + ' ') for k in key[key_file[:-4]]]

replaced_text = {}
for file_name in text.keys():
    replaced_text[file_name] = text[file_name]
    for i in range(len(key[file_name])):
        k = key[file_name][i]
        replaced_text[file_name] = replaced_text[file_name].replace(k, f' xxxkeyxxx{i} ')

if args.glove_encoding or args.get_embedding:
    # Get vocab and embedding from Glove
    glove_path = './KeyPhrase_Extraction/glove/glove.6B.100d.txt'
    vocab,embeddings = [],[]
    with open(glove_path,'r', encoding='utf8') as fi:
        full_content = fi.read().strip().split('\n')
        
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    START_TAG = '<start>'
    END_TAG = '<end>'

    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    vocab_npa = np.insert(vocab_npa, 2, '<START_TAG>')
    vocab_npa = np.insert(vocab_npa, 3, '<END_TAG>')

    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
    start_emb_npa = np.random.randn(1, embs_npa.shape[1])
    end_emb_npa = np.random.randn(1, embs_npa.shape[1])
    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, start_emb_npa, end_emb_npa, embs_npa))

    vocab = {}
    for i, word in enumerate(vocab_npa):
        vocab[word] = i

output_tag = {'B': 0, 'I': 1, 'O': 2}

if args.glove_encoding:
# Encode and save to file
    file_names, encoded_texts, encoded_targets, targets = encode(text, key, replaced_text, vocab, output_tag)

    save = {'file_names': file_names, 'encoded_texts': encoded_texts, 'encoded_targets': encoded_targets, 'targets': targets}
    with open(f'./KeyPhrase_Extraction/data/{data_name}_glove_encoding.json', 'w') as f:
        f.write(json.dumps(save))

if args.get_embedding:
    save = {'vocab': vocab, 'embedding': embs_npa.tolist()}
    with open(f'./KeyPhrase_Extraction/data/glove_embedding.json', 'w') as f:
        f.write(json.dumps(save))

if args.bert_encoding:
    file_names, encoded_texts, encoded_targets, targets = encode_bert(text, key, replaced_text, output_tag)
    save = {'file_names': file_names, 'encoded_texts': encoded_texts, 'encoded_targets': encoded_targets, 'targets': targets}
    with open(f'./KeyPhrase_Extraction/data/{data_name}_bert_encoding.json', 'w') as f:
        f.write(json.dumps(save))

if args.doc_embedding:
    bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    bert = RobertaModel.from_pretrained('roberta-base')
    doc_embed = []
    for k in tqdm(text.keys()):
        x = bert_tokenizer.encode(text[k])
        try:
            doc_embed.append(bert(torch.LongTensor([x]))[1].tolist())
        except:
            doc_embed.append(None)
    save = {'doc_embed': doc_embed}
    with open(f'./KeyPhrase_Extraction/data/{args.data_name}_doc_embedding.json', 'w') as f:
        f.write(json.dumps(save))