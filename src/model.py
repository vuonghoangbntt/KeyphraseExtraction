from os import X_OK
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import RobertaModel
from torch.nn import Transformer
import math, random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, embedding, use_pretrain=True):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        if use_pretrain == True:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
            self.embedding.requires_grad_(False)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        
        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        input = input.unsqueeze(0)
        # input : [1, ,batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert encoder.n_layers == decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            input = trg[t] if teacher_force else top1
            
        return outputs
class BiLSTM_CRF(nn.Module):
  def __init__(self, vocab_size, emb_size, hidden_size, output_size, embedding, use_pretrain=True, num_layers=1):
    super(BiLSTM_CRF, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = emb_size
    self.hidden_size = hidden_size
    self.target_size = output_size
    if use_pretrain == True:
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
        self.embedding.requires_grad_(False)
    else:
        self.embedding = nn.Embedding(vocab_size, emb_size)

    self.lstm = nn.LSTM(emb_size, hidden_size//2, batch_first = True, bidirectional = True, num_layers=num_layers)
    self.dropout = nn.Dropout(0.2)
    self.hidden2tag = nn.Linear(self.hidden_size, self.target_size)
    #self.softmax = nn.LogSoftmax(dim=2)
    self.crf = CRF(self.target_size, True)
  def forward(self, batch, y, mask):
    emb = self.embedding(batch)
    #emb = self.dropout(emb)
    out, _ = self.lstm(emb)
    out = self.dropout(out)
    out = self.hidden2tag(out)
    return -self.crf(out, y, mask, 'mean')
  def decode(self, batch, mask):
    emb = self.embedding(batch)
    out, _ = self.lstm(emb)
    out = self.hidden2tag(out)
    return self.crf.decode(out, mask)


class BERT_Classification(nn.Module):
  def __init__(self, num_labels=3, hidden_size=64, dropout_rate=0.2):
    super(BERT_Classification, self).__init__()
    self.bert = RobertaModel.from_pretrained('roberta-base')
    self.bert.requires_grad_(False)
    self.dropout = torch.nn.Dropout(dropout_rate)
    self.linear = nn.Sequential(nn.Linear(768, hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, num_labels))
  def forward(self, batch, attention_mask):
    x = self.bert(input_ids=batch, attention_mask=attention_mask)
    x = x[0]
    x = self.dropout(x)
    x = self.linear(x)
    return x.permute(0, 2, 1)


class BERT_CRF(nn.Module):
  def __init__(self, output_size=3, dropout_rate=0.2):
    super(BERT_CRF, self).__init__()

    self.target_size = output_size

    self.bert = RobertaModel.from_pretrained('roberta-base')
    self.bert.requires_grad_(False)
    self.dropout = torch.nn.Dropout(dropout_rate)
    self.hidden2tag = nn.Linear(768, self.target_size)
    #self.softmax = nn.LogSoftmax(dim=2)
    self.crf = CRF(self.target_size, True)
  def forward(self, batch, y, mask):
    x = self.bert(input_ids=batch, attention_mask=mask)
    #emb = self.dropout(emb)
    x = x[0]
    # x = self.linear(x)
    # x = self.dropout(x)
    x = self.hidden2tag(x)
    return -self.crf(x, y, mask, 'mean')
  def decode(self, batch, mask):
    x = self.bert(input_ids=batch, attention_mask=mask)
    #emb = self.dropout(emb)
    x = x[0]
    # x = self.linear(x)
    # x = self.dropout(x)
    x = self.hidden2tag(x)
    return self.crf.decode(x, mask)

class BERT_BiLSTM_CRF(nn.Module):
  def __init__(self, hidden_size, num_layers=1, output_size=3, dropout_rate=0.2):
    super(BERT_BiLSTM_CRF, self).__init__()

    self.hidden_size = hidden_size
    self.target_size = output_size

    self.lstm = nn.LSTM(768, hidden_size//2, batch_first = True, bidirectional = True, num_layers=num_layers)
    self.dropout = nn.Dropout(0.2)

    self.bert = RobertaModel.from_pretrained('roberta-base')
    self.bert.requires_grad_(False)
    self.dropout = torch.nn.Dropout(dropout_rate)
    self.hidden2tag = nn.Linear(hidden_size, self.target_size)
    #self.softmax = nn.LogSoftmax(dim=2)
    self.crf = CRF(self.target_size, True)
  def forward(self, batch, y, mask):
    x = self.bert(input_ids=batch, attention_mask=mask)
    #emb = self.dropout(emb)
    x = x[0]
    #emb = self.dropout(emb)
    out, _ = self.lstm(x)
    out = self.dropout(out)
    out = self.hidden2tag(out)
    return -self.crf(out, y, mask, 'mean')
  def decode(self, batch, mask):
    x = self.bert(input_ids=batch, attention_mask=mask)
    #emb = self.dropout(emb)
    x = x[0]
    out, _ = self.lstm(x)
    out = self.hidden2tag(out)
    return self.crf.decode(out, mask)

class NewModel(nn.Module):
  def __init__(self, vocab_size, emb_size, output_size, embedding, hidden_size=100, use_pretrain=True, num_layers=1, dropout=0.2):
      super(NewModel, self).__init__()
      self.vocab_size = vocab_size
      self.embedding_size = emb_size
      self.hidden_size = hidden_size
      self.target_size = output_size
      if use_pretrain == True:
          self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
          self.embedding.requires_grad_(False)
      else:
          self.embedding = nn.Embedding(vocab_size, emb_size)
        
      self.lstm = nn.LSTM(emb_size, hidden_size//2, batch_first = True, bidirectional = True, num_layers=num_layers)
      self.dropout = nn.Dropout(dropout)
      # self.linear = nn.Linear(768, doc_embed_hidden)
      self.hidden2tag = nn.Linear(self.hidden_size, self.target_size)
      #self.softmax = nn.LogSoftmax(dim=2)
      self.crf = CRF(self.target_size, True)
  def forward(self, batch, y, mask):
      emb = self.embedding(batch)
      #emb = self.dropout(emb)
      doc_embed = torch.sum(emb, axis=-2)/torch.sum(mask, axis=1).unsqueeze(1)
      out = emb+ doc_embed.unsqueeze(1).repeat(1, emb.size(1), 1)
      out, _ = self.lstm(out)
      
      # out = torch.cat((out,doc_embed), dim=2)
      # out = out+doc_embed
      # out = self.dropout(out)
      out = self.hidden2tag(out)
      return -self.crf(out, y, mask, 'mean')
  def decode(self, batch, mask):
      emb = self.embedding(batch)
      #emb = self.dropout(emb)
      doc_embed = torch.sum(emb, axis=-2)/torch.sum(mask, axis=1).unsqueeze(1)
      out = emb+ doc_embed.unsqueeze(1).repeat(1, emb.size(1), 1)
      out, _ = self.lstm(out)
      # out = torch.cat((out,doc_embed), dim=2)
      # out = out+doc_embed
      # out = self.dropout(out)
      out = self.hidden2tag(out)
      return self.crf.decode(out, mask)

# class NewModel(nn.Module):
#     def __init__(self, vocab_size, emb_size, output_size, embedding, hidden_size=100, use_pretrain=True, num_layers=1, dropout=0.2):
#         super(NewModel, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = emb_size
#         self.hidden_size = hidden_size
#         self.target_size = output_size
#         if use_pretrain == True:
#             self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
#             self.embedding.requires_grad_(False)
#         else:
#             self.embedding = nn.Embedding(vocab_size, emb_size)

#         self.lstm = nn.LSTM(emb_size, hidden_size//2, batch_first = True, bidirectional = True, num_layers=num_layers)
#         self.dropout = nn.Dropout(dropout)
#         self.hidden2tag = nn.Linear(self.hidden_size+768, self.target_size)
#         #self.softmax = nn.LogSoftmax(dim=2)
#         self.crf = CRF(self.target_size, True)
#     def forward(self, batch, y, mask, doc_embed):
#         emb = self.embedding(batch)
#         #emb = self.dropout(emb)
#         out, _ = self.lstm(emb)
#         doc_embed = doc_embed.repeat(out.size(1), 1, 1).permute(1,0,2)
#         out = torch.cat((out,doc_embed), dim=2)
#         out = self.dropout(out)
#         out = self.hidden2tag(out)
#         return -self.crf(out, y, mask, 'mean')
#     def decode(self, batch, mask, doc_embed):
#         emb = self.embedding(batch)
#         out, _ = self.lstm(emb)
#         doc_embed = doc_embed.repeat(out.size(1), 1, 1).permute(1,0,2)
#         out = torch.cat((out,doc_embed), dim=2)
#         out = self.hidden2tag(out)
#         return self.crf.decode(out, mask)

class KeyBert(nn.Module):
  def __init__(self, hidden_size=10, output_size=3, dropout=0.2):
    super(KeyBert, self).__init__()
    self.target_size = output_size
    self.dropout = dropout
    self.bert = RobertaModel.from_pretrained('roberta-base')
    self.bert.requires_grad_(False)
    self.linear = nn.Sequential(nn.Linear(769, hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, self.target_size))
    self.similarity = nn.CosineSimilarity(dim=2)
  def forward(self, x, mask, embedding):
    x = self.bert(input_ids=x, attention_mask=mask)
    #emb = self.dropout(emb)
    x = x[0]
    embedding = embedding.repeat(x.size(1), 1, 1).permute(1,0,2)
    similarity = self.similarity(x, embedding)
    similarity = torch.unsqueeze(similarity, 2)
    x = torch.cat((x, similarity), dim=2)
    x = self.linear(x)
    return x.permute(0, 2, 1)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
