import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab


class simpleModel(nn.Module):

    def __init__(self, args):
        super(simpleModel, self).__init__()

        self.args = args

        if self.args.use_elmo:
            self.embedding_dim = self.args.elmo_num_layer * 1024
        else:
            self.embedding_dim = 300
            self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
            self.embedding.weight.data.fill_(0)
            self.embedding.weight.data[:2].normal_(0, 0.1)

        self.pos_embedding = nn.Embedding(len(pos_vocab), args.pos_emb_dim, padding_idx=0)
        self.pos_embedding.weight.data.normal_(0, 0.1)
        self.ner_embedding = nn.Embedding(len(ner_vocab), args.ner_emb_dim, padding_idx=0)
        self.ner_embedding.weight.data.normal_(0, 0.1)
        self.rel_embedding = nn.Embedding(len(rel_vocab), args.rel_emb_dim, padding_idx=0)
        self.rel_embedding.weight.data.normal_(0, 0.1)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}

        self.q_c_emb_match = layers.SeqAttnMatch(self.embedding_dim)  # choice-aware question representation

        # RNN question encoder: 2 * word emb + rel emb
        qst_input_size = 2 * self.embedding_dim + args.rel_emb_dim
        self.question_rnn = layers.StackedBRNN(
            input_size=qst_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # 2 * rnn emb (2 layers?) + pos emb + ner emb + manual features

        proj_input_size = 2 * args.hidden_size + args.pos_emb_dim + args.ner_emb_dim + 3 + args.rel_emb_dim

        self.project = nn.Linear(proj_input_size, 1)

    def forward(self, q, q_pos, q_ner, q_mask, c, c_mask, f_tensor, q_c_relation):
        if self.args.use_elmo:  # already are embeddings
            q_emb, c_emb = q, c
        else:
            q_emb, c_emb = self.embedding(q), self.embedding(c)

        q_pos_emb, q_ner_emb = self.pos_embedding(q_pos), self.ner_embedding(q_ner)
        q_c_rel_emb = self.rel_embedding(q_c_relation)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            q_emb = nn.functional.dropout(q_emb, p=self.args.dropout_emb, training=self.training)
            c_emb = nn.functional.dropout(c_emb, p=self.args.dropout_emb, training=self.training)
            q_pos_emb = nn.functional.dropout(q_pos_emb, p=self.args.dropout_emb, training=self.training)
            q_ner_emb = nn.functional.dropout(q_ner_emb, p=self.args.dropout_emb, training=self.training)
            q_c_rel_emb = nn.functional.dropout(q_c_rel_emb, p=self.args.dropout_emb, training=self.training)

        # word level match between q and c
        q_c_weighted_emb = self.q_c_emb_match(q_emb, c_emb, c_mask)
        q_c_weighted_emb = nn.functional.dropout(q_c_weighted_emb, p=self.args.dropout_emb,
                                                 training=self.training)  # B x T x H

        # q rnn 
        q_rnn_input = torch.cat([q_emb, q_c_weighted_emb, q_c_rel_emb], dim=2)
        q_hiddens = self.question_rnn(q_rnn_input, q_mask)

        # hiddens concat with hand-craft features
        # print(q_hiddens.size(), q_pos_emb.size(), q_ner_emb.size(), f_tensor.size(), q_c_rel_emb.size())
        concat = torch.cat([q_hiddens, q_pos_emb, q_ner_emb, f_tensor, q_c_rel_emb], dim=-1)

        logits = self.project(concat)

        proba = logits
        # proba = F.sigmoid(logits)
        # proba = F.softmax(proba, dim=-1)
        return proba

class simpleModelWOFeatures(nn.Module):

    def __init__(self, args):
        super(simpleModelWOFeatures, self).__init__()

        self.args = args

        if self.args.use_elmo:
            self.embedding_dim = self.args.elmo_num_layer * 1024
        else:
            self.embedding_dim = 300
            self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
            self.embedding.weight.data.fill_(0)
            self.embedding.weight.data[:2].normal_(0, 0.1)

        self.pos_embedding = nn.Embedding(len(pos_vocab), args.pos_emb_dim, padding_idx=0)
        self.pos_embedding.weight.data.normal_(0, 0.1)
        self.ner_embedding = nn.Embedding(len(ner_vocab), args.ner_emb_dim, padding_idx=0)
        self.ner_embedding.weight.data.normal_(0, 0.1)
        self.rel_embedding = nn.Embedding(len(rel_vocab), args.rel_emb_dim, padding_idx=0)
        self.rel_embedding.weight.data.normal_(0, 0.1)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}

        self.q_c_emb_match = layers.SeqAttnMatch(self.embedding_dim)  # choice-aware question representation

        # RNN question encoder: 2 * word emb + rel emb
        qst_input_size = 2 * self.embedding_dim
        self.question_rnn = layers.StackedBRNN(
            input_size=qst_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # 2 * rnn emb (2 layers?) + pos emb + ner emb + manual features

        proj_input_size = 2 * args.hidden_size

        self.project = nn.Linear(proj_input_size, 1)

    def forward(self, q, q_pos, q_ner, q_mask, c, c_mask, f_tensor, q_c_relation):
        if self.args.use_elmo:  # already are embeddings
            q_emb, c_emb = q, c
        else:
            q_emb, c_emb = self.embedding(q), self.embedding(c)

        q_pos_emb, q_ner_emb = self.pos_embedding(q_pos), self.ner_embedding(q_ner)
        q_c_rel_emb = self.rel_embedding(q_c_relation)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            q_emb = nn.functional.dropout(q_emb, p=self.args.dropout_emb, training=self.training)
            c_emb = nn.functional.dropout(c_emb, p=self.args.dropout_emb, training=self.training)
            q_pos_emb = nn.functional.dropout(q_pos_emb, p=self.args.dropout_emb, training=self.training)
            q_ner_emb = nn.functional.dropout(q_ner_emb, p=self.args.dropout_emb, training=self.training)
            q_c_rel_emb = nn.functional.dropout(q_c_rel_emb, p=self.args.dropout_emb, training=self.training)

        # word level match between q and c
        q_c_weighted_emb = self.q_c_emb_match(q_emb, c_emb, c_mask)
        q_c_weighted_emb = nn.functional.dropout(q_c_weighted_emb, p=self.args.dropout_emb,
                                                 training=self.training)  # B x T x H

        # q rnn
        q_rnn_input = torch.cat([q_emb, q_c_weighted_emb], dim=2)
        q_hiddens = self.question_rnn(q_rnn_input, q_mask)

        # hiddens concat with hand-craft features
        # print(q_hiddens.size(), q_pos_emb.size(), q_ner_emb.size(), f_tensor.size(), q_c_rel_emb.size())
        concat = torch.cat([q_hiddens], dim=-1)

        logits = self.project(concat)

        proba = logits
        # proba = F.sigmoid(logits)
        # proba = F.softmax(proba, dim=-1)
        return proba
