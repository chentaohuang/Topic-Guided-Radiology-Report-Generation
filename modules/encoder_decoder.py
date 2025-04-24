from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .att_model import pack_wrapper, AttModel
device = torch.device('cuda:0')

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)


    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, node_encoder,g_encoder,gram_encoder,encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.gram_encoder = gram_encoder

        self.encoder = encoder
        self.decoder = decoder
        self.g_encoder = g_encoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.node_encoder = node_encoder
        self.gram_linear = nn.Linear(768,512)
        self.node_linear = nn.Linear(768,512)


        self.linear = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
        )

    def forward(self,text_feature,txt_mean,src, tgt, src_mask, tgt_mask,topic_sigma,topic_features,hidden_states):



        node_embeddings = self.get_node_embedding()
        gram_embeddings = self.get_gram_embedding()
        ind,selected_feature = self.topic_infer(text_feature,  topic_sigma, topic_features)
        b_num = tgt.size(0)
        grams, gram_masks = self.get_topic_gram(b_num, ind, gram_embeddings)
        nodes, node_masks = self.get_topic_node(b_num, ind, node_embeddings)

        nodes = self.node_encoder(src[:,1:,:], nodes, node_masks)

        result = self._decode(selected_feature,txt_mean,hidden_states , src_mask, tgt, tgt_mask,grams,gram_masks,nodes,node_masks )


        return result

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def get_adjusted_topic_probs(self,feature,topic_sigma,topic_features):

        pos_portion = torch.tensor([68316, 69578, 13931, 31353, 49107, 10990, 11650, 66836, 67657,
             55532, 7241, 23797, 7516, 59323,
             556, 87058, 97344, 53375, 129649, 9466,2035, 8184, 0, 172482, 74, 22170, 191569, 941]).to(device)

        topic_pi = pos_portion / pos_portion.sum()
        theta = feature.unsqueeze(dim=1)
        topic_features = topic_features.unsqueeze(dim=0)
        residual = theta - topic_features


        tmp = torch.bmm(residual / (topic_sigma.unsqueeze(dim=0) ** 2), residual.transpose(2, 1)) / 512 ** (0.5)
        det = torch.zeros(28).to(device)
        for i in range(512):
            det = det + 0.5 * torch.log(topic_sigma[:, i] ** 2 + 1e-25)
        const = -(det)
        const = torch.clip(const, max=50)
        likelihood = torch.exp(const.unsqueeze(dim=0) + torch.diagonal(tmp, dim1=1, dim2=2) * (-1 / 2))
        t = likelihood * (topic_pi.unsqueeze(dim=0))**(1.5)
        prob = t / torch.sum(t, dim=1).unsqueeze(dim=1)

        return prob

    def get_topic_probs(self,feature,topic_sigma,topic_features):

        pos_portion = torch.tensor([68316, 69578, 13931, 31353, 49107, 10990, 11650, 66836, 67657,
                                    55532, 7241, 23797, 7516, 59323,
                                    556, 87058, 97344, 53375, 129649, 9466, 2035, 8184, 0, 172482, 74, 22170, 191569,
                                    941]).to(device)
        topic_pi = pos_portion / pos_portion.sum()
        theta = feature.unsqueeze(dim=1)
        topic_features = topic_features.unsqueeze(dim=0)
        residual = theta - topic_features
        tmp = torch.bmm(residual / (topic_sigma.unsqueeze(dim=0) ** 2 ), residual.transpose(2, 1)) / 512 ** (0.5)
        det = torch.zeros(28).to(device)
        for i in range(512):
            det = det + 0.5 * torch.log(topic_sigma[:, i] ** 2 + 1e-25)
        const = -(det)
        const = torch.clip(const, max=50)
        likelihood = torch.exp(const.unsqueeze(dim=0) + torch.diagonal(tmp, dim1=1, dim2=2) * (-1 / 2))
        t = likelihood *(topic_pi.unsqueeze(dim=0))
        prob = t / torch.sum(t, dim=1).unsqueeze(dim=1)

        return prob

    def get_node_embedding(self):
        with open("path of mimic_cxr_KG.json", "r", encoding="utf-8") as f:
            f_read = json.load(f)

        embeddings = []
        eye = torch.eye(768).to(device)
        i = 1
        topics = ['Atelectasis_True', 'Cardiomegaly_True', 'Consolidation_True', 'Edema_True',
                  'Enlarged Cardiomediastinum_True', 'Fracture_True', 'Lung Lesion_True', 'Lung Opacity_True',
                  'No Finding_True', 'Pleural Effusion_True', 'Pleural Other_True', 'Pneumonia_True',
                  'Pneumothorax_True', 'Support Devices_True',
                  'Atelectasis_False', 'Cardiomegaly_False', 'Consolidation_False', 'Edema_False',
                  'Enlarged Cardiomediastinum_False', 'Fracture_False', 'Lung Lesion_False', 'Lung Opacity_False',
                  'No Finding_False', 'Pleural Effusion_False', 'Pleural Other_False', 'Pneumonia_False',
                  'Pneumothorax_False', 'Support Devices_False']


        for topic in topics:
            if 'matrix' in f_read[topic]:
                embedding = torch.tensor(f_read[topic]['embedding']).to(device).unsqueeze(dim=0)
                matrix = torch.tensor(f_read[topic]['matrix']).to(device).unsqueeze(dim=0)
                embeddings.append(self.g_encoder(embedding,matrix).squeeze(dim=0))

            else:
                embeddings.append(self.node_linear(eye[i]).unsqueeze(dim=0))
                i = i+1
        return embeddings

    def get_gram_embedding(self):
        with open("path of mimic_cxr_gram_embed.json", "r", encoding="utf-8") as f:
            f_read = json.load(f)
        embeddings = []
        eye = torch.eye(768).to(device)
        k=1
        topics = ['Atelectasis_True', 'Cardiomegaly_True', 'Consolidation_True', 'Edema_True',
                  'Enlarged Cardiomediastinum_True', 'Fracture_True', 'Lung Lesion_True', 'Lung Opacity_True',
                  'No Finding_True', 'Pleural Effusion_True', 'Pleural Other_True', 'Pneumonia_True',
                  'Pneumothorax_True', 'Support Devices_True',
                  'Atelectasis_False', 'Cardiomegaly_False', 'Consolidation_False', 'Edema_False',
                  'Enlarged Cardiomediastinum_False', 'Fracture_False', 'Lung Lesion_False', 'Lung Opacity_False',
                  'No Finding_False', 'Pleural Effusion_False', 'Pleural Other_False', 'Pneumonia_False',
                  'Pneumothorax_False', 'Support Devices_False']

        for i in topics:
            if i not in f_read or len(f_read[i])==0:
                a = eye[k]
                a = self.gram_linear(a)
                embeddings.append(a.unsqueeze(dim=0))
                k = k+1
            else:
                q = torch.tensor(f_read[i]).to(device)
                a = self.gram_linear(q)
                embeddings.append(a)
        return embeddings

    def get_topic_node(self,b_num,ind,node_embeddings):
        max_seq_length = 0
        merged_node = []
        node_mask = []
        for i in range(b_num):
            merge_0 = node_embeddings[ind[i][0]]
            merge_1 = node_embeddings[ind[i][1]]
            merge_2 = node_embeddings[ind[i][2]]
            merge_3 = node_embeddings[ind[i][3]]
            merge_4 = node_embeddings[ind[i][4]]

            merge = torch.cat([merge_0, merge_1,merge_2,merge_3,merge_4], dim=0)

            length = merge.size(0)
            if length > max_seq_length:
                max_seq_length = length
            mask = [1] * length
            node_mask.append(mask)
            merged_node.append(merge)

        node_masks = np.zeros((b_num, max_seq_length), dtype=int)
        for i, mask in enumerate(node_mask):
            node_masks[i, :len(mask)] = mask
        node_masks = torch.FloatTensor(node_masks).unsqueeze(dim=-2).to(device)
        nodes = torch.empty((0, max_seq_length, 512)).to(device)
        for i, node in enumerate(merged_node):
            length = node.size(0)
            if length < max_seq_length:
                add = torch.zeros((max_seq_length - length, 512)).to(device)
                node = torch.cat([node, add], dim=0)
            nodes = torch.cat([nodes, node.unsqueeze(dim=0)], dim=0)
        return nodes,node_masks

    def get_topic_gram(self,b_num,ind,gram_embeddings):
        max_seq_length = 0
        merged_gram = []
        gram_mask = []
        for i in range(b_num):
            merge_0 = gram_embeddings[ind[i][0]]
            merge_1 = gram_embeddings[ind[i][1]]
            merge_2 = gram_embeddings[ind[i][2]]
            merge_3 = gram_embeddings[ind[i][3]]
            merge_4 = gram_embeddings[ind[i][4]]


            merge = torch.cat([merge_0,merge_1,merge_2,merge_3,merge_4], dim=0)

            length = merge.size(0)
            if length > max_seq_length:
                max_seq_length = length
            mask = [1] * length
            gram_mask.append(mask)
            merged_gram.append(merge)

        gram_masks = np.zeros((b_num, max_seq_length), dtype=int)
        for i, mask in enumerate(gram_mask):
            gram_masks[i, :len(mask)] = mask
        gram_masks = torch.FloatTensor(gram_masks).unsqueeze(dim=-2).to(device)

        grams = torch.empty((0, max_seq_length, 512)).to(device)
        for i, gram in enumerate(merged_gram):
            length = gram.size(0)
            if length < max_seq_length:
                add = torch.zeros((max_seq_length - length, 512)).to(device)
                gram = torch.cat([gram, add], dim=0)
            grams = torch.cat([grams, gram.unsqueeze(dim=0)], dim=0)
        return grams,gram_masks

    def topic_infer(self,text_feature,  topic_sigma, topic_features):

        topic_features = topic_features.squeeze(dim=0)
        prob = self.get_topic_probs(text_feature, topic_sigma,  topic_features)
        val, ind = prob.topk(5, dim=1)
        val = val / torch.sum(val, dim=1).unsqueeze(dim=1)
        val = val.unsqueeze(dim=2)
        eyes = torch.eye(28).to(device)
        prob = torch.sum(eyes[ind] * val.detach(), dim=1)
        selected_feature = prob @ topic_features
        selected_feature = selected_feature.unsqueeze(dim=1)

        return ind,selected_feature

    def _decode(self, selected_feature,txt_mean,hidden_states, src_mask, tgt, tgt_mask,grams,gram_masks,nodes,node_masks):

        b, l, _ = tgt_mask.size()
        add_mask = (torch.zeros([b, 1, l]) > 0).to(tgt_mask.device)
        tgt_mask = torch.cat([add_mask, tgt_mask], dim=1)
        add_mask = (torch.ones([b, l + 1, 1]) > 0).to(tgt_mask.device)
        tgt_mask = torch.cat([add_mask, tgt_mask], dim=2)
        hidden_states = hidden_states[:,1:,:]
        tgt_embed = self.tgt_embed(tgt)

        b_num,length = tgt.size()
        selected_feature = torch.repeat_interleave(selected_feature, length, dim=1)
        input = [tgt_embed,selected_feature]
        tgt_embed = torch.cat(input,dim=2)
        tgt_embed = self.linear(tgt_embed)


        tgt_embed = torch.cat([txt_mean.unsqueeze(dim=1),tgt_embed],dim=1)
        return self.decoder(tgt_embed, hidden_states, src_mask[:,:,1:], tgt_mask,grams,gram_masks, nodes,node_masks)



class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class G_Encoder(nn.Module):
    def __init__(self, layer, N):
        super(G_Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.linear =  nn.Linear(768, layer.d_model)
        nn.init.kaiming_normal_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, x, mask=None):
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TextEncoder(nn.Module):
    def __init__(self, layer, N,seqembed):
        super(TextEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.seqembed = seqembed

    def forward(self,x,mask):
        x = self.seqembed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class NodeEncoder(nn.Module):
    def __init__(self, layer, N):
        super(NodeEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, m, mask=None):
        for layer in self.layers:
            x = layer(x, m, mask)
        return self.norm(x)


class NodeEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(NodeEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, m, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, m, m, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, grams,gram_masks,nodes,node_masks):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, grams,gram_masks,nodes,node_masks)
        return self.norm(x)


class TextualLayer(nn.Module):
    def __init__(self, d_model, src_attn, feed_forward, dropout):
        super(TextualLayer, self).__init__()
        self.d_model = d_model
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, hidden_states, src_mask=None):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn,gram_attn,node_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.node_attn = node_attn
        self.gram_attn = gram_attn
        self.dropout = nn.Dropout(dropout)
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 5)



    def forward(self, x, hidden_states, src_mask, tgt_mask, grams,gram_masks,nodes,node_masks):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[3](x, lambda x: self.node_attn(x, nodes, nodes))
        x1 = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        x2 = self.sublayer[2](x, lambda x: self.gram_attn(x, grams, grams,gram_masks))
        x = x1+x2



        return self.sublayer[4](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]


        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderDecoder(AttModel):

    def make_textual_encoder(self):
        c = copy.deepcopy
        dim = 768
        attn = MultiHeadedAttention(self.num_heads,dim)
        ff = PositionwiseFeedForward(dim, self.d_ff, self.dropout)
        textual_encoder = TextualLayer(dim, c(attn), c(ff), self.dropout)
        for p in textual_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return textual_encoder

    def make_text_encoder(self):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        text_encoder = TextEncoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers, nn.Sequential(Embeddings(self.d_model, self.vocab_size+1), c(position)))
        for p in text_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return text_encoder

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        node_encoder = NodeEncoderLayer(self.d_model, c(attn), c(ff), self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        g_encoder = G_Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), 2)

        model = Transformer(
            NodeEncoder(c(node_encoder),self.num_layers),
            c(g_encoder),
            NodeEncoderLayer(self.d_model, c(attn), c(ff), self.dropout),
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(attn),c(attn),c(ff), self.dropout),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position))
            )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.text_encoder = self.make_text_encoder()
        self.textual_encoder = self.make_textual_encoder()

        self.norm1 = LayerNorm(self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.do = nn.Dropout(self.dropout)

        self.norm2 = LayerNorm(self.d_model)
        self.ff2 = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.do2 = nn.Dropout(self.dropout)


        self.logit = nn.Linear(args.d_model, tgt_vocab)
        self.logit1 = nn.Linear(args.d_model, tgt_vocab)

        self.linear = nn.Linear(768, args.d_model)
        self.linear1 = nn.Linear(768, args.d_model)



        self.text_sigma = nn.Parameter(torch.ones(512))
        self.image_sigma = nn.Parameter(torch.ones(512))
        self.topic_sigma = nn.Parameter(torch.ones([28,512]))

        nn.init.constant_(self.text_sigma, 1)
        nn.init.constant_(self.image_sigma, 1)
        nn.init.constant_(self.topic_sigma, 1)


    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks,topic_features,gram_embeddings,node_embeddings):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)

        memory = self.model.encode(att_feats, att_masks)
        att_feats = att_feats[:,1:,:]
        image_mean = memory[:, 0, :].squeeze(dim=1)
        image_mean = self.Linear1(image_mean)
        image_mean1 = self.Draw_infer(image_mean, self.image_sigma)
        ind, selected_feature = self.model.topic_infer( image_mean1, self.topic_sigma, topic_features)

        img_mean = self.Linear2(image_mean1)

        b_num = image_mean.size(0)
        grams, gram_masks = self.model.get_topic_gram(b_num, ind, gram_embeddings)
        nodes, node_masks = self.model.get_topic_node(b_num, ind, node_embeddings)
        nodes = self.model.node_encoder(att_feats, nodes,  node_masks)

        return selected_feature, img_mean, memory, att_masks,grams,gram_masks,nodes,node_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def prepare_bert_mask(self,seq):
        bert_mask = (seq.data > 0)
        bert_mask[:, 0] += True
        mask = bert_mask
        bert_mask = bert_mask.unsqueeze(-2)
        bert_mask = bert_mask.repeat(1,seq.size(-1),1)
        return bert_mask,mask

    def get_topic_feature(self,report_embed):
        topic_feature = torch.empty((0,512)).to(device)
        topics = ['Atelectasis_True', 'Cardiomegaly_True', 'Consolidation_True', 'Edema_True', 'Enlarged Cardiomediastinum_True', 'Fracture_True', 'Lung Lesion_True', 'Lung Opacity_True', 'No Finding_True', 'Pleural Effusion_True', 'Pleural Other_True', 'Pneumonia_True', 'Pneumothorax_True', 'Support Devices_True',
                  'Atelectasis_False', 'Cardiomegaly_False', 'Consolidation_False', 'Edema_False', 'Enlarged Cardiomediastinum_False', 'Fracture_False', 'Lung Lesion_False', 'Lung Opacity_False', 'No Finding_False', 'Pleural Effusion_False', 'Pleural Other_False', 'Pneumonia_False', 'Pneumothorax_False', 'Support Devices_False']

        for i in topics:
            if len(report_embed[i])==1:
                f = self.linear1(torch.tensor(report_embed[i][0]).to(device).unsqueeze(dim=0))
                topic_feature = torch.cat((topic_feature,f),dim=0)

            else:
                qkv = torch.tensor(report_embed[i]).to(device).unsqueeze(dim=0)
                feat = self.textual_encoder(qkv,qkv).squeeze(dim=0)
                feat = self.linear(feat[0]).unsqueeze(dim=0)
                topic_feature = torch.cat([topic_feature,feat],dim=0)

        return topic_feature

    def KL(self,img_mean,text_mean,text_sigma,image_sigma ):

        n = img_mean.size(0)
        det = 1
        for i in range(512):
            det = det * (text_sigma[i] / (image_sigma[i] + 1e-25))
        residual = text_mean - img_mean
        divergence = 1 / 2 * (torch.sum(
            torch.diagonal((residual / ((0.1 * image_sigma.unsqueeze(dim=0)) ** 2) @ residual.t()), dim1=0, dim2=1)) +
                                   +n * (torch.sum((text_sigma / image_sigma) ** 2) - torch.log((det) ** 2)))

        print('divergence loss:', divergence / n)
        return divergence/n

    def CE(self,labels,logits):
        ce_loss = 0
        b = logits.size(0)

        label = labels
        logit = torch.log(logits  + 1e-25)
        ce_loss = - torch.sum(
            torch.diagonal((label @ logit.t()), dim1=0, dim2=1))

        print('ce_loss', ce_loss / (b))
        return ce_loss / (b)

    def Draw_classify(self,mean,dev):
        b_num, dim_num = mean.size()
        k = 5
        rand = torch.normal(0, 1, (b_num, k, dim_num)).to(device)
        rand = torch.mean(rand, 1)
        mean = 0.1 * rand * dev + mean
        return mean

    def Draw_infer(self,mean,dev):
        b_num, dim_num = mean.size()
        k = 5
        rand = torch.normal(0, 1, (b_num, k, dim_num)).to(device)
        rand = torch.mean(rand, 1)
        mean = 0.1 * rand * dev + mean
        return mean

    def Linear1(self,x):
        return self.norm1(x+self.do(self.ff(x)))

    def Linear2(self,x):
        return self.norm2(x+self.do2(self.ff2(x)))

    def _forward(self, report_embed,fc_feats, att_feats, seq, labels,att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        bert_mask,mask = self.prepare_bert_mask(seq)

        mask_sum = torch.sum(mask, dim=-1)
        mask = (mask / (mask_sum.unsqueeze(dim=-1))).unsqueeze(dim=-1)

        text_features = self.text_encoder(seq,bert_mask)
        image_features = self.model.encode(att_feats, att_masks)

        #get global representations
        image_mean = image_features[:, 0, :].squeeze(dim=1)
        image_mean = self.Linear1(image_mean)
        text_mean = torch.sum((text_features * mask), dim=1)

        #sampling
        text_mean1 = self.Draw_classify(text_mean,self.text_sigma)
        image_mean1 = self.Draw_classify(image_mean, self.image_sigma)


        topic_features = self.get_topic_feature(report_embed)


        img_topic_logits = self.model.get_adjusted_topic_probs(image_mean1,  self.topic_sigma,
                                                        topic_features.detach())
        text_topic_logits = self.model.get_adjusted_topic_probs(text_mean1, self.topic_sigma,
                                                         topic_features)

        #get classification loss
        clfloss = self.CE(labels, img_topic_logits) + self.CE(labels, text_topic_logits)

        #get KL divergence
        divergence = self.KL(image_mean,text_mean.detach(),self.text_sigma, self.image_sigma)



        text_mean2 = self.Draw_infer(text_mean, self.text_sigma)
        txt_mean = self.Linear2(text_mean2)
        out= self.model(text_mean2,txt_mean,att_feats, seq, att_masks, seq_mask,self.topic_sigma,topic_features,image_features)


        out1 = self.logit1(out[:, 0, :].unsqueeze(dim=1))
        out2 = self.logit(out[:, 1:, :])
        out = torch.cat([out1, out2], dim=1)
        outputs = F.log_softmax(out, dim=-1)
        return outputs,clfloss,divergence

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask,gram,gram_masks,nodes,node_masks):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)


        selected_feature = fc_feats_ph
        img_mean = att_feats_ph


        out = self.model._decode(selected_feature, img_mean, memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device),gram,gram_masks,nodes,node_masks)

        return out[:, -1], [ys.unsqueeze(0)]
