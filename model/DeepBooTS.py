import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from snns.Frequency_LIF import Frequency_LIF
from snns.surrogate_gradient import SpikeGeneration
from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        #V = self.dropout(torch.softmax(scale * queries, dim=-1)) * values

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_block, d_ff=None, dropout=0.1, gate=1, configs=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=1)
        d_ff = d_model * 2 if attention else d_model
        self.conv5 = nn.Conv1d(d_ff, d_block, kernel_size=1)
        self.conv6 = nn.Conv1d(d_ff, d_block, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = gate

        self.activation_type = configs.activation
        if configs.activation == "origin":
            self.activation = F.gelu
        elif configs.activation == "Frequency_LIF":
            # initialize the learnable betas
            self.activation = nn.Sequential(
                Frequency_LIF(
                    v_threshold=configs.threshold,
                    surrogate_function=SpikeGeneration.apply,
                    hard_reset=False,
                    detach_reset=False,
                    alpha=configs.alpha,
                    use_dynamic_beta=configs.use_dynamic_beta,
                    beta=configs.beta,
                    beta_decay_factor=configs.beta_decay_factor
                )
            )

    def forward(self, x, attn_mask=None, t=0):
        if self.attention:
            x_att, _ = self.attention(x, x, x, attn_mask=attn_mask)
            x = x - self.dropout(x_att)

        x_ln = x = self.norm1(x)

        conv_out = self.conv1(x_ln.transpose(-1, 1))
        if self.activation_type == "Frequency_LIF":
            activation_out = self.activation[0](conv_out, t=t)
        else:
            activation_out = self.activation(conv_out)

        x_ln = self.dropout(activation_out)
        x_ln = self.dropout(self.conv2(x_ln).transpose(-1, 1))

        x = (x - x_ln).transpose(-1, 1)
        h_gate = F.sigmoid(self.conv3(x)) if self.gate else 1
        h = h_gate * self.conv4(x)

        out = torch.cat((x_att, x_ln), -1) if self.attention else x_ln
        out = out.transpose(-1, 1)
        gate = F.sigmoid(self.conv5(out)) if self.gate else 1
        out = gate * self.conv6(out)

        return self.norm2(h.transpose(-1, 1)), out.transpose(-1, 1)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None

    def forward(self, x, attn_mask=None, t=0):
        output = 0
        for attn_layer in self.attn_layers:
            x, out = attn_layer(x, attn_mask=attn_mask, t=t)
            output = out - output

        return output

class standard_scaler():
    def __init__(self, ts, sub_last=False, cat_std=False):
        self.sub_last = sub_last
        self.cat_std = cat_std
        self.mean = ts.mean(-1, keepdim=True)
        self.std = torch.sqrt(torch.var(ts-self.mean, dim=-1, keepdim=True, unbiased=False) + 1e-5)

    def transform(self, data):
        if self.sub_last:
            self.last_value = data[...,-1:].detach()
            data = data - self.last_value
        data = (data - self.mean) / self.std
        if self.cat_std:
            data = torch.cat((data, self.mean, self.std),-1)
        return data

    def inverted(self, data):
        if self.cat_std:
            data =  data[...,:-2] * data[...,-1:] + data[...,-2:-1]
        else:
            data = (data * self.std) + self.mean
        data = data + self.last_value if self.sub_last else data
        return data

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2402.02332
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.embed = nn.Linear(configs.seq_len, configs.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads) if configs.attn else None,
                    configs.d_model,
                    configs.d_block,
                    configs.d_ff,
                    dropout=configs.dropout,
                    gate = configs.gate,
                    configs=configs
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if configs.d_block != configs.pred_len:
            self.align = nn.Linear(configs.d_block, configs.pred_len)
        else:
            self.align = nn.Identity()

        self.T = configs.T

    def forward(self, multi_batch_x, multi_batch_x_mark, x_dec, x_mark_dec, mask=None):
        dec_out_list = []

        for t in range(self.T):
            x = multi_batch_x[:, t, :, :].squeeze(1)
            if multi_batch_x_mark is not None:
                x_mark = multi_batch_x_mark[:, t, :, :].squeeze(1)
            else:
                x_mark = None

            x = x.permute(0,2,1)
            scaler = standard_scaler(x)
            x = scaler.transform(x)
            if x_mark is not None:
                x_emb = self.embed(torch.cat((x, x_mark.permute(0,2,1)),1))
            else:
                x_emb = self.embed(x)
            output = self.encoder(x_emb, t=t)
            output = self.align(output)
            output = scaler.inverted(output[:, :x.size(1), :]).permute(0,2,1)

            dec_out_list.append(output.unsqueeze(-1))

        dec_out = torch.cat(dec_out_list, dim=-1).mean(dim=-1)

        return dec_out
 
