import torch
import torch.nn as nn
from bbbb8888.nn.utils import initial_parameter

from typing import List
from math import sqrt

import numpy as np
class SelfAttention(nn.Module):
    def __init__(
        self,
        input_size,
        attention_unit=300,
        attention_hops=10,
        drop=0.5,
        initial_method: List = None,
    ):
        """
        :param int input_size: 输入tensor的hidden维度
        :param int attention_unit: 输出tensor的hidden维度
        :param int attention_hops:
        :param float drop: dropout概率，默认值为0.5
        :param str initial_method: 初始化参数方法
        """
        super(SelfAttention, self).__init__()

        self.attention_hops = attention_hops
        self.ws1 = nn.Linear(input_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.I = torch.eye(attention_hops, requires_grad=False)
        self.I_origin = self.I

        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        initial_parameter(self, initial_method)

    def _penalization(self, attention):
        r"""
        compute the penalization term for lstm_attention module
        """
        baz = attention.size(0)
        size = self.I.size()
        if len(size) != 3 or size[0] != baz:
            self.I = self.I_origin.expand(baz, -1, -1)
            self.I = self.I.to(device=attention.device)
        attention_t = torch.transpose(attention, 1, 2).contiguous()
        mat = torch.bmm(attention, attention_t) - self.I[: attention.size(0)]
        ret = (torch.sum(torch.sum((mat**2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]

    def forward(self, input, input_origin):
        r"""
        :param torch.Tensor input: [batch_size, seq_len, hidden_size] 要做attention的矩阵
        :param torch.Tensor input_origin: [batch_size, seq_len] 原始token的index组成的矩阵，含有pad部分内容
        :return torch.Tensor output1: [batch_size, multi-head, hidden_size] 经过attention操作后输入矩阵的结果
        :return torch.Tensor output2: [1] attention惩罚项，是一个标量
        """
        input = input.contiguous()
        size = input.size()  # [bsz, len, nhid]
        input_origin = input_origin.expand(
            self.attention_hops, -1, -1
        )  # [hops,baz, len]
        input_origin = input_origin.transpose(0, 1).contiguous()  # [baz, hops,len]

        y1 = self.tanh(
            self.ws1(self.drop(input))
        )  # [baz,len,dim] -->[bsz,len, lstm_attention-unit]
        attention = self.ws2(y1).transpose(1, 2).contiguous()
        # [bsz,len, lstm_attention-unit]--> [bsz, len, hop]--> [baz,hop,len]
        attention = attention + (
            -999999 * (input_origin == 0).float()
        )  # remove the weight on padding token.
        attention = torch.nn.functional.softmax(attention, 2)  # [baz ,hop, len]
        return torch.bmm(attention, input), self._penalization(
            attention
        )  # output1 --> [baz ,hop ,nhid]


class Attention(nn.Module):
    def __init__(self, output_attention: bool = False):
        super(Attention, self).__init__()
        self.output_attention = output_attention

    def forward(self, queries, keys, values, mask: bool = True):
        B, L, H, E = queries.shape
        _, L_v, _, E_v = values.shape
        keys = keys.squeeze().permute(0, 2, 3, 1)
        queries = queries.permute(0, 2, 1, 3)
        scores = torch.matmul(queries, keys)
        if mask:
            with torch.no_grad():
                mask = torch.triu(
                    torch.ones([B, 1, L, L_v], dtype=bool), diagonal=1
                ).to(queries.device)
            scores = scores.masked_fill_(mask, -np.inf)
        scale = 1 / sqrt(E)
        attention_scale = self.dropout(
            torch.softmax(scores * scale, dim=-1)
        )  # [B, H, L, L_v]
        values = values.squeeze().permute(0, 2, 1, 3)  # [B, H, L_v, E_v]
        attention = torch.matmul(attention_scale, values)  # [B, H, L, E_v]

        attention = attention.permute(0, 2, 1, 3)  # [B, L, H, E_v]
        attention = attention.contiguous()  # [B, L, H, E_v]
        if self.output_attention:
            return attention, attention_scale
        else:
            return attention, None


class MultiHead_Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head=None,
        head_dim=None,
        d_query: int = None,
        d_key: int = None,
        d_value: int = None,
    ):
        super(MultiHead_Attention, self).__init__()
        self.scaled_dot = Attention(output_attention=True)

        d_key = d_key if d_key else d_model // n_head
        d_value = d_value if d_value else d_model // n_head
        self.q_linear = nn.Linear(d_model, n_head * d_key)
        self.k_linear = nn.Linear(d_model, n_head * d_key)
        self.v_linear = nn.Linear(d_model, n_head * d_value)

    def forward(self, queries, keys, values):
        B, L, d = queries.shape
        _, L_k, d_k = keys.shape
        queries = self.q_linear(queries).view(B, L, self.n_head, -1)

        keys = self.k_linear(keys).view(B, L_k, self.n_head, -1)
        values = self.v_linear(values).view(B, L_k, self.n_head, -1)
        output_attention, attention = self.scaled_dot(
            queries, keys, values, mask=False
        )  # [B, L, n_head, d_value], [B, H, L, L_v ]
        output_attention = output_attention.view(B, L, -1)
        return output_attention, attention


class selfattention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob: float = 0.1):
        super(selfattention, self).__init__()

        if hidden_size % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of lstm_attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
            # 参数定义
        self.num_attention_heads = num_attention_heads  # 8
        self.attention_head_size = int(
            hidden_size / num_attention_heads
        )  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)  # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # [bs, seqlen, 8, 16]

        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values):
        # 线性变换
        mixed_query_layer = self.query(queries)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(keys)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(values)  # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer
        )  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(
            mixed_value_layer
        )  # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw lstm_attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / sqrt(
            self.attention_head_size
        )  # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        # 加上mask，将padding所在的表示直接-10000

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(
            attention_scores
        )  # [bs, 8, seqlen, seqlen]
        attention_prob_numpy = attention_probs.cpu().detach().numpy()
        #np.save("lstm_attention.npy", attention_prob_numpy)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(
            attention_probs, value_layer
        )  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )  # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
