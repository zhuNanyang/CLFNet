import torch
import torch.nn as nn
from typing import List

import math


class TemporalEmbedding(nn.Module):
    def __init__(self, input_size: List[int], feature_size, embedding_t8: str = None):
        super(TemporalEmbedding, self).__init__()
        self.embedding = nn.Embedding if embedding_t8 == "embedding" else None
        self.year_emb = self.embedding(input_size[0], feature_size)
        self.month_emb = self.embedding(input_size[1], feature_size)
        self.day_emb = self.embedding(input_size[2], feature_size)
        self.hour_emb = self.embedding(input_size[3], feature_size)

        self.minute_emb = self.embedding(input_size[-1], feature_size)

    def forward(self, **inputs):
        year_output = self.year_emb(inputs["year"])
        month_output = self.month_emb(inputs["month"])
        day_output = self.day_emb(inputs["day"])
        hour_output = self.hour_emb(inputs["hour"])
        minute_output = self.minute_emb(inputs["minute"])
        return year_output + month_output + day_output + hour_output + minute_output


class TemporalPositional(nn.Module):
    def __init__(self, input_size, feature_size):
        super(TemporalPositional, self).__init__()
        weight = torch.zeros(input_size, feature_size).float()
        weight.requires_grad = False
        position = torch.arange(0, input_size).float().unsqueeze(1)
        div_term = (
            torch.arange(0, feature_size, 2).float()
            * -(math.log(10000.0) / feature_size)
        ).exp()
        weight[:, 0:2] = torch.sin(position * div_term)
        weight[:, 1:2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(input_size, feature_size)
        self.emb.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        x = self.emb(x)
        return x
