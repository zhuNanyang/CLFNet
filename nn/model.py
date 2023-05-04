import torch
import torch.nn as nn
from pathlib import Path


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def predict(self):
        return NotImplemented

    def predict_examples(self):
        return NotImplemented

    @classmethod
    def load(cls, p, d=None) -> "Model":
        if d:
            c = d
        else:
            c = "cpu"
        return torch.load(str(p), map_location=c)

    @property
    def d8(self):
        params = list(self.parameters())
        if len(params) == 0:
            raise RuntimeError("find no d")
        else:
            return params[0].device

    def extract_features(self):
        return NotImplemented
