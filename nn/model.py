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

    @classmethod
    def transfer_model(cls, p, model: "Model"):
        pre_model = torch.load(p + "/" + "best_model_state.pt")
        model0 = model.state_dict()
        state = {}
        for k0, k1 in pre_model.items():
            if k0 in model0.keys():
                state[k0] = k1
            else:
                continue
        model0.update(state)
        model.load_state_dict(model0)
        return model
