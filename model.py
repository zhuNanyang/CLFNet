import torch
import torch.nn as nn
from nn.model import Model

from params import represent_param, reg_param
from nn.losses import M_Loss, BCEWithLogitsLoss
from nn.loss.dice_loss import dice_loss
from nn.loss.info import InfoNCE
import random
from fastdtw import fastdtw
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)


class represent_model(Model):
    def __init__(
        self,
        feature: int = 128,
        x: int = 2,
        output_size: int = 18,
        num_layer: int = 2,
        initial_method=None,
    ):
        super(represent_model, self).__init__()
        self.lstm = nn.LSTM(x, feature,num_layer, batch_first=True)

        self.fc = nn.ModuleList(
            [
                nn.Linear(
                    feature,
                    feature * 2,
                ),
                nn.Linear(
                    feature * 2,
                    feature,
                ),
            ]
        )
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(feature, 6)

    def _forward(self, **inputs):
        x = inputs["x"]
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x, _ = self.lstm(x)
        for layer in self.fc:
            x = self.dropout(self.relu(layer(x)))
        output = x[:, -1, :]
        output = self.decoder(output)
        return output, x

    def dtw(self, t_x, x):
        t_n = []
        t_x = t_x.detach().cpu().numpy()
        for t, n in enumerate(range(x.shape[0])):
            x_n = x[n, :, :]
            x_n = x_n.detach().cpu().numpy()
            distance, _ = fastdtw(x_n, t_x)
            similarity = 1 / (1 + distance)
            if similarity > 0.6:
                t_n.append(t)
        return t_n


    def generate(self, x, x_output, dtw=None):
        x_t = []
        for n in range(x_output.shape[0]):
            x_n = x_output[n, :, :]
            x_n_new = []
            for t in range(x_n.shape[0]):
                if t == 0:
                    random_list = [t, t + 1]
                elif t == x_n.shape[0] - 1:
                    random_list = [t - 1, t]
                else:
                    random_list = [t - 1, t, t + 1]
                random_t = random.sample(random_list, 1)
                x_n_new.append(x_n[random_t, :])
            x_n_new = torch.cat(x_n_new, 0)
            x_t.append((x_n, x_n_new))

        p = []
        n = []
        l = list(range(len(x_t)))
        for t, xt in enumerate(x_t):
            t_x = x[t, :, :]
            t_n = self.dtw(t_x, x)
            if len(t_n) == 1:
                left_x_t = [x_t[l0] for l0 in l if l0 != t and l0 not in t_n]
            else:
                left_x_t = [x_t[l0] for l0 in l if l0 != t]
            n_left_x_t = []
            for l_x_t in left_x_t:
                n_left_x_t.append(l_x_t[0].unsqueeze(0))
                n_left_x_t.append(l_x_t[1].unsqueeze(0))
            n_left_x_t = torch.cat(n_left_x_t, 0)

            n.append(n_left_x_t.unsqueeze(0))
            p.append(xt[1].unsqueeze(0))
        p = torch.cat(p, 0)
        n = torch.cat(n, 0)
        return p, n

    def forward(self, **inputs):
        x = inputs["x"]
        output, x_output = self._forward(**inputs)
        p, n = self.generate(x, x_output)
        loss0 = InfoNCE(temperature=0.1, negative_mode="paired")
        l0 = 0
        for t in range(x_output.shape[1]):
            x_p = p[:, t, :]
            x_n = n[:, :, t, :]
            query = x_output[:, t, :]
            l = loss0(query, x_p, x_n)
            l0 += l
        l2 = l0
        return {"output": output, "input_y": inputs["y"], "loss": l2}

    def predict(self, **inputs):
        output, _ = self._forward(**inputs)

        x = inputs["x"]
        y = inputs["y"]
        output = output.squeeze()
        y = y.squeeze()
        output = output.unsqueeze(-1)
        y = y.unsqueeze(-1)
        return {
            "output": output,
            "input_x": x,
            "input_y": y,
        }

class power_model(Model):
    def __init__(self, feature=128, x=2, num_layer=2, losses=None):
        super(power_model, self).__init__()
        self.lstm = nn.LSTM(x, feature, num_layer, batch_first=True)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.fc_output = nn.Linear(feature, reg_param.pred_len)

    def _forward(self, **x):
        x = x["x"]
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x, _ = self.lstm(x)
        output = x[:, -1, :]
        output = self.dropout(self.relu(output))
        output = self.fc_output(output)
        return output

    def forward(self, **x):
        output = self._forward(**x)
        y = x["y"]
        l = M_Loss()
        loss = l(output, y)
        return {"output": output, "input_y": y, "loss": loss}

    def predict(self, **x):
        output = self._forward(**x)
        x_ = x["x"]
        y = x["y"]
        output = output.squeeze()
        y = y.squeeze()
        output = output.unsqueeze(-1)
        y = y.unsqueeze(-1)
        return {
            "output": output,
            "input_x": x_,
            "input_y": y,
        }