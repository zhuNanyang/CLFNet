import torch
from params import reg_param
from common import func_utils

from nn.utils import move_data_to_device
from nn.model import Model
from pathlib import Path
from typing import List
import numpy as np

class Predictor:
    def __init__(
        self,
        model: Model = None,
        dataloader=None,
        pre: bool = True,
        d8=None,
        use_amp: bool = False,
        model_p: Path = None,
    ):
        if model_p is None:
            model_p = Path(reg_param.model)
        self.model = model

        self.model.load_state_dict(torch.load(model_p + "/" + "model_state_ckpt.pt"))
        #self.model = torch.load(path=model_p + "/" + "best.pkl")

        # self.model.load_state_dict(model.load(model_path))
        self.dataloader = dataloader
        self.pre = pre
        self.d8 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp

    def predict(
        self, index: int=0
    ):
        self.model.eval()
        prediction = []
        target = []

        for batch in self.dataloader:
            batch = move_data_to_device(batch, self.model.d8)
            output = self.forward(batch)
            prediction.extend(output["output"].detach().cpu().numpy())
            target.extend(output["input_y"].detach().cpu().numpy())
        prediction = np.array(prediction)
        target = np.array(target)
        predcition = prediction.reshape(-1, prediction.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        prediction = prediction.tolist()
        target = target.tolist()
        return prediction, target

    def forward(self, batch):
        refined_arg = func_utils.refine_args(self.model.forward, **batch)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                output = self.model.predict(**refined_arg)
        else:
            output = self.model.predict(**refined_arg)
        return output
