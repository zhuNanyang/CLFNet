import json
import logging
from pathlib import Path

from typing import Dict, Sequence, List
import torch
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np
from nn.metrics import Metric
from nn.utils import move_data_to_device
from common import func_utils
from nn.model import Model

from nn.Metr3 import Metr3
logger = logging.getLogger(__name__)


class MetricResult(BaseModel):
    metric_name: str
    result: Dict[str, float]
    pred: List
    target: List
class EvalResult(BaseModel):
    dataset_name: str
    results: Dict[str, MetricResult]


class Evaluator:
    def __init__(
        self,
        dataset_name,
        data_loader,
        proba=None,
        quantile=None,
        model: Model = None,
        metrics: List[Metric] = None,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        use_amp: bool = False,
    ):

        self._model: Model = model.to(device)
        self.dataset_name = dataset_name
        self._metrics = metrics
        self.p = proba

        self.quantile = quantile
        self.dataloader = data_loader
        self.device = device
        self.use_amp = use_amp

    def evaluate(self):
        # total_loss = []
        pred = []
        target = []
        results = {}
        try:
            self._model.eval()
            with torch.no_grad():
                for batch in self.dataloader:
                    batch = move_data_to_device(batch, self.device)
                    pred_dict = self.predict(batch)
                    pred.extend(pred_dict["output"].detach().cpu().numpy())
                    target.extend(pred_dict["input_y"].detach().cpu().numpy())
                    # total_loss.append(pred_dict["loss"].item())
                if self._metrics:
                    pred = np.array(pred)
                    target = np.array(target)
                    pred0 = pred.tolist()
                    target0 = target.tolist()
                    for metric in self._metrics:
                        metric_name = metric.metric_name
                        results[metric_name] = MetricResult(
                            metric_name=metric_name,
                            result=metric.get_result(pred, target),  # source_and_load: pred[:, -1, -1], target[:, -1, -1]
                            pred=pred0,
                            target=target0
                        )
                else:
                    results = None
        finally:
            self._model.train()
        return EvalResult(dataset_name=self.dataset_name, results=results)

    def predict(self, batch):
        batch = func_utils.refine_args(self._model.forward, **batch)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                y = self._model.predict(**batch)
        else:
            y = self._model.predict(**batch)
        return y

    @classmethod
    def dump_result(cls, dump_dir: Path, eval_result: EvalResult, epoch: int):
        dump_dir.mkdir(exist_ok=True, parents=True)
        datasetname = eval_result.dataset_name

        for metric_name, result in eval_result.results.items():
            dump_path = dump_dir / f"{datasetname}_{metric_name}_{epoch}.json"
            logger.info(f"Dumping result to {dump_path}")


            with open(str(dump_path), "w", encoding="utf-8") as f:
                result = {"result": result.result, "pred": result.pred, "target": result.target}
                json.dump(result, f, ensure_ascii=False, indent=2)


class E_c:
    def __init__(
        self,
        dataset_name: str,
        model: Model,
        metrics: Sequence[Metr3],
        dataloader,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        use_amp: bool=False
    ):
        self._dataset_name = dataset_name
        self._model: Model = model.to(device)
        self._metrics = metrics
        self._dataloader = dataloader
        self.use_amp = use_amp
    def evaluate(self) -> EvalResult:
        model_device = self._model.device
        results = {}

        try:
            self._model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    self._dataloader, desc=f"Eval {self._dataset_name}", leave=True, dynamic_ncols=True
                ):

                    batch = move_data_to_device(batch, model_device)
                    pred_dict = self.predict(batch)

                    for metric in self._metrics:
                        metric(pred_dict, {**batch})

                for metric in self._metrics:
                    metric_result, predictions = metric.get_metric()
                    metric_name = metric.metric_name
                    results[metric_name] = MetricResult(
                        metric_name=metric_name,
                        result=metric_result,
                        predictions=predictions or [],
                    )
        finally:
            self._model.train()

        return EvalResult(dataset_name=self._dataset_name, results=results)

    def predict(self, batch_x):
        batch_x = func_utils.refine_args(self._model.predict, **batch_x)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                y = self._model.predict(**batch_x)
        else:
            y = self._model.predict(**batch_x)
        #y = self._model.predict(**batch_x)
        return y

    @classmethod
    def dump_result(cls, dump_dir: Path, eval_result: EvalResult, epoch: int):
        dump_dir.mkdir(exist_ok=True, parents=True)
        ds_name = eval_result.dataset_name

        for metric_name, result in eval_result.results.items():
            dump_path = dump_dir / f"{ds_name}_{metric_name}_{epoch}.json"
            logger.info(f"Dumping result to {dump_path}")
            with open(str(dump_path), "w", encoding="utf-8") as f:
                result_f = {"result": result.result, "pred": result.pred, "target": result.target}
                json.dump(result_f, f, ensure_ascii=False, indent=2)