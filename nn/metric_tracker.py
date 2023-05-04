from typing import Dict, List

import pandas as pd

from nn.ealuator import EvalResult


class MetricTracker:
    def __init__(self, metric_key: str, higher_is_better: bool = True):
        self._metric_key = metric_key
        self._higher_is_better = higher_is_better
        self._df: pd.DataFrame = pd.DataFrame()
        self._dataset_names: List[str] = []

    def to_dict(self):
        return {
            "metric_key": self._metric_key,
            "higher_is_better": self._higher_is_better,
            "df": self._df.to_dict(),
            "dataset_names": self._dataset_names,
        }

    @classmethod
    def from_dict(cls, d):
        metric_key = d["metric_key"]
        higher_is_better = d["higher_is_better"]
        tracker = cls(metric_key, higher_is_better)
        tracker._df = pd.DataFrame.from_dict(d["df"])
        tracker._dataset_names = d["dataset_names"]
        return tracker

    def add_result(self, epoch: int, step: int, result: EvalResult):
        if result.dataset_name not in self._dataset_names:
            self._dataset_names.append(result.dataset_name)

        if step not in self._df.index:
            self._df = self._df.append(pd.Series({"epoch": epoch, "step": step}, name=step))

        for metric_name, metric_results in result.results.items():
            for k, v in metric_results.result.items():
                key = f"{result.dataset_name}_{metric_name}.{k}"
                self._df.loc[self._df.step == step, key] = v

        self._df.epoch = self._df.epoch.astype(int)
        self._df.step = self._df.step.astype(int)

    def best_series(self) -> Dict[str, pd.Series]:
        result = {}
        for dataset_name in self._dataset_names:
            key = f"{dataset_name}_{self._metric_key}"
            #print("self.df[key]:{}".format(self._df[key]))
            if self._higher_is_better:
                idx = self._df[key].idxmax()
                #print("idx:{}".format(idx))
            else:
                idx = self._df[key].idxmin()
                #print("idx:{}".format(idx))
            result[dataset_name] = self._df.loc[idx]
        return result

    def best_epochs(self) -> Dict[str, int]:
        rows = self.best_series()
        #print("rows:{}".format(rows))
        return {k: v["epoch"] for k, v in rows.items()}

    def best_steps(self) -> Dict[str, int]:
        rows = self.best_series()
        return {k: v["step"] for k, v in rows.items()}

    def best_metrics(self) -> Dict[str, float]:
        rows = self.best_series()
        return {k: v[f"{k}_{self._metric_key}"] for k, v in rows.items()}


    def is_better(self, step: int, dataset: str) -> bool:
        key = f"{dataset}_{self._metric_key}"
        best_step = self.best_steps()[dataset]
        if best_step:
            if self._df.loc[step, key] >= self._df.loc[best_step, key]:
                return True
            elif abs(self._df.loc[step, key] - self._df.loc[best_step, key]) <= 0.006:
                return True
            else:
                return False
        else:
            return True
    def is_better0(self, epoch: int, dataset: str) -> bool:
        key = f"{dataset}_{self._metric_key}"
        best_step = self.best_epochs()[dataset]
        if best_step-1:
            indexs = self._df[self._df.epoch==epoch].index.to_list()
            last_indexs = self._df[self._df.epoch==best_step-1].index.to_list()
            return self._df.loc[indexs[0], key] > self._df.loc[last_indexs[-1], key]
        else:
            return True
    def __str__(self):
        meta = (
            f"key: {self._metric_key}\n"
            f"higher_is_better: {self._higher_is_better}\n"
            f"best_epochs: {self.best_epochs()}\n"
            f"best_metrics: {self.best_metrics()}\n"
        )

        return meta + self._df.T.__str__()
