import inspect
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import r2_score

from common.component import Component
from common.func_utils import get_func_signature, check_args, CheckRes, CheckError, refine_args
logger = logging.getLogger(__name__)


class Metr3(Component):
    def __init__(self, **kwargs):
        self._param_map = {}
        self._checked = False

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spec = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spec.args if arg != "self"]
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @property
    def metric_name(self) -> str:
        raise NotImplementedError

    def get_metric(self, reset=True) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def __call__(self, pred_dict, label_dict):
        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")
            # 1. check consistence between signature and _param_map
            func_spec = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spec.args if arg != "self"])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_label_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in label_dict:
                mapped_label_dict[mapped_arg] = label_dict[input_arg]

        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in label_dict:
                    duplicated.append(input_arg)
            check_res = check_args(self.evaluate, [mapped_pred_dict, mapped_label_dict])
            # only check missing.
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = (
                    f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` " f"in `{self.__class__.__name__}`)"
                )

            check_res = CheckRes(
                missing=replaced_missing,
                unused=check_res.unused,
                duplicated=duplicated,
                required=check_res.required,
                all_needed=check_res.all_needed,
                varargs=check_res.varargs,
            )

            if check_res.missing or check_res.duplicated:
                raise CheckError(
                    check_res=check_res, func_signature=get_func_signature(self.evaluate),
                )
            self._checked = True
        refined_args = refine_args(self.evaluate, **mapped_pred_dict, **mapped_label_dict)

        self.evaluate(**refined_args)

    def _init_param_map(self, **kwargs):
        value_counter = defaultdict(set)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")

        # check consistence between signature and _param_map
        func_spec = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spec.args if arg != "self"]
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

class NR2_granular(Metr3):
    def __init__(self):
        super(NR2_granular, self).__init__()
        self.pred = []
        self.target = []
    @property
    def metric_name(self) -> str:
        return "NR2"
    def evaluate(self, output, input_y):
        # output: [B, N, pred_len], input_y:[B, N, pred_len]
        self.pred.extend(output)
        self.target.extend(output)










    def get_metric(self, reset=True) -> Tuple[Dict, Dict]:
        pred = np.array(self.pred)
        target = np.array(self.target)
        pred = pred.reshape((-1, pred.shape[-2], pred.shape[-1]))

        target = target.reshape((-1, pred.shape[-2], target.shape[-1]))
        pred0 = pred.tolist()
        target0 = target.tolist()
        r2 = r2_score(target, pred)
        return r2, {"pred": pred0, "target": target0}
