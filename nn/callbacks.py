import numpy as np
import torch

from torch.optim.lr_scheduler import *
from nn.utils import save_checkpoint
from params import represent_param

import inspect
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
import torch

# from torch.utils.tensorboard import SummaryWriter
from nn.utils import save_model
from nn.ealuator import Evaluator, E_c
from nn.metric_tracker import MetricTracker


class LeaningRate:
    def __init__(self, step_size, gamma):
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, optimizer):
        return StepLR(optimizer=optimizer, step_size=self.step_size, gamma=self.gamma)

    def adjust_lr(self, optimizer, epoch):
        lr_adjust = {epoch: Param.lr * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


class Event(str, Enum):
    TRAINING_START = "TRAINING_START"

    EPOCH_START = "EPOCH_START"

    BATCH_START = "BATCH_START"

    VALIDATE_START = "VALIDATE_START"

    FORWARD = "FORWARD"

    BACKWARD = "BACKWARD"

    BATCH_END = "BATCH_END"

    VALIDATE_END = "VALIDATE_END"

    EPOCH_END = "EPOCH_END"

    TRAINING_END = "TRAINING_END"

    ERROR = "ERROR"


def handle_event(event: str):
    def wrapper(method: Callable):
        setattr(method, "_event", event)
        return method

    return wrapper


class Callback:
    def __init__(self):
        super(Callback, self).__init__()
        self._callbacks = None

    def fire_event(self, event: Event):
        self._callbacks.fire_event(event)

    @property
    def trainer(self):
        return self._callbacks.trainer

    @property
    def step(self):
        return self.trainer.step

    @property
    def n_steps(self):
        return self.trainer.n_steps

    @property
    def batch_size(self):
        return self.trainer.batch_size

    @property
    def epoch(self):
        return self.trainer.epoch

    @property
    def n_epochs(self):
        return self.trainer.n_epochs

    @property
    def optimizer(self):
        return self.trainer.optimizer

    @property
    def model(self):
        return self.trainer.model

    @property
    def valiation(self):
        return self.trainer.valiation_dataloader

    @property
    def pbar(self):
        return self.trainer.pbar

    @property
    def metric_tracker(self) -> MetricTracker:
        return self.trainer.metric_tracker

    @property
    def update_every(self):
        return self.trainer.update_every

    @property
    def batch_per_epoch(self):
        return self.trainer.batch_per_epoch


@dataclass
class EventHandler:
    name: str
    callback: Callback
    handler: Callable


class CallbackList:
    def __init__(self, trainer, callbacks: Sequence[Callback] = None):
        callbacks = callbacks or []
        self._trainer = trainer
        self._callbacks: Dict[Event, List[EventHandler]] = OrderedDict()

        for callback in callbacks:
            self.add_callback(callback)

    @property
    def trainer(self):
        return self._trainer

    def callbacks(self) -> List[Callback]:
        return list(
            {
                callback.callback
                for callback_list in self._callbacks.values()
                for callback in callback_list
            }
        )

    @classmethod
    def _is_event_handler(cls, member) -> bool:
        return inspect.ismethod(member) and hasattr(member, "_event")

    def add_callback(self, callback: Callback) -> None:
        for name, method in inspect.getmembers(callback, self._is_event_handler):
            event = getattr(method, "_event")
            callback._callbacks = self
            self._callbacks.setdefault(event, []).append(
                EventHandler(name=name, callback=callback, handler=method)
            )

    def fire_event(self, event: Event) -> None:
        for event_handler in self._callbacks.get(event, []):
            event_handler.handler()


class GradientClipCallback(Callback):
    def __init__(
        self, grad_norm: Optional[float] = None, grad_clipping: Optional[float] = None
    ) -> None:
        super().__init__()
        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping

    @handle_event(Event.BACKWARD)
    def rescale(self):
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

    @handle_event(Event.BACKWARD)
    def clip(self):
        if self.grad_clipping is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clipping)


class EvaluateCallback(Callback):
    def __init__(
        self,
        dataset_names: Sequence[str] = ("dev", "test"),
        dump_dir: Optional[Path] = None,
    ):
        super(EvaluateCallback, self).__init__()

        self._dataset_names = dataset_names
        self._evaluators: Dict[str, Evaluator] = {}
        self._dump_dir = dump_dir

    @handle_event(Event.TRAINING_START)
    def setup(self):
        for dataset_name in self._dataset_names:
            self._evaluators[dataset_name] = Evaluator(
                dataset_name=dataset_name,
                data_loader=self.valiation[dataset_name],
                model=self.model,
                metrics=self.trainer.metrics,
                device=self.trainer.device,)

        #
        # for dataset_name in self._dataset_names:
        #     self._evaluators[dataset_name] = E_c(
        #         dataset_name=dataset_name,
        #         data_loader=self.valiation[dataset_name],
        #         model=self.model,
        #         metrics=self.trainer.metrics,
        #         device=self.trainer.device,)

    @handle_event(Event.VALIDATE_START)
    def validate(self):
        for name, evaluator in self._evaluators.items():
            eval_result = evaluator.evaluate()
            #print("eval_result:{}".format(eval_result))
            self.metric_tracker.add_result(self.epoch, self.step, eval_result)
            if self._dump_dir:
                evaluator.dump_result(self._dump_dir, eval_result, self.epoch)
            self.pbar.write(f"{self.metric_tracker}")
            self.fire_event(Event.VALIDATE_END)


class EarlyStopCallback(Callback):
    def __init__(self, patience: int = 1, dataset="dev"):
        super(EarlyStopCallback, self).__init__()
        self.patience = patience

        self.dataset = dataset
        self.wait = 0

    @handle_event(Event.VALIDATE_END)
    def check(self):
        if not self.metric_tracker.is_better(self.step, self.dataset):
            if self.wait == self.patience:
                raise KeyboardInterrupt("Early stopping")
            else:
                self.wait += 1
        else:
            self.wait = 0


class LRScheduler(Callback):
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
        super(LRScheduler, self).__init__()

        self.scheduler = lr_scheduler

    @handle_event(Event.EPOCH_END)
    def update_lr(self):
        self.scheduler.step(self.epoch)


class CheckpointCallback(Callback):
    def __init__(self):
        super(CheckpointCallback, self).__init__()
        self.model_state_path = None
        self.best_model_state_path = None
        self.train_state_path = None

    @handle_event(Event.TRAINING_START)
    def restore_checkpoint(self):
        self.model_state_path = os.path.join(
            self.trainer.model_dir, "model_state_ckpt.pt"
        )

        self.best_model_state_path = os.path.join(
            self.trainer.model_dir, f"best_model_state.pt"
        )
        self.train_state_path = os.path.join(
            self.trainer.model_dir, "train_state_ckpt.pt"
        )
        if os.path.exists(self.model_state_path) and os.path.exists(
            self.train_state_path
        ):
            self.pbar.write(f"上一个检查点存在, 读取并恢复训练")
            model_state = torch.load(self.model_state_path)
            train_state = torch.load(self.train_state_path)
            model = self.model
            model.load_state_dict(model_state)
            self.optimizer.load_state_dict(train_state["optimizer"])
            self.trainer.epoch = train_state["epoch"] + 1
            self.trainer.step = train_state["step"]
            self.trainer.metric_tracker = MetricTracker.from_dict(
                train_state["tracker"]
            )

    @handle_event(Event.VALIDATE_END)
    def on_valid_end(self):
        #print(self.trainer.model_dir)
        os.makedirs(self.trainer.model_dir, exist_ok=True)
        if self.metric_tracker.is_better(self.step, "dev") or self.metric_tracker.is_better(self.step, "test"):
            model = self.model
            model_state = model.state_dict()
            self.pbar.write(f"保存当前最佳权重{self.epoch}_{self.step} best_model_state.pt")
            self.best_model_state_path = os.path.join(
            self.trainer.model_dir, f"best_model_state.pt")
            torch.save(model_state, self.best_model_state_path)

    @handle_event(Event.EPOCH_END)
    def save_checkpoint(self):
        model = self.model
        model_state = model.state_dict()
        os.makedirs(self.trainer.model_dir, exist_ok=True)
        train_state = {
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "tracker": self.metric_tracker.to_dict(),
        }
        torch.save(model_state, self.model_state_path)
        torch.save(train_state, self.train_state_path)
        self.metric_tracker._df.to_json(
            os.path.join(self.trainer.model_dir, "metrics.json"),
            force_ascii=False,
            indent=2,
        )
        self.pbar.write(f"模型检查点@epoch:{self.epoch},step:{self.step} 已保存")


class LrWarmupCallback(Callback):
    def __init__(self, warmup=0.1, schedule="linear"):
        super().__init__()
        self.warmup = max(warmup, 0.0)
        self.initial_lrs = []  # 存放param_group的learning rate
        self.t_steps = 0
        if schedule == "constant":
            self.get_lr = self._get_constant_lr
        elif schedule == "linear":
            self.get_lr = self._get_linear_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.0) / (self.warmup - 1.0), 0.0)

    @handle_event(Event.TRAINING_START)
    def on_train_begin(self):
        self.t_steps = (
            len(self.trainer.data) // (self.batch_size * self.update_every)
            + int(len(self.trainer.data) % (self.batch_size * self.update_every) != 0)
        ) * self.n_epochs
        if self.warmup > 1:
            self.warmup = self.warmup / self.t_steps
        self.t_steps = max(2, self.t_steps)  # 不能小于2
        # 获取param_group的初始learning rate
        for group in self.optimizer.param_groups:
            self.initial_lrs.append(group["lr"])

    @handle_event(Event.BACKWARD)
    def on_backward_end(self):
        if self.step % self.update_every == 0:
            progress = (self.step / self.update_every) / self.t_steps
            for lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
                group["lr"] = lr * self.get_lr(progress)


class SaveCallback(Callback):
    def __init__(self):
        super().__init__()

    @handle_event(Event.EPOCH_END)
    def save_best_model(self):
        if self.metric_tracker.is_better0(self.epoch, "dev") or self.metric_tracker.is_better0(self.epoch, "test"):
            self.pbar.write(f"保存当前最佳模型 {self.epoch}_best.pkl")
            os.makedirs(self.trainer.model_dir, exist_ok=True)
            save_model(self.model, self.trainer.model_dir, model_name=f"best")


class TensorBoardCallback(Callback):
    def __init__(self, dump_dir: Optional[Path] = None):
        super(TensorBoardCallback, self).__init__()

        self._dump_dir = dump_dir
        self.writer = SummaryWriter(log_dir=str(self._dump_dir))

    @handle_event(Event.BATCH_END)
    def add_avg_loss_per_step(self):
        self.writer.add_scalar("Loss", self.trainer.avg_loss, self.trainer.step)
