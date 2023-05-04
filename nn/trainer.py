from typing import List, Union, Optional, Sequence
from nn.callbacks import LeaningRate


from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from nn.utils import auto_choose_gpu, move_data_to_device
from nn.optimizers import Optimizer, Adam
import numpy

from nn.metrics import Metric
import time
from common import func_utils

from nn.metric_tracker import MetricTracker
from nn.callbacks import CallbackList, Callback, Event
import random


class Trainer:
    def __init__(
        self,
        data: Optional,
        dataloader: DataLoader,
        model,
        metrics: List[Metric] = None,
        metric_tracker: MetricTracker = None,
        valiation_data: Optional = None,
        valiation_dataloader: dict = None,
        optimizer: Optimizer = Adam(),
        callbacks: Sequence[Callback] = None,
        n_epochs: int = 5,
        batch_size: int = 32,
        dev_batch_size: int = 32,
        update_every: int = 1,
        print_every: int = 1,
        validate_every: int = -1,
        device: str = "auto",
        dist_backend=None,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        world_size=1,
        dist_rank=0,
        local_rank: Union[str, int] = "auto",
        adversial_fgm: bool = False,
        adversial_pgd: bool = False,
        proba: bool = False,
        quantile: List = None,
        use_amp: bool = False,
        model_p=None,
        seed=2333
    ):
        if seed is not None:
            self.set_seed(seed)
        # parse device and backend
        if device == "auto":
            device = "cuda" if  torch.cuda.is_available() else "cpu"
        if dist_backend is None:
            dist_backend = "nccl" if device == "cuda" else "gloo"

        # init distributed
        if local_rank == "auto":
            if  torch.cuda.is_available():
                self.local_ranks, self.local_rank = auto_choose_gpu()
            else:
                self.local_rank = 0
        else:
            self.local_rank = local_rank
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device(device)
        self.data = data
        self.world_size = world_size
        self.dist_rank = dist_rank
        self.adversial_fgm = adversial_fgm
        self.adversial_pgd = adversial_pgd
        model = model.to(self.device)
        # init DataParallel

        if self.distributed:
            dist.init_process_group(
                backend=dist_backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=self.world_size,
                rank=self.dist_rank,
            )
            print(self.local_ranks)
            self.model = DistributedDataParallel(
                model,
                device_ids=self.local_ranks,
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        else:
            self.model = model
        if self.distributed:
            sampler = DistributedSampler(self.data)
            self.data_iterator = DataLoader(
                dataset=self.data, batch_size=batch_size, num_workers=10, sampler=sampler,drop_last=True
            )
        else:
            self.data_iterator = dataloader
        self.model.train()
        self.optimizer = optimizer.to_torch(self.model.parameters())
        self.metrics = metrics
        self.batch_size = batch_size
        self.dev_batch_size = dev_batch_size
        self.n_epochs = n_epochs
        self.mean_loss = 0.0
        self.validate_every = int(validate_every) if validate_every != 0 else -1
        self.update_every = update_every
        self.print_every = print_every
        self._forward_func = self.model.forward
        self.train_step = len(self.data_iterator)

        self.step = 0
        self.epoch = 1
        self.valiation_data = valiation_data
        self.valiation_dataloader = valiation_dataloader
        self.metric_tracker = metric_tracker
        # self.earlystop = EarlystopCallback(patience=30, verbose=True)
        self.callbacks = CallbackList(self, callbacks)
        self.learningrate = LeaningRate(step_size=2, gamma=0.1)
        self.n_steps = len(self.data_iterator) * self.n_epochs
        self.proba = proba
        self.quantile = quantile
        self._forward_func = self.model.forward
        self.use_amp = use_amp
        self.scalar = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.pbar = None
        self.model_dir = model_p if model_p else None
        self.barrier()

    def barrier(self):
        if self.distributed:
            dist.barrier()

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @classmethod
    def set_seed(cls, seed: int):
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self):
        self.model.train()
        self._train()

    def _train(
        self,
    ):
        with tqdm(total=self.n_steps, leave=True, dynamic_ncols=True) as pbar:
            self.pbar = pbar
            self.callbacks.fire_event(Event.TRAINING_START)
            pbar.update(self.step)
            mean_loss = 0.0
            for epoch in range(self.epoch, self.n_epochs + 1):
                pbar.set_description_str(
                    desc="Epoch {}/{}".format(epoch, self.n_epochs)
                )
                if self.device.type == "cuda":
                    memory_summary = torch.cuda.memory_summary(
                        self.device, abbreviated=True
                    )
                    pbar.write(memory_summary)

                self.epoch = epoch
                self.callbacks.fire_event(Event.EPOCH_START)

                for batch in self.data_iterator:
                    self.step += 1
                    self.callbacks.fire_event(Event.BATCH_START)

                    batch = move_data_to_device(batch, self.device)
                    output = self.forward(batch)

                    loss = self.get_loss(output).mean()
                    loss = loss / self.update_every
                    mean_loss += loss.item()

                    self.bachward(loss)
                    self.callbacks.fire_event(Event.BACKWARD)
                    self.update()

                    if self.step % self.print_every == 0:
                        mean_loss = float(mean_loss) / self.print_every
                        self.mean_loss = mean_loss
                        pbar.update(self.print_every)
                        pbar.set_postfix_str("loss:{:<6.5f}".format(mean_loss))
                        mean_loss = 0.0

                    self.callbacks.fire_event(Event.BATCH_END)

                    if (
                        self.validate_every > 0 and self.step % self.validate_every == 0
                    ) or (
                        self.validate_every < 0
                        and self.step % len(self.data_iterator) == 0
                    ):

                        self.callbacks.fire_event(Event.VALIDATE_START)

                self.callbacks.fire_event(Event.EPOCH_END)

            self.callbacks.fire_event(Event.TRAINING_END)
            pbar.close()
            self.pbar = None

    def get_loss(self, output):
        loss = output["loss"]
        if len(loss.size()) != 0:
            loss = torch.sum(loss) / (loss.view(-1)).size(0)
        return loss

    def bachward(self, loss):
        if (self.step - 1) % self.update_every == 0:
            self.model.zero_grad()
        if self.use_amp:
            self.scalar.scale(loss).backward()

        else:
            loss.backward()

    def update(self):
        if self.step % self.update_every == 0:
            if self.use_amp:
                self.scalar.step(self.optimizer)
                self.scalar.update()
            else:
                self.optimizer.step()

    def forward(self, batch):
        refined_args = func_utils.refine_args(self._forward_func, **batch)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                output = self.model(**refined_args)
        else:
            output = self.model(**refined_args)
        return output


# class ProbabilityTrainer:
#     def __init__(
#         self,
#         data: Dataset,
#         dataloader: DataLoader,
#         model,
#         metrics: List[Metric] = None,
#         valiation_data: Dataset = None,
#         valiation_dataloader: DataLoader = None,
#         optimizer: Optimizer = Adam(),
#         n_epochs: int = 5,
#         batch_size: int = 32,
#         dev_batch_size: int = 32,
#         update_every: int = 1,
#         print_every: int = 1,
#         validate_every: int = -1,
#         num_workers: int = 1,
#         device: str = "auto",
#         opt_level=None,
#         dist_backend=None,
#         master_addr: str = "127.0.0.1",
#         master_port: int = 29500,
#         world_size=1,
#         dist_rank=0,
#         local_rank: Union[str, int] = "auto",
#         seed=2333,
#         adversial_fgm: bool = False,
#         adversial_pgd: bool = False,
#     ):
#         # parse device and backend
#         if device == "auto":
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         if dist_backend is None:
#             dist_backend = "nccl" if device == "cuda" else "gloo"
#
#         # init distributed
#         if local_rank == "auto":
#             if torch.cuda.is_available():
#                 self.local_rank = auto_choose_gpu()
#             else:
#                 self.local_rank = 0
#         else:
#             self.local_rank = local_rank
#         if device == "cuda":
#             self.device = torch.device("cuda", self.local_rank)
#         else:
#             self.device = torch.device(device)
#         self.data = data
#         self.world_size = world_size
#         self.dist_rank = dist_rank
#         self.adversial_fgm = adversial_fgm
#         self.adversial_pgd = adversial_pgd
#         model = model.to(self.device)
#         # init DataParallel
#
#         if self.distributed:
#             dist.init_process_group(
#                 backend=dist_backend,
#                 init_method=f"tcp://{master_addr}:{master_port}",
#                 world_size=self.world_size,
#                 rank=self.dist_rank,
#             )
#             self.model = DistributedDataParallel(
#                 model,
#                 device_ids=[self.local_rank],
#                 output_device=self.local_rank,
#                 find_unused_parameters=True,
#             )
#         else:
#             self.model = model
#         self.optimizer = optimizer.to_torch(self.model.parameters())
#         self.metrics = metrics
#         self.batch_size = batch_size
#         self.dev_batch_size = dev_batch_size
#         self.data_iterator = dataloader
#         self.n_epochs = n_epochs
#         self.mean_loss = 0.0
#         self.validate_every = int(validate_every) if validate_every != 0 else -1
#         self.update_every = update_every
#         self.print_every = print_every
#         self._forward_func = self.model.forward
#         self.train_step = len(self.data_iterator)
#         self.step = 0
#         self.valiation_data = valiation_data
#         self.valiation_dataloader = valiation_dataloader
#         self.earlystop = EarlystopCallback(patience=3, verbose=True)
#
#         self.learningrate = LeaningRate(step_size=2, gamma=0.1)
#
#     @property
#     def distributed(self) -> bool:
#         return self.world_size > 1
#
#     def valiation(self):
#
#         self.model.eval()
#         total_loss = []
#
#         batch_normal = []
#         batch_target = []
#         batch_pred = []
#
#         batch_sigma = []
#         batch_pred50 = []
#         batch_pred90 = []
#         batch_pred10 = []
#
#         for i, (batch_x, batch_y) in enumerate(self.valiation_dataloader):
#             batch_x = move_data_to_device(batch_x, self.device)
#             (
#                 input_mu,
#                 input_sigma,
#                 output,
#                 hidden_output,
#                 cell_output,
#                 batch_y,
#             ) = self.forward(
#                 batch_x, batch_y
#             )  # seq_len - label_len 用于训练的
#             # input_mu: [B, pred_len]  input_sigma: [B, pred_len] output: [B, pred_len] batch_y: [B, pred_len]
#
#             loss = self.get_loss(input_mu, input_sigma, output, batch_y).mean()
#             batch_size = batch_x.shape[0]
#             pred_len = Param.pred_len
#             samples = torch.zeros(
#                 Param.sample_times, batch_size, pred_len, device=self.device
#             )
#
#             samples = []
#             for j in range(Param.sample_times):
#
#                 mu, sigma, output, hidden_output, cell_output, batch_y = self.forward(
#                     batch_x, batch_y
#                 )  # [B, seq_len] 预测的部分
#
#                 if mu.ndim != 2 and sigma.ndim != 2:
#                     mu = mu.squeeze()
#                     sigma = sigma.squeeze()
#
#                 gaussian = torch.distributions.normal.Normal(mu, sigma)  # [B, seq_len]
#                 pred = gaussian.sample()  # [B, seq_len]
#
#                 # samples[j] = pred
#                 samples.append(pred.detach().cpu().numpy())
#
#             samples = np.concatenate(samples, axis=1)  # [B, seq_len, sample_time]
#             print(samples.shape)
#             p50 = np.quantile(samples, 0.56, axis=2)  # [B, seq_len]
#             p90 = np.quantile(samples, 0.9, axis=2)
#             p10 = np.quantile(samples, 0.10, axis=2)
#             total_loss.append(loss)
#             batch_target.append(batch_y.detach().cpu().numpy())
#             batch_pred10.append(p10)
#             batch_pred90.append(p90)
#             batch_pred50.append(p50)
#
#         batch_target = np.array(batch_target)
#         batch_pred10 = np.array(batch_pred10)
#         batch_pred90 = np.array(batch_pred90)
#         batch_pred50 = np.array(batch_pred50)
#
#         batch_target = batch_target.reshape(
#             -1, batch_target.shape[-1]
#         )  # [n*B, seq_len]
#         batch_pred10 = batch_pred10.reshape(-1, batch_pred10.shape[-1])
#         batch_pred90 = batch_pred90.reshape(-1, batch_pred90.shape[-1])
#         batch_pred50 = batch_pred50.reshape(-1, batch_pred50.shape[-1])
#         """
#             sample_normal = torch.median(samples, dim=0)[0]  # [B, seq_len]
#             sample_sigma = samples.std(dim=0)  # [B, seq_len]
#             total_loss.append(loss.item())
#             batch_target.append(batch_y.detach().cpu().numpy())
#
#             batch_normal.append(sample_normal.detach().cpu().numpy())
#             batch_pred.append(output.detach().cpu().numpy())
#             batch_sigma.append(sample_sigma.detach().cpu().numpy())
#
#         batch_target = np.array(batch_target)
#         batch_normal = np.array(batch_normal)
#         batch_target = batch_target.reshape(-1, batch_target.shape[-1])
#         batch_normal = batch_normal.reshape(-1, batch_normal.shape[-1])
#
#
#         """
#         proba_result = {}
#         for index, pred in zip(
#             [10, 50, 90], [batch_pred10, batch_pred50, batch_pred90]
#         ):
#
#             proba_pred = {}
#             for metric in self.metrics:
#                 result = metric(pred, batch_target)
#                 proba_pred[metric.metric_name] = result
#             proba_result[f"proab_{index}"] = proba_pred
#
#         self.model.train()
#
#         return np.average(total_loss), proba_result
#
#     def train(self):
#         for epoch in range(self.n_epochs):
#             self.model.train()
#             train_loss = []
#             epoch_time = time.time()
#             for i, (batch_x, batch_y) in enumerate(self.data_iterator):
#                 self.step += 1
#                 batch_x = move_data_to_device(batch_x, self.device)
#                 mu, sigma, output, hidden_output, cell_output, batch_y = self.forward(
#                     batch_x, batch_y
#                 )
#
#                 loss = self.get_loss(mu, sigma, output, batch_y).mean()
#                 self.bachward(loss)
#                 self.update()
#                 train_loss.append(loss.item())
#
#                 if (i + 1) % self.print_every == 0:
#                     print(
#                         "epoch:{0}, iters: {0} | loss: {2:.7f}".format(
#                             epoch, i + 1, loss.item()
#                         )
#                     )
#                 if self.step % self.validate_every == 0:
#                     valiation_loss, result = self.valiation()
#                     print(
#                         "step:{}, valiation_loss:{}, result:{}".format(
#                             self.step, valiation_loss, result
#                         )
#                     )
#
#             if Param.learning_rate:
#                 self.learningrate(optimizer=self.optimizer).step()
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             valiation_loss, valiation_result = self.valiation()
#             print(
#                 "epoch:{}, step:{}, train_loss:{}, valiation_loss:{}, valiation_result:{}".format(
#                     epoch, self.step, train_loss, valiation_loss, valiation_result
#                 )
#             )
#
#             self.earlystop(valiation_result, self.model, Param.model_path)
#             if self.earlystop.early_stop:
#                 break
#         best_model_path = Param.model_path + "/" + f"{Param.model_name}_checkpoint.pth"
#         self.model.load_state_dict(torch.load(best_model_path))
#
#         return self.model
#
#     def _valiation(self):
#         self.model.eval()
#         batch_mu = []
#         batch_sigma = []
#         batch_samples = []
#         batch_y8 = []
#         total_loss = []
#
#         for batch_x, batch_y in self.valiation_dataloader:
#             batch_x = move_data_to_device(batch_x, self.device)
#             mu, sigma, output, hidden_output, cell_output, batch_y = self.forward(
#                 batch_x, batch_y
#             )
#             loss = self.get_loss(mu, sigma, output, batch_y).mean()
#             samples = torch.zeros(
#                 Param.batch_size, Param.sample_times, Param.seq_len, device=self.device
#             )
#             for j in range(Param.sample_times):
#                 hidden = None
#                 cell = None
#
#                 for t in range(Param.seq_len):
#                     mu, sigma, hidden, cell, output = self.model(
#                         batch_x[:, t : t + 1, :], hidden, cell
#                     )
#                     gaussian = torch.distributions.normal.Normal(mu, sigma)
#                     pred = gaussian.sample()
#                     samples[:, j, t] = pred
#
#             sample_mu = torch.median(samples, dim=0)[0]  # [B, seq_len]
#             sample_sigma = samples.std(dim=0)
#
#             batch_mu.append(sample_mu)
#             batch_sigma.append(sample_sigma)
#             batch_samples.append(samples)
#             batch_y8.append(batch_y.squeeze())
#             total_loss.append(loss.item())
#         batch_y8 = np.array(batch_y8)
#         batch_samples = np.array(batch_samples)
#         batch_sigma = np.array(batch_sigma)
#         batch_mu = np.array(batch_mu)
#         batch_y8 = batch_y8.reshape(-1, batch_y8.shape[-1])
#         batch_sigma = batch_sigma.reshape(-1, batch_sigma.shape[-1])
#         batch_mu = batch_mu.reshape(-1, batch_mu.shape[-1])
#         batch_samples = batch_samples.reshape(
#             (-1, batch_samples.shape[-2], batch_samples.shape[-1])
#         )
#
#         result8 = {}
#         for metric in self.metrics:
#             if metric.metric_name != "rou8":
#                 result = metric(mu=batch_mu, target=batch_y8)
#             else:
#
#                 result = metric(mu=batch_mu, samples=batch_samples, target=batch_y8)
#             result8[metric.metric_name] = result
#         return result8
#
#     def predict(self, pred_loader):
#         if Param.pre:
#             best_model_path = (
#                 Param.model_path + "/" + f"{Param.model_name}_checkpoint.pth"
#             )
#             self.model.load_state_dict(torch.load(best_model_path))
#
#         self.model.eval()
#         batch_target = torch.Tensor(0)
#         batch_pred10 = torch.Tensor(0)
#         batch_pred50 = torch.Tensor(0)
#         batch_pred90 = torch.Tensor(0)
#         for i, (batch_x, batch_y) in enumerate(pred_loader):
#             batch_x = move_data_to_device(batch_x, self.device)
#             (
#                 input_mu,
#                 input_sigma,
#                 output,
#                 hidden_output,
#                 cell_output,
#                 batch_y,
#             ) = self.forward(
#                 batch_x, batch_y
#             )  # seq_len - label_len 用于训练的
#             # input_mu: [B, pred_len]  input_sigma: [B, pred_len] output: [B, pred_len] batch_y: [B, pred_len]
#             loss = self.get_loss(input_mu, input_sigma, output, batch_y).mean()
#             batch_size = batch_x.shape[0]
#             pred_len = Param.pred_len
#             samples = torch.zeros(
#                 Param.sample_times, batch_size, pred_len, device=self.device
#             )
#             samples = []
#             for j in range(Param.sample_times):
#                 mu, sigma, output, batch_y, hidden_outout, cell_output = self.forward(
#                     batch_x, batch_y
#                 )  # [B, seq_len] 预测的部分
#                 if mu.ndim != 2 and sigma.ndim != 2:
#                     mu = mu.squeeze()
#                     sigma = sigma.squeeze()
#                 gaussian = torch.distributions.normal.Normal(mu, sigma)  # [B, seq_len]
#                 pred = gaussian.sample()  # [B, seq_len]
#                 # samples[j] = pred
#                 samples.append(pred.detach().cpu().numpy())
#             samples = np.concatenate(samples, axis=2)  # [B, seq_len, sample_time]
#             p50 = np.quantile(samples, 0.56, axis=2)  # [B, seq_len]
#             p90 = np.quantile(samples, 0.9, axis=2)
#
#             p10 = np.quantile(samples, 0.10, axis=2)
#
#             batch_target = torch.cat((batch_target, batch_y[:, 0, :].view(-1).cpu()))
#             batch_pred10 = torch.cat((batch_pred10, p10[:, 0, :].view(-1).cpu()))
#             batch_pred50 = torch.cat((batch_pred50, p50[:, 0, :].view(-1).cpu()))
#             batch_pred90 = torch.cat((batch_pred90, p90[:, 0, :].view(-1).cpu()))
#
#         return batch_target, batch_pred10, batch_pred50, batch_pred90
#
#     def get_loss(self, mu, sigma, output, batch_y):
#         criterion = Probability_Loss()
#         loss = criterion(mu, sigma, output, batch_y)
#         return loss
#
#     def bachward(self, loss):
#         if (self.step - 1) % self.update_every == 0:
#             self.model.zero_grad()
#         loss.backward()
#
#     def update(self):
#         if self.step % self.update_every == 0:
#             self.optimizer.step()
#
#     def forward(self, batch_x, batch_y):
#         if batch_x.shape[1] < batch_y.shape[1]:  # 说明输入需要padding
#             batch_n = torch.zeros(
#                 [
#                     batch_y.shape[0],
#                     batch_y.shape[1] - batch_x.shape[1],
#                     batch_x.shape[-1],
#                 ]
#             ).float()
#
#             batch_x = torch.cat([batch_x, batch_n], dim=1).float().to(self.device)
#         mu, sigma, output_lstm, output, hidden_output, cell_output = self.model(batch_x)
#         if Param.feature == "more-one":
#             batch_y = batch_y[:, -Param.pred_len :, 0:].to(self.device)
#         else:
#             batch_y = batch_y[:, -Param.pred_len :, -1:].to(self.device)
#
#         return mu, sigma, output, hidden_output, cell_output, batch_y
