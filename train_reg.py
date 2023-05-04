import sys
sys.path.append("/home/zhunanyang/bbbb8888")

from pathlib import Path
from data import power_dataset_reg

from params import represent_param, reg_param
import pandas as pd
from torch.utils.data import DataLoader
from nn.trainer import Trainer
from model import power_model
from nn.metrics import Rm_Mtric, MaeMtric, r2, agreement, n, NR2
from nn.optimizers import AdamW

from nn.metric_tracker import MetricTracker
import numpy as np
from nn.callbacks import (
    EvaluateCallback,
    CheckpointCallback,
    SaveCallback,
)


#data_path = Path.cwd() / "data"
#random_excel(power_param.data_p / "df.csv")
train_df = pd.read_excel(represent_param.data_p / "train.xlsx")
#df = pd.read_csv(power_param.data_p / "train_all.csv")
train_data = power_dataset_reg(
    train_df, target=reg_param.target, x=represent_param.x
)
train_dataloader = DataLoader(
    train_data, batch_size=reg_param.batch_size, shuffle=True, drop_last=True
)

train8_df = pd.read_excel(reg_param.data_p / "train.xlsx")

train8_data = power_dataset_reg(
    train8_df,  target=reg_param.target, x=reg_param.x
)
train8_dataloader = DataLoader(
    train8_data, batch_size=reg_param.batch_size, shuffle=False, drop_last=True
)

callbacks = [
    EvaluateCallback(),
    CheckpointCallback(),
    SaveCallback(),]
model = power_model(x=len(reg_param.x))
model = model.transfer_model(reg_param.model_represent, model)

metrics = [Rm_Mtric(name="rm"), MaeMtric(name="mae"), r2(name="r2"),]

trainer = Trainer(
    data=train_data,
    dataloader=train_dataloader,
    model=model,
    metrics=metrics,
    valiation_data=train8_data,
    valiation_dataloader={"dev": train8_dataloader, "test": train8_dataloader},
    optimizer=AdamW(lr=reg_param.learning_rate),
    n_epochs=128,
    callbacks=callbacks,
    batch_size=reg_param.batch_size,

    print_every=1,
    metric_tracker=MetricTracker(metric_key="r2.r2"),
    model_p=reg_param.model)
trainer.train()
