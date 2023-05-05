import sys
sys.path.append("~/bbbb")

from pathlib import Path
from data import power_dataset

from params import represent_param
import pandas as pd
from torch.utils.data import DataLoader
from nn.trainer import Trainer
from model import represent_model
from nn.metrics import Rm_Mtric, MaeMtric, r2, agreement, n, NR2
from nn.optimizers import AdamW

from params import represent_param
from nn.metric_tracker import MetricTracker
from nn.callbacks import (
    CheckpointCallback,
    SaveCallback,
    EvaluateCallback)
train_df_p = represent_param.data_p / "train.xlsx"
train_df = pd.read_excel(train_df_p)
train_dataset = power_dataset(df=train_df, x=represent_param.x, target=represent_param.target)
train_dataloader = DataLoader(train_dataset, batch_size=represent_param.batch_size, shuffle=True)
#
train8_df_p = represent_param.data_p / "train.xlsx"
train8_df = pd.read_excel(train8_df_p)
train8_dataset = power_dataset(df=train8_df, x=represent_param.x, target=represent_param.target)
train8_dataloader = DataLoader(train8_dataset, batch_size=represent_param.batch_size, shuffle=True)
callbacks = [
    EvaluateCallback(),
    CheckpointCallback(),
    SaveCallback(),
]

model = represent_model(x=len(represent_param.x))
metric = [
    Rm_Mtric(name="rm"),
    MaeMtric(name="mae"),
    r2(name="r2", t=-1),
]


trainer = Trainer(
    data=train_dataset,
    dataloader=train_dataloader,
    model=model,
    metrics=metric,
    valiation_data=train8_dataset,
    valiation_dataloader={"dev": train8_dataloader, "test": train8_dataloader},

    optimizer=AdamW(lr=represent_param.learning_rate),
    n_epochs=128,
    callbacks=callbacks,
    batch_size=represent_param.batch_size,
    print_every=1,
    metric_tracker=MetricTracker(metric_key="rm.rmse"),
    model_p=represent_param.model_p

)
trainer.train()

