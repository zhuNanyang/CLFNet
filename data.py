import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from params import represent_param
class power_dataset(Dataset):
    def __init__(self, df, x=None, target=None):
        self.df = df
        self.x = x
        self.target=target
        self._read()
        super(power_dataset, self).__init__()


    def _read(self):
        if not self.x:
            self.x = list(self.df.columns)
        df_data = self.df
        if represent_param.standard:

            self.df_mean = df_data[self.x].mean()
            self.df_std = df_data[self.x].std()
            df_data[self.x] = (df_data[self.x] - self.df_mean) / self.df_std
        data_y = df_data[self.target].values
        data_x = df_data[self.x].values

        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, t):
        b = t
        e = b + 30
        t_b = e - represent_param.label_len
        t_e = t_b + represent_param.label_len + represent_param.pred_len
        x = self.data_x[b: e]
        y = self.data_y[t_b: t_e]

        return {
            "x": np.array(x),
            "y": np.array(y),
        }

    def __len__(self):
        return len(self.data_x) - 30 - represent_param.pred_len + 1

class power_dataset_reg(Dataset):
    def __init__(self, df, x=None, target=None):
        self.df = df
        self.x = x
        self.target=target
        self._read()
        super(power_dataset_reg, self).__init__()


    def _read(self):
        if not self.x:
            self.x = list(self.df.columns)
        df_data = self.df
        if represent_param.standard:

            self.df_mean = df_data[self.x].mean()
            self.df_std = df_data[self.x].std()
            df_data[self.x] = (df_data[self.x] - self.df_mean) / self.df_std
        data_y = df_data[self.target].values
        data_x = df_data[self.x].values

        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, t):
        b = t
        e = b + 30
        t_b = e - represent_param.label_len
        t_e = t_b + represent_param.label_len + represent_param.pred_len
        x = self.data_x[b: e]
        y = self.data_y[t_b: t_e]

        return {
            "x": np.array(x),
            "y": np.array(y),
        }

    def __len__(self):
        return len(self.data_x) - 30 - represent_param.pred_len + 1