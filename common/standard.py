import torch



import sklearn


class Standard_Torch(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.



    def fit_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else data.mean(0)
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else data.std(0)
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean




class Standard:
    @classmethod
    def torch8(cls):
        return Standard_Torch()

    @classmethod
    def sklearn8(cls):
        return sklearn.preprocessing.MinMaxScaler()








