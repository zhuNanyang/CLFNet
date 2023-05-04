import numpy as np
import torch
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

import torch.nn as nn
import math


class Metric:
    def __init__(self, name):
        self._name = name

    @property
    def metric_name(self):
        return self._name

    def evaluate(self, pred, target):
        return NotImplemented

    def get_result(self, pred, target):
        return NotImplemented

    def __call__(self, pred, target):
        result = self.evaluate(pred, target)
        return result


class MaeMtric(Metric):
    def __init__(self, name):
        super(MaeMtric, self).__init__(name=name)

    def evaluate(self, pred, target):
        if len(pred.shape) == 3:
            pred = pred[:, -1, -1]
            target = target[:, -1, -1]
        return np.mean(np.abs(pred - target))

    def get_result(self, pred, target):
        if len(pred.shape) == 3:
            pred = pred[:, -1, -1]
            target = target[:, -1, -1]
        # result = self.evaluate(pred, target)
        return {"mae": np.mean(np.abs(pred - target))}


class MapeMetric(Metric):
    def __init__(self, name):
        super(MapeMetric, self).__init__(name=name)

    def evaluate(self, pred, target):
        return np.mean(np.abs((pred - target) / target))


class Rm_Mtric(Metric):
    def __init__(self, name):
        super(Rm_Mtric, self).__init__(name=name)

    def evaluate(self, pred, target):
        if len(pred.shape) == 3:
            pred = pred[:, -1, -1]
            target = target[:, -1, -1]
        return np.sqrt(np.mean((pred - target) ** 2))

    def get_result(self, pred=None, target=None):
        if len(pred.shape) == 3:
            pred = pred[:, -1, -1]
            target = target[:, -1, -1]
        return {"rmse": np.sqrt(np.mean((pred - target) ** 2))}


class Mpe_Mtric(Metric):
    def __init__(self, name):
        super(Mpe_Mtric, self).__init__(name=name)

    def evaluate(self, pred, target):
        return np.mean(np.square((pred - target) / target))


class Corr_Mtric(Metric):
    def __init__(self, name):
        super(Corr_Mtric, self).__init__(name=name)

    def evaluate(self, pred, target):
        u = ((target - target.mean(0)) * (pred - pred.mean(0))).sum(0)
        d = np.sqrt(
            ((target - target.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)
        )
        return (u / d).mean(-1)


class Probability_Metric(Metric):
    def __init__(self, name):
        super(Probability_Metric, self).__init__(name=name)

    def evaluate(self, pred, target):
        pass


class Nd_Metric(Metric):
    def __init__(self, name):
        super(Nd_Metric, self).__init__(name=name)

    def evaluate(self, mu: torch.Tensor, target: torch.Tensor, relative=False):
        mu = mu.cpu().detach().numpy()
        labels = target.cpu().detach().numpy()

        mu[labels == 0] = 0.0

        diff = np.sum(np.abs(mu - labels), axis=1)
        if relative:
            summation = np.sum((labels != 0), axis=1)
            mask = summation == 0
            summation[mask] = 1
            result = diff / summation
            result[mask] = -1
            return result
        else:
            summation = np.sum(np.abs(labels), axis=1)
            mask = summation == 0
            summation[mask] = 1
            result = diff / summation
            result[mask] = -1
            return result


class AcRou_Metric(Metric):
    def __init__(self, name, rou):
        self.rou = rou
        super(AcRou_Metric, self).__init__(name=name)

    def evaluate(
        self,
        mu: torch.Tensor = None,
        samples: torch.Tensor = None,
        target: torch.Tensor = None,
        relative=False,
    ):
        samples = samples.cpu().detach().numpy()
        labels = target.cpu().detach().numpy()
        mask = labels == 0
        samples[:, mask] = 0.0

        pred_samples = samples.shape[0]
        rou_th = math.floor(pred_samples * self.rou)
        samples = np.sort(samples, axis=0)
        rou_pred = samples[rou_th]
        abs_diff = np.abs(labels - rou_pred)
        abs_diff_1 = abs_diff.copy()
        abs_diff_1[labels < rou_pred] = 0.0
        abs_diff_2 = abs_diff.copy()
        abs_diff_2[labels >= rou_pred] = 0.0
        numerator = 2 * (
            self.rou * np.sum(abs_diff_1, axis=1)
            + (1 - self.rou) * np.sum(abs_diff_2, axis=1)
        )
        denominator = np.sum(labels, axis=1)
        mask2 = denominator == 0
        denominator[mask2] = 1
        result = numerator / denominator
        result[mask2] = -1
        return result


class Prob_Rm(Metric):
    def __init__(self, name):
        super(Prob_Rm, self).__init__(name=name)

    def evaluate(self, mu: torch.Tensor, target: torch.Tensor, relative=None):
        mu = mu.cpu().detach().numpy()
        labels = target.cpu().detach().numpy()

        mask = labels == 0
        mu[mask] = 0.0

        diff = np.sum((mu - labels) ** 2, axis=1)
        summation = np.sum(np.abs(labels), axis=1)
        mask2 = summation == 0
        if relative:
            div = np.sum(~mask, axis=1)
            div[mask2] = 1
            result = np.sqrt(diff / div)
            result[mask2] = -1
            return result
        else:
            summation[mask2] = 1
            result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))

            result[mask2] = -1
            return result

class r2(Metric):
    def __init__(self, name, t=None):
        super(r2, self).__init__(name=name)
        self.t = t

    def evaluate(self, pred, target):
        if len(pred.shape) == 3:
            if self.t:
                pred = pred[:, self.t, -1]
                target = target[:, self.t, -1]
            else:
                pred = pred[:, :, -1]
                target = target[:, :, -1]
        return r2_score(y_pred=pred, y_true=target)

    def get_result(self, pred, target):
        if len(pred.shape) == 3:
            if self.t:
                pred = pred[:, self.t, -1]
                target = target[:, self.t, -1]
            else:
                pred = pred[:, :, -1]
                target = target[:, :, -1]
        return {"r2": r2_score(y_pred=pred, y_true=target)}


class NRmse(Metric):
    def __init__(self, name):
        super(NRmse, self).__init__(name=name)

    def evaluate(self, pred, target):
        rm = mean_squared_error(y_pred=pred, y_true=target, squared=False)
        nrm = rm / np.sqrt(np.mean(target**2))
        return nrm


class agreement(Metric):
    def __init__(self, name):
        super(agreement, self).__init__(name)

    def evaluate(self, pred, target):
        if len(pred.shape) == 3:
            pred = pred[:, -1, -1]
            target = target[:, -1, -1]
        return 1 - np.sum((target - pred) ** 2) / np.sum(
            (np.abs(target - np.mean(target)) + np.abs(pred - np.mean(target))) ** 2
        )

    def get_result(self, pred, target):
        result = self.evaluate(pred, target)
        return {"d": result}


class n(Metric):
    def __init__(self, name):
        super(n, self).__init__(name=name)

    def evaluate(self, pred, target):
        if len(pred.shape) == 3:
            pred = pred[:, -1, -1]
            target = target[:, -1, -1]
        return 1 - np.sum((pred - target) ** 2) / np.sum(
            (np.mean(target) - target) ** 2
        )

    def get_result(self, pred, target):
        result = self.evaluate(pred, target)
        return {"n": result}


class corr(Metric):
    def __init__(self, name):
        super(corr, self).__init__(name=name)

    def evaluate(self, pred, target):
        if len(pred.shape) == 3:
            pred = pred[:, -1, -1]
            target = target[:, -1, -1]
        sigma_x = pred.std(axis=0)
        sigma_y = target.std(axis=0)
        mean_x = pred.mean(axis=0)
        mean_y = target.mean(axis=0)
        cor = ((pred - mean_x) * (target - mean_y)).mean(0) / (
            sigma_x * sigma_y + 0.000000000001
        )
        return cor.mean()

    def get_result(self, pred, target):
        result = self.evaluate(pred, target)
        return {"CORR": result}


class NRMSE(Metric):
    def __init__(self, name, n):
        super(NRMSE, self).__init__(name)
        self.n = n

    def evaluate(self, pred, target):
        # pred: [B, N, pred_len] target: [B, N, pred_l]
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]
            score = mean_squared_error(p, t, squared=False)
            scores += score
        return scores / self.n

    def get_result(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]
            score = mean_squared_error(p, t, squared=False)
            scores += score
        return {"NRMSE": scores / self.n}


class NMAE(Metric):
    def __init__(self, name, n):
        super(NMAE, self).__init__(name)
        self.n = n

    def evaluate(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]
            score = np.mean(np.abs(p - t))
            scores += score
        return scores / self.n

    def get_result(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]
            score = np.mean(np.abs(p - t))
            scores += score
        return {"NMAE": scores / self.n}


class NR2(Metric):
    def __init__(self, name, n):
        super(NR2, self).__init__(name=name)
        self.n = n

    def evaluate(self, pred, target):
        scores = 0
        for t in range(self.n):
            r2_t = r2(name=f"r2{t}", t=t)
            result = r2_t.get_result(pred, target)
            score = result["r2"]
            scores += score
        return scores / self.n

    def get_result(self, pred, target):
        scores = 0
        for t in range(self.n):
            r2_t = r2(name=f"r2{t}", t=t)
            result = r2_t.get_result(pred, target)
            score = result["r2"]
            scores += score

        return {"NR2": scores / self.n}


class NSE(Metric):
    def __init__(self, name, n):
        super(NSE, self).__init__(name)
        self.n = n

    def evaluate(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]

            mse = np.sum(np.square(np.subtract(pred[:, :, -1], target[:, :, -1])))
            means = np.mean(target[:, :, -1])
            labels_mse = np.sum(np.square(np.subtract(target[:, :, -1], means)))
            score = np.sqrt(mse / labels_mse)
            scores += score
        return scores / self.n

    def get_result(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]
            mse = np.sum(np.square(np.subtract(p, t)))

            means = np.mean(t)
            labels_mse = np.sum(np.square(np.subtract(t, means)))
            score = np.sqrt(mse / labels_mse)
            scores += score
        return {"nse": scores / self.n}


class NCORR(Metric):
    def __init__(self, name, n):
        super(NCORR, self).__init__(name=name)
        self.n = n

    def evaluate(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]

            sigma_x = p.std(axis=0)
            sigma_y = t.std(axis=0)
            mean_x = p.mean(axis=0)
            mean_y = t.mean(axis=0)
            cor = ((p - mean_x) * (t - mean_y)).mean(0) / (
                sigma_x * sigma_y + 0.000000000001
            )
            score = cor.mean()
            scores += score
        return scores / self.n

    def get_result(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]
            sigma_x = p.std(axis=0)
            sigma_y = t.std(axis=0)
            mean_x = p.mean(axis=0)
            mean_y = t.mean(axis=0)
            cor = ((p - mean_x) * (t - mean_y)).mean(0) / (
                sigma_x * sigma_y + 0.000000000001
            )
            score = cor.mean()
            scores += score
        return {"NCORR": scores / self.n}


class NMAPE(Metric):
    def __init__(self, name, n):
        super(NMAPE, self).__init__(name=name)
        self.n = n

    def evaluate(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = pred[:, n, :]
            score = mean_absolute_percentage_error(p, t)
            scores += score
        return scores / self.n

    def get_result(self, pred, target):
        scores = 0
        for n in range(self.n):
            p = pred[:, n, :]
            t = target[:, n, :]
            score = mean_absolute_percentage_error(p, t)

            scores += score
        return {"NMAPE": scores / self.n}

