import torch
from torch import Tensor
from typing import Optional


from typing import List
from nn.utils import seq_len_to_mask
import torch.nn as nn


class Loss:
    pass


class DiceLoss(Loss):
    def __init__(
        self,
        smooth=1e-4,
        square_denominator=False,
        with_logits=True,
        ohem_ratio=0.0,
        alpha=0.0,
        reduction="mean",
        index_label_position=True,
    ):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def __call__(self, input, target, mask=None):
        # print(input.shape)
        # print(target.shape)
        logits_size = input.shape[-1]
        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask)
        else:
            loss = self._binary_class(input, target, logits_size)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input

        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - (
                (2 * interection + self.smooth)
                / (flat_input.sum() + flat_target.sum() + self.smooth)
            )
        else:
            loss = 1 - (
                (2 * interection + self.smooth)
                / (
                    torch.sum(
                        torch.square(
                            flat_input,
                        ),
                        -1,
                    )
                    + torch.sum(torch.square(flat_target), -1)
                    + self.smooth
                )
            )

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = (
            torch.nn.functional.one_hot(target, num_classes=logits_size).float()
            if self.index_label_position
            else target.float()
        )
        # print(flat_target.shape)
        flat_input = (
            torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input
        )
        # print(flat_input.shape)
        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0:
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx
                # print(pos_example)
                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(
                        flat_input, neg_example.view(-1, 1).bool()
                    ).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(
                        neg_scores_idx,
                    )

                    threshold = neg_scores_sort[-keep_num + 1]
                    # print(pos_example.view(-1))
                    cond1 = torch.argmax(flat_input, dim=1) == label_idx
                    cond2 = flat_input[:, label_idx] >= threshold
                    cond3 = cond1 & cond2
                    cond6 = pos_example.view(-1)
                    cond = cond3 | cond6
                    # cond = (
                    #     torch.argmax(flat_input, dim=1) == label_idx & flat_input[:, label_idx] >= threshold

                    # ) | pos_example.view(-1)
                    ohem_mask_idx = cond.int()

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]
                loss_idx = self._compute_dice_loss(
                    flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1)
                )
                if loss is None:
                    loss = loss_idx
                else:

                    loss += loss_idx
            return loss
        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]
                loss_idx = self._compute_dice_loss(
                    flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1)
                )
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(
                neg_scores,
            )
            threshold = neg_scores_sort[-keep_num + 1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)


class FocalLoss(Loss):
    def __init__(self, gamma=2, weight=None, reduction="none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

        self.weight = weight
        self.reduction = reduction

    def __call__(self, input, target):
        target = target.unsqueeze(-1)
        logit_pred = torch.nn.functional.log_softmax(input, dim=-1)
        pred = torch.nn.functional.softmax(input, dim=-1)
        pred = pred.gather(1, target).squeeze(-1)
        logit_pred = logit_pred.gather(1, target).squeeze(-1)
        if self.weight is not None:
            self.weight = torch.FloatTensor(self.weight)
            at = self.weight.gather(0, target.squeeze(-1))
            logit_pred = logit_pred * at
        loss = -1 * (1 - pred) ** self.gamma * logit_pred

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "none":
            return loss

        return loss.sum()


class CrossEntropyLoss(Loss):
    def __init__(self, class_in_dim=-1, padding_idx=-100, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.padding_idx = padding_idx
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.class_in_dim = class_in_dim

    def __call__(self, pred, target, seq_len=None):
        if seq_len is not None and target.dim() > 1:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1)).eq(False)
            target = target.masked_fill(mask, self.padding_idx)

        if pred.dim() > 2:
            if self.class_in_dim == -1:
                if pred.size(1) != target.size(1):
                    pred = pred.transpose(1, 2)
            else:
                pred = pred.transpose(-1, self.class_in_dim)
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)

        return torch.nn.functional.cross_entropy(
            input=pred,
            target=target,
            ignore_index=self.padding_idx,
            reduction=self.reduction,
        )


class BCELoss(Loss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def __call__(self, pred: Tensor, target: Tensor, seq_len: Optional[Tensor] = None):
        target = target.type_as(pred)
        if seq_len is not None and target.dim() >= 2:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1))  # [B, max_len]
            loss = torch.nn.functional.binary_cross_entropy(
                pred, target, reduction="none"
            )

            num = torch.sum(mask)
            if target.dim() == 3:
                loss = torch.sum(loss, dim=-1)
                num = num * target.size(-1)

            total_loss = torch.sum(loss * mask)
            loss = total_loss / num
        else:
            loss = torch.nn.functional.binary_cross_entropy(pred, target)

        return loss


class BCEWithLogitsLoss(Loss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, seq_len: torch.Tensor):
        target = target.type_as(pred)
        if seq_len is not None and target.dim() >= 2:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1))  # [B, max_len]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred, target, reduction="none"
            )

            num = torch.sum(mask)
            if target.dim() == 3:
                loss = torch.sum(loss, dim=-1)
                num = num * target.size(-1)
            total_loss = torch.sum(loss * mask)

            loss = total_loss / num
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
        return loss


class M_Loss(Loss):
    def __init__(self):
        super(M_Loss, self).__init__()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):

        # pred:[B, L, d] , target: [B, L, d]  d 一般为输出目标的个数 单预测为1
        return torch.nn.functional.mse_loss(pred, target)


# class Probability_Loss(Loss):  # Probability_trainer中的损失函数
#     def __init__(self):
#         super(Probability_Loss, self).__init__()
#
#     def __call__(
#         self,
#         mu: torch.Tensor,
#         sigma: torch.Tensor,
#         pred: torch.Tensor,
#         target: torch.Tensor,
#     ):
#
#         if pred.shape != target.shape:
#             print("输入大小和输出大小不匹配。。。。。。")
#         if mu.ndim != 2 and sigma.ndim != 2:
#             mu = mu.squeeze()
#             sigma = sigma.squeeze()
#         if target.ndim != 2:
#             target = target.squeeze()
#         distribution = torch.distributions.normal.Normal(mu, sigma)
#         likelihood = distribution.log_prob(target)
#
#         return -torch.mean(likelihood)


class Gaussian_Loss(Loss):  # Probability_trainer中的损失函数
    def __init__(self):
        super(Gaussian_Loss, self).__init__()

    def __call__(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        negative_likelihood = (
            torch.log(sigma + 1) + (target - mu) ** 2 / (2 * sigma ** 2) + 6
        )

        return negative_likelihood.mean()


class Quantile_Loss(Loss):  # trainer中的损失函数  proba=True的情况
    def __init__(self):
        super(Quantile_Loss, self).__init__()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, quantile):
        losses = []
        for i, q in enumerate(quantile):
            errors = target - pred[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
