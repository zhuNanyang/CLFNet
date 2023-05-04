from bbbb8888.nn.losses import Loss
import torch

import numpy as np

# https://blog.csdn.net/u011984148/article/details/107754554
class info_nce(Loss):
    def __init__(self, temperature, batch_size):
        self.tempeature = temperature
        self.batch_size = batch_size

        super(info_nce, self).__init__()

    def cosine(self, f1, f2):

        t = torch.nn.functional.cosine_similarity(
            f1.unsqueeze(0), f2.unsqueeze(1), dim=2
        )
        return t

    def dot_similarity(self, f1, f2):
        t = torch.tensordot(
            f1.unsqueeze(1), f2.T.unsqueeze(0), dims=2
        )  # torch.Matmul torch.mm  功能一样
        return t

    def negatives_mask(self):
        return (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).to(
            float
        )

    def __call__(self, f1, f2):
        # f1: [B, embedding] f2: [B, embedding]
        f1_embedding = torch.nn.functional.normalize(f1, dim=1)
        f2_embedding = torch.nn.functional.normalize(f2, dim=1)

        representations = torch.cat([f1_embedding, f2_embedding], dim=0)
        similarity_matrix = self.cosine(f1=representations, f2=representations)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)

        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.tempeature)
        negatives_mask = self.negatives_mask()

        negatives_mask = negatives_mask.to(similarity_matrix.device)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.tempeature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


if __name__ == "__main__":
    value = torch.randn((2, 8))

    loss = info_nce(temperature=0.1, batch_size=2)
    value_0 = loss(value, value)
