import torch
from torch import nn


class Gaussian(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()

        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)

        return mu_t, sigma_t

class NegativeBinomial(nn.Module):
    def __init__(self, input_size, output_size):
        super(NegativeBinomial, self).__init__()

        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t
