import torch.nn as nn
from bbbb8888.nn.utils import initial_parameter



class Conv1d(nn.Module):
    def __init__(self, input_size, output_size, padding: int = 1, kernel=None, initita_method=None):
        self.input_size = input_size
        self.output_size = output_size
        super(Conv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=output_size,kernel_size=kernel, padding=padding, padding_mode='circular')
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
    def forward(self, x):
        # x [B, seq_len, embedding_d]  conv1d是在最后一个维度上扫的 需要进行转换 https://blog.csdn.net/sunny_xsc1994/article/details/82969867
        x = self.conv1d(x.permute(0, 2, 1)).transpose(1, 2)
        return x