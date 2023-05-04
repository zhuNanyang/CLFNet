import torch
import torch.nn.utils.rnn as rnn


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int, # 输入数据的特征维度 embedding_dim
        hidden_size=100,
        num_layers=1,
        dropout=0.0,
        batch_first=True,
        bidirectional=False,
        bias=True,
    ):
        super(LSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.init_param()

    def init_param(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                # based on https://github.com/pytorch/pytorch/issues/750#issuecomment-280671871
                param.data.fill_(0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
            else:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x, seq_len=None, h0=None, c0=None):
        """
        :param x: [batch, seq_len, input_size]
        :param seq_len: [batch, ]
        :param h0: [batch, hidden_size] init hidden, 若为 ``None`` , 设为全0向量. Default: ``None``
        :param c0: [batch, hidden_size] init cell, 若为 ``None`` , 设为全0向量. Default: ``None``
        :return (output, (ht, ct)):
            output: [batch, seq_len, hidden_size * num_direction] output
            ht,ct: [num_layers * num_direction, batch, hidden_size] last hidden, cell 最后一个time_step的输出
        """
        batch_size, max_len, _ = x.size()
        if h0 is not None and c0 is not None:
            hx = (h0, c0)
        else:
            hx = None
        if seq_len is not None and not isinstance(x, rnn.PackedSequence):
            sort_lens, sort_idx = torch.sort(seq_len, dim=0, descending=True)
            if self.batch_first:
                x = x[sort_idx]
            else:
                x = x[:, sort_idx]
            x = rnn.pack_padded_sequence(x, sort_lens, batch_first=self.batch_first)
            output, hx = self.lstm(x, hx)
            output, _ = rnn.pad_packed_sequence(output, batch_first=self.batch_first, total_length=max_len)
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            if self.batch_first:
                output = output[unsort_idx]
            else:
                output = output[:, unsort_idx]
            hx = hx[0][:, unsort_idx], hx[1][:, unsort_idx]
        else:
            output, hx = self.lstm(x, hx)
        return output, hx