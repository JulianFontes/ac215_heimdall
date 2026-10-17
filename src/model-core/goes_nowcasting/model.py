"""2D ConvLSTM (https://arxiv.org/pdf/1506.04214) as an encoder-forecaster: the usual Shi et al. (2015) cell."""
import torch
import torch.nn as nn

from .config import HIDDEN, N_CH, TARGET_IDX, T_OUT

class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, c_hid, k=3):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, k, padding=k // 2) # input, forget, output, candidate gates
        nn.init.constant_(self.conv.bias[c_hid:2 * c_hid], 1.0) # start with the forget gate open

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    """Stacked ConvLSTM. x: (B, T, C, H, W) -> the next T_OUT frames of the target bands: (B, T_OUT, n_out, H, W).
    The encoder reads the T input frames. The forecaster starts from the encoder state and predicts one frame at a time,
    feeding each prediction back in. Every step predicts a change on top of the previous frame, so the untrained model is persistence."""
    def __init__(self, c_in=N_CH, hidden=HIDDEN, target_idx=TARGET_IDX, t_out=T_OUT, k=3):
        super().__init__()
        n_out = len(target_idx)
        self.target_idx, self.t_out = list(target_idx), t_out
        self.encoder = nn.ModuleList(ConvLSTMCell(a, b, k) for a, b in zip([c_in, *hidden[:-1]], hidden))
        self.forecaster = nn.ModuleList(ConvLSTMCell(a, b, k) for a, b in zip([n_out, *hidden[:-1]], hidden))
        self.head = nn.Conv2d(hidden[-1], n_out, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, T, _, *space = x.shape
        state = [(x.new_zeros(B, c.c_hid, *space), x.new_zeros(B, c.c_hid, *space)) for c in self.encoder]
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.encoder):
                state[l] = cell(inp, state[l])
                inp = state[l][0]
        prev, out = x[:, -1, self.target_idx], []
        for _ in range(self.t_out):
            inp = prev
            for l, cell in enumerate(self.forecaster):
                state[l] = cell(inp, state[l])
                inp = state[l][0]
            prev = prev + self.head(inp)
            out.append(prev)
        return torch.stack(out, dim=1)
