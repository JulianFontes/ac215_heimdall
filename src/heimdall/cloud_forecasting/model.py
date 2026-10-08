"""3D ConvLSTM (https://arxiv.org/pdf/1506.04214): the usual Shi et al. (2015) cell with Conv3d in place of Conv2d."""
import torch
import torch.nn as nn

from config import N_CH

class ConvLSTM3dCell(nn.Module):
    def __init__(self, c_in, c_hid, k=3):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv3d(c_in + c_hid, 4 * c_hid, k, padding=k // 2) # input, forget, output, candidate gates
        nn.init.constant_(self.conv.bias[c_hid:2 * c_hid], 1.0) # start with the forget gate open

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class ConvLSTM3d(nn.Module):
    """Stacked ConvLSTM. x: (B, T, C, D, H, W) -> logits for the hour after T: (B, n_out, D, H, W)."""
    def __init__(self, c_in=N_CH, hidden=(16, 16), n_out=3, k=3):
        super().__init__()
        sizes = [c_in, *hidden]
        self.cells = nn.ModuleList(ConvLSTM3dCell(a, b, k) for a, b in zip(sizes[:-1], sizes[1:]))
        self.head = nn.Conv3d(hidden[-1], n_out, kernel_size=1)

    def forward(self, x):
        B, T, _, *space = x.shape
        state = [(x.new_zeros(B, c.c_hid, *space), x.new_zeros(B, c.c_hid, *space)) for c in self.cells]
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                state[l] = cell(inp, state[l])
                inp = state[l][0]
        return self.head(inp)