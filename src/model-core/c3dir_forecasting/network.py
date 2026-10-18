"""ConvLSTM forecaster (Shi et al., 2015, https://arxiv.org/abs/1506.04214) of C3DIR cloud
probability frames.

frames are (B, T, C, H, W) with C = levels (see frames.py), each a probability in [0, 1].

  encoder    two 3x3 convs, each halving H and W, then stacked ConvLSTM cells read the input
             frames one at a time
  forecaster the same cells keep stepping, fed the encoding of their own last forecast; a decoder
             (two 2x upsamplings) turns each hidden state into a change to the last frame in
             logit space. the last layer starts at zero, so an untrained model is persistence
             (the next frame = the last one)

forward(x, output_steps) can roll past the number of steps it was trained on.
"""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NetConfig:
    channels: int = 20  # levels per frame, set from FrameConfig
    width: int = 32  # channels after the encoder convs
    hidden: list = field(default_factory=lambda: [32, 32])  # ConvLSTM layers
    kernel: int = 3

    def as_dict(self):
        return asdict(self)


class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, c_hid, kernel=3):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, kernel, padding=kernel // 2)  # input, forget, output, candidate
        nn.init.constant_(self.conv.bias[c_hid:2 * c_hid], 1.0)  # start with the forget gate open

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMForecaster(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c = self.config = config or NetConfig()
        self.encode = nn.Sequential(
            nn.Conv2d(c.channels, c.width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c.width, c.width, 3, stride=2, padding=1), nn.GELU())
        sizes = [c.width, *c.hidden]
        self.cells = nn.ModuleList(ConvLSTMCell(a, b, c.kernel) for a, b in zip(sizes[:-1], sizes[1:]))
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(c.hidden[-1], c.width, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(c.width, c.width, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c.width, c.channels, 3, padding=1))
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)

    def step(self, frame, state):
        inp = self.encode(frame)
        for k, cell in enumerate(self.cells):
            state[k] = cell(inp, state[k])
            inp = state[k][0]
        return inp

    def forward(self, x, output_steps):
        """x (B, T, C, H, W) frames up to t -> (B, output_steps, C, H, W) frames t+1 ...."""
        b, t, _, h, w = x.shape
        size = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        state = [(x.new_zeros(b, cell.c_hid, *size), x.new_zeros(b, cell.c_hid, *size)) for cell in self.cells]
        for k in range(t):
            top = self.step(x[:, k], state)
        last, out = x[:, -1], []
        for k in range(output_steps):
            if k:  # after the first forecast, feed the model its own last frame
                top = self.step(last, state)
            delta = F.interpolate(self.decode(top), size=(h, w), mode="bilinear", align_corners=False)
            last = torch.sigmoid(torch.logit(last.clamp(1e-4, 1 - 1e-4)) + delta)
            out.append(last)
        return torch.stack(out, dim=1)
