"""ConvLSTM forecaster (Shi et al., 2015, https://arxiv.org/abs/1506.04214) for C3DIR fields.

encoder-forecaster layout:
  encoder    two 3x3 convs, each halving lat / lon, then stacked ConvLSTM cells read the
             previous input_steps frames
  forecaster the same cells keep stepping, fed the encoding of their own last forecast, and a
             decoder (two 2x upsamplings) turns each hidden state into a change to the last
             frame: probabilities move in logit space, scaled water adds. the last layer starts
             at zero, so an untrained model is persistence (the next frame = the last one).

frames are (B, T, C, H, W) with C = fields x levels (see dataset.py); the first n_prob
channels are probabilities, the rest scaled water contents.
"""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ForecastConfig:
    channels: int = 80  # fields x levels, set from the data
    n_prob: int = 20  # leading probability channels, set from the data
    input_steps: int = 6  # x: previous frames seen (6 = 1 hour at 10 min)
    output_steps: int = 3  # frames forecast (t+1 .. t+3)
    width: int = 64  # channels after the encoder convs
    hidden: list = field(default_factory=lambda: [64, 64])  # ConvLSTM layers
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
        c = self.config = config or ForecastConfig()
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

    def forward(self, x, output_steps=None):
        """x (B, T, C, H, W) previous frames -> (B, output_steps, C, H, W) forecast frames."""
        b, t, _, h, w = x.shape
        size = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        state = [(x.new_zeros(b, cell.c_hid, *size), x.new_zeros(b, cell.c_hid, *size)) for cell in self.cells]
        for k in range(t):
            top = self.step(x[:, k], state)
        last, out = x[:, -1], []
        for k in range(output_steps or self.config.output_steps):
            if k:  # after the first forecast, feed the model its own last frame
                top = self.step(last, state)
            delta = F.interpolate(self.decode(top), size=(h, w), mode="bilinear", align_corners=False)
            last = self.apply_change(last, delta)
            out.append(last)
        return torch.stack(out, dim=1)

    def apply_change(self, frame, delta):
        n = self.config.n_prob
        prob = torch.sigmoid(torch.logit(frame[:, :n].clamp(1e-4, 1 - 1e-4)) + delta[:, :n])
        water = (frame[:, n:] + delta[:, n:]).clamp_min(0)
        return torch.cat([prob, water], dim=1)
