"""Shared constants and fixed settings"""
CLOUD_PROPERTIES = ["ice", "liquid", "rain"]
S0 = 5e-5                      # log scaling of water content, as in the C3DIR paper: log10(1 + w / s)
N_CH = 6                       # model input channels: 3 scaled water contents + 3 masks
ALT_MAX = 16.0                 # crop the volume below this altitude (km)
T_IN = 6                       # input hours
HIDDEN = (16, 16)              # hidden channels per ConvLSTM layer
VAL_FRAC = 0.2                 # last fraction of the time axis held out for validation (80/20)
K = 1                          # number of one-hour-ahead forecasts in a row
SEED = 0