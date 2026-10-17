"""Settings to change between runs. Fixed values (gradient clipping, seed) live in the module that uses them."""
BANDS = tuple(range(1, 17))    # ABI bands used as model inputs (1-16); visible bands 1-6 are only usable in daylight
TARGET_BANDS = BANDS           # bands to forecast, must be a subset of BANDS
T_IN = 6                       # input frames (30 min)
T_OUT = 12                     # forecast frames, predicted one after another (1 hour)
STEP_MIN = 5                   # minutes between frames (CONUS scan): labels lead times and sets the gap that breaks a sequence
HIDDEN = (32, 32)              # hidden channels per ConvLSTM layer
VAL_FRAC = 0.2                 # chronological split: first 60% train, next 20% validation, last 20% test
TEST_FRAC = 0.0                # each split needs >= T_IN + T_OUT consecutive frames
PLOT_EVERY_MIN = 10            # lead times shown in the forecast and bar plots: +10, +20, ... min

N_CH = len(BANDS)                                   # model input channels
TARGET_IDX = [BANDS.index(b) for b in TARGET_BANDS] # positions of the target bands among the input channels
