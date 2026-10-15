"""Settings to change between runs. Fixed values (data quality threshold, gradient clipping, seed) live in the module that uses them."""
BANDS = tuple(range(1, 17))    # ABI bands used as model inputs (1-16); visible bands 1-6 are only usable in daylight
TARGET_BANDS = BANDS           # bands to forecast, must be a subset of BANDS
T_IN = 3                       # input frames
T_OUT = 2                      # forecast frames, predicted one after another
STEP_MIN = 5                   # minutes between frames (CONUS scan): labels lead times and sets the gap that breaks a sequence
HIDDEN = (32, 32)              # hidden channels per ConvLSTM layer
VAL_FRAC = 0.25                # chronological split: first 50% train, next 25% validation, last 25% test
TEST_FRAC = 0.25               # each split needs >= T_IN + T_OUT consecutive frames
