"""constants shared by every part of C3DIR (labels, collocation, model, plots)."""
import numpy as np

LEVELS_KM = np.arange(80) * 0.25 + 0.125  # 80 levels every 0.25 km (paper section 2.3)
MASK_THRESHOLD = 1e-5  # g m^-3, cloud / no-cloud cutoff (paper section 2.2)
BANDS = [f"CMI_C{band:02d}" for band in range(1, 17)]  # all 16 abi bands
GOES19_LON = -75.2  # goes-19 sub-satellite longitude
VIEW_RADIUS_DEG = 65.0  # earth-center angle from the sub-satellite point counted as visible (~73° viewing
# angle); an assumption to limit the data, not directly from the paper

# the 9 C3DIR targets (paper sections 2.2-2.3), in the order used everywhere downstream
WATER = ["iwc", "lwc", "rwc"]  # g m^-3
RADII = ["ice_radius", "liquid_radius"]  # um
MASKS = ["ice_mask", "liquid_mask", "rain_mask", "hydrometeor_mask"]  # 0/1
TARGETS = WATER + RADII + MASKS

# model output order: masks any / ice / liquid / rain, water ice / liquid / rain, radii ice / liquid
SPECIES = ["any", "ice", "liquid", "rain"]
PHASES = ["ice", "liquid", "rain"]
