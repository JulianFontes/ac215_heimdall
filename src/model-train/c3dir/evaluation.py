"""evaluate a trained C3DIR model on held-out samples (paper section 5.2) and plot the results.

  table = predict_samples(model, files)   one row per matched voxel: labels and predictions
  detection_scores(table), water_scores(table)
  plot_history, plot_detection, plot_water
  curtain = predict_curtain(model, pair, sample); plot_curtain(pair, curtain), plot_species_curtain(pair, curtain), plot_profiles(pair, curtain)
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr

from c3dir.constants import LEVELS_KM, MASK_THRESHOLD, SPECIES, WATER
from c3dir.geometry import closest_ray, satellite_frame, scan_angles
from c3dir.model import MASK_COLUMNS, WATER_COLUMNS, decode_numpy

def predict_samples(model, files):
    """predictions at every matched voxel of each sample file; returns one DataFrame."""
    tables = []
    for path in files:
        with xr.open_dataset(path) as s:
            sample = s.load()
        probability, water, _ = decode_numpy(model.predict_voxels(sample))
        labels = sample["voxel_labels"].values
        table = pd.DataFrame({"file": path.name, "level": sample["voxel_level"].values,
                              "profile": sample["voxel_profile"].values})
        for k, name in enumerate(SPECIES):
            table[f"{name}_true"] = labels[:, MASK_COLUMNS[k]]
            table[f"{name}_prob"] = probability[:, k]
        for k, name in enumerate(WATER):
            table[f"{name}_true"] = labels[:, WATER_COLUMNS[k]]
            table[f"{name}_pred"] = water[:, k]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _detection(truth, predicted):
    tp, tn = np.sum(truth & predicted), np.sum(~truth & ~predicted)
    fp, fn = np.sum(~truth & predicted), np.sum(truth & ~predicted)
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    f1 = 0.5 * (2 * tp / max(2 * tp + fp + fn, 1) + 2 * tn / max(2 * tn + fn + fp, 1))  # macro F1
    return {"voxels": len(truth), "positive": truth.mean(), "F1": f1, "ACC": (tp + tn) / max(len(truth), 1),
            "TPR": tpr, "TNR": tnr, "BACC": (tpr + tnr) / 2}


def detection_scores(table, by_level=False):
    """paper eqs. 6-10 for each species (and level), probability >= 0.5 counts as detected."""
    rows = []
    groups = table.groupby("level") if by_level else [(None, table)]
    for level, group in groups:
        for name in SPECIES:
            row = _detection(group[f"{name}_true"].values == 1, group[f"{name}_prob"].values >= 0.5)
            rows.append({"species": name, **({"level_km": LEVELS_KM[level]} if by_level else {}), **row})
    return pd.DataFrame(rows)


def water_scores(table):
    """MALE (paper eq. 11), Pearson and Spearman r in log space, where both label and
    prediction are above the cloud threshold. TWC is the sum of the three species."""
    rows = []
    for name in ["twc"] + WATER:
        truth, pred = _water(table, name)
        ok = (truth > MASK_THRESHOLD) & (pred > MASK_THRESHOLD)
        lt, lp = np.log10(truth[ok]), np.log10(pred[ok])
        rows.append({"water": name, "voxels": int(ok.sum()), "MALE": np.mean(np.abs(lp - lt)) if ok.any() else np.nan,
                     "pearson_r": pearsonr(lt, lp)[0] if ok.sum() > 2 else np.nan,
                     "spearman_r": spearmanr(lt, lp)[0] if ok.sum() > 2 else np.nan})
    return pd.DataFrame(rows)


def _water(table, name):
    if name == "twc":
        return (table[[f"{w}_true" for w in WATER]].sum(1).values,
                table[[f"{w}_pred" for w in WATER]].sum(1).values)
    return table[f"{name}_true"].values, table[f"{name}_pred"].values


def curtain_voxels(pair):
    """for every earthcare voxel (profile, level) of a collocated pair, the goes voxel whose line
    of sight passes closest: level, row, col arrays of shape (profile, level)."""
    p = pair["goes_imager_projection"].attrs
    x, y = pair.x.values, pair.y.values
    n_prof, n_lev = pair.sizes["profile"], pair.sizes["level"]
    lat = np.repeat(pair["lat"].values, n_lev)
    lon = np.repeat(pair["lon"].values, n_lev)
    z = np.tile(pair["level"].values * 1000.0, n_prof)
    vx, vy = scan_angles(lat, lon, z, p)
    col = np.clip(np.rint((vx - x[0]) / (x[1] - x[0])), 0, len(x) - 1).astype(int)
    row = np.clip(np.rint((vy - y[0]) / (y[1] - y[0])), 0, len(y) - 1).astype(int)
    row, col, _ = closest_ray(np.stack(satellite_frame(lat, lon, z, p)), z, row, col, x, y, p)
    level = np.tile(np.arange(n_lev), n_prof)
    return level, row, col


def predict_curtain(model, pair, sample):
    """C3DIR at every earthcare voxel of a pair (closest line of sight): probability
    (profile, level, 4: any/ice/liquid/rain) and water content g m^-3 (profile, level, 3)."""
    shape = (pair.sizes["profile"], pair.sizes["level"])
    probability, water, _ = decode_numpy(model.predict_voxels(sample, curtain_voxels(pair)))
    return probability.reshape(*shape, 4), water.reshape(*shape, 3)
