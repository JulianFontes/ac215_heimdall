"""score forecasts against the C3DIR frames that came next, by lead time, next to persistence
(the last frame repeated), the baseline any nowcast has to beat.

  detection   cloud (prob_any >= 0.5) per voxel: critical success index (CSI), F1, accuracy
  cover       cloudy column (any level cloudy) per lat / lon cell: accuracy
  prob_mae    mean absolute error of prob_any
  water_mae   mean absolute error of each scaled water content (log10(1 + w / 5e-5) / 4)
"""
import numpy as np
import pandas as pd

from heimdall.c3dir_forecast.sequences import FIELDS


def _detection(truth, predicted):
    tp, fp = (truth & predicted).sum(), (~truth & predicted).sum()
    fn, tn = (truth & ~predicted).sum(), (~truth & ~predicted).sum()
    return {"csi": tp / max(tp + fp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1)}


def scores(forecast, target, valid, levels, cadence_minutes=10):
    """one row per lead time; forecast and target (N, steps, C, H, W), C = len(FIELDS) x levels."""
    rows, shape = [], (len(target), len(FIELDS), levels, *target.shape[-2:])
    for lead in range(target.shape[1]):
        f = forecast[:, lead].reshape(shape)[..., valid]  # (N, fields, levels, cells)
        t = target[:, lead].reshape(shape)[..., valid]
        cloud_f, cloud_t = f[:, 0] >= 0.5, t[:, 0] >= 0.5
        row = {"lead_minutes": (lead + 1) * cadence_minutes, **_detection(cloud_t, cloud_f),
               "cover_accuracy": (cloud_f.any(axis=1) == cloud_t.any(axis=1)).mean(),
               "prob_mae": np.abs(f[:, 0] - t[:, 0]).mean()}
        for k, name in enumerate(FIELDS[1:], 1):
            row[f"{name.split('_')[0]}_water_mae"] = np.abs(f[:, k] - t[:, k]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compare(model, dataset, valid, levels, output_steps=None, cadence_minutes=10):
    """scores of the model and of persistence on every example of the dataset."""
    x = np.stack([dataset[k][0].numpy() for k in range(len(dataset))])
    y = np.stack([dataset[k][1].numpy() for k in range(len(dataset))])
    forecast = model.predict(x, output_steps)
    persistence = np.repeat(x[:, -1:], y.shape[1], axis=1)
    table = pd.concat([scores(forecast, y, valid, levels, cadence_minutes).assign(method="convlstm"),
                       scores(persistence, y, valid, levels, cadence_minutes).assign(method="persistence")])
    return table, forecast, y, x
