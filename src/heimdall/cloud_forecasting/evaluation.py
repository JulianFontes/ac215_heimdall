"""Loss, CSI / F1 scores, validation pass and score printing."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from tqdm.auto import tqdm

from config import N_CH, CLOUD_PROPERTIES
from data import split_batch

def weighted_bce(logits, y, valid, pos_w):
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w.view(1, -1, 1, 1, 1), reduction="none")
    return (bce * valid).sum() / (valid.sum() * y.shape[1])

def make_scores(device):
    return nn.ModuleDict({prop: nn.ModuleDict(
        {"CSI": BinaryJaccardIndex(ignore_index=-1), 
         "F1": BinaryF1Score(ignore_index=-1)}) for prop in CLOUD_PROPERTIES}).to(device)

def update_scores(scores, prob, y, valid):
    t = torch.where(valid.bool().expand_as(y), y.long(), torch.full_like(y, -1, dtype=torch.long))
    for i, prop in enumerate(CLOUD_PROPERTIES):
        for m in scores[prop].values():
            m.update(prob[:, i], t[:, i])

def report(scores):
    out = {prop: {k: m.compute().item() for k, m in scores[prop].items()} for prop in CLOUD_PROPERTIES}
    for prop in CLOUD_PROPERTIES:
        for m in scores[prop].values():
            m.reset()
    return out

@torch.no_grad()
def evaluate(model, loader, pos_w, device, with_baseline=False, desc="val"):
    model.eval()
    scores, base = make_scores(device), make_scores(device)
    total, n = 0.0, 0
    for w in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        x, y, valid = (a.to(device) for a in split_batch(w))
        logits = model(x)
        total, n = total + weighted_bce(logits, y, valid, pos_w).item(), n + 1
        update_scores(scores, torch.sigmoid(logits), y, valid)
        if with_baseline:
            update_scores(base, x[:, -1, 3:N_CH], y, valid) # persistence: next hour = last observed masks
    return total / n, report(scores), (report(base) if with_baseline else None)