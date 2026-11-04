"""Shared helpers for the ECG notebooks: loading PTB-XL, the splits, the training loop and the scores."""
import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.stats import rankdata

DATA = Path("data/ptb-xl")
CACHE = Path("data/X100.npy")  # every ECG at 100 samples per second, shape (21799, 1000, 12)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_labels():
    """One row per ECG: the cardiologists' codes, which of the 5 big categories they fall under, and the fold."""
    db = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(DATA / "scp_statements.csv", index_col=0)
    category = scp[scp.diagnostic == 1].diagnostic_class
    db["classes"] = db.scp_codes.apply(lambda codes: sorted({category[c] for c in codes if c in category.index}))
    for c in CLASSES:
        db[c] = db.classes.apply(lambda found: c in found)
    return db


def load_signals(db):
    """Every ECG as one array. Reading 21,799 files is slow, so the first run saves a copy to data/X100.npy."""
    if CACHE.exists():
        return np.load(CACHE)
    X = np.stack([wfdb.rdsamp(str(DATA / f))[0] for f in db.filename_lr]).astype(np.float32)
    np.save(CACHE, X)
    return X


def plot_12lead(X, db, ecg_id, title, path=None, seconds=5):
    """Draw one ECG like 12-lead paper: a big square is 0.2 s wide and 0.5 mV tall."""
    import matplotlib.pyplot as plt

    sig = X[db.index.get_loc(ecg_id)]
    t = np.arange(len(sig)) / 100
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True, sharey=True)
    for k, ax in enumerate(axes.T.flat):  # left column I to aVF, right column V1 to V6
        ax.plot(t, sig[:, k], color="black", lw=0.8)
        ax.set_xticks(np.arange(0, seconds + 0.01, 0.2))
        ax.set_xticks(np.arange(0, seconds + 0.01, 0.04), minor=True)
        ax.set_yticks(np.arange(-2, 2.01, 0.5))
        ax.set_yticks(np.arange(-2, 2.01, 0.1), minor=True)
        ax.grid(which="major", color="#f19c9c", lw=0.6)
        ax.grid(which="minor", color="#fbe0e0", lw=0.3)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_xlim(0, seconds)
        ax.set_ylim(-2, 2)
        ax.text(0.05, 1.45, LEADS[k], fontsize=12, fontweight="bold")
    row = db.loc[ecg_id]
    fig.suptitle(f"{title}   (ECG {ecg_id}, age {row.age:.0f}, codes {row.scp_codes})", fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.show()


def split_masks(db):
    """Folds 1-8 train the model, fold 9 tunes it, fold 10 is the final test and gets used once."""
    return (db.strat_fold <= 8).values, (db.strat_fold == 9).values, (db.strat_fold == 10).values


def mi_vs_normal(db):
    """The first task: ECGs that show an MI (1) or are normal and nothing else (0). Everything else is left out."""
    normal_only = db.classes.apply(lambda c: c == ["NORM"]).values
    keep = db.MI.values | normal_only
    return keep, db.MI.values.astype(np.float32)


def standardize(X, train_mask):
    """Rescale each lead so the training ECGs average 0 with a spread of 1. Models learn faster that way."""
    mean = X[train_mask].mean(axis=(0, 1))
    std = X[train_mask].std(axis=(0, 1))
    return ((X - mean) / std).astype(np.float32)


def auc(y, scores):
    """The chance a random positive ECG scores higher than a random negative one. 0.5 = coin flip, 1.0 = perfect."""
    y = np.asarray(y, bool)
    ranks = rankdata(scores)
    n_pos, n_neg = y.sum(), (~y).sum()
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mean_auc(Y, scores):
    """AUC for each output column, averaged (the "macro AUC" that research papers report)."""
    Y, scores = np.asarray(Y).reshape(len(Y), -1), np.asarray(scores).reshape(len(scores), -1)
    return float(np.mean([auc(Y[:, k], scores[:, k]) for k in range(Y.shape[1])]))


class ECGNet(torch.nn.Module):
    """A small 1D convolutional network. Each block slides little pattern detectors along the 10 seconds
    (a QRS shape, an ST segment that sits too high...), then halves the length so the next block sees
    a wider stretch. At the end it averages over time, so it doesn't matter where in the strip a beat lands."""

    def __init__(self, n_out):
        super().__init__()

        def block(c_in, c_out, k):
            return torch.nn.Sequential(torch.nn.Conv1d(c_in, c_out, k, padding=k // 2), torch.nn.BatchNorm1d(c_out),
                                       torch.nn.ReLU(), torch.nn.MaxPool1d(2))

        self.features = torch.nn.Sequential(block(12, 32, 7), block(32, 64, 5), block(64, 128, 5), block(128, 128, 3))
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(128, n_out))

    def forward(self, x):  # x: (ECGs, 12 leads, 1000 time steps)
        return self.head(self.features(x).mean(dim=2))


@torch.no_grad()
def predict(model, X, batch=512):
    """The model's probabilities (0 to 1) for every ECG in X."""
    model.eval()
    X = torch.from_numpy(X)
    return torch.cat([torch.sigmoid(model(X[i:i + batch])) for i in range(0, len(X), batch)]).numpy()


def train(model, X_tr, Y_tr, X_va, Y_va, epochs=20, lr=1e-3, batch=128, seed=0):
    """Show the model a batch of ECGs, measure how wrong it is (the loss), nudge every weight downhill
    (backprop, same as micrograd), repeat. Keeps the version that scored best on the tuning fold."""
    torch.manual_seed(seed)
    X_tr, Y_tr = torch.from_numpy(X_tr), torch.from_numpy(np.asarray(Y_tr, np.float32).reshape(len(Y_tr), -1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order, total = torch.randperm(len(X_tr)), 0.0
        for i in range(0, len(X_tr), batch):
            idx = order[i:i + batch]
            loss = loss_fn(model(X_tr[idx]), Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        score = mean_auc(Y_va, predict(model, X_va))
        history.append({"epoch": epoch, "loss": total / len(X_tr), "tuning_auc": score})
        print(f"epoch {epoch:2d}   training loss {total / len(X_tr):.3f}   tuning-fold AUC {score:.3f}")
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return pd.DataFrame(history)
