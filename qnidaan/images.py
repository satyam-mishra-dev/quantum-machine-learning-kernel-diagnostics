"""Image track: hybrid CNN + quantum head on PneumoniaMNIST (MedMNIST v2).

Dressed-circuit transfer learning (Mari et al. 2020): a small CNN trunk
compresses 28x28 chest X-rays to n_qubits features, a VQC head classifies.
The classical twin is the identical trunk with a linear head — same params
budget, same training schedule, fair fight.

Usage: python -m qnidaan.images            # trains both, writes runs/ JSON
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
N_QUBITS = 4
EPOCHS = 5
BATCH = 128


def loaders():
    from medmnist import PneumoniaMNIST

    DATA_DIR.mkdir(exist_ok=True)
    tf = lambda img: torch.tensor(
        np.asarray(img, dtype=np.float32)[None] / 255.0)
    tr = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR),
                        transform=tf)
    te = PneumoniaMNIST(split="test", download=True, root=str(DATA_DIR),
                        transform=tf)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=BATCH, shuffle=sh)
    return mk(tr, True), mk(te, False)


class Trunk(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 7 * 7, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def quantum_head():
    import pennylane as qml

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(torch.tanh(inputs) * np.pi / 2,
                           wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    shape = qml.StronglyEntanglingLayers.shape(3, N_QUBITS)
    return qml.qnn.TorchLayer(circuit, {"weights": shape})


def build(kind):
    torch.manual_seed(SEED)
    if kind == "hybrid_qnn":
        head = nn.Sequential(quantum_head(), nn.Unflatten(0, (-1, 1)))
        return nn.Sequential(Trunk(N_QUBITS), head)
    # classical twin: same trunk width, linear head
    return nn.Sequential(Trunk(N_QUBITS), nn.Linear(N_QUBITS, 1))


def train_eval(kind, train_dl, test_dl):
    model = build(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            out = model(xb).squeeze(-1)
            loss = lossf(out, yb.float().squeeze(-1))
            loss.backward()
            opt.step()
        print(f"  {kind} epoch {epoch+1}/{EPOCHS} loss={loss.item():.3f}",
              flush=True)
    model.eval()
    scores, ys = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            scores.append(model(xb).squeeze(-1))
            ys.append(yb.squeeze(-1))
    s, y = torch.cat(scores).numpy(), torch.cat(ys).numpy()
    return {
        "auroc": float(roc_auc_score(y, s)),
        "accuracy": float(((s > 0).astype(int) == y).mean()),
        "train_seconds": round(time.time() - t0, 1),
        "n_test": int(len(y)),
    }


if __name__ == "__main__":
    train_dl, test_dl = loaders()
    result = {
        "dataset": {
            "name": "pneumoniamnist",
            "title": "PneumoniaMNIST chest X-rays (MedMNIST v2)",
            "n_features": 784,
            "feature_names": [],
        },
        "budget": {"n_qubits": N_QUBITS, "method": "cnn_trunk",
                   "explained_variance": None},
        "heldout": {},
        "image_track": True,
    }
    for kind in ["cnn_classical", "hybrid_qnn"]:
        result["heldout"][kind] = train_eval(kind, train_dl, test_dl)
        print(f"{kind}: {result['heldout'][kind]}", flush=True)
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / "pneumoniamnist.json").write_text(
        json.dumps(result, indent=2))
    print("wrote runs/pneumoniamnist.json")
