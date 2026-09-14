"""LSTM forecaster: the arm in which compressing the input actually pays.

WHY THIS EXISTS, AND WHAT IT IS MEANT TO SHOW.

An extreme learning machine draws its first layer at random and never trains
it, so the cost of fitting is a single pass to accumulate H'H plus one
N_h x N_h solve, and neither depends on the input dimension. In that
architecture there is no computational reason to compress anything, which is
exactly why the raw field beat every latent representation in our own
measurements. The compression argument is not wrong there; it is empty there.

For a model trained by backpropagation the input dimension drives four
distinct costs that the ELM does not pay:

  1  the number of TRAINED parameters in the input-to-hidden block,
  2  the memory of the stored activations during the backward pass,
  3  the time per epoch,
  4  the sample complexity, since more free parameters need more data.

This file measures those costs on the same data, the same splits and the same
metric as the MATLAB bench, for the raw field and for a compressed input of
the same series. Accuracy is reported, but the point is the cost.

THE COUNTER-ARGUMENT, STATED HERE RATHER THAN LEFT TO A REFEREE. A
convolutional recurrent network would weaken this case considerably: weight
sharing decouples the parameter count from the input size, so a ConvLSTM over
the full field can be small. The claim defended here is therefore precise, and
narrower than it may look: compression pays for models whose FIRST LAYER IS
DENSE AND LEARNED. It does not pay for an ELM, whose first layer is not
learned, and it pays much less for a convolutional model, whose first layer is
shared. Both limits belong in the paper.

METRICS ARE NOT COMPUTED HERE. This script writes predictions; the errors,
masks and references are evaluated once, in MATLAB, exactly as for every other
arm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def build(d_in: int, d_out: int, hidden: int, layers: int):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, d_out)

        def forward(self, x):            # x: (batch, steps, d_in)
            y, _ = self.lstm(x)
            return self.head(y[:, -1, :])

    return Net()


def count(model) -> dict:
    """Trained parameters, split so that the input-dependent part is visible."""
    import torch.nn as nn

    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ih = sum(p.numel() for n, p in model.named_parameters() if "weight_ih" in n)
    return {"total": tot, "input_block": ih, "rest": tot - ih}


def run(Xtr, Ytr, Xte, hidden, layers, epochs, batch, lr, seed, log):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, __import__("os").cpu_count() // 2))

    n, steps, d_in = Xtr.shape
    d_out = Ytr.shape[1]
    net = build(d_in, d_out, hidden, layers)
    par = count(net)

    # standardise on the training split only
    mu = Xtr.reshape(-1, d_in).mean(0)
    sd = Xtr.reshape(-1, d_in).std(0)
    sd[sd < 1e-8] = 1.0
    ymu, ysd = Ytr.mean(0), Ytr.std(0)
    ysd[ysd < 1e-8] = 1.0

    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Ytr - ymu) / ysd, dtype=torch.float32)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    t0 = time.time()
    per_epoch = []
    for ep in range(epochs):
        te = time.time()
        perm = torch.randperm(n)
        run_loss = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            l = lossf(net(Xt[b]), Yt[b])
            l.backward()
            opt.step()
            run_loss += l.item() * len(b)
        per_epoch.append(time.time() - te)
        log(f"    epoch {ep+1:2d}/{epochs}  loss {run_loss/n:.5f}  {per_epoch[-1]:.1f} s")

    net.eval()
    with torch.no_grad():
        Xs = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
        out = np.concatenate(
            [net(Xs[i:i + 256]).numpy() for i in range(0, len(Xs), 256)], 0)
    pred = out * ysd + ymu

    return pred, {
        "params": par,
        "d_in": int(d_in),
        "d_out": int(d_out),
        "hidden": hidden,
        "layers": layers,
        "epochs": epochs,
        "batch": batch,
        "train_seconds": time.time() - t0,
        "seconds_per_epoch": float(np.median(per_epoch)),
        "n_train": int(n),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    d = loadmat(a.infile)
    Xtr, Ytr, Xte = (np.asarray(d[k], dtype=np.float64) for k in ("Xtr", "Ytr", "Xte"))
    steps = int(np.atleast_1d(d["steps"]).ravel()[0])
    Xtr = Xtr.reshape(Xtr.shape[0], steps, -1)
    Xte = Xte.reshape(Xte.shape[0], steps, -1)

    def log(s):
        print(s, flush=True)

    log(f"LSTM: {Xtr.shape[0]} train sequences of {steps} steps x {Xtr.shape[2]} inputs")
    pred, info = run(Xtr, Ytr, Xte, a.hidden, a.layers, a.epochs, a.batch,
                     a.lr, a.seed, log)
    savemat(a.outfile, {"pred": pred, "info": json.dumps(info)}, do_compression=False)
    log(f"trained parameters {info['params']['total']:,} "
        f"of which {info['params']['input_block']:,} in the input block")
    log(f"{info['seconds_per_epoch']:.1f} s per epoch, "
        f"{info['train_seconds']/60:.1f} min total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
