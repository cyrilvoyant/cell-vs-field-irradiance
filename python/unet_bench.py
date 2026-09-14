"""U-Net forecaster: the spatial reference the comparison was missing.

WHY THIS EXISTS. Two independent readers of the manuscript made the same
objection and it is the strongest one against the paper: a study that concludes
"no spatial representation beats a per-pixel model" while testing only models
whose first layer is DENSE has measured its own architecture, not the value of
spatial information. A dense layer costs a parameter per input, so a whole-field
model is forced to be enormous and loses on frugality before it starts. A
convolution shares its weights, so its parameter count does not grow with the
grid at all. That is precisely the architecture the objection is about, and
until it is run the conclusion has to be narrowed to dense models.

WHY A U-NET AND NOT A CONVOLUTIONAL LSTM. Both share weights and both answer
the objection; the U-Net is the cheaper and the more standard of the two here.
The twenty-four input hours enter as CHANNELS and the twenty-four output hours
leave as channels, so the whole time axis is processed in parallel -- a
recurrent network unrolls it, which on a CPU costs several times more for the
same parameter count and adds truncation and initialisation choices nobody
would tune here. It is also what the nowcasting literature actually uses on
stacks of images.

TWO LEVELS, NOT FOUR. The domain is 32 by 32. Three downsampling steps reach
4 by 4, where a convolution sees the whole island and the architecture stops
being convolutional in any useful sense. Two levels, 32 to 16 to 8, with a base
of sixteen filters, put the parameter count near a hundred thousand -- the
budget of the per-pixel machine this arm has to beat. That is a deliberate
match: the comparison is at equal storage, not at equal capacity.

ONE STANDARD CONFIGURATION, NOT TUNED, exactly as for every other arm. Adam at
1e-3, fifteen epochs, batch 32, seed fixed, mean squared error on the support
cells. Nothing is searched. An arm tuned when the others were not would answer
a different question.

METRICS ARE NOT COMPUTED HERE. This script writes predictions; every error,
mask and reference is evaluated once, in MATLAB, on the same targets and the
same scored cells as every other arm in the study.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def load_mat(path):
    """Read a .mat whatever version MATLAB wrote it in.

    The handover volumes are gigabytes, so the MATLAB side saves them with
    -v7.3, which is HDF5 and which scipy.io.loadmat refuses outright. The
    fallback is h5py, with one subtlety that silently corrupts the data if it
    is missed: MATLAB is column-major and writes its dimensions REVERSED into
    the HDF5 descriptors, so an (origins, lags, rows, cols) array comes back
    shaped (cols, rows, lags, origins). Reversing every axis restores it. A
    reader that only transposed the last two would have trained the network on
    a volume whose origins and columns were swapped, and nothing would have
    raised.
    """
    try:
        return {k: v for k, v in loadmat(path).items() if not k.startswith('__')}
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                out[k] = np.asarray(f[k]).T
        return out


def build(n_in: int, n_out: int, base: int):
    """A two-level U-Net over a small square grid."""
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = block(n_in, base)
            self.e2 = block(base, base * 2)
            self.b = block(base * 2, base * 4)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.d2 = block(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.d1 = block(base * 2, base)
            self.out = nn.Conv2d(base, n_out, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            c1 = self.e1(x)
            c2 = self.e2(self.pool(c1))
            b = self.b(self.pool(c2))
            d2 = self.d2(torch.cat([self.u2(b), c2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), c1], 1))
            return self.out(d1)

    return UNet()


def run(Xtr, Ytr, Xte, mask, base, epochs, batch, lr, seed, log):
    """Xtr: (n, L, H_grid, W_grid); Ytr: (n, H, H_grid, W_grid)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, __import__("os").cpu_count() // 2))

    n, n_in, gh, gw = Xtr.shape
    n_out = Ytr.shape[1]
    net = build(n_in, n_out, base)
    npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"    U-Net base {base}, {n_in} in, {n_out} out, {npar} parameters")

    # standardised on the training split only, one scalar for the whole field:
    # a per-cell standardisation would give the network the climatology for
    # free and is not what the other arms receive.
    mu = float(Xtr.mean())
    sd = float(Xtr.std()) or 1.0

    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Ytr - mu) / sd, dtype=torch.float32)
    W = torch.tensor(mask.astype(np.float32))          # (gh, gw)
    W = W / W.mean()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    per_epoch = []
    for ep in range(epochs):
        te = time.time()
        perm = torch.randperm(n)
        run_loss = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            pred = net(Xt[b])
            # THE LOSS IS RESTRICTED TO THE SUPPORT. Three hundred of the
            # thousand cells are sea, where the product reports nothing; a
            # network rewarded for predicting zeros there would spend capacity
            # on a constant.
            l = (((pred - Yt[b]) ** 2) * W).mean()
            l.backward()
            opt.step()
            run_loss += l.item() * len(b)
        per_epoch.append(time.time() - te)
        log(f"    epoch {ep+1:2d}/{epochs}  loss {run_loss/n:.5f}  "
            f"{per_epoch[-1]:.1f} s")

    net.eval()
    with torch.no_grad():
        Xs = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
        out = np.concatenate(
            [net(Xs[i:i + 64]).numpy() for i in range(0, len(Xs), 64)], 0)
    pred = out * sd + mu

    return pred, {
        "params": {"total": int(npar)},
        "base": base,
        "n_in": int(n_in),
        "n_out": int(n_out),
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
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    d = load_mat(a.infile)
    Xtr = np.asarray(d["Xtr"], dtype=np.float64)
    Ytr = np.asarray(d["Ytr"], dtype=np.float64)
    Xte = np.asarray(d["Xte"], dtype=np.float64)
    mask = np.asarray(d["mask"], dtype=np.float64)

    def log(s):
        print(s, flush=True)

    log(f"  U-Net on {Xtr.shape[0]} training windows of "
        f"{Xtr.shape[1]}x{Xtr.shape[2]}x{Xtr.shape[3]}")
    pred, info = run(Xtr, Ytr, Xte, mask, a.base, a.epochs, a.batch,
                     a.lr, a.seed, log)
    # SINGLE PRECISION ON THE WAY BACK. The prediction volume is origins by
    # horizons by rows by columns, about 1.7 GB in double, which is close
    # enough to the v5 format ceiling to be a risk for no benefit: the
    # network computed it in single and MATLAB casts it to double on
    # arrival anyway.
    savemat(a.outfile, {"pred": np.asarray(pred, dtype=np.float32),
                        "info": json.dumps(info)},
            do_compression=True)
    log(f"  written {a.outfile}  ({info['train_seconds']:.0f} s, "
        f"{info['params']['total']} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
