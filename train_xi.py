#!/usr/bin/env python3
"""Train xi mode only on 1000-sample dataset."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from src.dataset_generation import load_dataset
from src.rfb_bubble import KANBubble1D

NAME = "rfb_1k"
N_HIDDEN = 10
N_EPOCHS = 400
LR = 1e-3
BATCH_SIZE = 128
PATIENCE = 100
N_PTS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ds = load_dataset(NAME)
xi_f = ds["train"]["constant"]["xi"]
xi_s = np.linspace(0, 1, N_PTS)
xi_t = torch.tensor(xi_s, dtype=torch.float32, device=device)

for mode in ["xi"]:
    print(f"\n{'='*50}\n{mode}\n{'='*50}")
    tr, va, te = ds["train"][mode], ds["val"][mode], ds["test"][mode]
    b_t = np.array([np.interp(xi_s, xi_f, tr["b"][i]) for i in range(tr["b"].shape[0])])
    b_v = np.array([np.interp(xi_s, xi_f, va["b"][i]) for i in range(va["b"].shape[0])])

    p_t = torch.tensor(tr["pe"], dtype=torch.float32, device=device)
    r_t = torch.tensor(tr["rho"], dtype=torch.float32, device=device)
    b_t = torch.tensor(b_t, dtype=torch.float32, device=device)
    p_v = torch.tensor(va["pe"], dtype=torch.float32, device=device)
    r_v = torch.tensor(va["rho"], dtype=torch.float32, device=device)
    b_v = torch.tensor(b_v, dtype=torch.float32, device=device)

    model = KANBubble1D(n_hidden=N_HIDDEN, n_grid=8, spline_order=3).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    n_t = p_t.shape[0]
    n_v = p_v.shape[0]
    idx = np.arange(n_t)
    best = float('inf'); best_st = None; wait = 0
    t0 = time.time()

    for ep in range(N_EPOCHS):
        model.train()
        np.random.shuffle(idx)
        el = 0.
        for s in range(0, n_t, BATCH_SIZE):
            bi = idx[s:s+BATCH_SIZE]
            bs = len(bi)
            opt.zero_grad()
            xi_e = xi_t.unsqueeze(0).expand(bs, -1).reshape(-1)
            pe_e = p_t[bi].unsqueeze(1).expand(-1, N_PTS).reshape(-1)
            r_e = r_t[bi].unsqueeze(1).expand(-1, N_PTS).reshape(-1)
            pr = model(xi_e, pe_e, r_e).reshape(bs, N_PTS)
            F.mse_loss(pr, b_t[bi]).backward()
            opt.step()
            el += F.mse_loss(pr.detach(), b_t[bi]).item() * bs
        el /= n_t

        model.eval()
        with torch.no_grad():
            xi_v = xi_t.unsqueeze(0).expand(n_v, -1).reshape(-1)
            pv = p_v.unsqueeze(1).expand(-1, N_PTS).reshape(-1)
            rv = r_v.unsqueeze(1).expand(-1, N_PTS).reshape(-1)
            pr_v = model(xi_v, pv, rv).reshape(n_v, N_PTS)
            vl = F.mse_loss(pr_v, b_v).item()
        sched.step()

        if vl < best - 1e-10:
            best = vl; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop ep {ep+1}"); break

        if (ep+1) % 100 == 0:
            print(f"  ep {ep+1:3d}  train={el:.4e}  val={vl:.4e}  best={best:.4e}  t={time.time()-t0:.0f}s")

    print(f"  Best val: {best:.4e}  ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_st); model.eval()
    xi_ft = torch.tensor(xi_f, dtype=torch.float32, device=device)
    p_te = torch.tensor(te["pe"], dtype=torch.float32, device=device)
    r_te = torch.tensor(te["rho"], dtype=torch.float32, device=device)
    n_te = p_te.shape[0]; n_fd = len(xi_f)
    with torch.no_grad():
        xi_et = xi_ft.unsqueeze(0).expand(n_te, -1).reshape(-1)
        p_et = p_te.unsqueeze(1).expand(-1, n_fd).reshape(-1)
        r_et = r_te.unsqueeze(1).expand(-1, n_fd).reshape(-1)
        pr_t = model(xi_et, p_et, r_et).reshape(n_te, n_fd)
        b_te_t = torch.tensor(te["b"], dtype=torch.float32, device=device)
        te_l = F.mse_loss(pr_t, b_te_t).item()
        rms = np.sqrt(np.mean((pr_t.cpu().numpy() - te["b"])**2, axis=1))
    print(f"  Test MSE: {te_l:.4e}")
    print(f"  Test RMSE: mean={rms.mean():.4e}  median={np.median(rms):.4e}  max={rms.max():.4e}")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(best_st, f"models/kan_bubble_{mode}_{NAME}.pt")
    print(f"  Saved models/kan_bubble_{mode}_{NAME}.pt")
