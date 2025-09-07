
from __future__ import annotations
import os, json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def archive(run_dir_base: str, t, X, U, Fw, refs, metrics, meta):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path(run_dir_base) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(outdir / "trace.npz",
                        t=t, X=X, U=U, Fw=Fw,
                        x_ref=refs['x_ref'], theta_ref=refs['theta_ref'])

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "meta": meta}, f, indent=2)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.subplot(3,1,1); plt.plot(t, X[:,0]); plt.grid(True); plt.ylabel('theta [rad]')
    plt.subplot(3,1,2); plt.plot(t, X[:,2]); plt.grid(True); plt.ylabel('x [m]')
    plt.subplot(3,1,3); plt.plot(t, U);      plt.grid(True); plt.ylabel('u [N]'); plt.xlabel('t [s]')
    plt.tight_layout()
    plt.savefig(outdir / "overview.png", dpi=150)
    plt.close()
    print(f"Archive: {outdir}")
