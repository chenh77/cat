"""experiments/common.py

Paths, data loaders and plot style shared by the experiment scripts.

Data (under `data/` at the repository root):
  - `posteriordb/`: the posteriordb datasets of the IOHMM and LDA targets;
  - `reference/`: long-NUTS reference draws of SUR, IOHMM and LDA (the LDA pool is split
    into one file per chain);
  - `sur/`: the frozen SUR dataset and the three local maxima of the SUR posterior.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import zipfile

import numpy as np

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXPERIMENTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
FIGURES_DIR = os.path.join(EXPERIMENTS_DIR, "figures")

TARGETS = ("skewnormal", "sur", "iohmm", "lda")


# ---------------- results ----------------
def benchmark_dir(target: str) -> str:
    """Directory of the benchmark runs of `target` (one `seed_<seed>/` per run)."""
    return os.path.join(RESULTS_DIR, "benchmark", target)


def seed_dirs(results_dir: str) -> list[str]:
    dirs = sorted(glob.glob(os.path.join(results_dir, "seed_*")),
                  key=lambda p: int(os.path.basename(p).replace("seed_", "")))
    if not dirs:
        raise FileNotFoundError(f"no seed_* subdirectories found under {results_dir}")
    return dirs


def npz_key(label: str) -> str:
    return label.replace(" ", "_")


def json_safe(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "tolist"):
        return json_safe(obj.tolist())
    return obj


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2)


# ---------------- data ----------------
def posteriordb_data(name: str) -> dict:
    """The posteriordb dataset `name` (e.g. "three_men1")."""
    with zipfile.ZipFile(os.path.join(DATA_DIR, "posteriordb", f"{name}.json.zip")) as zf:
        with zf.open(f"{name}.json") as f:
            return json.load(f)


def build_iohmm():
    """The IOHMM regression posterior (posteriordb `iohmm_reg_simulated-iohmm_reg`)."""
    from src.targets import IohmmReg
    data = posteriordb_data("iohmm_reg_simulated")
    return IohmmReg(u=np.array(data["u"]), y=np.array(data["y"]), K=data["K"])


def build_lda():
    """The LDA posterior (posteriordb `three_men1-ldaK2`, prior-Gaussian coordinates)."""
    from src.targets import LdaK2
    data = posteriordb_data("three_men1")
    return LdaK2(w=np.array(data["w"]), doc=np.array(data["doc"]), V=data["V"], M=data["M"], K=2,
                 parameterization="prior_gaussian")


def build_sur_case2(T=10, M=5, k_per_eq=1, seed=300):
    """SUR Case 2 data (Y, X_list, beta_true, Sigma_true), loaded from the frozen file the
    NUTS reference was run on (regenerating it from `seed` depends on the LAPACK build)."""
    dat = np.load(os.path.join(DATA_DIR, "sur", f"sur_case2_T{T}_M{M}_k{k_per_eq}_seed{seed}.npz"))
    return dat["Y"], list(dat["X"]), list(dat["beta_true"]), dat["Sigma_true"]


SUR_MODES_PATH = os.path.join(DATA_DIR, "sur", "sur_ref_modes.npz")
SUR_REFERENCE_PATH = os.path.join(DATA_DIR, "reference", "ref_sur_case2.npz")
IOHMM_REFERENCE_PATH = os.path.join(DATA_DIR, "reference", "ref_iohmm_reg_simulated-iohmm_reg.npz")


def lda_reference_stan() -> np.ndarray:
    """LDA NUTS reference in Stan's unconstrained coordinates, (4 * 20000, 502), chain-major
    (chains 0-1 in one topic-label basin, 2-3 in the other)."""
    return np.concatenate([np.load(os.path.join(DATA_DIR, "reference", f"ref_three_men1-ldaK2_chain{c}.npz"))["draws"]
                           for c in range(4)])


# ---------------- plot style ----------------
TITLE_COLOR = "#1F3864"   # bold navy panel titles
RC = {
    "font.family": "sans-serif", "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.8, "axes.edgecolor": "#444444", "axes.grid": True,
    "grid.linestyle": "--", "grid.linewidth": 0.5, "grid.alpha": 0.35, "svg.fonttype": "none",
}
# print-size style: figures are drawn at the width they are printed at (one-column A4,
# ~6.3 in), so the font sizes below are the sizes on paper
RC_PRINT = {**RC, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
            "axes.linewidth": 0.6, "grid.linewidth": 0.4, "xtick.major.width": 0.6,
            "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2}


def save_figure(fig, path):
    """Save `fig` as `path`.png and `path`.pdf."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".png", bbox_inches="tight", dpi=150)
    fig.savefig(path + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {path}.png/.pdf")


def grid_shape(n: int, max_cols: int = 4) -> tuple[int, int]:
    """(rows, cols) of a panel grid for n panels."""
    cols = min(n, max_cols)
    return math.ceil(n / cols), cols
