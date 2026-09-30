"""Style commun des figures (pour que toutes les figures du rendu soient homogènes).

- Régimes : ils sont ORDONNÉS par volatilité (0 = calme ... 2 = crise), donc une rampe
  bleue du clair au foncé : plus c'est foncé, plus le marché est agité.
- Séries (stratégies, actifs) : palette catégorielle dans un ordre fixe, toujours la même
  couleur pour la même stratégie d'une figure à l'autre.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from .data import ROOT

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"

REGIME_COLORS = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}
REGIME_NAMES = {0: "calme", 1: "intermédiaire", 2: "crise"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHADE = "#c3c2b7"    # gris neutre pour surligner des périodes (crise) derrière des courbes


def set_style():
    mpl.rcParams.update({
        "figure.figsize": (10, 4.2), "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False,
        "lines.linewidth": 1.5, "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "font.size": 10,
    })


def savefig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")


def save_table(df, name: str, floatfmt: str = ".3f"):
    """Sauve un tableau en CSV (données) et en Markdown (pour le README)."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv")
    (TAB_DIR / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt), encoding="utf-8")


def shade_regime(ax, regime, state: int = 2, color: str = SHADE, alpha: float = 0.45, label=None):
    """Surligne en fond les périodes où `regime == state` (une seule entrée de légende)."""
    s = (regime == state).astype(int)
    starts = s.index[(s.diff() == 1) | ((s == 1) & (s.index == s.index[0]))]
    ends = s.index[(s.diff() == -1)]
    ends = list(ends) + ([s.index[-1]] if s.iloc[-1] == 1 else [])
    for i, (a, b) in enumerate(zip(starts, ends)):
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, label=label if i == 0 else None)
