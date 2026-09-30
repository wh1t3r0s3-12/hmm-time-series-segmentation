"""Calcule (ou recharge depuis le cache) tous les walk-forwards utilisés dans les notebooks.

Usage : python scripts/run_walkforwards.py      (environ 10 minutes au premier lancement)
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.pipeline import TEST_START, get_walk_forward  # noqa: E402

END = "2026-08-31"          # dernière date commune aux prix et aux facteurs de K. French
LONG_START = "2004-01-02"   # premier mois avec 2707 jours d'historique SPY disponibles

RUNS = [
    ("realized", TEST_START, END, True),   # stratégie principale (partie D)
    ("paper", TEST_START, END, False),     # mêmes réglages avec la volatilité du papier
    ("realized", LONG_START, END, True),   # test long 2004-2026 (robustesse)
]

if __name__ == "__main__":
    for kind, start, end, ins in RUNS:
        t = time.time()
        get_walk_forward(kind, start, end, store_insample=ins)
        print(f"{kind:9s} {start} -> {end} : {time.time() - t:6.1f} s", flush=True)
