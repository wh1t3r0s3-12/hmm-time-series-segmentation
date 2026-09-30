"""Allocation statique de référence pour le backtest (partie D.3).

- max_sharpe_weights : portefeuille long-only de Sharpe maximal (poids >= 0, somme = 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def max_sharpe_weights(excess: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    """Maximise moyenne / écart-type des rendements excédentaires, sous w >= 0 et sum(w) = 1.

    L'optimisation porte sur des estimations bruitées (moyennes surtout) : c'est un
    optimum IN-SAMPLE, rien ne garantit qu'il le reste hors échantillon.
    """
    mu = excess.mean().values
    cov = excess.cov().values
    n = len(mu)

    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w @ cov @ w)

    res = minimize(neg_sharpe, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=excess.columns)
