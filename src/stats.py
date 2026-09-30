"""Indicateurs de performance (fonction `performance_stats`).

Indicateurs calculés : rendement annualisé, volatilité annualisée, ratio de Sharpe,
perte maximale (max drawdown), pourcentage de mois positifs, rotation (turnover).
J'ajoute l'erreur-type du Sharpe (utile pour la critique : un Sharpe estimé sur 2,5 ans
est très imprécis) et, en option, le ratio d'information par rapport à un benchmark.

Conventions : rendements SIMPLES journaliers en décimal (0.01 = 1 %), 252 jours par an.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252


def max_drawdown(returns: pd.Series) -> float:
    """Plus forte baisse depuis un plus haut, en valeur positive (0.34 = -34 %)."""
    wealth = (1 + returns.fillna(0)).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(-dd.min())


def sharpe_ratio(returns: pd.Series, rf: pd.Series | float = 0.0, periods: int = PERIODS) -> float:
    """Sharpe annualisé = moyenne(r - rf) / écart-type(r - rf) * sqrt(252)."""
    ex = (returns - rf).dropna()
    return float(ex.mean() / ex.std() * np.sqrt(periods))


def sharpe_standard_error(sharpe: float, n_years: float) -> float:
    """Erreur-type approchée du Sharpe annualisé pour des rendements iid (Lo, 2002) :
    se(SR) ~ sqrt((1 + SR^2 / 2) / T_années)."""
    return float(np.sqrt((1 + 0.5 * sharpe ** 2) / n_years))


def turnover(weights: pd.DataFrame, legs: dict | None = None, periods: int = PERIODS) -> float:
    """Rotation annualisée = moyenne de sum_i legs_i |w_i,t - w_i,t-1| * 252.

    `legs` permet de compter 2 jambes pour un facteur long/short (on traite la jambe
    longue ET la jambe courte). 1 = on remplace 100 % du portefeuille une fois par an.
    """
    dw = weights.diff().abs()
    if legs is not None:
        dw = dw * pd.Series(legs).reindex(weights.columns).fillna(1.0)
    return float(dw.sum(axis=1).mean() * periods)


def performance_stats(returns: pd.Series, rf: pd.Series | float = 0.0,
                      weights: pd.DataFrame | None = None, legs: dict | None = None,
                      benchmark: pd.Series | None = None, periods: int = PERIODS) -> pd.Series:
    """Tableau d'indicateurs pour une série de rendements journaliers."""
    r = returns.dropna()
    if isinstance(rf, pd.Series):
        rf = rf.reindex(r.index).fillna(0.0)
    n_years = len(r) / periods
    wealth = (1 + r).cumprod()
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    sr = sharpe_ratio(r, rf, periods)

    out = {
        "rendement_annualise": wealth.iloc[-1] ** (1 / n_years) - 1,   # CAGR
        "volatilite_annualisee": r.std() * np.sqrt(periods),
        "sharpe": sr,
        "sharpe_se": sharpe_standard_error(sr, n_years),
        "max_drawdown": max_drawdown(r),
        "pct_mois_positifs": float((monthly > 0).mean()),
        "rendement_total": wealth.iloc[-1] - 1,
        "n_annees": n_years,
    }
    if weights is not None:
        out["turnover_annuel"] = turnover(weights.reindex(r.index).ffill().fillna(0.0), legs, periods)
    if benchmark is not None:
        active = (r - benchmark.reindex(r.index)).dropna()
        out["ratio_information"] = float(active.mean() / active.std() * np.sqrt(periods))
    return pd.Series(out)


def stats_table(returns: dict[str, pd.Series], rf: pd.Series | float = 0.0,
                weights: dict[str, pd.DataFrame] | None = None, legs: dict | None = None,
                benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Applique performance_stats à plusieurs stratégies -> une ligne par stratégie."""
    weights = weights or {}
    rows = {name: performance_stats(r, rf, weights.get(name), legs, benchmark)
            for name, r in returns.items()}
    return pd.DataFrame(rows).T
