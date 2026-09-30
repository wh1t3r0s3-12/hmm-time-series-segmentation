"""Construction des observations du HMM.

Deux jeux de variables :

1. "paper" : exactement la définition de Wang et al. (2020), section 3.1
   - rendement journalier  r_t = (P_t - P_{t-1}) / P_{t-1}          (en %)
   - "volatilité"          v_t = (1/10) * sum_{i=0..9} (P_{t-i} - MA10_t)^2
     c'est l'erreur quadratique moyenne (MSE) des prix de clôture autour de leur
     moyenne mobile 10 jours, donc une VARIANCE DE PRIX, exprimée en dollars^2.

2. "realized" : l'alternative que je propose (question A.1)
   - même rendement journalier (en %)
   - volatilité réalisée = écart-type des 10 derniers rendements journaliers (en %)
   Elle ne dépend pas du niveau de prix : un marché à 70 $ et un marché à 700 $ qui
   bougent de 1 % par jour ont la même volatilité, ce qui n'est pas le cas avec "paper".

Toutes les variables en t n'utilisent que des prix connus à la clôture de t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOL_WINDOW = 10  # fenêtre de 10 jours comme dans le papier


def daily_return_pct(close: pd.Series) -> pd.Series:
    """Rendement simple journalier en pourcentage."""
    return 100.0 * close.pct_change()


def paper_volatility(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """MSE des prix autour de leur moyenne mobile sur `window` jours (définition du papier).

    Remarque : c'est la variance "population" (ddof=0) du prix sur la fenêtre,
    d'où l'unité en $^2 qui dépend fortement du niveau de l'indice.
    """
    return close.rolling(window).var(ddof=0)


def realized_volatility(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Écart-type glissant des rendements journaliers (en %), sans annualisation."""
    return daily_return_pct(close).rolling(window).std(ddof=1)


def build_features(close: pd.Series, kind: str = "paper", window: int = VOL_WINDOW) -> pd.DataFrame:
    """Renvoie un DataFrame (ret, vol) sans valeurs manquantes.

    kind = "paper"    -> vol = MSE des prix (dollars^2)
    kind = "realized" -> vol = écart-type réalisé des rendements (%)
    kind = "logrv"    -> vol = log de la volatilité réalisée (plus proche d'une gaussienne)
    """
    ret = daily_return_pct(close)
    if kind == "paper":
        vol = paper_volatility(close, window)
    elif kind == "realized":
        vol = realized_volatility(close, window)
    elif kind == "logrv":
        vol = np.log(realized_volatility(close, window))
    else:
        raise ValueError(f"kind inconnu : {kind}")
    return pd.DataFrame({"ret": ret, "vol": vol}).dropna()


class Standardizer:
    """Centrage-réduction dont les paramètres sont estimés sur la période d'entraînement SEULEMENT.

    Estimer la moyenne et l'écart-type sur tout l'échantillon (test inclus) serait une
    petite fuite d'information : on utiliserait la volatilité future pour normaliser le passé.
    """

    def fit(self, X: pd.DataFrame | np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(Z, index=X.index, columns=X.columns)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z) * self.std_ + self.mean_
