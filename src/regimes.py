"""Méthodes de classification plus simples que le HMM (partie C.4) et mesures de persistance.

- kmeans_regimes       : KMeans à K groupes sur les mêmes variables standardisées
- vol_threshold_regime : règle manuelle "crise si volatilité > son 90e centile historique"
- persistence_table    : fréquence, nombre d'épisodes, durée moyenne, changements par an
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def kmeans_regimes(Z: pd.DataFrame, n_clusters: int = 3, vol_col: str = "vol",
                   seed: int = 0) -> tuple[pd.Series, KMeans]:
    """KMeans sur les observations standardisées ; groupes renumérotés par volatilité croissante
    (même règle d'identification que pour le HMM, pour pouvoir comparer les étiquettes)."""
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(Z.values)
    order = np.argsort(km.cluster_centers_[:, list(Z.columns).index(vol_col)])
    relabel = np.empty(n_clusters, dtype=int)
    relabel[order] = np.arange(n_clusters)          # ancien label -> nouveau label
    return pd.Series(relabel[km.labels_], index=Z.index, name="kmeans"), km


def vol_threshold_regime(vol: pd.Series, q: float = 0.90, expanding: bool = True,
                         min_periods: int = 252) -> pd.Series:
    """1 = crise si la volatilité dépasse son q-ième centile, 0 sinon.

    expanding=True : le centile est calculé sur tout l'historique DISPONIBLE en t
    (règle causale, utilisable en temps réel). expanding=False : centile de tout
    l'échantillon (règle in-sample, uniquement descriptive).
    """
    if expanding:
        thr = vol.expanding(min_periods=min_periods).quantile(q)
    else:
        thr = pd.Series(vol.quantile(q), index=vol.index)
    out = (vol > thr).astype(float)
    out[thr.isna()] = np.nan
    return out.rename("vol_rule")


def spells(labels: pd.Series) -> pd.DataFrame:
    """Découpe une série d'étiquettes en épisodes (suites de jours consécutifs dans le même état)."""
    s = labels.dropna().astype(int)
    new_spell = s.ne(s.shift()).cumsum()
    g = s.groupby(new_spell)
    return pd.DataFrame({"state": g.first(), "start": g.apply(lambda x: x.index[0]),
                         "length": g.size()})


def n_switches_per_year(labels: pd.Series, periods_per_year: int = 252) -> float:
    s = labels.dropna()
    n_switch = int((s != s.shift()).iloc[1:].sum())
    return n_switch / (len(s) / periods_per_year)


def persistence_table(labels: pd.Series) -> pd.DataFrame:
    """Pour chaque état : fréquence d'occurrence, nombre d'épisodes, durée moyenne (jours)."""
    sp = spells(labels)
    s = labels.dropna().astype(int)
    tab = pd.DataFrame({
        "frequence": s.value_counts(normalize=True).sort_index(),
        "n_episodes": sp.groupby("state").size(),
        "duree_moyenne_j": sp.groupby("state")["length"].mean(),
        "duree_mediane_j": sp.groupby("state")["length"].median(),
    })
    tab.index.name = "etat"
    return tab
