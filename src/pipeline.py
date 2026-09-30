"""Fonctions "haut niveau" partagées par les notebooks (données prêtes à l'emploi + cache).

Les walk-forwards prennent quelques minutes : leurs résultats sont mis en cache dans
results/cache/ pour que les notebooks se relancent vite. Supprimer ce dossier force
un recalcul complet.
"""
from __future__ import annotations

import pickle
from dataclasses import asdict

import pandas as pd

from .backtest import WFConfig, walk_forward_regimes
from .data import ROOT, load_french, load_prices
from .features import build_features

CACHE = ROOT / "results" / "cache"

# Dates de l'étude (papier : entraînement janv. 2007 -> sept. 2017, test sept. 2017 -> avr. 2020)
TRAIN_START, TRAIN_END = "2007-01-01", "2017-08-31"
TEST_START, PAPER_TEST_END = "2017-09-01", "2020-04-30"

# Nombre de "jambes" traitées quand on achète/vend 1 unité de chaque actif (frais)
LEGS = {"SPY": 1, "MKT": 1, "SMB": 2, "HML": 2, "MOM": 2, "CASH": 0}


def spy_close(adjusted: bool = False) -> pd.Series:
    return load_prices(adjusted=adjusted)["SPY"]


def factor_returns() -> pd.DataFrame:
    """Rendements journaliers (décimal) des "actifs" de la partie D.

    Un facteur long/short (SMB, HML, MOM) est un portefeuille à coût nul : l'investisseur
    place son capital au taux sans risque et superpose le long/short -> RF + facteur.
    MKT = Mkt-RF + RF (marché total de French), SPY = ETF réel (dividendes réinvestis).
    """
    ff = load_french()
    spy = load_prices(adjusted=True)["SPY"].pct_change()
    df = pd.DataFrame({"SPY": spy, "MKT": ff["Mkt-RF"] + ff["RF"], "SMB": ff["SMB"] + ff["RF"],
                       "HML": ff["HML"] + ff["RF"], "MOM": ff["MOM"] + ff["RF"], "CASH": ff["RF"]})
    return df.dropna()


def get_walk_forward(kind: str, start: str, end: str, cfg: WFConfig = WFConfig(),
                     store_insample: bool = False, tag: str = ""):
    """Walk-forward avec cache disque (clé = variables + dates + configuration)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{kind}_{start}_{end}_{cfg.refit}_{cfg.train_window}_{cfg.n_states}{tag}"
    f_probs, f_params, f_ins = (CACHE / f"{key}_probs.csv", CACHE / f"{key}_params.csv",
                                CACHE / f"{key}_insample.pkl")
    if f_probs.exists() and f_params.exists() and (not store_insample or f_ins.exists()):
        probs = pd.read_csv(f_probs, index_col=0, parse_dates=["Date", "refit_date"])
        params = pd.read_csv(f_params, index_col=0, parse_dates=True)
        for c in ("train_start", "train_end"):
            params[c] = pd.to_datetime(params[c])
        if store_insample:
            with open(f_ins, "rb") as fh:
                return probs, params, pickle.load(fh)
        return probs, params

    feats = build_features(spy_close(adjusted=False), kind)
    out = walk_forward_regimes(feats, start, end, cfg, store_insample=store_insample)
    out[0].to_csv(f_probs)
    out[1].to_csv(f_params)
    if store_insample:
        with open(f_ins, "wb") as fh:
            pickle.dump(out[2], fh)
    with open(CACHE / f"{key}_config.txt", "w") as fh:
        fh.write(str(asdict(cfg)))
    return out
