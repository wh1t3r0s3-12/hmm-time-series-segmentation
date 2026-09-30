"""Walk-forward sans biais d'anticipation et backtest avec frais (partie D).

Chronologie retenue (à la clôture du jour t) :
    1. on connaît les prix jusqu'à t inclus -> observations x_1..x_t
    2. si t est le premier jour d'un nouveau bloc (mois), le HMM a été réestimé sur les
       `train_window` jours qui précèdent STRICTEMENT le bloc
    3. on calcule la probabilité filtrée P(S_t | x_1..x_t) avec le forward (pas Viterbi,
       pas les probabilités lissées), puis la prédiction P(S_{t+1} | x_1..x_t)
    4. on choisit l'allocation détenue pendant la journée t+1 (rendement de t à t+1)
Aucune de ces étapes n'utilise une donnée postérieure à t : c'est vérifié par le test
tests/test_no_lookahead.py (même résultat en tronquant les données après t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import Standardizer
from .forward import filter_gaussian_hmm
from .hmm_model import expected_durations, fit_hmm, sort_states

PERIODS = 252


# ---------------------------------------------------------------------------
# 1) Probabilités de régime hors échantillon
# ---------------------------------------------------------------------------
@dataclass
class WFConfig:
    n_states: int = 3
    train_window: int = 2707      # longueur de la fenêtre glissante (celle du papier)
    refit: str = "M"              # "M" mensuel, "W" hebdomadaire, "D" quotidien
    n_init: int = 5               # initialisations aléatoires de l'EM à chaque réestimation
    n_iter: int = 500
    tol: float = 1e-4
    covariance_type: str = "full"
    warm_start: bool = True       # ajoute une initialisation depuis le modèle du mois précédent
    seed: int = 0


def walk_forward_regimes(features: pd.DataFrame, start, end, cfg: WFConfig = WFConfig(),
                         verbose: bool = False, store_insample: bool = False):
    """Renvoie
        probs  : DataFrame indexé par les dates hors échantillon avec
                 p0..p{K-1} = P(S_t = k | x_1..x_t)       (filtrée)
                 q0..q{K-1} = P(S_{t+1} = k | x_1..x_t)   (prédite, sert à l'allocation)
        params : une ligne par réestimation (dates de la fenêtre, durées moyennes, moyennes)
        insample (si store_insample) : {date de réestimation: régime prédit (argmax de q)
                 sur la fenêtre d'entraînement}, pour réapprendre la table régime -> actif
                 à chaque réestimation avec les seules données de la fenêtre.
    Les états sont triés par volatilité moyenne croissante à chaque réestimation :
    0 = calme, K-1 = crise (règle contre le label switching).
    """
    feats = features.dropna()
    oos = feats.loc[start:end]
    K = cfg.n_states
    keys = np.arange(len(oos)) if cfg.refit == "D" else oos.index.to_period(cfg.refit)

    blocks, params, insample, prev = [], [], {}, None
    for _, block in oos.groupby(keys, sort=True):
        i0 = feats.index.get_loc(block.index[0])
        i1 = feats.index.get_loc(block.index[-1])
        if i0 < cfg.train_window:
            raise ValueError("Pas assez d'historique avant le début du test")

        # --- réestimation sur la fenêtre [i0 - W, i0[ : uniquement du passé ---
        train = feats.iloc[i0 - cfg.train_window:i0]
        scaler = Standardizer().fit(train)                     # moyennes/écarts-types du passé
        Z_train = scaler.transform(train.values)
        model = fit_hmm(Z_train, K, cfg.covariance_type, cfg.n_init, cfg.n_iter, cfg.tol,
                        cfg.seed, init_model=prev if cfg.warm_start else None)
        model, _ = sort_states(model, by="mean", feature=1)    # 0 = vol faible ... K-1 = vol forte
        prev = model

        # --- filtre forward sur fenêtre + bloc : en t il n'a vu que x_1..x_t ---
        Z_run = scaler.transform(feats.iloc[i0 - cfg.train_window:i1 + 1].values)
        filt, _ = filter_gaussian_hmm(Z_run, model.startprob_, model.transmat_,
                                      model.means_, model.covars_)
        filt_b = filt[-len(block):]
        pred_b = filt_b @ model.transmat_
        df = pd.DataFrame(np.hstack([filt_b, pred_b]), index=block.index,
                          columns=[f"p{k}" for k in range(K)] + [f"q{k}" for k in range(K)])
        df["refit_date"] = block.index[0]
        blocks.append(df)
        if store_insample:
            n_tr = len(train)
            insample[block.index[0]] = pd.Series((filt[:n_tr] @ model.transmat_).argmax(axis=1),
                                                 index=train.index)

        means_orig = scaler.inverse_transform(model.means_)   # moyennes en unités d'origine
        row = {"refit_date": block.index[0], "train_start": train.index[0],
               "train_end": train.index[-1], "loglik": model.score(Z_train)}
        for k, d in enumerate(expected_durations(model.transmat_)):
            row[f"duree_{k}"] = d
            row[f"ret_moy_{k}"] = means_orig[k, 0]
            row[f"vol_moy_{k}"] = means_orig[k, 1]
        params.append(row)
        if verbose:
            print(block.index[0].date(), "LL=%.1f" % row["loglik"])

    probs, params = pd.concat(blocks), pd.DataFrame(params).set_index("refit_date")
    if store_insample:
        return probs, params, insample
    return probs, params


# ---------------------------------------------------------------------------
# 2) Du vecteur de probabilités au régime retenu (avec filtres anti allers-retours)
# ---------------------------------------------------------------------------
def regime_from_probs(q: pd.DataFrame, threshold: float | None = None,
                      ema_span: int | None = None, min_duration: int | None = None,
                      rebalance: str | None = None) -> pd.Series:
    """Régime retenu à la clôture de t à partir des probabilités prédites q (colonnes q0..).

    Options pour limiter les allers-retours (D.4) :
      - ema_span     : lissage exponentiel CAUSAL des probabilités avant l'argmax
      - threshold    : hystérésis, on ne quitte le régime courant que si la probabilité
                       du nouveau régime dépasse `threshold`
      - min_duration : on reste au moins `min_duration` jours dans un régime après un changement
      - rebalance="M": on ne change d'allocation que le dernier jour de bourse du mois
    """
    P = q.copy()
    if ema_span:
        P = P.ewm(span=ema_span, adjust=False).mean()   # n'utilise que le passé
    arg = P.values.argmax(axis=1)
    pmax = P.values.max(axis=1)

    out = np.empty(len(P), dtype=int)
    cur, last_switch = arg[0], 0
    for t in range(len(P)):
        if arg[t] != cur:
            ok = True
            if threshold is not None and pmax[t] < threshold:
                ok = False
            if min_duration is not None and t - last_switch < min_duration:
                ok = False
            if ok:
                cur, last_switch = arg[t], t
        out[t] = cur
    s = pd.Series(out, index=P.index, name="regime")

    if rebalance == "M":
        dates = s.index.to_series()
        is_month_end = dates.groupby(s.index.to_period("M")).transform("max") == dates
        s_m = s.where(is_month_end)
        s_m.iloc[0] = s.iloc[0]          # allocation initiale décidée le premier jour
        s = s_m.ffill().astype(int)
    return s


def weights_from_regime(regime: pd.Series, mapping: dict, columns) -> pd.DataFrame:
    """mapping = {régime: {actif: poids}} -> poids cibles décidés à la clôture de chaque jour."""
    W = pd.DataFrame(0.0, index=regime.index, columns=list(columns))
    for k, alloc in mapping.items():
        mask = (regime == k).values
        for asset, w in alloc.items():
            W.loc[mask, asset] = w
    return W


# ---------------------------------------------------------------------------
# 3) Backtest avec frais
# ---------------------------------------------------------------------------
def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 0.0,
                 legs: dict | None = None) -> pd.DataFrame:
    """Backtest journalier.

    weights : poids cibles DÉCIDÉS à la clôture de la date d'index
    returns : rendement de chaque actif entre la clôture t-1 et la clôture t (index t)
    Le poids détenu pendant la journée t est donc weights.shift(1) : c'est ce décalage
    qui empêche d'utiliser le rendement du jour pour choisir la position du jour.

    Frais : cost_bps points de base par unité de notionnel échangé, multiplié par le
    nombre de jambes de l'actif (2 pour un facteur long/short). Le coût d'un ordre passé
    à la clôture de t-1 est imputé au rendement du jour t.
    """
    # on s'arrête à la dernière date de décision (sinon le dernier poids serait prolongé)
    r = returns.loc[weights.index[0]:weights.index[-1], list(weights.columns)].dropna()
    W = weights.reindex(r.index).ffill()
    held = W.shift(1)
    gross = (held * r).sum(axis=1)

    legs_s = (pd.Series(legs).reindex(W.columns).fillna(1.0) if legs else
              pd.Series(1.0, index=W.columns))
    trades = W.diff()
    trades.iloc[0] = W.iloc[0]                     # achat initial du portefeuille
    traded = (trades.abs() * legs_s).sum(axis=1)   # notionnel échangé à la clôture de t
    cost = traded.shift(1) * cost_bps * 1e-4

    out = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                        "traded": traded.shift(1)})
    return out.iloc[1:]                            # le 1er jour ne porte aucune position


# ---------------------------------------------------------------------------
# 4) Performances conditionnelles aux régimes (D.1)
# ---------------------------------------------------------------------------
def regime_conditional_stats(excess: pd.DataFrame, regime: pd.Series, lag: int = 1,
                             annualize: bool = True) -> pd.DataFrame:
    """Moyenne, volatilité et Sharpe de chaque actif selon le régime.

    lag=1 : régime connu à la clôture de t, rendement du jour t+1 -> relation EXPLOITABLE
    lag=0 : régime et rendement du même jour (ce que fait le papier) -> relation en partie
            mécanique, car le régime du jour t est déterminé à partir du rendement du jour t.
    """
    # on coupe AVANT de décaler : le dernier jour de la période n'a pas de "lendemain" connu
    excess = excess.loc[regime.index[0]:regime.index[-1]]
    fwd = excess.shift(-lag)
    df = fwd.join(regime.rename("regime"), how="inner").dropna()
    g = df.groupby("regime")
    f = PERIODS if annualize else 1
    mean, vol = g.mean() * f, g.std() * np.sqrt(f)
    res = pd.concat({"moyenne": mean, "volatilite": vol, "sharpe": mean / vol}, axis=1)
    res["n_jours"] = g.size()
    return res


def choose_mapping(sharpe: pd.DataFrame, cash: str = "CASH", n_states: int | None = None) -> dict:
    """Dans chaque régime, on choisit l'actif de meilleur Sharpe ; si aucun n'a un Sharpe
    positif (ou si le régime n'a jamais été observé), on reste en monétaire."""
    mapping = {}
    for k, row in sharpe.iterrows():
        best = row.idxmax()
        mapping[int(k)] = {best: 1.0} if row[best] > 0 else {cash: 1.0}
    for k in range(n_states or 0):
        mapping.setdefault(k, {cash: 1.0})
    return mapping


def rolling_mappings(insample: dict, excess: pd.DataFrame, n_states: int = 3,
                     chooser=None) -> dict:
    """Table régime -> allocation réapprise à chaque réestimation, uniquement avec les
    données de la fenêtre d'entraînement correspondante (pas de fuite vers le test)."""
    chooser = chooser or (lambda sh: choose_mapping(sh, n_states=n_states))
    return {d: chooser(regime_conditional_stats(excess, reg, lag=1)["sharpe"])
            for d, reg in insample.items()}


def weights_from_rolling_mapping(regime: pd.Series, refit_dates: pd.Series, mappings: dict,
                                 columns) -> pd.DataFrame:
    """Comme weights_from_regime, mais chaque bloc utilise la table de SA réestimation."""
    W = pd.DataFrame(0.0, index=regime.index, columns=list(columns))
    for d, mp in mappings.items():
        in_block = (refit_dates == d).values
        W.loc[in_block] = weights_from_regime(regime[in_block], mp, columns).values
    return W
