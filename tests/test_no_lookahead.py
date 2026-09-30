"""Vérifie qu'aucune donnée future n'est utilisée par le walk-forward et le backtest (D.2)."""
import numpy as np
import pandas as pd

from src.backtest import (WFConfig, regime_from_probs, run_backtest, walk_forward_regimes,
                          weights_from_regime)
from src.features import build_features
from src.forward import simulate_hmm

# Série de prix simulée par un HMM à 2 régimes (calme / crise)
PI = np.array([1.0, 0.0])
A = np.array([[0.99, 0.01], [0.03, 0.97]])
X, _ = simulate_hmm(1400, PI, A, [[0.05], [-0.15]], [[[0.64]], [[6.25]]], seed=42)
IDX = pd.bdate_range("2010-01-01", periods=len(X))
CLOSE = pd.Series(100 * np.cumprod(1 + X[:, 0] / 100), index=IDX)
FEATS = build_features(CLOSE, "realized")
CFG = WFConfig(n_states=2, train_window=800, n_init=2, n_iter=100)
START, END = IDX[1000], IDX[-1]


def test_walk_forward_does_not_use_future():
    """Les probabilités calculées jusqu'à la date d sont identiques, que l'on dispose ou non
    des données postérieures à d (et même si ces données futures sont délirantes)."""
    probs_full, _ = walk_forward_regimes(FEATS, START, END, CFG)
    cut = IDX[1150]                                   # en plein milieu d'un mois
    close_cut = CLOSE.copy()
    close_cut.loc[cut + pd.Timedelta(days=1):] *= 3.0  # le "futur" est falsifié...
    feats_cut = build_features(close_cut, "realized")
    probs_cut, _ = walk_forward_regimes(feats_cut, START, END, CFG)
    pd.testing.assert_frame_equal(probs_full.loc[:cut], probs_cut.loc[:cut])  # ...sans effet sur le passé


def test_training_window_strictly_before_block():
    _, params = walk_forward_regimes(FEATS, START, END, CFG)
    assert (params["train_end"] < params.index).all()
    assert (params.index.to_period("M") != params["train_end"].dt.to_period("M")).all()


def test_backtest_applies_weights_next_day():
    idx = pd.bdate_range("2021-01-01", periods=5)
    rets = pd.DataFrame({"A": [0.01, 0.02, -0.03, 0.04, 0.05], "B": 0.0}, index=idx)
    # poids "tricheurs" : on est investi en A exactement les jours où A monte
    w = pd.DataFrame({"A": (rets["A"] > 0).astype(float)}, index=idx)
    w["B"] = 1 - w["A"]
    bt = run_backtest(w, rets)
    # le poids décidé en t s'applique au rendement de t+1 -> la triche ne sert à rien
    expected = w["A"].shift(1).iloc[1:] * rets["A"].iloc[1:]
    np.testing.assert_allclose(bt["gross"].values, expected.values)


def test_backtest_stops_at_last_decision():
    idx = pd.bdate_range("2021-01-01", periods=10)
    rets = pd.DataFrame({"A": 0.01}, index=idx)
    w = pd.DataFrame({"A": 1.0}, index=idx[:5])
    bt = run_backtest(w, rets)
    assert bt.index[-1] == idx[4]          # pas de position au-delà des poids fournis


def test_costs_charged_on_switch():
    idx = pd.bdate_range("2021-01-01", periods=4)
    rets = pd.DataFrame({"A": 0.0, "B": 0.0}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]}, index=idx)
    bt = run_backtest(w, rets, cost_bps=10, legs={"A": 1, "B": 2})
    # jour 2 : achat initial de A (1 unité) ; jour 3 : rien ;
    # jour 4 : switch A -> B décidé à la clôture du jour 3 : 1 (vente A) + 1 x 2 jambes (achat B)
    np.testing.assert_allclose(bt["cost"].values, [1e-3, 0.0, 3e-3])


def test_regime_filters_are_causal():
    rng = np.random.default_rng(0)
    q = pd.DataFrame(rng.dirichlet([1, 1, 1], size=200), index=pd.bdate_range("2020", periods=200),
                     columns=["q0", "q1", "q2"])
    q2 = q.copy()
    q2.iloc[100:] = [0.0, 0.0, 1.0]
    for kw in [dict(), dict(threshold=0.6), dict(ema_span=5), dict(min_duration=5)]:
        s1, s2 = regime_from_probs(q, **kw), regime_from_probs(q2, **kw)
        pd.testing.assert_series_equal(s1.iloc[:100], s2.iloc[:100])


def test_weights_from_regime():
    reg = pd.Series([0, 1, 2], index=pd.bdate_range("2020", periods=3))
    W = weights_from_regime(reg, {0: {"A": 1.0}, 1: {"A": 0.5, "B": 0.5}, 2: {"B": 1.0}}, ["A", "B"])
    np.testing.assert_allclose(W.values, [[1, 0], [0.5, 0.5], [0, 1]])
