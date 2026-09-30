"""Tests des variables observées, de la règle de label switching et des statistiques."""
import numpy as np
import pandas as pd
import pytest

from src.features import Standardizer, build_features, paper_volatility, realized_volatility
from src.hmm_model import fit_hmm, sort_states
from src.forward import simulate_hmm
from src.stats import max_drawdown, performance_stats, sharpe_ratio, turnover

IDX = pd.bdate_range("2020-01-01", periods=300)


def test_paper_vol_constant_price_is_zero():
    close = pd.Series(100.0, index=IDX)
    assert paper_volatility(close).dropna().eq(0).all()


def test_paper_vol_linear_price():
    # prix 1, 2, ..., n : variance population de 10 entiers consécutifs = (10^2 - 1) / 12
    close = pd.Series(np.arange(1, 301, dtype=float), index=IDX)
    np.testing.assert_allclose(paper_volatility(close).dropna(), 99 / 12)


def test_paper_vol_depends_on_price_level_but_realized_does_not():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, len(IDX))
    p1 = pd.Series(100 * np.cumprod(1 + r), index=IDX)
    p2 = 5 * p1                                   # même dynamique, prix 5 fois plus élevé
    ratio = (paper_volatility(p2) / paper_volatility(p1)).dropna()
    np.testing.assert_allclose(ratio, 25.0)       # la variance de prix est multipliée par 25
    np.testing.assert_allclose(realized_volatility(p2).dropna(), realized_volatility(p1).dropna())


def test_features_are_causal():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(IDX))), index=IDX)
    close2 = close.copy()
    close2.iloc[150:] *= 2.0                      # on modifie le "futur"
    for kind in ("paper", "realized"):
        f1, f2 = build_features(close, kind), build_features(close2, kind)
        pd.testing.assert_frame_equal(f1.loc[:IDX[149]], f2.loc[:IDX[149]])


def test_standardizer_uses_train_only():
    X = np.array([[0.0], [2.0], [100.0]])
    sc = Standardizer().fit(X[:2])
    assert sc.mean_[0] == 1.0 and sc.std_[0] == 1.0
    assert sc.transform(X)[2, 0] == 99.0


def test_sort_states_fixes_label_switching():
    pi = np.array([0.5, 0.5])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    X, _ = simulate_hmm(3000, pi, A, [[0.05], [-0.1]], [[[0.49]], [[6.25]]], seed=0)
    variances = []
    for seed in range(5):
        m = fit_hmm(X, 2, n_init=1, seed=seed)
        m, _ = sort_states(m, by="variance", feature=0)
        variances.append(m.covars_[:, 0, 0])
    variances = np.array(variances)
    # après tri, l'état 0 est toujours le calme et l'état 1 toujours la crise
    assert np.all(variances[:, 0] < 1.0) and np.all(variances[:, 1] > 4.0)


def test_sharpe_known_series():
    # rendements alternés mu +/- s : moyenne mu, écart-type (ddof=1) connu exactement
    n, mu, s = 1000, 0.001, 0.01
    r = pd.Series(np.where(np.arange(n) % 2 == 0, mu + s, mu - s), index=pd.bdate_range("2000", periods=n))
    expected = mu / (s * np.sqrt(n / (n - 1))) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-10)


def test_sharpe_large_sample_gaussian():
    # Sharpe théorique 1.0 : rendement 10 %/an et volatilité 10 %/an -> SR = 0.10 / 0.10 = 1
    rng = np.random.default_rng(0)
    n = 252 * 400
    r = pd.Series(rng.normal(0.1 / 252, 0.1 / np.sqrt(252), n), index=pd.bdate_range("1900", periods=n))
    assert sharpe_ratio(r) == pytest.approx(1.0, abs=0.1)


def test_max_drawdown_known_path():
    # 100 -> 120 -> 60 -> 90 : pire baisse 120 -> 60 = 50 %
    r = pd.Series([0.2, -0.5, 0.5], index=pd.bdate_range("2020", periods=3))
    assert max_drawdown(r) == pytest.approx(0.5)


def test_turnover_and_stats_keys():
    idx = pd.bdate_range("2020", periods=252)
    w = pd.DataFrame({"A": [1.0, 0.0] * 126, "B": [0.0, 1.0] * 126}, index=idx)
    # chaque jour on vend 100 % de A et on achète 100 % de B (ou l'inverse) : 2 par jour
    assert turnover(w) == pytest.approx(2.0 * 251 / 252 * 252)
    r = pd.Series(0.001, index=idx)
    st = performance_stats(r, weights=w)
    for key in ["rendement_annualise", "volatilite_annualisee", "sharpe", "max_drawdown",
                "pct_mois_positifs", "turnover_annuel"]:
        assert key in st.index
    assert st["pct_mois_positifs"] == 1.0
