"""Tests de l'allocation (Sharpe maximal) et du rééquilibrage mensuel."""
import numpy as np
import pandas as pd

from src.allocation import max_sharpe_weights
from src.backtest import regime_from_probs


def test_max_sharpe_matches_analytical_tangency():
    # Sans contrainte active, le portefeuille tangent est proportionnel à Sigma^-1 mu
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.multivariate_normal([0.0004, 0.0003, 0.0002],
                                             [[1e-4, 2e-5, 0], [2e-5, 4e-5, 0], [0, 0, 2e-5]], size=5000),
                     columns=["A", "B", "C"])
    w_num = max_sharpe_weights(X)
    w_th = np.linalg.solve(X.cov().values, X.mean().values)
    w_th = w_th / w_th.sum()
    assert np.all(w_th > 0)                       # on vérifie qu'on est bien dans le cas sans contrainte active
    np.testing.assert_allclose(w_num.values, w_th, atol=1e-3)
    assert abs(w_num.sum() - 1) < 1e-9


def test_monthly_rebalance_changes_only_at_month_end():
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    rng = np.random.default_rng(1)
    q = pd.DataFrame(rng.dirichlet([1, 1, 1], size=len(idx)), index=idx, columns=["q0", "q1", "q2"])
    s = regime_from_probs(q, rebalance="M")
    changes = s.index[s.ne(s.shift())][1:]
    month_ends = idx.to_series().groupby(idx.to_period("M")).max()
    # la série est indexée par la date de DÉCISION : un changement n'apparaît qu'un dernier jour de mois
    assert len(changes) > 0 and set(changes) <= set(month_ends)
