"""Tests de l'algorithme forward codé à la main (cas dont on connaît la réponse)."""
import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from src.forward import (filter_gaussian_hmm, forward_filter, forward_unscaled,
                         gaussian_logpdf, simulate_hmm)
from src.hmm_model import expected_durations, n_params, stationary_distribution

# HMM de la partie B.1 (rendements en %)
PI = np.array([0.5, 0.5])
A = np.array([[0.98, 0.02], [0.05, 0.95]])
MU = np.array([[0.05], [-0.10]])
COV = np.array([[[0.7 ** 2]], [[2.5 ** 2]]])


def _hmmlearn_model():
    m = GaussianHMM(n_components=2, covariance_type="full")
    m.n_features = 1
    m.startprob_, m.transmat_, m.means_, m.covars_ = PI, A, MU, COV
    return m


def test_logpdf_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    means = np.array([[0.0, 0.0], [1.0, -1.0]])
    covs = np.array([[[1.0, 0.3], [0.3, 2.0]], [[0.5, 0.0], [0.0, 0.5]]])
    ours = gaussian_logpdf(X, means, covs)
    for k in range(2):
        ref = multivariate_normal(means[k], covs[k]).logpdf(X)
        np.testing.assert_allclose(ours[:, k], ref, rtol=1e-10)


def test_loglik_equals_hmmlearn_score():
    X, _ = simulate_hmm(3000, PI, A, MU, COV, seed=1)
    _, ll = filter_gaussian_hmm(X, PI, A, MU, COV)
    assert ll == pytest.approx(_hmmlearn_model().score(X), rel=1e-9)


def test_filtered_are_probabilities():
    X, _ = simulate_hmm(1000, PI, A, MU, COV, seed=2)
    filt, _ = filter_gaussian_hmm(X, PI, A, MU, COV)
    assert np.all(filt >= 0) and np.allclose(filt.sum(axis=1), 1.0)


def test_last_filtered_equals_last_smoothed():
    # En t = T, filtrée et lissée coïncident : P(S_T | x_1..x_T)
    X, _ = simulate_hmm(500, PI, A, MU, COV, seed=3)
    filt, _ = filter_gaussian_hmm(X, PI, A, MU, COV)
    smooth = _hmmlearn_model().predict_proba(X)
    np.testing.assert_allclose(filt[-1], smooth[-1], atol=1e-10)
    # ... mais pas avant (les lissées regardent le futur)
    assert np.abs(filt[:-1] - smooth[:-1]).max() > 1e-3


def test_filtered_is_causal():
    # Modifier les observations après t ne change pas la probabilité filtrée en t
    X, _ = simulate_hmm(400, PI, A, MU, COV, seed=4)
    X2 = X.copy()
    X2[200:] = 10.0
    f1, _ = filter_gaussian_hmm(X, PI, A, MU, COV)
    f2, _ = filter_gaussian_hmm(X2, PI, A, MU, COV)
    np.testing.assert_allclose(f1[:200], f2[:200])


def test_uninformative_emissions_give_markov_prediction():
    # Si toutes les densités sont égales, le filtre = pi A^(t-1) (pure chaîne de Markov)
    T = 30
    log_b = np.zeros((T, 2))
    filt, _ = forward_filter(log_b, PI, A)
    expected = PI @ np.linalg.matrix_power(A, T - 1)
    np.testing.assert_allclose(filt[-1], expected, atol=1e-12)


def test_unscaled_forward_agrees_then_underflows():
    X, _ = simulate_hmm(3000, PI, A, MU, COV, seed=5)
    log_b = gaussian_logpdf(X, MU, COV)
    # série courte : sum_j alpha_T(j) = P(x_1..x_T) = exp(loglik)
    alpha = forward_unscaled(log_b[:50], PI, A)
    _, ll = forward_filter(log_b[:50], PI, A)
    assert np.log(alpha[-1].sum()) == pytest.approx(ll, rel=1e-10)
    # série longue : le forward non normalisé tombe à 0 (underflow), pas la version normalisée
    alpha_long = forward_unscaled(log_b, PI, A)
    assert alpha_long[-1].sum() == 0.0
    _, ll_long = forward_filter(log_b, PI, A)
    assert np.isfinite(ll_long)


def test_expected_durations_and_stationary():
    np.testing.assert_allclose(expected_durations(A), [50.0, 20.0])
    pi_star = stationary_distribution(A)
    np.testing.assert_allclose(pi_star @ A, pi_star)
    np.testing.assert_allclose(pi_star, [0.05 / 0.07, 0.02 / 0.07])


def test_n_params_matches_hmmlearn():
    # même décompte de paramètres libres que hmmlearn (utilisé pour son BIC)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    for K in (2, 3, 4):
        m = GaussianHMM(n_components=K, covariance_type="full", n_iter=5, random_state=0).fit(X)
        ll = m.score(X)
        bic_ours = -2 * ll + n_params(K, 2, "full") * np.log(len(X))
        assert bic_ours == pytest.approx(m.bic(X), rel=1e-9)
