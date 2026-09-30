"""Algorithme forward codé à la main + simulation d'un HMM gaussien.

Notations (Rabiner 1989) pour un HMM à K états et des observations x_1..x_T :
    pi_k      = P(S_1 = k)                         loi initiale
    A[i, j]   = P(S_{t+1} = j | S_t = i)           matrice de transition
    b_k(x)    = N(x ; mu_k, Sigma_k)               densité d'émission de l'état k

Forward "brut" : alpha_t(j) = P(x_1..x_t, S_t = j)
    alpha_1(j)     = pi_j b_j(x_1)
    alpha_t(j)     = [ sum_i alpha_{t-1}(i) A[i, j] ] b_j(x_t)
Problème : alpha_t décroît géométriquement avec t (produit de densités) et passe sous
le plus petit flottant représentable après quelques centaines de pas (underflow).

Forward normalisé : on divise à chaque pas par c_t = sum_j alpha_t(j), ce qui donne
directement la probabilité FILTRÉE
    alpha_hat_t(j) = P(S_t = j | x_1..x_t)
et la log-vraisemblance log P(x_1..x_T) = sum_t log c_t.

C'est la seule probabilité légitime pour trader en t : elle n'utilise que le passé.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Densités gaussiennes multivariées (en log, pour la stabilité numérique)
# ---------------------------------------------------------------------------
def gaussian_logpdf(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """log N(x_t ; mu_k, Sigma_k) pour tout t et tout k -> tableau (T, K).

    X      : (T, d)  observations
    means  : (K, d)
    covars : (K, d, d) matrices de covariance pleines
    On passe par la décomposition de Cholesky Sigma = L L' :
        log N = -d/2 log(2 pi) - sum(log diag L) - 1/2 ||L^{-1}(x - mu)||^2
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:                                # série 1D -> colonne (T, 1)
        X = X[:, None]
    T, d = X.shape
    K = means.shape[0]
    out = np.empty((T, K))
    for k in range(K):
        L = np.linalg.cholesky(covars[k])
        diff = X - means[k]                        # (T, d)
        sol = np.linalg.solve(L, diff.T)           # L^{-1}(x - mu), (d, T)
        maha = np.sum(sol ** 2, axis=0)            # distance de Mahalanobis au carré
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        out[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    return out


# ---------------------------------------------------------------------------
# Algorithme forward
# ---------------------------------------------------------------------------
def forward_unscaled(log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """Forward SANS normalisation, écrit uniquement pour montrer l'underflow (partie B.2).

    Renvoie alpha_t(j) = P(x_1..x_t, S_t = j). Devient 0.0 (underflow) sur de longues séries.
    """
    b = np.exp(log_b)
    T, K = b.shape
    alpha = np.zeros((T, K))
    alpha[0] = startprob * b[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ transmat) * b[t]
    return alpha


def forward_filter(log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray):
    """Forward normalisé à chaque pas.

    Paramètres
    ----------
    log_b     : (T, K) log-densités d'émission log b_k(x_t)
    startprob : (K,)   loi initiale pi
    transmat  : (K, K) matrice de transition A (lignes = état de départ)

    Renvoie
    -------
    filtered : (T, K) probabilités filtrées P(S_t = k | x_1..x_t)
    loglik   : log P(x_1..x_T)
    """
    T, K = log_b.shape
    filtered = np.empty((T, K))
    loglik = 0.0
    prior = np.asarray(startprob, dtype=float)       # P(S_1) puis P(S_t | x_1..x_{t-1})
    for t in range(T):
        # On retranche le max des log-densités avant d'exponentier : évite exp(-800) = 0
        m = log_b[t].max()
        unnorm = prior * np.exp(log_b[t] - m)        # proportionnel à alpha_t(j)
        c = unnorm.sum()
        filtered[t] = unnorm / c                     # normalisation -> somme = 1
        loglik += np.log(c) + m                      # log c_t (en remettant le max)
        prior = filtered[t] @ transmat               # prédiction P(S_{t+1} | x_1..x_t)
    return filtered, loglik


def filter_gaussian_hmm(X, startprob, transmat, means, covars):
    """Raccourci : log-densités gaussiennes + forward normalisé."""
    log_b = gaussian_logpdf(X, np.asarray(means), np.asarray(covars))
    return forward_filter(log_b, startprob, transmat)


def predict_next(filtered: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """P(S_{t+1} = k | x_1..x_t) = sum_i P(S_t = i | x_1..x_t) A[i, k].

    C'est la probabilité à utiliser pour choisir l'allocation détenue en t+1.
    """
    return filtered @ transmat


# ---------------------------------------------------------------------------
# Simulation (partie B.1)
# ---------------------------------------------------------------------------
def simulate_hmm(n: int, startprob, transmat, means, covars, seed: int | None = None):
    """Simule n observations d'un HMM gaussien.

    Renvoie (X, states) avec X de forme (n, d) et states de forme (n,).
    """
    rng = np.random.default_rng(seed)
    startprob = np.asarray(startprob, dtype=float)
    transmat = np.asarray(transmat, dtype=float)
    means = np.atleast_2d(np.asarray(means, dtype=float))
    covars = np.asarray(covars, dtype=float)
    K, d = means.shape
    if covars.ndim == 1:                    # variances scalaires (cas 1D)
        covars = covars.reshape(K, 1, 1)

    states = np.empty(n, dtype=int)
    states[0] = rng.choice(K, p=startprob)
    for t in range(1, n):
        # la chaîne de Markov : l'état suivant ne dépend que de l'état courant
        states[t] = rng.choice(K, p=transmat[states[t - 1]])

    X = np.empty((n, d))
    for k in range(K):
        idx = states == k
        X[idx] = rng.multivariate_normal(means[k], covars[k], size=idx.sum())
    return X, states
