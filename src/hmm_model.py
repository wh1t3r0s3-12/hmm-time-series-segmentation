"""Estimation du HMM gaussien avec hmmlearn + outils autour du modèle.

- fit_hmm           : Baum-Welch (EM) avec plusieurs initialisations, on garde la meilleure
- sort_states       : règle d'identification contre le "label switching"
- n_params / bic    : critère d'information bayésien (C.3)
- expected_durations, stationary_distribution : lecture de la matrice de transition (A.4)
- filtered_probs / smoothed_probs : les deux types de probabilités, à ne pas confondre
"""
from __future__ import annotations

import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM

from .forward import filter_gaussian_hmm, predict_next

# hmmlearn affiche un warning à chaque EM qui ne converge pas exactement : trop verbeux
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _new_hmm(n_states, covariance_type, n_iter, tol, seed, init_params="stmc"):
    return GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                       n_iter=n_iter, tol=tol, random_state=seed, init_params=init_params)


def fit_hmm(X, n_states: int = 3, covariance_type: str = "full", n_init: int = 10,
            n_iter: int = 1000, tol: float = 1e-4, seed: int = 0, init_model=None,
            return_all: bool = False):
    """Estime un GaussianHMM avec `n_init` initialisations aléatoires (+ éventuellement
    une initialisation "à chaud" depuis `init_model`) et garde celle de meilleure
    log-vraisemblance.

    L'EM ne trouve qu'un maximum LOCAL : une seule initialisation peut tomber sur une
    mauvaise solution (voir partie B.3), d'où les initialisations multiples.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    candidates = []
    for i in range(n_init):
        m = _new_hmm(n_states, covariance_type, n_iter, tol, seed + i)
        try:
            m.fit(X)
            candidates.append((m.score(X), m))
        except (ValueError, np.linalg.LinAlgError):
            continue  # initialisation dégénérée (covariance non définie positive...) : on l'ignore

    if init_model is not None:
        # initialisation à chaud : on part des paramètres du modèle précédent (walk-forward)
        m = _new_hmm(n_states, covariance_type, n_iter, tol, seed, init_params="")
        m.startprob_ = init_model.startprob_.copy()
        m.transmat_ = init_model.transmat_.copy()
        m.means_ = init_model.means_.copy()
        m.covars_ = _covars_for_setter(init_model)
        try:
            m.fit(X)
            candidates.append((m.score(X), m))
        except (ValueError, np.linalg.LinAlgError):
            pass

    if not candidates:
        raise RuntimeError("Aucune initialisation n'a convergé")
    candidates.sort(key=lambda c: c[0], reverse=True)
    best = candidates[0][1]
    if return_all:
        return best, [c[0] for c in candidates], [c[1] for c in candidates]
    return best


def _covars_for_setter(model):
    """hmmlearn renvoie toujours des matrices pleines via .covars_, mais le setter attend
    la forme propre au type de covariance."""
    full = model.covars_
    if model.covariance_type == "full":
        return full.copy()
    if model.covariance_type == "diag":
        return np.array([np.diag(c) for c in full])
    raise NotImplementedError(model.covariance_type)


# ---------------------------------------------------------------------------
# Label switching
# ---------------------------------------------------------------------------
def permute_model(model, perm):
    """Renvoie une copie du modèle dont l'état `new` est l'ancien état `perm[new]`."""
    perm = np.asarray(perm)
    new = GaussianHMM(n_components=model.n_components, covariance_type=model.covariance_type,
                      n_iter=model.n_iter, tol=model.tol)
    new.n_features = model.n_features
    new.startprob_ = model.startprob_[perm]
    new.transmat_ = model.transmat_[np.ix_(perm, perm)]
    new.means_ = model.means_[perm]
    full = model.covars_[perm]
    new.covars_ = full if model.covariance_type == "full" else np.array([np.diag(c) for c in full])
    return new


def sort_states(model, by: str = "mean", feature: int = 1):
    """Règle d'identification : on renumérote les états par ordre CROISSANT
        - by="mean"     : de la moyenne de la variable `feature` (ex : la volatilité)
        - by="variance" : de la variance de la variable `feature` (ex : cas simulé 1D)
    Ainsi l'état 0 est toujours le plus calme et l'état K-1 le plus agité, quel que soit
    l'ordre arbitraire renvoyé par l'EM.
    Renvoie (modèle_trié, permutation).
    """
    if by == "mean":
        key = model.means_[:, feature]
    elif by == "variance":
        key = model.covars_[:, feature, feature]
    else:
        raise ValueError(by)
    perm = np.argsort(key)
    return permute_model(model, perm), perm


# ---------------------------------------------------------------------------
# Probabilités filtrées / lissées
# ---------------------------------------------------------------------------
def filtered_probs(model, X):
    """P(S_t | x_1..x_t) avec MON forward (causal : utilisable pour trader)."""
    filt, _ = filter_gaussian_hmm(np.asarray(X), model.startprob_, model.transmat_,
                                  model.means_, model.covars_)
    return filt


def predicted_probs(model, X):
    """P(S_{t+1} | x_1..x_t) : régime attendu pour le jour suivant."""
    return predict_next(filtered_probs(model, X), model.transmat_)


def smoothed_probs(model, X):
    """P(S_t | x_1..x_T) (forward-backward de hmmlearn) : utilise le FUTUR, interdit pour trader."""
    return model.predict_proba(np.asarray(X))


# ---------------------------------------------------------------------------
# Sélection de modèle
# ---------------------------------------------------------------------------
def n_params(n_states: int, n_features: int, covariance_type: str = "full") -> int:
    """Nombre de paramètres libres d'un HMM gaussien.
        loi initiale : K - 1
        transitions  : K (K - 1)          (chaque ligne somme à 1)
        moyennes     : K d
        covariances  : K d (d + 1) / 2 (pleine) ou K d (diagonale)
    """
    K, d = n_states, n_features
    cov = K * d * (d + 1) // 2 if covariance_type == "full" else K * d
    return (K - 1) + K * (K - 1) + K * d + cov


def information_criteria(model, X):
    """Renvoie (loglik, AIC, BIC) calculés à la main."""
    X = np.asarray(X)
    ll = model.score(X)
    p = n_params(model.n_components, X.shape[1], model.covariance_type)
    T = X.shape[0]
    return ll, -2 * ll + 2 * p, -2 * ll + p * np.log(T)


# ---------------------------------------------------------------------------
# Lecture de la matrice de transition
# ---------------------------------------------------------------------------
def expected_durations(transmat) -> np.ndarray:
    """Durée moyenne d'un séjour dans l'état k.

    Le nombre de jours consécutifs D passés dans k suit une loi géométrique :
    P(D = n) = a_kk^(n-1) (1 - a_kk), donc E[D] = 1 / (1 - a_kk).
    """
    return 1.0 / (1.0 - np.diag(np.asarray(transmat)))


def stationary_distribution(transmat) -> np.ndarray:
    """Loi stationnaire pi* telle que pi* A = pi* (vecteur propre à gauche pour la valeur 1)."""
    A = np.asarray(transmat)
    vals, vecs = np.linalg.eig(A.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()
