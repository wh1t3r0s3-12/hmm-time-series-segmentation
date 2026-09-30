"""Détection de régimes de marché avec un HMM gaussien.

Réplication critique de Wang, Lin & Mikhelson (2020),
"Regime-Switching Factor Investing with Hidden Markov Models", JRFM 13(12), 311.

Modules :
    data       : téléchargement / chargement des données (yfinance, Kenneth French)
    features   : construction des observations du HMM (rendement, volatilité)
    forward    : algorithme forward codé à la main, simulation d'un HMM
    hmm_model  : estimation multi-initialisations, label switching, BIC, durées
    regimes    : méthodes de comparaison (KMeans, règle de seuil), persistance
    backtest   : walk-forward sans biais d'anticipation, frais de transaction
    allocation : allocation statique de Sharpe maximal (référence du backtest)
    stats      : indicateurs de performance
    pipeline   : dates de l'étude, jeux de rendements, cache des walk-forwards
    plotting   : style commun des figures
"""
