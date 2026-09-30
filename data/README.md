# Données

Aucune donnée n'est versionnée dans ce dépôt. Elles sont téléchargées par :

```bash
python -m src.data
```

| Fichier créé | Source | Contenu |
|---|---|---|
| `close.csv` | Yahoo Finance (via `yfinance`) | clôture **brute** journalière de SPY (sert à la « volatilité » du papier) |
| `adj_close.csv` | Yahoo Finance (via `yfinance`) | clôture **ajustée** des dividendes de SPY (sert aux rendements du backtest) |
| `french_factors_daily.csv` | [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Mkt-RF, SMB, HML, RF (*F-F Research Data Factors daily*) et MOM (*Momentum Factor daily*), en décimal |

Les résultats de `results/` ont été produits avec des données téléchargées fin septembre 2026 (prix jusqu'au
25/09/2026, facteurs jusqu'au 31/08/2026). Toutes les études s'arrêtent au **31/08/2026**, dernière date commune.
Yahoo révise parfois ses séries : un nouveau téléchargement peut modifier légèrement les chiffres.

**Conditions d'utilisation.** Les données Yahoo Finance sont soumises aux conditions de Yahoo (usage
personnel), c'est pourquoi elles ne sont pas redistribuées ici. Les facteurs de Kenneth French sont mis à
disposition gratuitement par la Tuck School of Business (Dartmouth) ; la source est citée.
