"""Téléchargement et chargement des données.

Sources (toutes gratuites) :
    - yfinance : prix journaliers de SPY (clôture brute + clôture ajustée)
    - Kenneth French Data Library : facteurs journaliers Mkt-RF, SMB, HML, RF et Mom

Les fichiers bruts sont mis en cache dans data/ pour que tout le projet puisse être
relancé hors ligne une fois le premier téléchargement effectué.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# Racine du projet = dossier parent de src/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TICKERS = ["SPY"]
START = "1993-01-01"  # SPY existe depuis janvier 1993

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}


# ---------------------------------------------------------------------------
# Prix (yfinance)
# ---------------------------------------------------------------------------
def download_prices(tickers=TICKERS, start=START, end=None) -> None:
    """Télécharge les prix journaliers et les écrit dans data/close.csv et data/adj_close.csv.

    - close.csv     : clôture brute (non ajustée) -> sert à construire la "volatilité"
                      du papier, qui est définie sur les prix bruts (données OHLC).
    - adj_close.csv : clôture ajustée des dividendes -> sert à calculer les rendements
                      réellement obtenus par un investisseur (backtest).
    """
    import yfinance as yf  # import local : pas besoin de yfinance si les CSV existent déjà

    DATA_DIR.mkdir(exist_ok=True)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      progress=False, threads=False)
    close = raw["Close"][tickers]
    adj = raw["Adj Close"][tickers]
    close.index.name = adj.index.name = "Date"
    close.to_csv(DATA_DIR / "close.csv", float_format="%.6f")
    adj.to_csv(DATA_DIR / "adj_close.csv", float_format="%.6f")


def load_prices(adjusted: bool = True) -> pd.DataFrame:
    """Charge les prix depuis le cache (les télécharge si besoin)."""
    fname = DATA_DIR / ("adj_close.csv" if adjusted else "close.csv")
    if not fname.exists():
        download_prices()
    df = pd.read_csv(fname, index_col=0, parse_dates=True)
    return df.sort_index()


# ---------------------------------------------------------------------------
# Facteurs Fama-French (Kenneth French Data Library)
# ---------------------------------------------------------------------------
def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse un CSV de la librairie de K. French (entête libre + tableau + copyright).

    On garde uniquement le premier tableau : les lignes qui commencent par une date
    AAAAMMJJ. La ligne d'entête est la dernière ligne non vide qui précède la première
    ligne de données. Les valeurs sont en pourcentage -> on divise par 100.
    """
    lines = text.splitlines()
    date_re = re.compile(r"^\s*\d{8}\s*,")
    first = next(i for i, l in enumerate(lines) if date_re.match(l))
    header_line = next(lines[j] for j in range(first - 1, -1, -1) if lines[j].strip())
    cols = [c.strip() for c in header_line.split(",")][1:]

    rows = []
    for l in lines[first:]:
        if not date_re.match(l):
            break  # fin du premier tableau (les fichiers n'en ont qu'un en journalier)
        parts = [p.strip() for p in l.split(",")]
        rows.append([parts[0]] + [float(x) for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["Date"] + cols)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date") / 100.0


def download_french() -> None:
    """Télécharge les deux zip de facteurs journaliers et les sauve en CSV propres."""
    DATA_DIR.mkdir(exist_ok=True)
    frames = []
    for key, fname in FRENCH_FILES.items():
        req = urllib.request.Request(FRENCH_BASE + fname, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        inner = z.namelist()[0]
        text = z.read(inner).decode("latin-1")
        frames.append(_parse_french_csv(text))
    ff = frames[0].join(frames[1], how="inner")
    ff.columns = [c.strip() for c in ff.columns]
    ff = ff.rename(columns={"Mom": "MOM"})
    ff.to_csv(DATA_DIR / "french_factors_daily.csv", float_format="%.6f")


def load_french() -> pd.DataFrame:
    """Facteurs journaliers en décimal : Mkt-RF, SMB, HML, RF, MOM."""
    fname = DATA_DIR / "french_factors_daily.csv"
    if not fname.exists():
        download_french()
    return pd.read_csv(fname, index_col=0, parse_dates=True).sort_index()


def download_all() -> None:
    """Point d'entrée unique pour (re)télécharger toutes les données brutes."""
    download_prices()
    download_french()


if __name__ == "__main__":
    download_all()
    print(load_prices().tail())
    print(load_french().tail())
