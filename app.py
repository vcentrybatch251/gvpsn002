"""
NSE Pre-Breakout Stock Scanner — v2
====================================
Scans NSE stocks via Yahoo Finance and scores each one across **14 signals**
spanning technical structure, fundamentals, institutional activity, base
quality, sector rotation, and market-timing cycles.

SCORING (14 points total)
  Phase 1 — Technical (price/volume data, fast, applies to every symbol):
      1. Volume Accumulation (surge ≥1.5x 20-day avg)
      2. VCP / Price Contraction (5-day range < 3%)
      3. Proximity to 52-week High (within 5%)
      4. Moving Average Setup (Stage 2 / >50 EMA)
      5. Bollinger Band Squeeze (BBW < 6%)
      6. ATR Contraction (recent ATR < 85% of prior)
      7. NR7 (narrowest range bar in 7 sessions)
      8. Relative Strength (1M > 5%, 3M > 10%)
     11. Base Quality (base ≥ 6 weeks, upper-half, ≤2 prior tests)
     14. Fresh Breakout Timing (≥120 days since price last near this level)

  Phase 2 — Fundamentals & Institutions (yfinance .info, only run on
            Phase-1 candidates to keep it fast):
      9. Fundamental Catalyst (EPS growth ≥25% or Revenue growth ≥20%)
     10. Institutional / Promoter footprint (institutions ≥10% AND
         insiders/promoters ≥30%)
     13. Float & Market-Cap Sweet Spot (₹500cr–₹15,000cr, float ≤75%)

  Cross-cutting:
     12. Sector Rotation — bonus point if ≥2 candidates share a sector
         (static sector map, always available)

  Plus a full trade plan (Buy / SL / T1-T2-T3 / Days-to-target / R:R)
  and a Market-Timing context banner (day-of-week, results season,
  expiry week).

Install:
    pip install streamlit yfinance pandas numpy requests

Run:
    streamlit run nse_scanner.py

Note on Phase 2 data: Yahoo's fundamental/holder fields for NSE stocks are
often sparse or missing. Where data is unavailable, that factor simply
contributes 0 (shown as "N/A") rather than penalizing the stock.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import io
import calendar
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Pre-Breakout Scanner",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 600; color: #212529; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stProgress > div > div { border-radius: 4px; }
    .signal-pill {
        display: inline-block;
        background: #e7f1ff;
        color: #0d47a1;
        border-radius: 4px;
        font-size: 11px;
        padding: 2px 7px;
        margin: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# NSE SYMBOL UNIVERSE  (5000+ tickers)
# Strategy: fetch live from NSE's public CSV + supplement with
# known indices. Falls back to a curated 600-stock list if
# the live fetch fails.
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nse_universe():
    """Fetch all NSE-listed equity symbols from NSE's public data."""
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com",
    }
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                col = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
                if col:
                    syms = df[col].dropna().astype(str).str.strip().tolist()
                    syms = [s for s in syms if s and len(s) <= 20]
                    return syms
        except Exception:
            pass

    # ── fallback: curated ~600 symbols ──
    return [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
        "ITC","BAJFINANCE","LT","KOTAKBANK","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA",
        "TITAN","ULTRACEMCO","WIPRO","POWERGRID","NTPC","ONGC","JSWSTEEL","NESTLEIND",
        "M&M","TECHM","BAJAJFINSV","HCLTECH","GRASIM","COALINDIA","INDUSINDBK","TATASTEEL",
        "TATAMOTORS","ADANIENT","HINDALCO","EICHERMOT","HEROMOTOCO","DRREDDY","DIVISLAB",
        "BPCL","CIPLA","BRITANNIA","APOLLOHOSP","SBILIFE","HDFCLIFE","BAJAJ-AUTO",
        "TATACONSUM","SHRIRAMFIN","ADANIPORTS","UPL","SIEMENS","HAVELLS","PIDILITIND",
        "BERGEPAINT","MUTHOOTFIN","DMART","NAUKRI","TORNTPHARM","COLPAL","BANDHANBNK",
        "AMBUJACEM","ACC","VEDL","LUPIN","BIOCON","MRF","PAGEIND","SRF","POLYCAB",
        "CONCOR","GODREJCP","DABUR","MARICO","BOSCHLTD","BATAINDIA","PETRONET","GAIL",
        "ICICIPRULI","LTIM","OFSS","PERSISTENT","COFORGE","MPHASIS","JSWENERGY","TRENT",
        "ZOMATO","NYKAA","ABCAPITAL","APLAPOLLO","ASHOKLEY","ASTRAL","AUBANK","BSOFT",
        "CANBK","CESC","CROMPTON","DELHIVERY","ESCORTS","FEDERALBNK","GLENMARK",
        "GODREJPROP","GRANULES","HFCL","IDFCFIRSTB","INDHOTEL","INDIAMART","INDUSTOWER",
        "JKCEMENT","LAURUSLABS","LICHSGFIN","LTTS","METROPOLIS","MFSL","NMDC","OBEROIRLTY",
        "PHOENIXLTD","PIIND","RAMCOCEM","RBLBANK","SJVN","SONACOMS","SUMICHEM","SUPREMEIND",
        "SYNGENE","TATACOMM","TVSMOTOR","VOLTAS","AUROPHARMA","BALRAMCHIN","CHAMBLFERT",
        "DEEPAKNTR","EIDPARRY","FLUOROCHEM","GNFC","GSFC","IOLCP","JUBLFOOD","KANSAINER",
        "LINDEINDIA","NAVINFLUOR","NOCIL","TATAPOWER","TORNTPOWER","ZEEL","IDEA","PNB",
        "BANKBARODA","CANBK","UNIONBANK","IOB","CENTRALBK","MAHABANK","UCOBANK","JKBANK",
        "KARURVYSYA","DCBBANK","SOUTHBANK","CITYUNIONB","TMKEMICAL","TATACHEM","PCBL",
        "AAVAS","APTUS","HOMEFIRST","REPCO","CHOLAFIN","M&MFIN","MANAPPURAM","BAJAJHFL",
        "CANFINHOME","HDFC","PNBHOUSING","GRUH","LALPATHLAB","THYROCARE","HEALTHSNS",
        "MAXHEALTH","FORTIS","ASTER","KIMS","RAINBOW","MEDANTA","HCG","NARAYANA",
        "AEGISCHEM","ALKYLAMINE","ATUL","BALCHEMLTD","BASF","CAMLIN","CLEAN","COROMANDEL",
        "DFMFOODS","DODLA","DHANUKA","EXCEL","FINEORG","GHCL","GODREJAGRO","GRINDWELL",
        "GSPL","GULFOILLUB","IIFLSEC","JINDALSAW","KALYANKJIL","KRBL","LAXMIMACH",
        "MAZDOCK","MIDHANI","MOLDTKPAC","MSTCLTD","NATCOPHARM","NAVNETEDUL","NBCC",
        "NIITTECH","ORIENTELEC","PARAGMILK","PCJEWELLER","PENIND","PFIZER","PIL",
        "PRINCEPIPE","RATNAMANI","RENUKA","ROUTE","SAFARI","SAPPHIRE","SCHAEFFLER",
        "SEQUENT","SHARDACROP","SHYAMMETL","SOMANYCERA","SSWL","STARCEMENT","STYRENIX",
        "SUDARSCHEM","SUPRAJIT","SUTLEJTEX","SWSOLAR","TANLA","TATAINVEST","TEAMLEASE",
        "TECHNOCRAFT","TEJASNET","THYROCARE","TIMKEN","TINPLATE","TIRUMALCHM","TNPETRO",
        "TORNTPHARM","TRENT","TRIDENT","TRIVENI","TTKPRESTIG","UNIPARTS","UTIAMC",
        "VGUARD","VINATIORGA","VINDHYATEL","VOLTAMP","VRLLOG","WABAG","WELCORP","WELENT",
        "WESLEYAN","WHIRLPOOL","WINDLAS","WOCKPHARMA","XCHANGING","XPROINDIA","YATHARTH",
        "ZAGGLE","ZENSARTECH","ZENTEC","ZYDUSLIFE","ZYDUSWELL",
    ]


# ─────────────────────────────────────────────────────────────
# SECTOR MAP  (for factor #12 — Sector & Index Rotation)
# Static mapping so sector grouping works instantly without any
# extra network calls. Symbols not listed fall back to "Unknown"
# and simply don't qualify for the sector-rotation bonus.
# ─────────────────────────────────────────────────────────────

SECTOR_GROUPS = {
    "IT Services": [
        "TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","OFSS","PERSISTENT","COFORGE",
        "MPHASIS","LTTS","BSOFT","TANLA","ZENSARTECH","CYIENT","MASTEK","SONATSOFTW",
        "INTELLECT","NEWGEN","KPITTECH","TATAELXSI","NAUKRI","ROUTE","NIITTECH",
    ],
    "Banking": [
        "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","BANDHANBNK",
        "IDFCFIRSTB","AUBANK","FEDERALBNK","PNB","BANKBARODA","CANBK","UNIONBANK",
        "RBLBANK","KARURVYSYA","DCBBANK","SOUTHBANK","CITYUNIONB","IOB","CENTRALBK",
        "MAHABANK","UCOBANK","JKBANK","YESBANK",
    ],
    "NBFC & Financial Services": [
        "BAJFINANCE","BAJAJFINSV","MUTHOOTFIN","CHOLAFIN","M&MFIN","MANAPPURAM",
        "SHRIRAMFIN","LICHSGFIN","PNBHOUSING","CANFINHOME","AAVAS","APTUS","HOMEFIRST",
        "ICICIPRULI","SBILIFE","HDFCLIFE","MFSL","ABCAPITAL","IIFLSEC","UTIAMC","BAJAJHFL",
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA","DRREDDY","DIVISLAB","CIPLA","LUPIN","BIOCON","TORNTPHARM",
        "AUROPHARMA","GLENMARK","LAURUSLABS","NATCOPHARM","GRANULES","SYNGENE",
        "ZYDUSLIFE","APOLLOHOSP","FORTIS","MAXHEALTH","LALPATHLAB","METROPOLIS",
        "THYROCARE","ASTER","KIMS","NARAYANA","WOCKPHARMA","PFIZER","HCG","MEDANTA",
        "RAINBOW","ZYDUSWELL",
    ],
    "Auto & Ancillaries": [
        "MARUTI","M&M","TATAMOTORS","EICHERMOT","HEROMOTOCO","BAJAJ-AUTO","TVSMOTOR",
        "ASHOKLEY","BOSCHLTD","MRF","SONACOMS","SUPRAJIT","ESCORTS","SCHAEFFLER",
        "TIMKEN","BALKRISHNA","APOLLOTYRE","MOTHERSON","ENDURANCE","SSWL",
    ],
    "FMCG": [
        "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM","DABUR","MARICO",
        "COLPAL","GODREJCP","BERGEPAINT","EMAMILTD","DODLA","RADICO","GODREJAGRO",
        "PGHH","PARAGMILK","DFMFOODS",
    ],
    "Metals & Mining": [
        "TATASTEEL","JSWSTEEL","HINDALCO","VEDL","NMDC","JINDALSAW","SAIL","NATIONALUM",
        "HINDZINC","MOIL","SHYAMMETL","WELCORP","TINPLATE",
    ],
    "Energy & Oil/Gas": [
        "RELIANCE","ONGC","BPCL","GAIL","PETRONET","IOC","OIL","MGL","IGL","GSPL",
    ],
    "Power": [
        "NTPC","POWERGRID","TATAPOWER","TORNTPOWER","SJVN","NHPC","CESC","JSWENERGY",
        "ADANIPOWER",
    ],
    "Cement": [
        "ULTRACEMCO","AMBUJACEM","ACC","SHREECEM","JKCEMENT","RAMCOCEM","STARCEMENT",
        "HEIDELBERG",
    ],
    "Capital Goods & Engineering": [
        "LT","SIEMENS","ABB","HAVELLS","CUMMINSIND","THERMAX","BHEL","CGPOWER",
        "POLYCAB","KEI","VOLTAS","CROMPTON","VGUARD","FINOLEX","GRINDWELL","VOLTAMP",
    ],
    "Defence": [
        "HAL","BEL","MAZDOCK","BDL","COCHINSHIP","GRSE","DATAPATTNS","SOLARINDS",
        "MIDHANI",
    ],
    "Chemicals & Agrochem": [
        "SRF","PIDILITIND","UPL","AARTIIND","DEEPAKNTR","NAVINFLUOR","ATUL","ALKYLAMINE",
        "FINEORG","BALRAMCHIN","GNFC","GSFC","CHAMBLFERT","COROMANDEL","PIIND","CLEAN",
        "VINATIORGA","SUMICHEM","DHANUKA","FLUOROCHEM","NOCIL",
    ],
    "Realty": [
        "DLF","GODREJPROP","OBEROIRLTY","PHOENIXLTD","PRESTIGE","BRIGADE","SOBHA",
    ],
    "Telecom": [
        "BHARTIARTL","IDEA","INDUSTOWER","TATACOMM","HFCL","TEJASNET","VINDHYATEL",
    ],
    "Retail & Consumer": [
        "DMART","TRENT","NYKAA","ZOMATO","TITAN","PAGEIND","VMART","SHOPERSTOP",
        "BATAINDIA","INDIAMART",
    ],
    "PSU & Diversified": [
        "COALINDIA","ADANIENT","ADANIPORTS","GRASIM","CONCOR","NBCC",
    ],
}

SECTOR_MAP = {sym: sector for sector, syms in SECTOR_GROUPS.items() for sym in syms}


# ─────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────

def ema(series: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    out = np.empty(len(series))
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def sma(series: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(series), np.nan)
    for i in range(period - 1, len(series)):
        out[i] = series[i - period + 1 : i + 1].mean()
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))
    atr_arr = np.empty(len(tr))
    atr_arr[0] = tr[0]
    for i in range(1, len(tr)):
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period
    return atr_arr


def bollinger_width(close: np.ndarray, period: int = 20) -> float:
    if len(close) < period:
        return np.nan
    window = close[-period:]
    mid = window.mean()
    std = window.std(ddof=0)
    return (2 * std / mid) * 100 if mid else np.nan


def analyze_base_structure(c: np.ndarray, h: np.ndarray, l: np.ndarray) -> dict:
    """
    Factor 11 (Base Count & Quality) + Factor 14a (Fresh Breakout Timing).

    • base_weeks  — length of the current consolidation (trading days / 5)
    • base_count  — how many times price has approached this resistance
                    zone (≥97% of it) within the lookback window
    • in_upper_half — is price sitting in the upper half of the current base?
    • fresh_breakout_days — how long since price was last near (≥95% of)
                    this resistance level, before the current base started.
                    A large number = breakout after a long, fresh consolidation.
    """
    n = len(c)
    if n < 10:
        return {
            "base_weeks": 0.0, "base_count": 0, "in_upper_half": False,
            "fresh_breakout_days": 0, "base_high": float(c[-1]), "base_low": float(c[-1]),
        }

    lookback = min(252, n)
    hwin = h[-lookback:]
    cur = c[-1]
    resistance = hwin.max()

    # length of current base: walk back while price stays within 12% of resistance
    threshold = resistance * 0.88
    base_len = 0
    for i in range(n - 1, -1, -1):
        if c[i] < threshold:
            break
        base_len += 1
    base_len = max(base_len, 1)

    base_window_h = h[-base_len:]
    base_window_l = l[-base_len:]
    base_high = float(base_window_h.max())
    base_low  = float(base_window_l.min())
    base_mid  = (base_high + base_low) / 2
    in_upper_half = bool(cur >= base_mid)
    base_weeks = round(base_len / 5, 1)

    # base_count: clusters of bars that came within 3% of resistance
    near_res = hwin >= resistance * 0.97
    clusters, prev = 0, False
    for flag in near_res:
        f = bool(flag)
        if f and not prev:
            clusters += 1
        prev = f
    base_count = max(clusters, 1)

    # fresh_breakout_days: days since price was last near this level,
    # measured BEFORE the current base started
    prior_end = n - base_len
    if prior_end > 0:
        prior_highs = h[:prior_end]
        hits = np.where(prior_highs >= resistance * 0.95)[0]
        if len(hits) > 0:
            fresh_days = (prior_end - 1 - hits[-1]) + base_len
        else:
            fresh_days = prior_end + base_len
    else:
        fresh_days = base_len

    return {
        "base_weeks": base_weeks,
        "base_count": base_count,
        "in_upper_half": in_upper_half,
        "fresh_breakout_days": int(fresh_days),
        "base_high": round(base_high, 2),
        "base_low": round(base_low, 2),
    }


# ─────────────────────────────────────────────────────────────
# PHASE 1 — TECHNICAL SCORING  (factors 1-8, 11, 14a → max 10 pts)
# ─────────────────────────────────────────────────────────────

def score_stock(symbol: str, df: pd.DataFrame):
    """
    Returns (row_dict, (c,h,l,v)) or None if data insufficient.
    row_dict contains Phase1Score (0-10), Phase1Signals (list), and all
    technical detail columns. Trade levels & Phase 2 fields are added later.
    """
    try:
        if df is None or len(df) < 50:
            return None

        c = df["Close"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        l = df["Low"].to_numpy(dtype=float)
        v = df["Volume"].to_numpy(dtype=float)
        n = len(c)

        score = 0
        signals = []
        detail = {}

        # ── 1. Volume Accumulation ──
        if n >= 25:
            avg_vol20 = v[n - 25 : n - 5].mean()
            recent_vols = v[n - 5 :]
            vol_ratio = recent_vols.max() / avg_vol20 if avg_vol20 > 0 else 0
            detail["vol_ratio"] = round(vol_ratio, 2)

            up_vol = sum(v[i] for i in range(n - 10, n) if c[i] > c[i - 1])
            dn_vol = sum(v[i] for i in range(n - 10, n) if c[i] <= c[i - 1])
            ud_ratio = (up_vol / dn_vol) if dn_vol > 0 else 99
            detail["ud_ratio"] = round(ud_ratio, 2)

            if vol_ratio >= 1.5:
                score += 1
                signals.append(f"Vol {vol_ratio:.1f}x")
        else:
            detail["vol_ratio"] = np.nan
            detail["ud_ratio"] = np.nan

        # ── 2. VCP / Price Contraction ──
        if n >= 5:
            last5 = c[n - 5 :]
            rng_pct = (last5.max() - last5.min()) / last5.min() * 100 if last5.min() > 0 else 99
            detail["range_pct_5d"] = round(rng_pct, 2)
            if rng_pct < 3.0:
                score += 1
                signals.append("VCP")
        else:
            detail["range_pct_5d"] = np.nan

        # ── 3. Near 52-Week High ──
        lookback = min(252, n)
        hi52 = h[-lookback:].max()
        dist_pct = (hi52 - c[-1]) / hi52 * 100 if hi52 > 0 else 99
        detail["dist_52wk_high"] = round(dist_pct, 2)
        if dist_pct <= 5:
            score += 1
            signals.append("<5% Hi")

        # ── 4. Moving Average (Stage 2) ──
        if n >= 200:
            e10 = ema(c, 10); e21 = ema(c, 21); e50 = ema(c, 50); e200 = ema(c, 200)
            detail["ema50"] = round(e50[-1], 2)
            detail["ema200"] = round(e200[-1], 2)
            detail["above_50ema"] = bool(c[-1] > e50[-1])
            if c[-1] > e50[-1] > e200[-1]:
                score += 1
                signals.append("Stage2")
            if c[-1] > e10[-1] > e21[-1] > e50[-1]:
                signals.append("EMA✓")
        elif n >= 50:
            e50 = ema(c, 50)
            detail["ema50"] = round(e50[-1], 2)
            detail["above_50ema"] = bool(c[-1] > e50[-1])
            if c[-1] > e50[-1]:
                score += 1
                signals.append(">50EMA")
        else:
            detail["above_50ema"] = False

        # ── 5. Bollinger Band Squeeze ──
        bbw = bollinger_width(c)
        detail["bb_width"] = round(bbw, 2) if not np.isnan(bbw) else np.nan
        if not np.isnan(bbw) and bbw < 6:
            score += 1
            signals.append("BBsqz")

        # ── 6. ATR Contraction ──
        if n >= 30:
            atr_arr = atr(h, l, c)
            na = len(atr_arr)
            if na >= 20:
                recent_atr = atr_arr[na - 10 :].mean()
                prior_atr  = atr_arr[na - 20 : na - 10].mean()
                detail["atr_ratio"] = round(recent_atr / prior_atr, 3) if prior_atr > 0 else np.nan
                if prior_atr > 0 and recent_atr < prior_atr * 0.85:
                    score += 1
                    signals.append("ATR↓")
            else:
                detail["atr_ratio"] = np.nan
        else:
            detail["atr_ratio"] = np.nan

        # ── 7. NR7 ──
        if n >= 8:
            ranges = (h - l)
            nr7 = (h[-1] - l[-1]) == ranges[-7:].min()
            detail["nr7"] = bool(nr7)
            if nr7:
                score += 1
                signals.append("NR7")
        else:
            detail["nr7"] = False

        if n >= 2:
            detail["inside_bar"] = bool(h[-1] < h[-2] and l[-1] > l[-2])
        else:
            detail["inside_bar"] = False

        # ── 8. Relative Strength ──
        ret_1m = (c[-1] - c[max(0, n - 21)]) / c[max(0, n - 21)] * 100 if n >= 21 else 0
        ret_3m = (c[-1] - c[max(0, n - 63)]) / c[max(0, n - 63)] * 100 if n >= 63 else 0
        ret_6m = (c[-1] - c[max(0, n - 126)]) / c[max(0, n - 126)] * 100 if n >= 126 else 0
        detail["ret_1m"] = round(ret_1m, 2)
        detail["ret_3m"] = round(ret_3m, 2)
        detail["ret_6m"] = round(ret_6m, 2)
        if ret_1m > 5 and ret_3m > 10:
            score += 1
            signals.append("RS+")

        # ── 11. Base Quality  &  14. Fresh Breakout Timing ──
        base = analyze_base_structure(c, h, l)
        if base["base_weeks"] >= 6 and base["in_upper_half"] and base["base_count"] <= 2:
            score += 1
            signals.append("BaseQ")
        if base["fresh_breakout_days"] >= 120:
            score += 1
            signals.append("FreshBO")

        change_1d = (c[-1] - c[-2]) / c[-2] * 100 if n >= 2 else 0

        row = {
            "Symbol":      symbol,
            "Sector":      SECTOR_MAP.get(symbol, "Unknown"),
            "Price":       round(c[-1], 2),
            "Chg%":        round(change_1d, 2),
            "Vol Surge":   detail.get("vol_ratio", np.nan),
            "U/D Ratio":   detail.get("ud_ratio", np.nan),
            "VCP Range%":  detail.get("range_pct_5d", np.nan),
            "Dist 52Hi%":  detail.get("dist_52wk_high", np.nan),
            "BB Width":    detail.get("bb_width", np.nan),
            "ATR Ratio":   detail.get("atr_ratio", np.nan),
            "NR7":         detail.get("nr7", False),
            "Inside Bar":  detail.get("inside_bar", False),
            "Base Wks":    base["base_weeks"],
            "Base#":       base["base_count"],
            "FreshBO(d)":  base["fresh_breakout_days"],
            "Ret 1M%":     detail.get("ret_1m", 0),
            "Ret 3M%":     detail.get("ret_3m", 0),
            "Ret 6M%":     detail.get("ret_6m", 0),
            "Above 50EMA": detail.get("above_50ema", False),
            "Phase1Score":   score,
            "Phase1Signals": signals,
        }
        return row, (c, h, l, v)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# PHASE 2 — FUNDAMENTALS / INSTITUTIONAL / FLOAT  (factors 9,10,13 → max 3 pts)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    """Best-effort fetch of fundamental/holder/float data via yfinance .info"""
    try:
        info = yf.Ticker(symbol + ".NS").info or {}
    except Exception:
        info = {}
    return {
        "eps_growth":   info.get("earningsQuarterlyGrowth"),
        "rev_growth":   info.get("revenueGrowth"),
        "inst_pct":     info.get("heldPercentInstitutions"),
        "insider_pct":  info.get("heldPercentInsiders"),
        "market_cap":   info.get("marketCap"),
        "float_shares": info.get("floatShares"),
        "shares_out":   info.get("sharesOutstanding"),
    }


def score_fundamentals(fund: dict) -> dict:
    """
    Factor 9  — Fundamental Catalyst Alignment (EPS growth ≥25% or Rev growth ≥20%)
    Factor 10 — Institutional / Promoter footprint
    Factor 13 — Float & Market-Cap Sweet Spot (₹500cr–₹15,000cr, float ≤75%)
    """
    score_add = 0
    sigs = []

    eps_g = fund.get("eps_growth")
    rev_g = fund.get("rev_growth")
    eps_g_pct = round(eps_g * 100, 1) if eps_g is not None else None
    rev_g_pct = round(rev_g * 100, 1) if rev_g is not None else None

    if (eps_g is not None and eps_g >= 0.25) or (rev_g is not None and rev_g >= 0.20):
        score_add += 1
        sigs.append("Catalyst")

    inst = fund.get("inst_pct")
    insider = fund.get("insider_pct")
    inst_pct = round(inst * 100, 1) if inst is not None else None
    promoter_pct = round(insider * 100, 1) if insider is not None else None

    if inst is not None and insider is not None and inst >= 0.10 and insider >= 0.30:
        score_add += 1
        sigs.append("SmartMoney")

    mcap = fund.get("market_cap")
    mcap_cr = round(mcap / 1e7, 0) if mcap else None
    float_shares = fund.get("float_shares")
    shares_out = fund.get("shares_out")
    float_pct = round(float_shares / shares_out * 100, 1) if (float_shares and shares_out) else None

    if mcap is not None and 5e9 <= mcap <= 1.5e11:
        if float_pct is None or float_pct <= 75:
            score_add += 1
            sigs.append("MidCapFloat")

    return {
        "score_add": score_add,
        "signals": sigs,
        "EPS Gr%": eps_g_pct,
        "Rev Gr%": rev_g_pct,
        "Inst%": inst_pct,
        "Promoter%": promoter_pct,
        "MCap(Cr)": mcap_cr,
        "Float%": float_pct,
    }


# ─────────────────────────────────────────────────────────────
# GRADE & TRADE-LEVEL CALCULATOR
# ─────────────────────────────────────────────────────────────

def grade(score: int) -> str:
    if score >= 11: return "🟢 Strong"
    if score >= 8:  return "🟡 Watch"
    if score >= 5:  return "🟠 Weak"
    return "⚪ Skip"


def compute_trade_levels(
    c: np.ndarray, h: np.ndarray, l: np.ndarray, v: np.ndarray,
    score: int, atr_mult_sl: float = 1.5,
) -> dict:
    """
    BUY    = highest high of last 5 bars + 0.25% breakout buffer
    SL     = max( Buy − ATR14×mult , 5-bar structural low )
    T1/T2/T3 = Buy + 1.5R / 3R / 5R  (7R if total score ≥ 11/14 — "Strong")
    Days→target = (target move %) / (20-day ADR%), discounted for stronger setups
    """
    n = len(c)
    cur = c[-1]

    atr_arr = atr(h, l, c, 14)
    current_atr = atr_arr[-1] if len(atr_arr) > 0 else cur * 0.015

    resistance = h[-5:].max()
    buy_price  = round(resistance * 1.0025, 2)

    atr_sl        = buy_price - atr_mult_sl * current_atr
    structural_sl = l[-5:].min()
    sl = round(max(atr_sl, structural_sl), 2)
    if sl >= buy_price:
        sl = round(buy_price * 0.97, 2)

    risk = buy_price - sl

    t3_mult = 7.0 if score >= 11 else 5.0
    t1 = round(buy_price + 1.5 * risk, 2)
    t2 = round(buy_price + 3.0 * risk, 2)
    t3 = round(buy_price + t3_mult * risk, 2)

    risk_pct = round((risk / buy_price) * 100, 2)

    lookback = min(20, n - 1)
    daily_moves = np.abs(c[-lookback:] - c[-lookback - 1:-1]) / c[-lookback - 1:-1] * 100
    adr_pct = daily_moves.mean() if len(daily_moves) > 0 else 0.5
    adr_pct = max(adr_pct, 0.2)

    def est_days(target: float) -> int:
        move_needed_pct = (target - buy_price) / buy_price * 100
        raw = move_needed_pct / adr_pct
        discount = 1.0 - min(score, 14) * 0.02
        return min(max(3, int(np.ceil(raw * discount))), 90)

    return {
        "Buy": buy_price, "SL": sl, "Risk%": risk_pct,
        "T1": t1, "T2": t2, "T3": t3,
        "RR(T1)": 1.5, "RR(T2)": 3.0, "RR(T3)": t3_mult,
        "Days→T1": est_days(t1), "Days→T2": est_days(t2), "Days→T3": est_days(t3),
        "Up%→T1": round((t1 - cur) / cur * 100, 1),
        "Up%→T2": round((t2 - cur) / cur * 100, 1),
        "Up%→T3": round((t3 - cur) / cur * 100, 1),
        "ATR14": round(current_atr, 2), "ADR%": round(adr_pct, 2),
    }


# ─────────────────────────────────────────────────────────────
# DATA FETCHER (batch via yfinance download)
# ─────────────────────────────────────────────────────────────

def fetch_batch(symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """Download OHLCV for a batch of NSE tickers in one yfinance call."""
    tickers = [s + ".NS" for s in symbols]
    try:
        raw = yf.download(
            tickers, period=period, interval="1d", group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception:
        return {}

    result = {}
    if len(symbols) == 1:
        sym = symbols[0]
        if not raw.empty:
            result[sym] = raw
    else:
        for sym in symbols:
            ticker = sym + ".NS"
            try:
                if ticker in raw.columns.get_level_values(0):
                    df = raw[ticker].dropna(how="all")
                    if len(df) >= 30:
                        result[sym] = df
            except Exception:
                pass
    return result


# ─────────────────────────────────────────────────────────────
# MARKET TIMING CONTEXT  (factor 14b — calendar-based, shown as a banner)
# ─────────────────────────────────────────────────────────────

def market_timing_context() -> dict:
    now = datetime.now()
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday = now.weekday()
    last_day = calendar.monthrange(now.year, now.month)[1]
    days_left_in_month = last_day - now.day
    return {
        "date_str": now.strftime("%A, %d %b %Y"),
        "day_name": day_names[weekday],
        "favorable_day": weekday in (0, 1),       # Mon/Tue
        "results_season": now.month in (1, 4, 7, 10),
        "expiry_week": days_left_in_month <= 6,    # last week of the month
    }


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Scanner Settings")

universe_choice = st.sidebar.selectbox(
    "Universe",
    ["Nifty 50", "Nifty 100", "Nifty 500", "All NSE (~5000)", "Custom"],
)

if universe_choice == "Custom":
    custom_input = st.sidebar.text_area(
        "Enter symbols (comma or newline separated)",
        placeholder="RELIANCE\nINFY\nTCS",
        height=120,
    )
else:
    custom_input = ""

min_score = st.sidebar.slider("Minimum Total Score (out of 14)", 1, 14, 8)
batch_size = st.sidebar.slider("Batch size (yfinance)", 20, 100, 50, step=10)
period_map = {"3 months": "3mo", "6 months": "6mo", "1 year": "1y"}
period_label = st.sidebar.selectbox("Data Period", list(period_map.keys()), index=2)
data_period = period_map[period_label]
delay = st.sidebar.slider("Delay between batches (sec)", 0.5, 5.0, 1.5, step=0.5)
st.sidebar.caption("💡 Use **1 year** of data for the Base Quality & Fresh Breakout factors to work properly.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Phase 2 — Fundamentals")
enable_phase2 = st.sidebar.checkbox(
    "Enable Fundamentals/Institutional/Float check", value=True,
    help="Fetches EPS/Revenue growth, institutional & promoter holding, "
         "market cap and float via yfinance .info — only for Phase-1 candidates."
)
phase1_prefilter = st.sidebar.slider(
    "Phase 1 pre-filter score (out of 10)", 0, 10, 6,
    help="Only stocks scoring at least this much on technicals proceed to "
         "Phase 2 (fundamentals) and the final results. Lower = more API calls."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Trade Level Settings")
atr_sl_mult = st.sidebar.slider(
    "SL = ATR × multiplier", 1.0, 3.0, 1.5, step=0.1,
    help="Stop loss = Buy price minus (ATR14 x this). Lower = tighter SL."
)
st.sidebar.caption(
    "Buy = 5-bar high + 0.25% buffer  \n"
    "T1 = 1.5R  |  T2 = 3R  |  T3 = 5R (7R if Score ≥ 11/14)  \n"
    "Days estimated from the stock's own ADR%"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Signal Filters")
require_vcp    = st.sidebar.checkbox("Must have VCP", False)
require_bbsqz  = st.sidebar.checkbox("Must have BB Squeeze", False)
require_stage2 = st.sidebar.checkbox("Must be Stage 2", False)
require_nr7    = st.sidebar.checkbox("Must have NR7", False)
require_base   = st.sidebar.checkbox("Must have Base Quality (BaseQ)", False)
require_fresh  = st.sidebar.checkbox("Must have Fresh Breakout (FreshBO)", False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data source:** Yahoo Finance\n\n"
    "Phase 1 batches multiple tickers per call for speed.\n\n"
    "⚠️ Yahoo Finance may throttle at high volume — use delays.\n\n"
    "⚠️ Phase 2 fundamental/holder fields are often sparse for NSE stocks; "
    "missing data shows as N/A and contributes 0 (never penalizes)."
)


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────

st.markdown("# 🔭 NSE Pre-Breakout Scanner")
st.markdown(
    "Scans NSE stocks via **Yahoo Finance** · Scores **14 signals** across "
    "Technical, Fundamental, Institutional, Base Quality & Sector Rotation · "
    "Generates a full Buy / SL / Target trade plan"
)

# ── Market Timing banner ──
mt = market_timing_context()
mt_bits = []
mt_bits.append(f"📅 **{mt['date_str']}** ({mt['day_name']})")
mt_bits.append("✅ Favorable entry day (Mon/Tue)" if mt['favorable_day']
                else "⚠️ Late-week — follow-through less reliable")
mt_bits.append("📈 Results season (Jan/Apr/Jul/Oct) — catalyst-driven breakouts more likely"
               if mt['results_season'] else "📉 Outside core results season")
mt_bits.append("⚠️ Expiry week — breakouts prone to reversal, size down"
               if mt['expiry_week'] else "✅ Not expiry week")
st.info("  ·  ".join(mt_bits))

col1, col2, col3, col4 = st.columns(4)
metric_scanned  = col1.empty()
metric_hits     = col2.empty()
metric_strong   = col3.empty()
metric_avg      = col4.empty()

def show_metrics(scanned=0, hits=0, strong=0, avg=0.0):
    metric_scanned.metric("📦 Scanned", scanned, help="Total stocks processed")
    metric_hits.metric("🎯 Candidates", hits, help=f"Final score ≥ {min_score}")
    metric_strong.metric("🟢 Strong", strong, help="Score ≥ 11/14")
    metric_avg.metric("📊 Avg Score", f"{avg:.1f}/14", help="Average score of candidates")

show_metrics()

st.markdown("---")

run_btn = st.button("🚀 Start Scan", type="primary", use_container_width=True)

progress_bar = st.progress(0)
status_text  = st.empty()
results_area = st.empty()

# ─────────────────────────────────────────────────────────────
# SYMBOL LISTS
# ─────────────────────────────────────────────────────────────

NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "ITC","BAJFINANCE","LT","KOTAKBANK","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA",
    "TITAN","ULTRACEMCO","WIPRO","POWERGRID","NTPC","ONGC","JSWSTEEL","NESTLEIND",
    "M&M","TECHM","BAJAJFINSV","HCLTECH","GRASIM","COALINDIA","INDUSINDBK","TATASTEEL",
    "TATAMOTORS","ADANIENT","HINDALCO","EICHERMOT","HEROMOTOCO","DRREDDY","DIVISLAB",
    "BPCL","CIPLA","BRITANNIA","APOLLOHOSP","SBILIFE","HDFCLIFE","BAJAJ-AUTO",
    "TATACONSUM","SHRIRAMFIN","ADANIPORTS","UPL",
]

NIFTY100_EXTRA = [
    "SIEMENS","HAVELLS","PIDILITIND","BERGEPAINT","MUTHOOTFIN","DMART","NAUKRI",
    "TORNTPHARM","COLPAL","BANDHANBNK","AMBUJACEM","ACC","VEDL","LUPIN","BIOCON",
    "MRF","PAGEIND","SRF","POLYCAB","CONCOR","GODREJCP","DABUR","MARICO","BOSCHLTD",
    "BATAINDIA","PETRONET","GAIL","ICICIPRULI","LTIM","OFSS","PERSISTENT","COFORGE",
    "MPHASIS","JSWENERGY","TRENT","ZOMATO","NYKAA","ABCAPITAL","APLAPOLLO","ASHOKLEY",
    "ASTRAL","AUBANK","CESC","CROMPTON","ESCORTS","FEDERALBNK","GLENMARK","GODREJPROP",
    "IDFCFIRSTB","INDHOTEL","INDUSTOWER","LICHSGFIN","OBEROIRLTY","PHOENIXLTD",
]

NIFTY500_EXTRA = [
    "AUROPHARMA","BALRAMCHIN","CHAMBLFERT","DEEPAKNTR","GNFC","GSFC","JUBLFOOD",
    "KANSAINER","LINDEINDIA","NAVINFLUOR","TATAPOWER","TORNTPOWER","ZEEL","IDEA",
    "PNB","BANKBARODA","CANBK","UNIONBANK","KARURVYSYA","DCBBANK","SOUTHBANK",
    "CITYUNIONB","TATACHEM","PCBL","AAVAS","APTUS","HOMEFIRST","CHOLAFIN","M&MFIN",
    "MANAPPURAM","CANFINHOME","LALPATHLAB","MAXHEALTH","FORTIS","ASTER","KIMS",
    "ALKYLAMINE","ATUL","COROMANDEL","DHANUKA","FINEORG","GODREJAGRO","GRINDWELL",
    "JINDALSAW","KALYANKJIL","KRBL","LAURUSLABS","LTTS","METROPOLIS","MFSL","NMDC",
    "PIIND","RAMCOCEM","RBLBANK","SJVN","SONACOMS","SUMICHEM","SUPREMEIND","SYNGENE",
    "TATACOMM","TVSMOTOR","VOLTAS","ZYDUSLIFE","ZYDUSWELL","TRIDENT","TRIVENI",
    "POLYPLEX","SAFARI","SCHAEFFLER","SEQUENT","STARCEMENT","SUPRAJIT","TANLA",
    "TEAMLEASE","TECHNOCRAFT","TIMKEN","TINPLATE","VOLTAMP","VRLLOG","WELCORP",
    "XCHANGING","ZENSARTECH","VINATIORGA","WABAG","UNIPARTS","ROUTE","PRINCEPIPE",
    "RATNAMANI","RENUKA","PARAGMILK","ORIENTELEC","NATCOPHARM","MOLDTKPAC","MIDHANI",
    "LAXMIMACH","JKCEMENT","INDIAMART","HFCL","GHCL","GRANULES","ESCORTS","DELHIVERY",
    "BSOFT","CANBK","CROMPTON","GODREJPROP",
]


def get_symbol_list(choice: str, custom_text: str) -> list[str]:
    if choice == "Custom":
        raw = custom_text.replace(",", "\n").splitlines()
        return list({s.strip().upper() for s in raw if s.strip()})
    elif choice == "Nifty 50":
        return NIFTY50
    elif choice == "Nifty 100":
        return list(dict.fromkeys(NIFTY50 + NIFTY100_EXTRA))
    elif choice == "Nifty 500":
        return list(dict.fromkeys(NIFTY50 + NIFTY100_EXTRA + NIFTY500_EXTRA))
    else:  # All NSE
        with st.spinner("Fetching full NSE symbol list…"):
            live = fetch_nse_universe()
        base = list(dict.fromkeys(NIFTY50 + NIFTY100_EXTRA + NIFTY500_EXTRA))
        return list(dict.fromkeys(base + live))


# ─────────────────────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────────────────────

if run_btn:
    symbols = get_symbol_list(universe_choice, custom_input)
    if not symbols:
        st.error("No symbols to scan. Check your custom input.")
        st.stop()

    total = len(symbols)
    st.info(f"🔍 Phase 1 (Technical): scanning **{total} symbols** in batches of {batch_size}…")

    phase1_candidates = {}   # symbol -> (row, (c,h,l,v))
    batches = [symbols[i : i + batch_size] for i in range(0, total, batch_size)]
    processed = 0

    for bi, batch in enumerate(batches):
        status_text.markdown(
            f"⏳ Phase 1 · Batch {bi+1}/{len(batches)} · fetching {len(batch)} symbols "
            f"({processed}/{total} done) …"
        )

        data_map = fetch_batch(batch, period=data_period)

        for sym in batch:
            df = data_map.get(sym)
            result = score_stock(sym, df)
            processed += 1
            if result is None:
                continue
            row, arrays = result
            if row["Phase1Score"] >= phase1_prefilter:
                phase1_candidates[sym] = (row, arrays)

        pct = int(processed / total * 100)
        progress_bar.progress(pct)

        # live preview (Phase 1 only)
        if phase1_candidates:
            prev_rows = [r for r, _ in phase1_candidates.values()]
            preview = pd.DataFrame(prev_rows).sort_values("Phase1Score", ascending=False)
            preview["P1/10"] = preview["Phase1Score"].apply(lambda s: "█" * s + "░" * (10 - s))
            preview["Signals"] = preview["Phase1Signals"].apply(lambda s: ", ".join(s) if s else "—")
            cols_show = ["Symbol","Sector","Price","Chg%","Phase1Score","P1/10","Base Wks","FreshBO(d)","Signals"]
            results_area.dataframe(
                preview[cols_show].rename(columns={"Phase1Score": "P1 Score"}).style.format({
                    "Price": "₹{:.1f}", "Chg%": "{:+.2f}%",
                }),
                use_container_width=True,
                height=min(500, 40 + len(preview) * 36),
            )
            show_metrics(processed, len(phase1_candidates), 0, preview["Phase1Score"].mean())

        if bi < len(batches) - 1:
            time.sleep(delay)

    progress_bar.progress(100)
    status_text.markdown(
        f"✅ Phase 1 complete · {processed} scanned · "
        f"**{len(phase1_candidates)} candidates** (Phase-1 score ≥ {phase1_prefilter}/10)"
    )

    if not phase1_candidates:
        results_area.warning(
            "No stocks passed the Phase 1 pre-filter. Try lowering the "
            "'Phase 1 pre-filter score' or broadening the universe."
        )
        st.stop()

    # ── Sector rotation counts (factor 12) ──
    sector_counts: dict[str, int] = {}
    for sym, (row, _) in phase1_candidates.items():
        sec = row["Sector"]
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # ── Phase 2: Fundamentals / Institutional / Float ──
    fund_map = {}
    if enable_phase2:
        n_cand = len(phase1_candidates)
        status_text.markdown(f"⏳ Phase 2 · Fetching fundamentals for {n_cand} candidates…")
        p2_progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(fetch_fundamentals, sym): sym for sym in phase1_candidates}
            done_count = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    fund_map[sym] = fut.result()
                except Exception:
                    fund_map[sym] = {}
                done_count += 1
                p2_progress.progress(int(done_count / n_cand * 100))
        status_text.markdown("✅ Phase 2 complete.")
    else:
        for sym in phase1_candidates:
            fund_map[sym] = {}

    # ── Combine Phase 1 + Phase 2 + Sector bonus → final score (0-14) ──
    final_results = []
    for sym, (row, arrays) in phase1_candidates.items():
        c, h, l, v = arrays
        fund = fund_map.get(sym, {})
        fres = score_fundamentals(fund)

        sector = row["Sector"]
        sector_bonus = 1 if (sector != "Unknown" and sector_counts.get(sector, 0) >= 2) else 0

        total_score = row["Phase1Score"] + fres["score_add"] + sector_bonus
        signals = list(row["Phase1Signals"]) + fres["signals"]
        if sector_bonus:
            signals.append("SectorRot")

        # signal filters
        sig_str = ", ".join(signals)
        if require_vcp    and "VCP"     not in sig_str: continue
        if require_bbsqz  and "BBsqz"   not in sig_str: continue
        if require_stage2 and "Stage2"  not in sig_str: continue
        if require_nr7    and "NR7"     not in sig_str: continue
        if require_base   and "BaseQ"   not in sig_str: continue
        if require_fresh  and "FreshBO" not in sig_str: continue
        if total_score < min_score: continue

        tl = compute_trade_levels(c, h, l, v, total_score, atr_mult_sl=atr_sl_mult)

        final_row = {k: v2 for k, v2 in row.items() if k not in ("Phase1Score", "Phase1Signals")}
        final_row["Score"] = total_score
        final_row["Grade"] = grade(total_score)
        final_row["Signals"] = sig_str if sig_str else "—"
        final_row["SectorCount"] = sector_counts.get(sector, 0)
        final_row.update(fres)
        final_row.pop("score_add", None)
        final_row.pop("signals", None)
        final_row.update(tl)
        final_results.append(final_row)

    if not final_results:
        results_area.warning(
            f"No stocks met the final filters (Score ≥ {min_score}/14 plus any "
            "required signals). Try lowering the minimum score or relaxing filters."
        )
        st.stop()

    final_df = pd.DataFrame(final_results).sort_values("Score", ascending=False).reset_index(drop=True)
    final_df["Score/14"] = final_df["Score"].apply(lambda s: "█" * s + "░" * (14 - s))

    strong_count = int((final_df["Score"] >= 11).sum())
    show_metrics(processed, len(final_df), strong_count, final_df["Score"].mean())

    st.markdown("### 📋 Final Results")
    st.markdown(
        f"**{len(final_df)} stocks** passed all filters — sorted by total score "
        f"(out of 14). Click column headers to sort."
    )

    display_cols_full = [
        "Symbol","Sector","Price","Chg%",
        "Score","Score/14","Grade","Signals",
        "Buy","SL","Risk%",
        "T1","Days→T1","Up%→T1","RR(T1)",
        "T2","Days→T2","Up%→T2","RR(T2)",
        "T3","Days→T3","Up%→T3","RR(T3)",
        "ATR14","ADR%",
        "Vol Surge","VCP Range%","Dist 52Hi%","BB Width","ATR Ratio",
        "Base Wks","Base#","FreshBO(d)",
        "EPS Gr%","Rev Gr%","Inst%","Promoter%","MCap(Cr)","Float%",
        "SectorCount","Ret 1M%","Ret 3M%","Ret 6M%","Above 50EMA",
    ]
    display_cols_full = [c for c in display_cols_full if c in final_df.columns]

    def color_score(v):
        if isinstance(v, (int, np.integer)):
            if v >= 11: return "color: #198754; font-weight: 600"
            if v >= 8:  return "color: #fd7e14; font-weight: 600"
            return "color: #dc3545"
        return ""

    def color_chg(v):
        if isinstance(v, (int, float)) and not pd.isna(v):
            return "color: #198754" if v >= 0 else "color: #dc3545"
        return ""

    def color_targets(v):
        if isinstance(v, (int, float)) and not pd.isna(v) and v > 0:
            return "color: #198754; font-weight: 500"
        return ""

    def color_sl(v):
        if isinstance(v, (int, float)) and not pd.isna(v):
            return "color: #dc3545; font-weight: 500"
        return ""

    def color_days(v):
        if isinstance(v, (int, float)) and not pd.isna(v):
            if v <= 10: return "color: #198754; font-weight: 600"
            if v <= 30: return "color: #fd7e14"
            return "color: #6c757d"
        return ""

    fmt = {
        "Price": "₹{:.2f}", "Chg%": "{:+.2f}%",
        "Buy": "₹{:.2f}", "SL": "₹{:.2f}", "Risk%": "{:.2f}%",
        "T1": "₹{:.2f}", "T2": "₹{:.2f}", "T3": "₹{:.2f}",
        "Up%→T1": "{:+.1f}%", "Up%→T2": "{:+.1f}%", "Up%→T3": "{:+.1f}%",
        "RR(T1)": "{:.1f}:1", "RR(T2)": "{:.1f}:1", "RR(T3)": "{:.1f}:1",
        "ATR14": "₹{:.2f}", "ADR%": "{:.2f}%",
        "Vol Surge": "{:.2f}x", "VCP Range%": "{:.2f}%", "Dist 52Hi%": "{:.2f}%",
        "BB Width": "{:.2f}", "ATR Ratio": "{:.3f}",
        "Base Wks": "{:.1f}",
        "EPS Gr%": "{:+.1f}%", "Rev Gr%": "{:+.1f}%",
        "Inst%": "{:.1f}%", "Promoter%": "{:.1f}%",
        "MCap(Cr)": "₹{:,.0f} Cr", "Float%": "{:.1f}%",
        "Ret 1M%": "{:+.1f}%", "Ret 3M%": "{:+.1f}%", "Ret 6M%": "{:+.1f}%",
    }
    fmt = {k: v2 for k, v2 in fmt.items() if k in display_cols_full}

    styler = (
        final_df[display_cols_full].style
            .map(color_score,   subset=["Score"])
            .map(color_chg,     subset=[c for c in ["Chg%","Ret 1M%","Ret 3M%","Ret 6M%","EPS Gr%","Rev Gr%"] if c in display_cols_full])
            .map(color_targets, subset=[c for c in ["T1","T2","T3","Up%→T1","Up%→T2","Up%→T3"] if c in display_cols_full])
            .map(color_sl,      subset=[c for c in ["SL","Risk%"] if c in display_cols_full])
            .map(color_days,    subset=[c for c in ["Days→T1","Days→T2","Days→T3"] if c in display_cols_full])
            .format(fmt, na_rep="N/A")
    )

    st.dataframe(styler, use_container_width=True, height=600)

    # ── Charts ──
    st.markdown("### 📊 Score Distribution")
    st.bar_chart(final_df["Score"].value_counts().sort_index())

    st.markdown("### 🏭 Candidates by Sector")
    sector_chart = final_df["Sector"].value_counts()
    st.bar_chart(sector_chart)

    st.markdown("### 🏆 Top 20 by Score")
    top_cols = ["Symbol","Sector","Price","Score","Grade","Buy","SL","Risk%","T1","T2","T3","Days→T1","Days→T2","Days→T3","Signals"]
    top_cols = [c for c in top_cols if c in final_df.columns]
    st.dataframe(final_df.head(20)[top_cols], use_container_width=True, hide_index=True)

    # ── Download ──
    csv = final_df.drop(columns=["Score/14"]).to_csv(index=False)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name=f"nse_prebreakout_{now}.csv",
        mime="text/csv",
        use_container_width=True,
    )
