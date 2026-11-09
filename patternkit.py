"""
차트 패턴 스캐너 공통 도구 (bottoms·wedge가 같이 씀)
- 종목 목록(S&P 500 + watchlist.txt + etfs.txt), 시세 다운로드, 디스코드 전송
- 스윙 고점·저점 찾기, 추세선 맞추기, 캔들 차트 그림 그리기, 재알림 방지
"""
import datetime as dt
import io
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (pattern-scanner)"}
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"
DIAGNOSE = [t.strip().upper() for t in os.environ.get("DIAGNOSE", "").replace(" ", ",").split(",") if t.strip()]
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "10000000"))       # 종목: 50일 평균 거래대금($)
MIN_DOLLAR_VOL_ETF = float(os.environ.get("MIN_DOLLAR_VOL_ETF", "5000000"))  # ETF


# ───────────── 종목 목록 ─────────────
def _sp500(state: Path) -> pd.DataFrame:
    cache = state / "sp500.csv"
    if not cache.exists() or time.time() - cache.stat().st_mtime > 7 * 86400:
        try:
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({"ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False), "name": t["Security"]})
            if len(df) > 400:
                df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] S&P 500 목록 갱신 실패: {e}")
    return pd.read_csv(cache).fillna("") if cache.exists() else pd.DataFrame(columns=["ticker", "name"])


def _txt(root: Path, name: str) -> pd.DataFrame:
    rows = []
    p = root / name
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tk, _, nm = ln.partition(",")
                rows.append({"ticker": re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper()), "name": nm.strip()})
    return pd.DataFrame(rows, columns=["ticker", "name"]).drop_duplicates("ticker")


def load_all(root: Path, state: Path) -> dict[str, tuple[str, str]]:
    """티커 → (이름, 'stock' / 'etf')"""
    out = {}
    for r in _txt(root, "etfs.txt").itertuples():
        out[r.ticker] = (r.name, "etf")
    for df in (_sp500(state), _txt(root, "watchlist.txt")):
        for r in df.itertuples():
            if r.ticker not in out:
                out[r.ticker] = (r.name, "stock")
    return out


def download(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 200):
        part = tickers[i:i + 200]
        try:
            data = yf.download(part, period=period, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({part[0]}…): {e}")
            continue
        if data is None or data.empty:
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({part[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in part:
            if t in have:
                df = data[t].dropna(subset=["Open", "High", "Low", "Close"])
                if len(df):
                    out[t] = df
        time.sleep(1)
    return out


def liquid(df: pd.DataFrame, kind: str) -> str:
    """유동성 필터. 통과하면 빈 문자열, 아니면 탈락 이유"""
    c = float(df["Close"].iloc[-1])
    dv = float((df["Close"] * df["Volume"]).tail(50).mean())
    if c < MIN_PRICE or dv < (MIN_DOLLAR_VOL_ETF if kind == "etf" else MIN_DOLLAR_VOL):
        return f"주가 또는 거래대금 미달 (${c:.2f}, 50일 평균 ${dv/1e6:.0f}M)"
    return ""


# ───────────── 스윙 고점·저점 ─────────────
def pivots(h: np.ndarray, l: np.ndarray, k: int = 4, min_move: float = 3.0) -> list[tuple[int, str, float]]:
    """앞뒤 k봉보다 높은(낮은) 봉 → 고점 H / 저점 L. 같은 종류가 연속이면 더 극단값만 남기고,
    직전 반대 꼭짓점 대비 min_move% 미만 움직임은 잡음으로 제거 → H·L이 번갈아 나오는 목록"""
    n = len(h)
    raw = []
    for i in range(k, n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        if h[i] == h[lo:hi].max():
            raw.append((i, "H", float(h[i])))
        if l[i] == l[lo:hi].min():
            raw.append((i, "L", float(l[i])))
    raw.sort(key=lambda x: (x[0], x[1] == "H"))
    out: list[tuple[int, str, float]] = []
    for p in raw:
        if out and out[-1][1] == p[1]:
            if (p[1] == "H" and p[2] >= out[-1][2]) or (p[1] == "L" and p[2] <= out[-1][2]):
                out[-1] = p
            continue
        if out and abs(p[2] / out[-1][2] - 1) * 100 < min_move:
            if len(out) >= 2 and out[-2][1] == p[1]:
                if (p[1] == "H" and p[2] > out[-2][2]) or (p[1] == "L" and p[2] < out[-2][2]):
                    out.pop()
                    out[-1] = p
            continue
        out.append(p)
    return out


def line_at(i1: int, v1: float, i2: int, v2: float, i: float) -> float:
    return v1 + (v2 - v1) * (i - i1) / (i2 - i1)


# ───────────── 디스코드 ─────────────
def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content[:1990]}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def fit(s: str, w: int, right: bool = False) -> str:
    out = ""
    for ch in s:
        if width(out + ch) > w:
            break
        out += ch
    pad = " " * (w - width(out))
    return pad + out if right else out + pad


def send_lines(title: str, head: str, lines: list[str]):
    first, part = True, 1
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else f"**{title} ({part})**\n"
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first, part = False, part + 1


def px(x: float) -> str:
    return f"{x:,.2f}" if x < 1000 else f"{x:,.0f}"


# ───────────── 재알림 방지 ─────────────
class Sent:
    def __init__(self, state: Path, days: int):
        self.f, self.days = state / "sent.json", days
        self.db = json.loads(self.f.read_text()) if self.f.exists() else {}
        self.today = dt.datetime.now(NY).date()

    def fresh(self, key: str) -> bool:
        d = self.db.get(key)
        return d is None or (self.today - dt.date.fromisoformat(d)).days >= self.days

    def mark(self, key: str):
        self.db[key] = str(self.today)

    def save(self):
        cut = self.today - dt.timedelta(days=60)
        self.db = {k: v for k, v in self.db.items() if dt.date.fromisoformat(v) >= cut}
        self.f.write_text(json.dumps(self.db))


# ───────────── 캔들 차트 그림 ─────────────
def candle_chart(df: pd.DataFrame, start: int, title: str, subtitle: str = "",
                 lines: list[dict] | None = None, marks: list[dict] | None = None, hlines: list[dict] | None = None):
    """df의 start번째 봉부터 끝까지 캔들. lines = [{i1,v1,i2,v2,color,ls,label}] (봉 번호 기준, 오른쪽 끝까지 연장 가능),
    marks = [{i, v, text, color}], hlines = [{v, color, ls, label}]"""
    import charts
    import matplotlib.pyplot as plt
    seg = df.iloc[start:]
    n = len(seg)
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04})
    o, h, l, c, v = (seg[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    up = c >= o
    x = np.arange(n)
    wbody = 0.6 if n < 150 else 0.75
    ax.vlines(x, l, h, color=np.where(up, charts.GREEN, charts.RED), lw=0.8)
    ax.bar(x, np.maximum(np.abs(c - o), (h - l) * 0.02), bottom=np.minimum(o, c), width=wbody,
           color=np.where(up, charts.GREEN, charts.RED), edgecolor="none")
    axv.bar(x, v, width=wbody, color=np.where(up, charts.GREEN, charts.RED), alpha=0.6)
    for ln in lines or []:
        x1, x2 = ln["i1"] - start, ln.get("i3", ln["i2"]) - start
        y1 = ln["v1"]
        y2 = line_at(ln["i1"], ln["v1"], ln["i2"], ln["v2"], ln.get("i3", ln["i2"]))
        ax.plot([x1, x2], [y1, y2], color=ln.get("color", charts.YELLOW), lw=1.6, ls=ln.get("ls", "-"),
                label=ln.get("label"))
    for hl in hlines or []:
        ax.axhline(hl["v"], color=hl.get("color", charts.BLUE), lw=1.2, ls=hl.get("ls", "--"), label=hl.get("label"))
    for m in marks or []:
        ax.annotate(m["text"], (m["i"] - start, m["v"]), xytext=(0, -16 if m.get("below", True) else 12),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold",
                    color=m.get("color", charts.YELLOW))
    ax.set_xlim(-1, n + 3)
    ymin, ymax = l.min(), h.max()
    for hl in hlines or []:
        ymax, ymin = max(ymax, hl["v"]), min(ymin, hl["v"])
    pad = (ymax - ymin) * 0.06
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, alpha=0.4)
    axv.grid(True, alpha=0.3)
    axv.set_yticks([])
    for a in (ax, axv):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ticks = list(range(0, n, max(1, n // 8)))
    axv.set_xticks(ticks, [seg.index[i].strftime("%y/%m/%d") for i in ticks], fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    if any(ln.get("label") for ln in (lines or []) + (hlines or [])):
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=4, frameon=False, labelcolor=charts.TEXT, fontsize=9)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=charts.TEXT, pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color=charts.SUB, va="bottom")
    return fig
