"""
아일랜드 갭(아일랜드 리버설) 스캐너 — 추세 전환 신호
- S&P 500 + 관심종목(watchlist.txt) + ETF(etfs.txt)
- 🟢 아일랜드 바텀: 갭 하락 → 아래에서 1~60봉 머묾 → 갭 상승  (하락 → 상승 전환)
- 🔴 아일랜드 톱  : 갭 상승 → 위에서 1~60봉 머묾 → 갭 하락  (상승 → 하락 전환)
- 두 번째 갭이 최근 3봉 안에 생겼고 아직 메워지지 않은 종목을 디스코드로 보냅니다.
- 수동 실행 때 티커를 넣으면 최근 섬과 탈락 이유를 보여 줍니다.
"""
import datetime as dt
import io
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
MAX_ISLAND = int(os.environ.get("MAX_ISLAND", "60"))           # 섬 최대 길이(봉). 1봉짜리 섬부터 인정
MIN_GAP = float(os.environ.get("MIN_GAP", "0.5"))              # 두 갭 각각 최소 크기(%)
RECENT_BARS = int(os.environ.get("RECENT_BARS", "3"))          # 두 번째 갭이 최근 이 봉 수 안에 생긴 것만
TREND_BARS = int(os.environ.get("TREND_BARS", "20"))           # 섬 직전 추세를 볼 기간(봉)
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "10000000"))       # 종목: 50일 평균 거래대금($)
MIN_DOLLAR_VOL_ETF = float(os.environ.get("MIN_DOLLAR_VOL_ETF", "5000000"))  # ETF
MAX_ROWS = int(os.environ.get("MAX_ROWS", "30"))
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"
DIAGNOSE = [t.strip().upper() for t in os.environ.get("DIAGNOSE", "").replace(" ", ",").split(",") if t.strip()]
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (island-scanner)"}


# ───────────── 종목 목록 ─────────────
def _sp500() -> pd.DataFrame:
    cache = STATE / "sp500.csv"
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


def _txt(name: str) -> pd.DataFrame:
    rows = []
    p = ROOT / name
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tk, _, nm = ln.partition(",")
                rows.append({"ticker": re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper()), "name": nm.strip()})
    return pd.DataFrame(rows, columns=["ticker", "name"]).drop_duplicates("ticker")


def load_all() -> dict[str, tuple[str, str]]:
    """티커 → (이름, 'stock' / 'etf')"""
    out = {}
    for r in _txt("etfs.txt").itertuples():
        out[r.ticker] = (r.name, "etf")
    for df in (_sp500(), _txt("watchlist.txt")):
        for r in df.itertuples():
            if r.ticker not in out:
                out[r.ticker] = (r.name, "stock")
    return out


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 200):
        part = tickers[i:i + 200]
        try:
            data = yf.download(part, period="1y", interval="1d", group_by="ticker",
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


# ───────────── 판정 ─────────────
def island_at(H: np.ndarray, L: np.ndarray, e: int) -> dict | None:
    """e번째 봉이 섬을 끝내는 두 번째 갭인지. 가장 짧은 섬을 찾음"""
    if e < 2:
        return None
    # 🟢 바텀: e에서 갭 상승 → 앞쪽에 갭 하락으로 시작한 섬이 아래에 고립
    if L[e] > H[e - 1]:
        top_isl = -np.inf
        for k in range(1, MAX_ISLAND + 1):
            s = e - k
            if s < 1:
                break
            top_isl = max(top_isl, H[s])
            if top_isl >= L[e]:
                break                                                 # 섬이 두 번째 갭 위로 올라감 → 고립 안 됨
            if H[s] < L[s - 1] and top_isl < L[s - 1]:                # s에서 갭 하락으로 섬 시작, 섬 전체가 그 갭 아래
                g1 = (L[s - 1] - top_isl) / L[s - 1] * 100
                g2 = (L[e] - top_isl) / top_isl * 100
                if g1 >= MIN_GAP and g2 >= MIN_GAP:
                    return {"kind": "바텀", "start": s, "end": e, "len": k, "gap1": -g1, "gap2": g2,
                            "zone_lo": float(top_isl), "zone_hi": float(min(L[s - 1], L[e]))}
    # 🔴 톱: e에서 갭 하락 → 앞쪽에 갭 상승으로 시작한 섬이 위에 고립
    if H[e] < L[e - 1]:
        bot_isl = np.inf
        for k in range(1, MAX_ISLAND + 1):
            s = e - k
            if s < 1:
                break
            bot_isl = min(bot_isl, L[s])
            if bot_isl <= H[e]:
                break
            if L[s] > H[s - 1] and bot_isl > H[s - 1]:
                g1 = (bot_isl - H[s - 1]) / H[s - 1] * 100
                g2 = (bot_isl - H[e]) / bot_isl * 100
                if g1 >= MIN_GAP and g2 >= MIN_GAP:
                    return {"kind": "톱", "start": s, "end": e, "len": k, "gap1": g1, "gap2": -g2,
                            "zone_lo": float(max(H[s - 1], H[e])), "zone_hi": float(bot_isl)}
    return None


def check(df: pd.DataFrame, kind: str) -> tuple[dict | None, str, dict | None]:
    """(최근 섬 결과, 탈락 이유, 진단용 최근 120봉 안 마지막 섬)"""
    df = df.dropna(subset=["High", "Low", "Close"])
    n = len(df)
    if n < 60:
        return None, "시세 기간 부족 (3개월 미만)", None
    H, L, C = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    V = df["Volume"].fillna(0).to_numpy(float)
    dollar = float((df["Close"] * df["Volume"]).tail(50).mean())
    last_any = None
    for e in range(n - 1, max(1, n - 120), -1):
        isl = island_at(H, L, e)
        if isl:
            last_any = {**isl, "date": df.index[e].date().isoformat(), "ago": n - 1 - e}
            break
    if C[-1] < MIN_PRICE or dollar < (MIN_DOLLAR_VOL_ETF if kind == "etf" else MIN_DOLLAR_VOL):
        return None, f"주가 또는 거래대금 미달 (${C[-1]:.2f}, 50일 평균 ${dollar/1e6:.0f}M)", last_any
    for e in range(n - 1, n - 1 - RECENT_BARS, -1):
        isl = island_at(H, L, e)
        if not isl:
            continue
        # 두 번째 갭이 메워졌는지 (그 뒤 봉이 섬 쪽 가격대로 되돌아옴)
        after = slice(e + 1, n)
        if isl["kind"] == "바텀" and (L[after] <= isl["zone_lo"]).any():
            return None, f"{df.index[e].date()} 아일랜드 바텀 → 이미 갭이 메워짐", last_any
        if isl["kind"] == "톱" and (H[after] >= isl["zone_hi"]).any():
            return None, f"{df.index[e].date()} 아일랜드 톱 → 이미 갭이 메워짐", last_any
        s = isl["start"]
        t0 = max(0, s - 1 - TREND_BARS)
        trend = (C[s - 1] / C[t0] - 1) * 100
        vavg = V[max(0, e - 50):e].mean()
        isl.update({
            "date": df.index[e].date().isoformat(), "ago": n - 1 - e, "close": float(C[-1]),
            "trend": float(trend), "vol": float(V[e] / vavg) if vavg > 0 else float("nan"),
            "chg_since": float((C[-1] / C[e - 1] - 1) * 100),
            "fit": (isl["kind"] == "바텀" and trend < 0) or (isl["kind"] == "톱" and trend > 0),
        })
        return isl, "", last_any
    return None, f"최근 {RECENT_BARS}봉 안에 섬을 완성하는 갭 없음", last_any


# ───────────── 디스코드 ─────────────
def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        res = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if res.status_code == 429:
            time.sleep(float(res.json().get("retry_after", 2)) + 0.5)
            continue
        res.raise_for_status()
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


def px(x: float) -> str:
    return f"{x:,.2f}" if x < 1000 else f"{x:,.0f}"


def send_table(title: str, rows: list[dict]):
    head = (fit("TICKER", 7) + fit("섬", 4, True) + fit("갭1%", 7, True) + fit("갭2%", 7, True) + fit("거래량", 7, True)
            + fit("선행%", 7, True) + fit("갭존", 17, True) + fit("종가", 9, True) + fit("이후%", 7, True) + "  이름")
    lines = [fit(r["t"], 7) + fit(f"{r['len']}봉", 4, True) + f"{r['gap1']:>+7.1f}{r['gap2']:>+7.1f}{r['vol']:>7.1f}"
             + fit(f"{'⭐' if r['fit'] else ''}{r['trend']:+.0f}", 7, True)
             + fit(f"{px(r['zone_lo'])}~{px(r['zone_hi'])}", 17, True) + fit(px(r["close"]), 9, True)
             + f"{r['chg_since']:>+7.1f}  " + r["name"][:18]
             for r in rows]
    first, part = True, 1
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else f"**{title} ({part})**\n"
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first, part = False, part + 1


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE and not DIAGNOSE:
        print("주말. 종료.")
        return
    universe = load_all()
    tickers = DIAGNOSE or list(universe)
    print(f"대상 {len(tickers)}종목")
    data = download(tickers)
    if not data:
        post("⚠️ 아일랜드 갭: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    found, why, last = [], {}, {}
    for t in tickers:
        if t not in data:
            why[t] = "시세를 받지 못함 (티커 확인 필요)"
            continue
        name, kind = universe.get(t, ("", "stock"))
        try:
            r, w, la = check(data[t], kind)
        except Exception as e:
            r, w, la = None, f"계산 오류: {e}", None
        why[t], last[t] = w, la
        if r:
            found.append({"t": t, "name": name, "type": kind, **r})

    if DIAGNOSE:
        lines = []
        for t in DIAGNOSE:
            r = next((f for f in found if f["t"] == t), None)
            if r:
                mark = "🟢 아일랜드 바텀" if r["kind"] == "바텀" else "🔴 아일랜드 톱"
                lines.append(f"**{t}** {mark} · {r['date']} 완성 · 섬 {r['len']}봉 · 갭 {r['gap1']:+.1f}% / {r['gap2']:+.1f}% · "
                             f"갭존 {px(r['zone_lo'])}~{px(r['zone_hi'])} · 갭날 거래량 {r['vol']:.1f}배 · 선행 추세 {r['trend']:+.0f}%")
                continue
            lines.append(f"**{t}** ❌ {why.get(t, '알 수 없음')}")
            la = last.get(t)
            if la:
                lines.append(f"　최근 섬: {la['date']} 아일랜드 {la['kind']} (섬 {la['len']}봉, {la['ago']}봉 전)")
        post("**아일랜드 갭 진단**\n" + "\n".join(lines))
        return

    session = max(df.index[-1] for df in data.values()).date()
    bottoms = sorted([f for f in found if f["kind"] == "바텀"], key=lambda r: (r["ago"], -r["gap2"]))
    tops = sorted([f for f in found if f["kind"] == "톱"], key=lambda r: (r["ago"], r["gap2"]))
    post(f"**🏝️ 아일랜드 갭 | {session} 마감**\n"
         f"🟢 바텀 {len(bottoms)} · 🔴 톱 {len(tops)} (최근 {RECENT_BARS}봉 안에 완성, 갭 안 메워짐 · 확인 {len(data)}종목)")
    for title, rows in (("🟢 아일랜드 바텀 · 하락 → 상승 전환", bottoms), ("🔴 아일랜드 톱 · 상승 → 하락 전환", tops)):
        for typ, label in (("stock", "종목"), ("etf", "ETF")):
            sub = [r for r in rows if r["type"] == typ]
            if sub:
                more = f" · 상위 {MAX_ROWS}개 (전체 {len(sub)})" if len(sub) > MAX_ROWS else ""
                send_table(f"{title} · {label}{more}", sub[:MAX_ROWS])
    if found:
        post("섬 = 두 갭 사이 봉 수 · 갭1% = 섬으로 들어간 갭 · 갭2% = 섬에서 빠져나온 갭 (전환 방향) · "
             "거래량 = 갭2 날 거래량 ÷ 50일 평균 · 선행% = 섬 직전 20봉 등락 (⭐ = 반전 방향과 맞음: 바텀은 하락 후, 톱은 상승 후) · "
             "갭존 = 두 갭이 겹치는 가격대 (바텀은 지지, 톱은 저항) · 이후% = 섬 마지막 봉 종가 대비 지금")
    if len(data) < len(tickers) * 0.5:
        post(f"⚠️ 아일랜드 갭: 시세를 {len(data)}/{len(tickers)}종목만 받았습니다.")
    print(f"완료: 확인 {len(data)}종목, 바텀 {len(bottoms)}, 톱 {len(tops)}")


if __name__ == "__main__":
    sys.exit(main())
