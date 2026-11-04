"""
강세주 눌림목 스캐너 (일봉)
- S&P 500 + 관심종목(watchlist.txt)에서
- ① 먼저 크게 오른 강세주가 ② 지지선(10EMA·21EMA·50일선)으로 눌리면서 ③ 변동폭과 거래량이 줄어드는(응축) 종목을 찾고
- 단계별로 나눠 디스코드로 보냅니다.
    📍 접근 중  : 지지선 위 2~6%, 응축하며 내려오는 중 (곧 좋은 자리에 올 종목)
    🎯 지지 도착: 최근 3일 안에 지지선에 닿음. 트리거가(오늘 고가)·손절가·리스크% 계산
    ✅ 발동     : 어제 '지지 도착'이었고 오늘 어제 고가 위로 마감
- 수동 실행 때 티커를 넣으면 조건별 통과/탈락 이유를 보여 줍니다.
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
# ① 강세주
RS_MIN = float(os.environ.get("RS_MIN", "80"))                 # RS 백분위 최소 (1~99)
RUN_MIN = float(os.environ.get("RUN_MIN", "25"))               # 선행 상승률 최소 % (최근 고점 ÷ 그 전 3개월 저점)
NEAR_HIGH = float(os.environ.get("NEAR_HIGH", "25"))           # 52주 고점에서 이 % 이내
# ② 눌림
PB_MIN, PB_MAX = float(os.environ.get("PB_MIN", "3")), float(os.environ.get("PB_MAX", "25"))  # 최근 고점 대비 조정폭 %
HIGH_LOOKBACK = int(os.environ.get("HIGH_LOOKBACK", "30"))     # 최근 고점을 찾는 기간(봉)
TOUCH_PCT = float(os.environ.get("TOUCH_PCT", "2"))            # 지지선 ±이 % 안에 저점이 닿으면 '지지 도착'
TOUCH_BARS = int(os.environ.get("TOUCH_BARS", "3"))            # 최근 이 봉 수 안에 닿았는지
APPROACH_MAX = float(os.environ.get("APPROACH_MAX", "6"))      # 지지선 위 이 % 이내면 '접근 중'
# ③ 응축
RANGE_MAX = float(os.environ.get("RANGE_MAX", "0.75"))         # 최근 5일 평균 변동폭 ÷ 50일 평균 변동폭
VOL_MAX = float(os.environ.get("VOL_MAX", "0.9"))              # 최근 5일 평균 거래량 ÷ 50일 평균 거래량
# ④ 리스크
RISK_MAX = float(os.environ.get("RISK_MAX", "6"))              # 트리거가 → 손절가 거리 최대 %
# 기본 필터
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "10000000"))   # 50일 평균 거래대금($)
MAX_ROWS = int(os.environ.get("MAX_ROWS", "25"))
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"
DIAGNOSE = [t.strip().upper() for t in os.environ.get("DIAGNOSE", "").replace(" ", ",").split(",") if t.strip()]
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (pullback-scanner)"}
SUPPORTS = [("10E", "10EMA"), ("21E", "21EMA"), ("50D", "50일선")]


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


def _watchlist() -> pd.DataFrame:
    rows = []
    p = ROOT / "watchlist.txt"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tk, _, nm = ln.partition(",")
                rows.append({"ticker": re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper()), "name": nm.strip()})
    return pd.DataFrame(rows, columns=["ticker", "name"])


def load_all() -> dict[str, str]:
    out = {}
    for df in (_sp500(), _watchlist()):
        for r in df.itertuples():
            if r.ticker not in out or not out[r.ticker]:
                out[r.ticker] = r.name
    return out


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 200):
        part = tickers[i:i + 200]
        try:
            data = yf.download(part, period="2y", interval="1d", group_by="ticker",
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
                df = data[t].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                if len(df):
                    out[t] = df
        time.sleep(1)
    return out


def rs_raw(c: pd.Series) -> float:
    num = den = 0.0
    for n, w in ((63, 2), (126, 1), (189, 1), (252, 1)):
        if len(c) > n:
            num += (c.iloc[-1] / c.iloc[-1 - n] - 1) * w
            den += w
    return num / den if den else np.nan


# ───────────── 판정 ─────────────
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["10E"] = c.ewm(span=10, adjust=False).mean()
    df["21E"] = c.ewm(span=21, adjust=False).mean()
    df["50D"] = c.rolling(50).mean()
    df["200D"] = c.rolling(200).mean()
    df["rng"] = df["High"] - df["Low"]
    return df


def setup_at(df: pd.DataFrame, i: int) -> tuple[dict | None, str]:
    """i번째 봉 기준 눌림 셋업 판정 (강세 조건 제외, RS는 밖에서). (결과, 탈락 이유)"""
    if i < 260:
        return None, "시세 기간 부족 (약 1년 미만)"
    row = df.iloc[i]
    c, h, l = df["Close"], df["High"], df["Low"]
    close = float(row["Close"])
    if not (close > row["200D"] and row["50D"] > row["200D"]):
        return None, "상승 추세 아님 (주가 > 200일선, 50일선 > 200일선 불충족)"
    hi52 = float(h.iloc[i - 251:i + 1].max())
    if close < hi52 * (1 - NEAR_HIGH / 100):
        return None, f"52주 고점에서 멂 ({(close / hi52 - 1) * 100:.0f}%)"
    # 최근 고점과 선행 상승
    win = h.iloc[i - HIGH_LOOKBACK + 1:i + 1]
    hi_pos = int(np.argmax(win.to_numpy())) + i - HIGH_LOOKBACK + 1
    swing_hi = float(h.iloc[hi_pos])
    if i - hi_pos < 2:
        return None, "아직 고점 근처 (눌림 시작 전)"
    base_lo = float(l.iloc[max(0, hi_pos - 63):hi_pos + 1].min())
    run = (swing_hi / base_lo - 1) * 100
    if run < RUN_MIN:
        return None, f"선행 상승 부족 ({run:.0f}% < {RUN_MIN:g}%)"
    pb = (1 - close / swing_hi) * 100
    if not (PB_MIN <= pb <= PB_MAX):
        return None, f"조정폭 범위 밖 ({pb:.1f}%, 기준 {PB_MIN:g}~{PB_MAX:g}%)"
    # 응축
    rng_r = float(df["rng"].iloc[i - 4:i + 1].mean() / df["rng"].iloc[i - 49:i + 1].mean())
    vol_r = float(df["Volume"].iloc[i - 4:i + 1].mean() / df["Volume"].iloc[i - 49:i + 1].mean())
    if rng_r > RANGE_MAX:
        return None, f"변동폭 응축 부족 ({rng_r:.2f} > {RANGE_MAX:g})"
    if vol_r > VOL_MAX:
        return None, f"거래량 감소 부족 ({vol_r:.2f} > {VOL_MAX:g})"
    # 지지선: 상승 중인 선만 (10봉 전보다 높음)
    best = None
    for key, label in SUPPORTS:
        s = float(df[key].iloc[i])
        if not s > float(df[key].iloc[i - 10]):
            continue
        lows = l.iloc[i - TOUCH_BARS + 1:i + 1]
        sup_hist = df[key].iloc[i - TOUCH_BARS + 1:i + 1]
        touched = bool(((lows <= sup_hist * (1 + TOUCH_PCT / 100)) & (lows >= sup_hist * (1 - TOUCH_PCT / 100))).any()
                       or ((lows < sup_hist) & (c.iloc[i - TOUCH_BARS + 1:i + 1] >= sup_hist * (1 - TOUCH_PCT / 100))).any())
        dist = (close / s - 1) * 100
        if close < s * (1 - TOUCH_PCT / 100):
            continue                                                  # 이미 아래로 깨짐 → 다음 지지선
        if touched:
            cand = ("도착", key, label, s, dist)
        elif 0 < dist <= APPROACH_MAX:
            cand = ("접근", key, label, s, dist)
        else:
            continue
        best = cand
        break                                                         # 가장 가까운(짧은) 지지선 우선
    if best is None:
        return None, "지지선(상승 중인 10EMA·21EMA·50일선) 근처가 아님"
    stage, key, label, s, dist = best
    trigger = float(row["High"])
    stop = float(min(l.iloc[i - TOUCH_BARS + 1:i + 1].min(), s * (1 - TOUCH_PCT / 100))) * 0.995
    risk = (trigger / stop - 1) * 100
    if stage == "도착" and risk > RISK_MAX:
        return None, f"손절까지 너무 멂 (리스크 {risk:.1f}% > {RISK_MAX:g}%)"
    return {"stage": stage, "sup": key, "sup_label": label, "sup_px": s, "dist": dist, "pb": pb, "run": run,
            "rng": rng_r, "vol": vol_r, "trigger": trigger, "stop": stop, "risk": risk, "close": close,
            "hi_ago": i - hi_pos}, ""


def check(df: pd.DataFrame) -> tuple[dict | None, str]:
    df = prepare(df.dropna(subset=["Close"]))
    n = len(df)
    if n < 262:
        return None, "시세 기간 부족 (약 1년 미만)"
    close = float(df["Close"].iloc[-1])
    if close < MIN_PRICE or float((df["Close"] * df["Volume"]).tail(50).mean()) < MIN_DOLLAR_VOL:
        return None, "주가 또는 거래대금 미달"
    # ✅ 발동: 어제 '지지 도착' + 오늘 어제 고가 위 마감
    y, _ = setup_at(df, n - 2)
    if y and y["stage"] == "도착" and close > y["trigger"]:
        vol_today = float(df["Volume"].iloc[-1] / df["Volume"].iloc[-51:-1].mean())
        stop = y["stop"]
        return {**y, "stage": "발동", "close": close, "trigger": y["trigger"], "stop": stop,
                "risk": (close / stop - 1) * 100, "vol_today": vol_today,
                "chg": float((close / df["Close"].iloc[-2] - 1) * 100)}, ""
    r, why = setup_at(df, n - 1)
    if r:
        r["chg"] = float((close / df["Close"].iloc[-2] - 1) * 100)
        r["vol_today"] = float(df["Volume"].iloc[-1] / df["Volume"].iloc[-51:-1].mean())
    return r, why


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


def table_trade(title: str, rows: list[dict], fired: bool):
    head = (fit("TICKER", 7) + fit("지지", 5) + fit("고점%", 7, True) + fit("응축", 6, True) + fit("거래량", 7, True)
            + fit("종가" if fired else "트리거", 9, True) + fit("손절", 9, True) + fit("리스크", 7, True)
            + fit("RS", 4, True) + "  이름")
    lines = [fit(r["t"], 7) + fit(r["sup"], 5) + f"{-r['pb']:>+7.1f}{r['rng']:>6.2f}"
             + f"{(r['vol_today'] if fired else r['vol']):>7.2f}"
             + f"{(r['close'] if fired else r['trigger']):>9.2f}{r['stop']:>9.2f}{r['risk']:>6.1f}%{r['rs']:>4}  {r['name'][:18]}"
             for r in rows]
    send_lines(title, head, lines)


def table_approach(title: str, rows: list[dict]):
    head = (fit("TICKER", 7) + fit("지지", 5) + fit("지지가", 9, True) + fit("거리%", 7, True) + fit("고점%", 7, True)
            + fit("응축", 6, True) + fit("거래량", 7, True) + fit("RS", 4, True) + "  이름")
    lines = [fit(r["t"], 7) + fit(r["sup"], 5) + f"{r['sup_px']:>9.2f}{r['dist']:>+7.1f}{-r['pb']:>+7.1f}"
             f"{r['rng']:>6.2f}{r['vol']:>7.2f}{r['rs']:>4}  {r['name'][:18]}"
             for r in rows]
    send_lines(title, head, lines)


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE and not DIAGNOSE:
        print("주말. 종료.")
        return
    names = load_all()
    tickers = list(names) if not DIAGNOSE else list(dict.fromkeys(DIAGNOSE + list(names)))
    print(f"대상 {len(tickers)}종목")
    data = download(tickers)
    if not data:
        post("⚠️ 눌림목 스캐너: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    rs = pd.Series({t: rs_raw(df["Close"].dropna()) for t, df in data.items() if len(df) > 252})
    rs = (rs.rank(pct=True) * 98 + 1).round().fillna(0).astype(int)
    found, why = [], {}
    for t in (DIAGNOSE or list(data)):
        if t not in data:
            why[t] = "시세를 받지 못함 (티커 확인 필요)"
            continue
        try:
            r, w = check(data[t])
        except Exception as e:
            r, w = None, f"계산 오류: {e}"
        score = int(rs.get(t, 0))
        if r and score < RS_MIN:
            w, r = f"RS 부족 ({score} < {RS_MIN:g})", None
        why[t] = w
        if r:
            found.append({"t": t, "name": names.get(t, ""), "rs": score, **r})

    if DIAGNOSE:
        lines = []
        for t in DIAGNOSE:
            r = next((f for f in found if f["t"] == t), None)
            if r is None:
                lines.append(f"**{t}** ❌ {why.get(t, '알 수 없음')} (RS {int(rs.get(t, 0))})")
                continue
            mark = {"발동": "✅ 발동", "도착": "🎯 지지 도착", "접근": "📍 접근 중"}[r["stage"]]
            lines.append(f"**{t}** {mark} · {r['sup_label']} {r['sup_px']:.2f} (거리 {r['dist']:+.1f}%) · 고점 대비 {-r['pb']:+.1f}% · "
                         f"선행 상승 +{r['run']:.0f}% · 응축 {r['rng']:.2f} · 거래량 {r['vol']:.2f} · "
                         f"트리거 {r['trigger']:.2f} · 손절 {r['stop']:.2f} (리스크 {r['risk']:.1f}%) · RS {r['rs']}")
        post("**눌림목 스캐너 진단**\n" + "\n".join(lines))
        return

    by = {s: sorted([f for f in found if f["stage"] == s], key=lambda r: -r["rs"]) for s in ("발동", "도착", "접근")}
    session = max(df.index[-1] for df in data.values()).date()
    post(f"**🪜 강세주 눌림목 | {session} 마감**\n"
         f"✅ 발동 {len(by['발동'])} · 🎯 지지 도착 {len(by['도착'])} · 📍 접근 중 {len(by['접근'])} (확인 {len(data)}종목)")

    def cap(rows):
        return rows[:MAX_ROWS], (f" · 상위 {MAX_ROWS}개 (전체 {len(rows)})" if len(rows) > MAX_ROWS else "")
    if by["발동"]:
        rows, more = cap(by["발동"])
        table_trade(f"✅ 발동 · 어제 지지 도착 → 오늘 어제 고가 위 마감{more}", rows, fired=True)
    if by["도착"]:
        rows, more = cap(by["도착"])
        table_trade(f"🎯 지지 도착 · 내일 트리거가(오늘 고가) 넘으면 진입 후보{more}", rows, fired=False)
    if by["접근"]:
        rows, more = cap(by["접근"])
        table_approach(f"📍 접근 중 · 응축하며 지지선 위 {APPROACH_MAX:g}% 이내로 내려오는 중{more}", rows)
    if found:
        post("지지 = 10E(10EMA) · 21E(21EMA) · 50D(50일선) · 고점% = 최근 고점 대비 · 응축 = 최근 5일 변동폭 ÷ 50일 평균 "
             "(작을수록 조용함) · 거래량 = 최근 5일 ÷ 50일 평균 (발동 표는 오늘 ÷ 50일 평균) · "
             "손절 = 최근 3일 저점과 지지선 −2% 중 낮은 값 아래 · 리스크 = 트리거(발동은 종가) → 손절 거리")
    # 탈락 이유 집계 (어느 조건에서 많이 빠지는지)
    from collections import Counter
    strong = [t for t in data if int(rs.get(t, 0)) >= RS_MIN]
    cnt = Counter(why[t].split(" (")[0] for t in strong if why.get(t))
    if cnt:
        post(f"🔎 RS {RS_MIN:g} 이상 {len(strong)}종목 중 탈락 이유: "
             + " · ".join(f"{k} {v}" for k, v in cnt.most_common(6)))
    if len(data) < len(tickers) * 0.5:
        post(f"⚠️ 눌림목 스캐너: 시세를 {len(data)}/{len(tickers)}종목만 받았습니다.")
    print(f"완료: 확인 {len(data)}종목, 발동 {len(by['발동'])}, 도착 {len(by['도착'])}, 접근 {len(by['접근'])}")


if __name__ == "__main__":
    sys.exit(main())
