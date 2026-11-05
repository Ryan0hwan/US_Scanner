"""
강세주 눌림목 스캐너 v2 — 마틴 룩(Martin Luk) 방식 (일봉)
S&P 500 + 관심종목(watchlist.txt)

1) 시장 내부 강도: 전 종목을 EMA 배열로 Lead / Mediocre / Lag 분류, 어제와 비교
     Lead     = 9 EMA > 21 EMA > 50 EMA
     Lag      = 9 EMA < 21 EMA < 50 EMA
     Mediocre = 그 사이
2) 🆕 새로 Lead 진입: 같은 업종에서 여러 종목이 동시에 올라오면 자금 유입 신호
3) ⚡ Potent: 21·50 EMA 위 + 변동성·거래대금 충분 + 오늘 확실히 움직인 종목 (새 강세 후보)
4) 눌림목: Lead 종목이 최근 고점에서 조정받아 **여러 지지선이 겹치는 자리**에 도착
     지지선 = 9 EMA · 21 EMA · 50 EMA · 이전 고점(돌파한 저항) · AVWAP(상승 시작점 기준)
     단계: 📍 접근 중 → 🎯 지지 도착 (내일 오늘 고가 돌파 시 진입, 손절 = 오늘 저가) → ✅ 발동
     실적 발표 5일 이내는 ⚠️ 표시 (마틴 룩은 대부분 신규 진입을 피함)
진입 타이밍(5분봉 돌파)은 장중에 직접 확인합니다. 이 시스템은 후보·지지 구간·손절가를 미리 준비해 줍니다.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
# 기본 필터 (마틴 룩 스크리너 기준을 일봉으로 옮김)
MIN_AVG_VOL = float(os.environ.get("MIN_AVG_VOL", "500000"))          # 10일 평균 거래량(주)
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "50000000"))  # 20일 평균 거래대금($) — Price × Volume
ADR_MIN = float(os.environ.get("ADR_MIN", "2.5"))                     # 20일 평균 일중 변동폭(%) — 원래 주간 변동성 3.5%, 대형주 위주라 완화
# Leader (강세주)
LEADER_PCT = float(os.environ.get("LEADER_PCT", "70"))                # 1개월 또는 3개월 수익률 백분위가 이 이상
# 눌림목
PB_MIN, PB_MAX = float(os.environ.get("PB_MIN", "3")), float(os.environ.get("PB_MAX", "20"))  # 최근 20일 고점 대비 조정폭(%)
BAND = float(os.environ.get("BAND", "1.5"))                           # 지지선 ±이 % 안이면 '닿음'
MIN_CONFLUENCE = int(os.environ.get("MIN_CONFLUENCE", "2"))           # 겹쳐야 하는 지지선 최소 개수
APPROACH_MAX = float(os.environ.get("APPROACH_MAX", "5"))             # 지지 구간 위 이 % 이내면 '접근 중'
RISK_MAX = float(os.environ.get("RISK_MAX", "7"))                     # 트리거(오늘 고가) → 손절(오늘 저가) 최대 %
EARN_DAYS = int(os.environ.get("EARN_DAYS", "5"))                     # 실적 발표가 이 일수 안이면 ⚠️
# Potent
POTENT_MOVE = float(os.environ.get("POTENT_MOVE", "0.5"))             # 시가 대비 ±이 % 밖으로 움직인 날
POTENT_ADR = float(os.environ.get("POTENT_ADR", "3.5"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "20"))
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"
DIAGNOSE = [t.strip().upper() for t in os.environ.get("DIAGNOSE", "").replace(" ", ",").split(",") if t.strip()]
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (pullback-scanner)"}
CLASS_ICON = {"Lead": "🟢", "Mediocre": "🟡", "Lag": "🔴"}


# ───────────── 종목 목록 ─────────────
def _sp500() -> pd.DataFrame:
    cache = STATE / "sp500_v2.csv"
    if not cache.exists() or time.time() - cache.stat().st_mtime > 7 * 86400:
        try:
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({"ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
                               "name": t["Security"], "industry": t["GICS Sub-Industry"]})
            if len(df) > 400:
                df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] S&P 500 목록 갱신 실패: {e}")
    return pd.read_csv(cache).fillna("") if cache.exists() else pd.DataFrame(columns=["ticker", "name", "industry"])


def _watchlist() -> pd.DataFrame:
    rows = []
    p = ROOT / "watchlist.txt"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tk, _, nm = ln.partition(",")
                rows.append({"ticker": re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper()), "name": nm.strip(), "industry": ""})
    return pd.DataFrame(rows, columns=["ticker", "name", "industry"])


def load_all() -> dict[str, dict]:
    out = {}
    for df in (_sp500(), _watchlist()):
        for r in df.itertuples():
            if r.ticker not in out:
                out[r.ticker] = {"name": r.name, "industry": r.industry or "관심종목"}
            elif not out[r.ticker]["name"]:
                out[r.ticker]["name"] = r.name
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


def earnings_in(t: str) -> int | None:
    """다음 실적 발표까지 남은 일수 (모르면 None)"""
    try:
        import yfinance as yf
        cal = yf.Ticker(t).calendar
        ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not ds:
            return None
        today = dt.datetime.now(NY).date()
        days = [(pd.Timestamp(d).date() - today).days for d in ds]
        days = [d for d in days if d >= 0]
        return min(days) if days else None
    except Exception:
        return None


# ───────────── 지표 ─────────────
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    for n in (9, 21, 50, 150):
        df[f"E{n}"] = c.ewm(span=n, adjust=False).mean()
    df["adr"] = ((df["High"] / df["Low"] - 1) * 100).rolling(20).mean()
    df["dvol"] = (c * df["Volume"]).rolling(20).mean()
    df["vol10"] = df["Volume"].rolling(10).mean()
    return df


def classify(row) -> str:
    if row["E9"] > row["E21"] > row["E50"]:
        return "Lead"
    if row["E9"] < row["E21"] < row["E50"]:
        return "Lag"
    return "Mediocre"


def avwap(df: pd.DataFrame, start: int, end: int) -> float:
    seg = df.iloc[start:end + 1]
    tp = (seg["High"] + seg["Low"] + seg["Close"]) / 3
    v = seg["Volume"]
    return float((tp * v).sum() / v.sum()) if v.sum() > 0 else np.nan


def pivot_highs(h: np.ndarray, k: int = 5) -> list[int]:
    return [i for i in range(k, len(h) - k) if h[i] == h[i - k:i + k + 1].max()]


def weekly_up(df: pd.DataFrame, i: int) -> bool:
    w = df["Close"].iloc[:i + 1].resample("W-FRI").last().dropna()
    if len(w) < 30:
        return False
    e = w.ewm(span=10, adjust=False).mean()                           # 주봉 10 EMA ≈ 일봉 50 EMA
    return bool(w.iloc[-1] > e.iloc[-1] and e.iloc[-1] > e.iloc[-4])


def higher_lows(l: np.ndarray, i: int) -> bool:
    """최근 두 개의 스윙 저점(앞뒤 3봉)이 높아지는 중인지"""
    lows = [j for j in range(i - 60, i - 2) if j > 3 and l[j] == l[j - 3:j + 4].min()]
    return len(lows) >= 2 and l[lows[-1]] > l[lows[-2]]


def setup_at(df: pd.DataFrame, i: int) -> tuple[dict | None, str]:
    """i번째 봉 기준 눌림목 판정 (Lead·필터·리더 조건은 밖에서)"""
    row = df.iloc[i]
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    hi_pos = int(np.argmax(h[i - 19:i + 1])) + i - 19
    swing_hi = float(h[hi_pos])
    if i - hi_pos < 1:
        return None, "오늘이 고점 (아직 눌림 아님)"
    pb = (1 - c[i] / swing_hi) * 100
    if not (PB_MIN <= pb <= PB_MAX):
        return None, f"조정폭 범위 밖 ({pb:.1f}%, 기준 {PB_MIN:g}~{PB_MAX:g}%)"
    # 지지선 후보
    run_lo = int(np.argmin(l[max(0, hi_pos - 60):hi_pos + 1])) + max(0, hi_pos - 60)
    levels = {"9E": float(row["E9"]), "21E": float(row["E21"]), "50E": float(row["E50"]),
              "VWAP": avwap(df, run_lo, i)}                            # 상승 시작 저점 기준 AVWAP
    piv = [j for j in pivot_highs(h[:max(0, hi_pos - 3)]) if j >= i - 150 and h[j] < c[i] * 1.02]
    if piv:
        levels["전고"] = float(h[max(piv, key=lambda j: h[j])])        # 돌파한 이전 고점 중 가장 높은 것 (지금 주가 아래)
    zone_ref = float(min(l[i], l[i - 1]))                              # 최근 2일 저점
    touched = {k: v for k, v in levels.items() if v == v and abs(zone_ref / v - 1) * 100 <= BAND}
    above = {k: v for k, v in levels.items() if v == v and 0 < (c[i] / v - 1) * 100 <= APPROACH_MAX}
    broken = [k for k in ("21E",) if c[i] < levels[k] * (1 - BAND / 100)]
    if broken and c[i] < levels["50E"]:
        return None, "21 EMA·50 EMA 아래로 이탈"
    if len(touched) >= MIN_CONFLUENCE and c[i] >= min(touched.values()) * (1 - BAND / 100):
        stage, used = "도착", touched
    elif len(above) >= MIN_CONFLUENCE and len(touched) < MIN_CONFLUENCE:
        stage, used = "접근", above
    else:
        return None, (f"지지선 겹침 부족 (닿은 선 {len(touched)}개: {'+'.join(touched) or '없음'}, "
                      f"기준 {MIN_CONFLUENCE}개)")
    zone_lo, zone_hi = min(used.values()), max(used.values())
    trigger, stop = float(h[i]), float(min(l[i], l[i - 1])) * 0.998
    risk = (trigger / stop - 1) * 100
    if stage == "도착" and risk > RISK_MAX:
        return None, f"손절까지 너무 멂 (리스크 {risk:.1f}% > {RISK_MAX:g}%)"
    order = ["9E", "21E", "50E", "VWAP", "전고"]
    return {"stage": stage, "levels": "+".join(k for k in order if k in used), "n": len(used),
            "zone_lo": zone_lo, "zone_hi": zone_hi, "dist": (c[i] / zone_hi - 1) * 100, "pb": pb,
            "trigger": trigger, "stop": stop, "risk": risk, "close": float(c[i])}, ""


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


def earn_txt(d) -> str:
    return "" if d is None else (f"⚠️D-{d}" if d <= EARN_DAYS else f"D-{d}")


def table_setup(title: str, rows: list[dict], fired: bool = False):
    head = (fit("TICKER", 7) + fit("겹친 지지선", 15) + fit("고점%", 6, True) + fit("1M%", 6, True) + fit("ADR", 5, True)
            + fit("종가" if fired else "트리거", 9, True) + fit("손절", 9, True) + fit("리스크", 7, True) + "  실적  이름")
    lines = [fit(r["t"], 7) + fit(r["levels"], 15) + f"{-r['pb']:>+6.1f}{r['r1m']:>+6.0f}{r['adr']:>5.1f}"
             + f"{(r['close'] if fired else r['trigger']):>9.2f}{r['stop']:>9.2f}{r['risk']:>6.1f}%  "
             + fit(earn_txt(r.get("earn")), 6) + r["name"][:16]
             for r in rows]
    send_lines(title, head, lines)


def table_approach(title: str, rows: list[dict]):
    head = (fit("TICKER", 7) + fit("지지선", 15) + fit("지지구간", 17, True) + fit("거리%", 6, True)
            + fit("고점%", 6, True) + fit("1M%", 6, True) + fit("ADR", 5, True) + "  이름")
    lines = [fit(r["t"], 7) + fit(r["levels"], 15) + fit(f"{r['zone_lo']:.2f}~{r['zone_hi']:.2f}", 17, True)
             + f"{r['dist']:>+6.1f}{-r['pb']:>+6.1f}{r['r1m']:>+6.0f}{r['adr']:>5.1f}  " + r["name"][:16]
             for r in rows]
    send_lines(title, head, lines)


def table_potent(title: str, rows: list[dict]):
    head = (fit("TICKER", 7) + fit("등락%", 6, True) + fit("시가대비", 8, True) + fit("ADR", 5, True)
            + fit("거래대금", 9, True) + fit("1M%", 6, True) + fit("배열", 5, True) + "  업종 / 이름")
    lines = [fit(r["t"], 7) + f"{r['chg']:>+6.1f}{r['oc']:>+8.1f}{r['adr']:>5.1f}" + fit(f"${r['dvol_today']/1e6:,.0f}M", 9, True)
             + f"{r['r1m']:>+6.0f}" + fit(CLASS_ICON[r['cls']], 5, True) + "  " + (r["name"][:14] + " · " + r["ind"][:18])
             for r in rows]
    send_lines(title, head, lines)


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE and not DIAGNOSE:
        print("주말. 종료.")
        return
    info = load_all()
    tickers = list(dict.fromkeys(DIAGNOSE + list(info)))
    print(f"대상 {len(tickers)}종목")
    raw = download(tickers)
    if not raw:
        post("⚠️ 눌림목 스캐너: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    data = {t: prepare(df) for t, df in raw.items() if len(df) > 160}
    session = max(df.index[-1] for df in data.values()).date()

    # 1·3개월 수익률 백분위 (Leader 판정)
    r1 = pd.Series({t: df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1 for t, df in data.items()})
    r3 = pd.Series({t: df["Close"].iloc[-1] / df["Close"].iloc[-64] - 1 for t, df in data.items()})
    p1, p3 = r1.rank(pct=True) * 100, r3.rank(pct=True) * 100

    # 1) Lead / Mediocre / Lag
    cls_now = {t: classify(df.iloc[-1]) for t, df in data.items()}
    sf = STATE / "classes.json"
    st = json.loads(sf.read_text()) if sf.exists() else {}
    prev = st.get("prev", {}) if st.get("date") == str(session) else st.get("cur", {})
    if not DIAGNOSE:
        sf.write_text(json.dumps({"date": str(session), "cur": cls_now, "prev": prev}))

    found, why, potent, funnel = [], {}, [], Counter()
    for t in (DIAGNOSE or list(data)):
        if t not in data:
            why[t] = "시세를 받지 못함 (티커 확인 또는 상장 8개월 미만)"
            continue
        df = data[t]
        n, last = len(df), df.iloc[-1]
        meta = info.get(t, {"name": "", "industry": "관심종목"})
        base = {"t": t, "name": meta["name"], "ind": meta["industry"], "cls": cls_now[t],
                "r1m": float(r1[t] * 100), "adr": float(last["adr"])}
        # 기본 필터
        if last["vol10"] < MIN_AVG_VOL or last["dvol"] < MIN_DOLLAR_VOL:
            why[t] = f"거래량·거래대금 미달 (10일 평균 {last['vol10']/1e3:,.0f}K주, 20일 평균 ${last['dvol']/1e6:,.0f}M)"
            funnel["① 거래대금 미달"] += 1
            continue
        if last["adr"] < ADR_MIN:
            why[t] = f"변동성 부족 (ADR {last['adr']:.1f}% < {ADR_MIN:g}%)"
            funnel["② 변동성 부족"] += 1
            continue
        # ⚡ Potent (오늘 강하게 움직인 새 강세 후보)
        oc = (last["Close"] / last["Open"] - 1) * 100
        chg = (last["Close"] / df["Close"].iloc[-2] - 1) * 100
        if (last["Close"] >= last["E21"] and last["Close"] > last["E50"] and last["adr"] >= POTENT_ADR
                and abs(oc) > POTENT_MOVE and chg > 0):
            potent.append({**base, "chg": chg, "oc": oc, "dvol_today": float(last["Close"] * last["Volume"])})
        # 강세주(Leader) 조건
        if cls_now[t] != "Lead":
            why[t] = f"EMA 배열이 Lead 아님 ({cls_now[t]})"
            funnel["③ Lead 아님"] += 1
            continue
        if max(p1[t], p3[t]) < LEADER_PCT:
            why[t] = f"주도주 아님 (1개월 상위 {100 - p1[t]:.0f}% · 3개월 상위 {100 - p3[t]:.0f}%, 기준 상위 {100 - LEADER_PCT:g}%)"
            funnel["④ 수익률 순위 부족"] += 1
            continue
        if not weekly_up(df, n - 1):
            why[t] = "주봉 추세 아님 (주봉 10 EMA 아래거나 하락 중)"
            funnel["⑤ 주봉 추세 아님"] += 1
            continue
        # ✅ 발동: 어제 '지지 도착' + 오늘 어제 고가 위 마감
        y, _ = setup_at(df, n - 2)
        if y and y["stage"] == "도착" and last["Close"] > y["trigger"] and data[t]["E9"].iloc[-2] > data[t]["E21"].iloc[-2]:
            found.append({**base, **y, "stage": "발동", "close": float(last["Close"]),
                          "risk": (last["Close"] / y["stop"] - 1) * 100, "hl": higher_lows(df["Low"].to_numpy(), n - 1)})
            continue
        r, w = setup_at(df, n - 1)
        if not r:
            why[t] = w
            funnel["⑥ " + w.split(" (")[0]] += 1
            continue
        found.append({**base, **r, "hl": higher_lows(df["Low"].to_numpy(), n - 1)})
        why[t] = ""

    # 실적 일정 (후보만 조회)
    for f in found:
        if f["stage"] in ("도착", "발동"):
            f["earn"] = earnings_in(f["t"])

    if DIAGNOSE:
        lines = []
        for t in DIAGNOSE:
            f = next((x for x in found if x["t"] == t), None)
            c = f"[{cls_now.get(t, '?')}] " if t in cls_now else ""
            if f:
                mark = {"발동": "✅ 발동", "도착": "🎯 지지 도착", "접근": "📍 접근 중"}[f["stage"]]
                lines.append(f"**{t}** {c}{mark} · 지지선 {f['levels']} ({f['zone_lo']:.2f}~{f['zone_hi']:.2f}) · "
                             f"고점 대비 {-f['pb']:+.1f}% · 트리거 {f['trigger']:.2f} · 손절 {f['stop']:.2f} "
                             f"(리스크 {f['risk']:.1f}%) · ADR {f['adr']:.1f}% · 1개월 {f['r1m']:+.0f}% {earn_txt(f.get('earn'))}")
            else:
                lines.append(f"**{t}** {c}❌ {why.get(t, '알 수 없음')}")
        post("**눌림목 스캐너 진단**\n" + "\n".join(lines))
        return

    # ── 1) 시장 내부 강도
    cnt = Counter(cls_now.values())
    total = sum(cnt.values())
    moved = Counter((prev[t], c) for t, c in cls_now.items() if t in prev and prev[t] != c)
    trend = ""
    if prev:
        pc = Counter(v for k, v in prev.items() if k in cls_now)
        trend = f" (어제 Lead {pc['Lead']})"
    post(f"**🪜 강세주 눌림목 · 마틴 룩 방식 | {session} 마감**\n"
         f"시장 내부 강도: 🟢 Lead {cnt['Lead']} ({cnt['Lead']/total*100:.0f}%){trend} · 🟡 Mediocre {cnt['Mediocre']} · "
         f"🔴 Lag {cnt['Lag']} (전체 {total})\n"
         f"오늘 이동: Lead 진입 +{moved[('Mediocre','Lead')] + moved[('Lag','Lead')]} · "
         f"Lead 이탈 −{moved[('Lead','Mediocre')] + moved[('Lead','Lag')]} · Lag 진입 +{moved[('Mediocre','Lag')] + moved[('Lead','Lag')]}")

    # ── 2) 새로 Lead 진입 (업종별 묶음)
    new_lead = [t for t, c in cls_now.items() if c == "Lead" and t in prev and prev[t] != "Lead"]
    if new_lead:
        by_ind = defaultdict(list)
        for t in new_lead:
            by_ind[info.get(t, {}).get("industry", "관심종목")].append(t)
        items = sorted(by_ind.items(), key=lambda kv: -len(kv[1]))
        lines = [("🔥 " if len(v) >= 2 and k != "관심종목" else "   ") + fit(k[:26], 27) + " " + ", ".join(sorted(v)) for k, v in items]
        send_lines(f"🆕 오늘 새로 Lead 진입 {len(new_lead)}종목 · 🔥 = 같은 업종에서 2종목 이상 (자금 유입 신호)", "업종 / 종목", lines)

    # ── 3) 눌림목
    by = {s: sorted([f for f in found if f["stage"] == s],
                    key=lambda r: (r.get("earn") is not None and r["earn"] <= EARN_DAYS, -r["n"], r["risk"]))
          for s in ("발동", "도착", "접근")}
    if by["발동"]:
        table_setup(f"✅ 발동 · 어제 지지 도착 → 오늘 어제 고가 위 마감 ({len(by['발동'])})", by["발동"][:MAX_ROWS], fired=True)
    if by["도착"]:
        table_setup(f"🎯 지지 도착 · 내일 트리거(오늘 고가) 돌파 시 진입 후보, 손절 = 최근 2일 저가 ({len(by['도착'])})",
                    by["도착"][:MAX_ROWS])
    if by["접근"]:
        table_approach(f"📍 접근 중 · 지지선 {MIN_CONFLUENCE}개 이상이 겹친 구간 위 {APPROACH_MAX:g}% 이내 ({len(by['접근'])})",
                       sorted(by["접근"], key=lambda r: r["dist"])[:MAX_ROWS])
    if not any(by.values()):
        post("눌림목 후보 없음")

    # ── 4) Potent
    potent.sort(key=lambda r: -r["chg"])
    if potent:
        table_potent(f"⚡ Potent · 21·50 EMA 위에서 오늘 강하게 오른 종목 (새 강세 후보, 상위 {min(15, len(potent))})", potent[:15])

    post("겹친 지지선: 9E·21E·50E = EMA · VWAP = 상승 시작 저점부터 앵커드 VWAP · 전고 = 돌파한 이전 고점\n"
         f"고점% = 최근 20일 고점 대비 · ADR = 20일 평균 일중 변동폭 · 리스크 = 트리거(발동은 종가) → 손절 · "
         f"⚠️D-n = {EARN_DAYS}일 안 실적 발표 (신규 진입 주의)\n"
         "진입은 장중 5분봉에서 지지 구간 반등 후 작은 박스 돌파를 확인하고, 손절은 오늘 저가 또는 돌파 캔들 저가")
    if funnel:
        post("🔎 탈락 단계: " + " · ".join(f"{k} {v}" for k, v in sorted(funnel.items())))
    print(f"완료: {len(data)}종목, 발동 {len(by['발동'])}, 도착 {len(by['도착'])}, 접근 {len(by['접근'])}, Potent {len(potent)}")


if __name__ == "__main__":
    sys.exit(main())
