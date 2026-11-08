"""
실적 플레이 리포트 (무료 데이터: 야후 파이낸스 옵션 체인 + 나스닥 실적 캘린더)
- 자동: 앞으로 3거래일 안에 실적 발표하는 S&P 500 + 관심종목 중 시가총액 큰 순 최대 10개
- 수동: Run workflow의 diagnose 칸에 티커를 넣으면 그 종목만 (실적 일정과 상관없이)
종목마다 디스코드로:
  ① 요약: 실적 일정·예상치, 옵션 예상 변동폭(ATM 스트래들), 과거 실적 반응 평균, IV(실적 만기 vs 다음 만기),
          GEX 레벨(콜월·풋월·감마 플립·맥스페인), 풋/콜 비율
  ② 과거 실적 반응 표 (최근 8회)
  ③ 거래량 많은 옵션 계약 (거래량 > 미결제약정이면 🔥 = 새 포지션 가능성)
  ④ GEX 히트맵 그림 (행사가 × 만기 3개)
한계: 체결 방향(매수/매도), 스윕, 다크풀, 장중 실시간 흐름은 무료 데이터에 없음.
     미결제약정은 하루 1회(전날 기준) 갱신, GEX는 '딜러가 콜·풋을 판 쪽' 단순 가정으로 계산.
"""
import datetime as dt
import io
import math
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 ─────────────
AHEAD_DAYS = int(os.environ.get("AHEAD_DAYS", "3"))           # 오늘부터 이 거래일 안의 실적
MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "10"))         # 자동 모드 최대 종목 수 (시가총액 큰 순)
MIN_MCAP = float(os.environ.get("MIN_MCAP", "2000000000"))     # 자동 모드 최소 시가총액($)
EXTRA = [t.strip().upper() for t in os.environ.get("EARN_TICKERS", "").split(",") if t.strip()]  # 실적 일정과 상관없이 항상 볼 종목
STRIKE_RANGE = float(os.environ.get("STRIKE_RANGE", "25"))     # 히트맵 행사가 범위: 현재가 ±%
MAX_STRIKES = int(os.environ.get("MAX_STRIKES", "36"))          # 히트맵 최대 줄 수 (넘으면 GEX 큰 행사가 위주)
ALL_DAYS = int(os.environ.get("ALL_DAYS", "90"))               # '전체 만기' 집계 범위(일): 총거래량·Net GEX·콜월·풋월
HIST_N = int(os.environ.get("HIST_N", "8"))                    # 과거 실적 반응 횟수
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"
DIAGNOSE = [t.strip().upper() for t in os.environ.get("DIAGNOSE", "").replace(" ", ",").split(",") if t.strip()]
# ───────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}
WD = "월화수목금토일"


# ───────────── 공통 ─────────────
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


def money(x: float) -> str:
    a = abs(x)
    s = f"{a/1e9:.1f}B" if a >= 1e9 else f"{a/1e6:.1f}M" if a >= 1e6 else f"{a/1e3:.0f}K" if a >= 1e3 else f"{a:.0f}"
    return ("-" if x < 0 else "") + s


def universe() -> dict[str, str]:
    out = {}
    cache = STATE / "sp500.csv"
    if not cache.exists() or time.time() - cache.stat().st_mtime > 7 * 86400:
        try:
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({"ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False), "name": t["Security"]})
            if len(df) > 400:
                df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] S&P 500 목록 실패: {e}")
    if cache.exists():
        for r in pd.read_csv(cache).fillna("").itertuples():
            out[r.ticker] = r.name
    p = ROOT / "watchlist.txt"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tk, _, nm = ln.partition(",")
                out.setdefault(re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper()), nm.strip())
    return out


# ───────────── 실적 캘린더 (나스닥) ─────────────
def nasdaq_calendar(day: dt.date) -> list[dict]:
    try:
        r = requests.get(f"https://api.nasdaq.com/api/calendar/earnings?date={day:%Y-%m-%d}", headers=UA, timeout=30)
        rows = ((r.json() or {}).get("data") or {}).get("rows") or []
    except Exception as e:
        print(f"[경고] 나스닥 실적 캘린더 {day} 실패: {e}")
        return []
    out = []
    for x in rows:
        mc = re.sub(r"[^\d.]", "", str(x.get("marketCap", "")))
        t = str(x.get("time", ""))
        out.append({"ticker": str(x.get("symbol", "")).upper().replace(".", "-"), "date": day,
                    "when": "장 시작 전" if "pre" in t else "장 마감 후" if "after" in t else "시간 미정",
                    "mcap": float(mc) if mc else 0.0, "eps_fc": str(x.get("epsForecast", "") or "").strip()})
    return out


def upcoming(univ: dict[str, str]) -> list[dict]:
    today = dt.datetime.now(NY).date()
    days, d = [], today
    while len(days) < AHEAD_DAYS:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = [x for day in days for x in nasdaq_calendar(day)]
    return [x for x in rows if x["ticker"] in univ]


# ───────────── 옵션 계산 ─────────────
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_gamma(S: float, K: np.ndarray, T: float, iv: np.ndarray) -> np.ndarray:
    iv = np.clip(np.nan_to_num(iv, nan=0.5), 0.05, 5.0)
    T = max(T, 0.5 / 365)
    d1 = (np.log(S / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
    return np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * iv * np.sqrt(T))


def years_to(exp: str) -> float:
    e = dt.datetime.combine(dt.date.fromisoformat(exp), dt.time(16, 0), NY)
    return max((e - dt.datetime.now(NY)).total_seconds() / (365 * 86400), 0.5 / 365)


def mid(row) -> float:
    def num(x):
        try:
            x = float(x)
            return x if x == x else 0.0                                  # NaN → 0
        except (TypeError, ValueError):
            return 0.0
    b, a, last = num(row.get("bid")), num(row.get("ask")), num(row.get("lastPrice"))
    return (b + a) / 2 if b > 0 and a > 0 else last


def gex_table(S: float, chains: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """행사가 × 만기 GEX ($, 주가 1% 움직일 때 딜러 헤지 금액). 콜 +, 풋 −"""
    cols = {}
    for exp, (calls, puts) in chains.items():
        T = years_to(exp)
        s = pd.Series(dtype=float)
        for df, sign in ((calls, 1), (puts, -1)):
            if df is None or df.empty:
                continue
            k = df["strike"].to_numpy(float)
            g = bs_gamma(S, k, T, df["impliedVolatility"].to_numpy(float))
            v = sign * g * df["openInterest"].fillna(0).to_numpy(float) * 100 * S * S * 0.01
            s = s.add(pd.Series(v, index=k), fill_value=0)
        cols[exp] = s
    return pd.DataFrame(cols).sort_index()


def total_gex_at(S2: float, chains) -> float:
    tot = 0.0
    for exp, (calls, puts) in chains.items():
        T = years_to(exp)
        for df, sign in ((calls, 1), (puts, -1)):
            if df is None or df.empty:
                continue
            g = bs_gamma(S2, df["strike"].to_numpy(float), T, df["impliedVolatility"].to_numpy(float))
            tot += float(np.sum(sign * g * df["openInterest"].fillna(0).to_numpy(float))) * 100 * S2 * S2 * 0.01
    return tot


def gamma_flip(S: float, chains) -> float | None:
    grid = np.linspace(S * 0.8, S * 1.2, 81)
    vals = np.array([total_gex_at(x, chains) for x in grid])
    flips = [grid[i] - vals[i] * (grid[i + 1] - grid[i]) / (vals[i + 1] - vals[i])
             for i in range(len(grid) - 1) if vals[i] == 0 or vals[i] * vals[i + 1] < 0]
    return min(flips, key=lambda x: abs(x - S)) if flips else None


def max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    ks = sorted(set(calls["strike"]).union(puts["strike"]))
    if not ks:
        return None
    co, po = calls.set_index("strike")["openInterest"].fillna(0), puts.set_index("strike")["openInterest"].fillna(0)
    pain = {k: float(sum(max(k - c, 0) * oi for c, oi in co.items()) + sum(max(p - k, 0) * oi for p, oi in po.items())) for k in ks}
    return min(pain, key=pain.get)


def atm(calls: pd.DataFrame, puts: pd.DataFrame, S: float):
    ks = sorted(set(calls["strike"]).intersection(puts["strike"]))
    if not ks:
        return None
    k = min(ks, key=lambda x: abs(x - S))
    c, p = calls[calls["strike"] == k].iloc[0], puts[puts["strike"] == k].iloc[0]
    return {"k": k, "straddle": mid(c) + mid(p), "iv": float(np.nanmean([c["impliedVolatility"], p["impliedVolatility"]]))}


# ───────────── 과거 실적 반응 ─────────────
def past_reactions(edates: pd.DataFrame | None, px: pd.DataFrame) -> list[dict]:
    out = []
    if edates is None or edates.empty:
        return out
    c, o = px["Close"], px["Open"]
    idx = c.index.tz_localize(None) if c.index.tz is not None else c.index
    c, o = pd.Series(c.values, index=idx), pd.Series(o.values, index=idx)
    for ts, row in edates.iterrows():
        if pd.isna(row.get("Reported EPS")):
            continue
        t = pd.Timestamp(ts)
        t = t.tz_convert(NY).tz_localize(None) if t.tzinfo else t
        d = pd.Timestamp(t.date())
        if d not in c.index:
            continue
        i = c.index.get_loc(d)
        amc = t.hour >= 12
        try:
            if amc:
                prev_c, nxt_o, nxt_c = c.iloc[i], o.iloc[i + 1], c.iloc[i + 1]
            else:
                prev_c, nxt_o, nxt_c = c.iloc[i - 1], o.iloc[i], c.iloc[i]
        except IndexError:
            continue
        out.append({"date": d.date(), "when": "마감 후" if amc else "시작 전", "gap": (nxt_o / prev_c - 1) * 100,
                    "move": (nxt_c / prev_c - 1) * 100, "surprise": row.get("Surprise(%)")})
        if len(out) >= HIST_N:
            break
    return out


# ───────────── 데이터 받기 (야후) ─────────────
def fetch(t: str, ev: dict | None) -> dict:
    import yfinance as yf
    tk = yf.Ticker(t)
    px = tk.history(period="3y", interval="1d", auto_adjust=False)
    if px is None or px.empty:
        raise RuntimeError("시세 없음")
    S = float(px["Close"].iloc[-1])
    chg = float((px["Close"].iloc[-1] / px["Close"].iloc[-2] - 1) * 100)
    try:
        ed = tk.get_earnings_dates(limit=HIST_N + 6)
    except Exception as e:
        print(f"[경고] {t} 실적 날짜 실패: {e}")
        ed = None
    # 다가오는 실적 일정 (캘린더 없으면 야후)
    if ev is None:
        ev = {"ticker": t, "date": None, "when": "시간 미정", "eps_fc": ""}
        if ed is not None and not ed.empty:
            fut = [pd.Timestamp(x) for x in ed.index if pd.Timestamp(x) >= pd.Timestamp.now(tz=NY) - pd.Timedelta(hours=12)]
            if fut:
                f = min(fut).tz_convert(NY)
                ev.update(date=f.date(), when="장 마감 후" if f.hour >= 12 else "장 시작 전")
    est = {}
    for name, attr in (("eps", "earnings_estimate"), ("rev", "revenue_estimate")):
        try:
            df = getattr(tk, attr)
            if df is not None and "0q" in df.index:
                est[name] = df.loc["0q"].to_dict()
        except Exception:
            pass
    # 옵션 체인: 실적 반영 첫 만기 + 다음 2개
    now = dt.datetime.now(NY)
    exps = [e for e in (tk.options or []) if dt.date.fromisoformat(e) > now.date()
            or (dt.date.fromisoformat(e) == now.date() and now.time() < dt.time(16, 0))]   # 이미 끝난 만기 제외
    if not exps:
        raise RuntimeError("옵션 없음")
    edate = ev.get("date")
    first = 0
    if edate:
        for i, e in enumerate(exps):
            ed_ = dt.date.fromisoformat(e)
            if ed_ > edate or (ed_ == edate and ev["when"] == "장 시작 전"):
                first = i
                break
    chosen = exps[first:first + 3]
    chains = {}
    for e in chosen:
        oc = tk.option_chain(e)
        chains[e] = (oc.calls, oc.puts)
        time.sleep(0.3)
    allx = {}                                               # ALL_DAYS 이내 모든 만기 (총거래량·Net GEX·월)
    today = now.date()
    for e in exps:
        if (dt.date.fromisoformat(e) - today).days > ALL_DAYS:
            break
        if e in chains:
            allx[e] = chains[e]
        else:
            oc = tk.option_chain(e)
            allx[e] = (oc.calls, oc.puts)
            time.sleep(0.3)
    near = {e: v for e, v in allx.items() if (dt.date.fromisoformat(e) - today).days <= 45}
    return {"t": t, "S": S, "chg": chg, "px": px, "ed": ed, "ev": ev, "est": est, "chains": chains, "near": near, "allx": allx}


def vol_history(t: str, vol: float) -> dict:
    """종목별 하루 옵션 총거래량을 state에 쌓아 3·7·30일 평균 대비 %"""
    import json
    f = STATE / "optvol.json"
    db = json.loads(f.read_text()) if f.exists() else {}
    day = dt.datetime.now(NY).date()
    if dt.datetime.now(NY).time() < dt.time(9, 30) or day.weekday() >= 5:
        day = (pd.Timestamp(day) - pd.offsets.BDay(1)).date()    # 장 시작 전·주말 실행은 직전 거래일 거래량
    h = db.get(t, {})
    past = [v for k, v in sorted(h.items()) if k != str(day)]
    out = {"n": len(past)}
    for n in (3, 7, 30):
        if len(past) >= n:
            avg = sum(past[-n:]) / n
            out[n] = vol / avg * 100 if avg else None
    h[str(day)] = vol
    db[t] = dict(sorted(h.items())[-40:])
    f.write_text(json.dumps(db))
    return out


# ───────────── 분석 ─────────────
def analyze(d: dict) -> dict:
    S, chains = d["S"], d["chains"]
    exps = list(chains)
    e0 = exps[0]
    a0 = atm(*chains[e0], S)
    a1 = atm(*chains[exps[1]], S) if len(exps) > 1 else None
    g = gex_table(S, chains)
    calls_all = pd.concat([c.assign(exp=e) for e, (c, p) in chains.items()])
    puts_all = pd.concat([p.assign(exp=e) for e, (c, p) in chains.items()])
    cg = gex_table(S, {e: (c, c.iloc[0:0]) for e, (c, p) in chains.items()}).sum(axis=1)
    pg = gex_table(S, {e: (p.iloc[0:0], p) for e, (c, p) in chains.items()}).sum(axis=1)
    near = d["near"]
    cv = sum(c["volume"].fillna(0).sum() for c, p in near.values())
    pv = sum(p["volume"].fillna(0).sum() for c, p in near.values())
    coi = sum(c["openInterest"].fillna(0).sum() for c, p in near.values())
    poi = sum(p["openInterest"].fillna(0).sum() for c, p in near.values())
    rows = []
    today = dt.datetime.now(NY).date().isoformat()
    for e, (c, p) in near.items():
        if e == today:                                                     # 오늘 만기(0DTE)는 실적과 무관 → 계약 표에서 제외
            continue
        for df, cp in ((c, "C"), (p, "P")):
            for _, r in df.iterrows():
                vol, oi = float(r.get("volume") or 0), float(r.get("openInterest") or 0)
                if vol > 0 and abs(float(r["strike"]) / S - 1) <= 0.2:     # 현재가 ±20% 행사가만 (깊은 내가격 제외)
                    rows.append({"exp": e, "k": float(r["strike"]), "cp": cp, "vol": vol, "oi": oi,
                                 "prem": vol * mid(r) * 100, "iv": float(r.get("impliedVolatility") or 0)})
    top = sorted(rows, key=lambda x: -x["prem"])[:10]
    # 전체 만기(ALL_DAYS 이내) 집계
    allx = d["allx"]
    tv_c = sum(c["volume"].fillna(0).sum() for c, p in allx.values())
    tv_p = sum(p["volume"].fillna(0).sum() for c, p in allx.values())
    oi_c = sum(c["openInterest"].fillna(0).sum() for c, p in allx.values())
    oi_p = sum(p["openInterest"].fillna(0).sum() for c, p in allx.values())
    pr_c = sum(sum(float(r.get("volume") or 0) * mid(r) * 100 for _, r in c.iterrows()) for c, p in allx.values())
    pr_p = sum(sum(float(r.get("volume") or 0) * mid(r) * 100 for _, r in p.iterrows()) for c, p in allx.values())
    cg_all = gex_table(S, {e: (c, c.iloc[0:0]) for e, (c, p) in allx.items()}).sum(axis=1)
    pg_all = gex_table(S, {e: (p.iloc[0:0], p) for e, (c, p) in allx.items()}).sum(axis=1)
    first_exp = next(iter(allx))
    allstat = {
        "days": ALL_DAYS, "n_exp": len(allx), "vol": tv_c + tv_p,
        "share_vol": tv_c / (tv_c + tv_p) * 100 if tv_c + tv_p else None,
        "share_oi": oi_c / (oi_c + oi_p) * 100 if oi_c + oi_p else None,
        "share_prem": pr_c / (pr_c + pr_p) * 100 if pr_c + pr_p else None,
        "pc_vol": tv_p / tv_c if tv_c else None, "pc_oi": oi_p / oi_c if oi_c else None,
        "net_gex": float(cg_all.sum() + pg_all.sum()),
        "call_wall": float(cg_all.idxmax()) if len(cg_all) and cg_all.max() > 0 else None,
        "put_wall": float(pg_all.idxmin()) if len(pg_all) and pg_all.min() < 0 else None,
        "flip": gamma_flip(S, allx), "near_exp": first_exp, "near_pain": max_pain(*allx[first_exp]),
        "hist": vol_history(d["t"], tv_c + tv_p),
    }
    hist = past_reactions(d["ed"], d["px"])
    return {
        "e0": e0, "exps": exps, "atm0": a0, "atm1": a1, "gex": g,
        "em": a0["straddle"] / S * 100 if a0 else None,
        "call_wall": float(cg.idxmax()) if len(cg) and cg.max() > 0 else None,
        "put_wall": float(pg.idxmin()) if len(pg) and pg.min() < 0 else None,
        "flip": gamma_flip(S, chains), "max_pain": max_pain(*chains[e0]),
        "pc_vol": pv / cv if cv else None, "pc_oi": poi / coi if coi else None,
        "call_prem": sum(x["prem"] for x in rows if x["cp"] == "C"), "put_prem": sum(x["prem"] for x in rows if x["cp"] == "P"),
        "top": top, "hist": hist, "all": allstat,
    }


# ───────────── 디스코드 ─────────────
def report(name: str, d: dict, a: dict):
    t, S, ev = d["t"], d["S"], d["ev"]
    date_txt = f"{ev['date']:%m/%d}({WD[ev['date'].weekday()]}) {ev['when']}" if ev.get("date") else "일정 확인 안 됨"
    L = [f"**🎯 {(t + ' ' + name[:24]).strip()} 실적 플레이 | {date_txt}** · 현재 ${S:,.2f} ({d['chg']:+.1f}%)", "```"]
    if a["em"]:
        lo, hi = S * (1 - a["em"] / 100), S * (1 + a["em"] / 100)
        L.append(f"옵션 예상 변동   ±{a['em']:.1f}%  → ${lo:,.2f} ~ ${hi:,.2f}  ({a['e0']} 만기 ATM ${a['atm0']['k']:g} 스트래들)")
    if a["hist"]:
        mv = [abs(h["move"]) for h in a["hist"]]
        avg = sum(mv) / len(mv)
        up, dn = max(h["move"] for h in a["hist"]), min(h["move"] for h in a["hist"])
        cmp = ""
        if a["em"]:
            cmp = " → 옵션이 과거보다 크게 가격 매김" if a["em"] > avg * 1.15 else " → 옵션이 과거보다 작게 가격 매김" if a["em"] < avg * 0.85 else " → 과거와 비슷"
        L.append(f"과거 {len(mv)}회 평균 반응 ±{avg:.1f}%  (최대 {up:+.1f}% / 최저 {dn:+.1f}%){cmp}")
    if a["atm0"]:
        crush = ""
        if a["atm1"] and a["atm1"]["iv"] > 0:
            crush = f" / 다음 만기 {a['atm1']['iv']*100:.0f}%  → 실적 후 IV 하락(크러시) 예상" if a["atm0"]["iv"] > a["atm1"]["iv"] * 1.1 else f" / 다음 만기 {a['atm1']['iv']*100:.0f}%"
        L.append(f"IV               실적 만기 {a['atm0']['iv']*100:.0f}%{crush}")
    est = d["est"]
    parts = []
    if "eps" in est and est["eps"].get("avg") == est["eps"].get("avg"):
        parts.append(f"EPS 예상 ${est['eps']['avg']:.2f}")
    elif ev.get("eps_fc"):
        parts.append(f"EPS 예상 {ev['eps_fc']}")
    if "rev" in est and est["rev"].get("avg") == est["rev"].get("avg"):
        g = est["rev"].get("growth")
        parts.append(f"매출 예상 ${money(est['rev']['avg'])}" + (f" ({g*100:+.0f}%)" if g == g and g is not None else ""))
    beats = [h for h in a["hist"] if h["surprise"] == h["surprise"] and h["surprise"] is not None]
    if beats:
        parts.append(f"최근 {len(beats)}회 EPS 상회 {sum(1 for h in beats if h['surprise'] > 0)}회")
    if parts:
        L.append("예상치           " + " · ".join(parts))
    lv = []
    for k, lab in (("call_wall", "콜월"), ("put_wall", "풋월"), ("flip", "감마 플립"), ("max_pain", "맥스페인")):
        if a[k]:
            lv.append(f"{lab} ${a[k]:,.2f}".replace(".00", ""))
    if lv:
        L.append(f"실적 만기 GEX    " + " · ".join(lv) + f"  (만기 {', '.join(e[5:] for e in a['exps'])} 합산, 맥스페인은 {a['e0'][5:]})")
    if a["pc_vol"] is not None:
        L.append(f"45일 이내 만기   풋/콜 거래량 {a['pc_vol']:.2f} · 미결제 {a['pc_oi']:.2f} · 오늘 프리미엄 콜 ${money(a['call_prem'])} / 풋 ${money(a['put_prem'])}")
    A = a["all"]
    if A["vol"]:
        bar = lambda x: "-" if x is None else f"콜 {x:.0f} · 풋 {100 - x:.0f}"
        hv = A["hist"]
        vs = " · ".join(f"{n}일 평균 대비 {hv[n]:.0f}%" for n in (3, 7, 30) if hv.get(n)) or f"평소 대비: 기록 쌓는 중 ({hv['n']}일)"
        L.append("```")
        post("\n".join(L))
        L = [f"**{t} 옵션 전체 현황** (만기 {A['days']}일 이내 {A['n_exp']}개)", "```"]
        L.append(f"오늘 옵션 거래량  {A['vol']:,.0f}계약 · {vs}")
        L.append(f"콜/풋 비중       거래량 {bar(A['share_vol'])} · 미결제 {bar(A['share_oi'])} · 프리미엄 {bar(A['share_prem'])}")
        L.append(f"풋/콜 비율       거래량 {A['pc_vol']:.2f} · 미결제 {A['pc_oi']:.2f}" if A["pc_vol"] is not None else "풋/콜 비율       -")
        lv = [f"콜월 ${A['call_wall']:g}" if A["call_wall"] else "", f"풋월 ${A['put_wall']:g}" if A["put_wall"] else "",
              f"감마 플립 ${A['flip']:,.2f}" if A["flip"] else "", f"Net GEX {'-' if A['net_gex'] < 0 else ''}${money(abs(A['net_gex']))}"]
        L.append("감마 레벨        " + " · ".join(x for x in lv if x) + ("  (마이너스 = 변동성 커지기 쉬움)" if A["net_gex"] < 0 else "  (플러스 = 변동성 눌림)"))
        if A["near_pain"]:
            L.append(f"맥스페인 (가장 가까운 만기 {A['near_exp'][5:]})  ${A['near_pain']:g}")
    L.append("```")
    post("\n".join(L))

    if a["hist"]:
        h = ["날짜        시점     갭%    종가%  EPS서프라이즈"]
        for x in a["hist"]:
            s = f"{x['surprise']:+.1f}%" if x["surprise"] == x["surprise"] and x["surprise"] is not None else "-"
            h.append(f"{x['date']}  {x['when']}  {x['gap']:+6.1f}  {x['move']:+6.1f}  {s:>8}")
        post(f"**{t} 과거 실적 반응** (마감 후 발표 = 다음 날, 시작 전 발표 = 당일 기준)\n```\n" + "\n".join(h) + "\n```")

    if a["top"]:
        h = ["만기    행사가  C/P   거래량   미결제  Vol/OI  프리미엄   IV"]
        for x in a["top"]:
            fire = "🔥" if x["vol"] > x["oi"] and x["vol"] >= 500 else "  "
            ratio = f"{x['vol'] / x['oi']:.1f}" if x["oi"] else "신규"
            h.append(f"{x['exp'][5:]}  {x['k']:>7g}   {x['cp']}  {x['vol']:>7,.0f}  {x['oi']:>7,.0f}  {ratio:>5}{fire}"
                     f"{'$' + money(x['prem']):>8}  {x['iv']*100:>3.0f}%")
        post(f"**{t} 오늘 프리미엄 큰 옵션 계약** (45일 이내 만기, 오늘 만기 제외 · 현재가 ±20% 행사가 · 🔥 = 거래량이 미결제보다 많음 → 새 포지션 가능성, 매수/매도 방향은 모름)\n"
             f"```\n" + "\n".join(h) + "\n```")

    try:
        import charts
        charts.safe(heatmap, t, S, a, ev)
    except Exception as e:
        print(f"[경고] 차트 건너뜀: {e}")


def heatmap(t: str, S: float, a: dict, ev: dict):
    import charts
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    g = a["gex"]
    g = g[(g.index >= S * (1 - STRIKE_RANGE / 100)) & (g.index <= S * (1 + STRIKE_RANGE / 100))]
    if g.empty:
        return
    if len(g) > MAX_STRIKES:
        near = set(sorted(g.index, key=lambda k: abs(k - S))[:10])          # 현재가 근처 10줄은 항상
        big = g.abs().sum(axis=1).sort_values(ascending=False).index          # 나머지는 GEX 큰 행사가
        keep = list(near) + [k for k in big if k not in near][:MAX_STRIKES - len(near)]
        g = g.loc[sorted(keep)]
    g = g.iloc[::-1]                                                       # 위 = 높은 행사가
    vals = g.to_numpy(float)
    vmax = np.nanmax(np.abs(vals)) or 1
    fig, ax = plt.subplots(figsize=(4.5 + 3.0 * g.shape[1], 0.42 * len(g) + 2.2))
    fig.subplots_adjust(bottom=0.06)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("gex", ["#ff2a2a", "#5a1a1d", "#16181d", "#14482a", "#2bff6a"])
    ax.imshow(np.nan_to_num(vals), cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if v == v and abs(v) > 0:
                ax.text(j, i, money(v), ha="center", va="center", fontsize=10,
                        color="black" if abs(v) > vmax * 0.6 else "white")
    ax.set_xticks(np.arange(-0.5, vals.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, vals.shape[0]), minor=True)
    ax.grid(which="minor", color=charts.BG, lw=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(g.shape[1]), [e[5:].replace("-", "/") + (" ★" if e == a["e0"] else "") for e in g.columns])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(g)), [f"{k:g}" for k in g.index])
    ks = list(g.index)
    si = min(range(len(ks)), key=lambda i: abs(ks[i] - S))
    ax.get_yticklabels()[si].set_color(charts.YELLOW)
    ax.get_yticklabels()[si].set_fontweight("bold")
    ax.axhline(si, color=charts.YELLOW, lw=0.6, alpha=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    date_txt = f"{ev['date']:%m/%d} {ev['when']}" if ev.get("date") else ""
    ax.set_title(f"{t} GEX 히트맵 · ${S:,.2f}  (실적 {date_txt}, ★ = 실적 반영 첫 만기)", loc="left", fontsize=12,
                 color=charts.TEXT, pad=22)
    fig.text(0.01, 0.005, "초록 = 플러스 감마 (지지·저항·자석) · 빨강 = 마이너스 감마 (뚫리면 가속) · 단위 $ (1% 움직일 때 딜러 헤지 금액) · 노란 행 = 현재가",
             fontsize=8, color=charts.SUB, ha="left", va="bottom")
    charts.send(fig, filename=f"{t}_gex.png")


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE and not DIAGNOSE:
        print("주말. 종료.")
        return
    univ = universe()
    targets: list[dict] = []
    if DIAGNOSE:
        targets = [{"ticker": t, "ev": None} for t in DIAGNOSE]
    else:
        ups = upcoming(univ)
        ups.sort(key=lambda x: (x["date"], -x["mcap"]))
        if ups:
            lines = [f"{x['date']:%m/%d}({WD[x['date'].weekday()]}) {x['when']:<6} {x['ticker']:<6} {univ.get(x['ticker'], '')[:22]:<22}"
                     f" 시총 ${money(x['mcap'])}" for x in ups]
            post(f"**📅 실적 예정 · 앞으로 {AHEAD_DAYS}거래일 (S&P 500 + 관심종목 {len(ups)}개)**\n```\n" + "\n".join(lines[:40]) + "\n```")
        else:
            post(f"📅 앞으로 {AHEAD_DAYS}거래일 안에 실적 예정인 S&P 500·관심종목이 없거나, 캘린더를 받지 못했습니다.")
        big = sorted([x for x in ups if x["mcap"] >= MIN_MCAP], key=lambda x: -x["mcap"])[:MAX_TICKERS]
        targets = [{"ticker": x["ticker"], "ev": x} for x in sorted(big, key=lambda x: (x["date"], -x["mcap"]))]
        targets += [{"ticker": t, "ev": None} for t in EXTRA if t not in {x["ticker"] for x in targets}]
    for x in targets:
        t = x["ticker"]
        try:
            d = fetch(t, x["ev"])
            a = analyze(d)
            report(univ.get(t, ""), d, a)
        except Exception as e:
            post(f"⚠️ {t} 실적 리포트 실패: {e}")
        time.sleep(1)
    if targets:
        post("ℹ️ 무료 데이터 한계: 체결 방향(매수/매도·스윕)·다크풀·장중 실시간 흐름은 없음 · 미결제약정은 전날 기준 · "
             "GEX는 '딜러가 콜·풋을 판 쪽' 단순 가정 · 예상 변동 = 실적 반영 첫 만기 ATM 스트래들 ÷ 주가")


if __name__ == "__main__":
    sys.exit(main())
