"""
ETF 거래량 감시 (하루 1회, 최근 3거래일 분석)
- 매일 미국 장 마감 후 etfs.txt의 ETF들을 확인해서
- 최근 3거래일 거래량이 "그 전 20거래일 평균"보다 크게 늘어난 ETF를
- 3일 가격 흐름(유입·이탈·혼조)과 거래량 패턴(증가·지속·급증·감소)으로 정리해 디스코드로 보냅니다.
"""
import datetime as dt
import os
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ───────────── 설정 ─────────────
DAYS = int(os.environ.get("DAYS", "3"))                          # 분석할 최근 거래일 수
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "20"))       # 평소 거래량 = 분석 구간 직전 N거래일 평균
RVOL3_MIN = float(os.environ.get("RVOL3_MIN", "1.5"))            # 3일 평균 거래량이 평소의 이 배수 이상이면 알림
RVOL1_MIN = float(os.environ.get("RVOL1_MIN", "2.0"))            # 또는 마지막 날 하루가 이 배수 이상이면 알림
HIGH_DAY = float(os.environ.get("HIGH_DAY", "1.3"))              # 이 배수 이상인 날 = "거래량 많은 날"
FLOW_MIN = float(os.environ.get("FLOW_MIN", "0.5"))              # 3일 등락이 ±이 %보다 작으면 방향 없음(혼조)
MIN_AVG_DOLLAR = float(os.environ.get("MIN_AVG_DOLLAR", "5000000"))  # 평소 하루 거래대금이 이보다 적으면 제외 ($)
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
# ───────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent


def load_etfs() -> dict[str, str]:
    out = {}
    for ln in (ROOT / "etfs.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            t, _, desc = ln.partition(",")
            out[t.strip().upper()] = desc.strip()
    return out


def download_daily(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker",
                       auto_adjust=False, threads=True, progress=False)
    if not isinstance(data.columns, pd.MultiIndex):
        data = pd.concat({tickers[0]: data}, axis=1)
    have = set(data.columns.get_level_values(0))
    return {t: data[t].dropna(subset=["Close", "Volume"]) for t in tickers if t in have}


def drop_unfinished(df: pd.DataFrame) -> pd.DataFrame:
    """장중에 수동 실행하면 오늘 봉은 아직 미완성 → 빼고 마감된 날만 분석"""
    now = dt.datetime.now(NY)
    if df.empty or now.weekday() >= 5 or not (dt.time(9, 30) <= now.time() < dt.time(16, 15)):
        return df
    last = pd.Timestamp(df.index[-1])
    return df.iloc[:-1] if last.date() == now.date() else df


def pattern(r: list[float]) -> str:
    """최근 3일 RVOL 모양"""
    if r[0] < r[1] < r[2] and r[2] >= HIGH_DAY:
        return "📶증가"
    if all(x >= HIGH_DAY for x in r):
        return "🔁지속"
    if r[-1] >= RVOL1_MIN and all(x < HIGH_DAY for x in r[:-1]):
        return "⚡급증"
    if r[0] > r[1] > r[2] and r[0] >= HIGH_DAY:
        return "🔻감소"
    return "  ─ "


def analyze(df: pd.DataFrame) -> dict | None:
    df = drop_unfinished(df)
    if len(df) < DAYS + LOOKBACK_DAYS + 1:
        return None
    win = df.iloc[-DAYS:]
    base = df.iloc[-(DAYS + LOOKBACK_DAYS):-DAYS]
    avg_vol = float(base["Volume"].mean())
    avg_dollar = float((base["Volume"] * base["Close"]).mean())
    if avg_vol <= 0 or avg_dollar < MIN_AVG_DOLLAR:
        return None
    rv = [float(v) / avg_vol for v in win["Volume"]]
    closes = df["Close"].iloc[-(DAYS + 1):].tolist()                 # 분석 구간 직전 종가 포함
    day_chg = [closes[i + 1] / closes[i] - 1 for i in range(DAYS)]
    chg = (closes[-1] / closes[0] - 1) * 100
    chg = 0.0 if abs(chg) < 0.05 else chg
    hi, lo = float(win["High"].max()), float(win["Low"].min())
    pos = (closes[-1] - lo) / (hi - lo) * 100 if hi > lo else 50.0
    acc = sum(1 for c, r in zip(day_chg, rv) if c > 0 and r >= HIGH_DAY)   # 거래량 동반 상승일
    dist = sum(1 for c, r in zip(day_chg, rv) if c < 0 and r >= HIGH_DAY)  # 거래량 동반 하락일
    if chg >= FLOW_MIN and pos >= 60 and acc >= dist:
        flow = "유입"
    elif chg <= -FLOW_MIN and pos <= 40 and dist >= acc:
        flow = "이탈"
    else:
        flow = "혼조"
    return {"date": pd.Timestamp(win.index[-1]).date().isoformat(),
            "dates": [pd.Timestamp(d).strftime("%m/%d") for d in win.index],
            "rv": rv, "rv3": sum(rv) / DAYS, "chg": chg, "pos": pos, "acc": acc, "dist": dist,
            "dollar": float((win["Volume"] * win["Close"]).sum()), "flow": flow, "pat": pattern(rv)}


def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def money(x: float) -> str:
    return f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M"


def width(s: str) -> int:
    """디스코드 고정폭 글꼴에서 한글·이모지는 두 칸"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def fit(s: str, w: int, right: bool = False) -> str:
    out = ""
    for ch in s:
        if width(out + ch) > w:
            break
        out += ch
    pad = " " * (w - width(out))
    return pad + out if right else out + pad


def send_table(title: str, rows: list[dict], dates: list[str]):
    head = (fit("TICKER", 7) + fit("3일", 5, True) + "".join(fit(d, 6, True) for d in dates)
            + fit("3일%", 7, True) + fit("위치", 5, True) + fit("거래대금", 9, True) + "  패턴   설명")
    lines = [fit(r["t"], 7) + f"{r['rv3']:>5.1f}" + "".join(f"{x:>6.1f}" for x in r["rv"])
             + f"{r['chg']:>+7.1f}{r['pos']:>5.0f}{money(r['dollar']):>9}  " + fit(r["pat"], 7) + r["desc"]
             for r in rows]
    first = True
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else ""
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first = False


def main():
    etfs = load_etfs()
    data = download_daily(list(etfs))
    results = {t: analyze(df) for t, df in data.items()}
    results = {t: r for t, r in results.items() if r}
    missing = [t for t in etfs if t not in data]
    if not results:
        post("⚠️ ETF 거래량 감시: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    session_date = max(r["date"] for r in results.values())
    dates = next(r["dates"] for r in results.values() if r["date"] == session_date)

    hits = [{"t": t, "desc": etfs[t], **r} for t, r in results.items()
            if r["date"] == session_date and (r["rv3"] >= RVOL3_MIN or r["rv"][-1] >= RVOL1_MIN)]
    span = f"{dates[0]}~{dates[-1]}"
    if not hits:
        post(f"**ETF 거래량 | 최근 3일({span})**\n평소 대비 거래량이 크게 늘어난 ETF 없음 (확인 {len(results)}개)")
    else:
        hits.sort(key=lambda r: -r["rv3"])
        groups = [("유입", "🟢 유입 (3일 상승 + 3일 고가권 + 거래량 실린 상승일이 더 많음)"),
                  ("이탈", "🔴 이탈 (3일 하락 + 3일 저가권 + 거래량 실린 하락일이 더 많음)"),
                  ("혼조", "⚪ 혼조 (그 외)")]
        counts = " · ".join(f"{k} {sum(h['flow'] == k for h in hits)}" for k, _ in groups)
        post(f"**ETF 거래량 | 최근 3일({span}) 마감 기준**\n"
             f"3일 평균 {RVOL3_MIN:g}배 이상 또는 마지막 날 {RVOL1_MIN:g}배 이상 {len(hits)}개 ({counts})")
        for key, title in groups:
            rows = [h for h in hits if h["flow"] == key]
            if rows:
                send_table(title, rows, dates)
        post(f"숫자 = 평소(직전 {LOOKBACK_DAYS}거래일 평균) 대비 거래량 배수 · 3일% = 3일 등락 · "
             f"위치 = 3일 저가(0)~고가(100) 중 종가\n"
             f"패턴: 📶증가 매일 늘어남 · 🔁지속 3일 모두 {HIGH_DAY:g}배↑ · ⚡급증 마지막 날만 · 🔻감소 줄어드는 중")
    if missing:
        post("⚠️ 시세를 받지 못한 ETF: " + ", ".join(missing))
    print(f"완료: 확인 {len(results)}개, 알림 {len(hits)}개")


if __name__ == "__main__":
    sys.exit(main())
