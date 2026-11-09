"""
바닥 반전 패턴 스캐너 — 역헤드앤숄더 + 쌍바닥(이중 바닥)
S&P 500 + 관심종목 + ETF, 일봉

판정 방식 (차트 그림이 아니라 매일의 고가·저가·종가 숫자로 계산)
  1) 스윙 저점·고점 찾기: 앞뒤 4봉보다 낮은(높은) 날, 3% 미만 잔물결은 무시
  2) 역헤드앤숄더: 저점 3개(왼쪽 어깨·머리·오른쪽 어깨) + 반등 고점 2개를 이은 넥라인
     쌍바닥: 비슷한 높이 저점 2개 + 가운데 고점(넥라인)
  3) 단계: 🚀 돌파 (최근 5봉 안에 넥라인 위로 마감) / 🎯 돌파 임박 (넥라인 아래 5% 이내) / 🔨 형성 중 (오른쪽 바닥에서 반등 시작)
  4) 목표가 = 넥라인 + 패턴 높이, 손절 = 오른쪽 어깨(두 번째 바닥) 저점 아래
잡힌 종목은 표 + 패턴 선을 그린 캔들 차트 그림으로 보냅니다.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import patternkit as K

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
LOOKBACK = 220            # 최근 이 봉 수 안에서 패턴 찾기
PIVOT_K = 4               # 스윙 고점·저점 판별 폭 (앞뒤 몇 봉)
PIVOT_MIN_MOVE = 3.0      # 이 % 미만 잔물결은 무시
RECENT_LAST_LOW = 30      # 마지막 바닥(오른쪽 어깨·두 번째 바닥)이 최근 이 봉 수 안
PRIOR_DROP = 10.0         # 패턴 전 하락폭 최소 % (바닥 반전이므로 앞에 하락이 있어야 함)
# 역헤드앤숄더
HEAD_DEPTH = 3.0          # 머리가 두 어깨 중 낮은 쪽보다 최소 % 더 낮음
SHOULDER_TOL = 10.0       # 두 어깨 높이 차이 최대 %
NECK_SLOPE_MAX = 12.0     # 넥라인 두 고점 높이 차이 최대 %
SYMMETRY_MAX = 3.0        # 왼쪽·오른쪽 기간 비율 최대
# 쌍바닥
DB_TOL = 3.0              # 두 바닥 높이 차이 최대 %
DB_MIN_BARS, DB_MAX_BARS = 10, 150   # 두 바닥 사이 봉 수
# 공통
MIN_HEIGHT = 8.0          # 패턴 높이(넥라인 ~ 최저점) 최소 %
BREAKOUT_BARS = 5         # 돌파로 볼 최근 봉 수
NEAR_PCT = 5.0            # 넥라인 아래 이 % 이내 = 돌파 임박
FORMING_PCT = 12.0        # 넥라인 아래 이 % 이내 + 바닥에서 3% 이상 반등 = 형성 중
MAX_ABOVE = 10.0          # 돌파 후 넥라인 위 이 % 넘게 간 종목은 제외
SHOW_FORMING = False      # 형성 중 단계도 표로 보내기 (켜면 결과가 많아짐)
IMG_MAX = 10               # 패턴별 그림 최대 장 수 (돌파·임박만)
REALERT_DAYS = 10         # 같은 종목·패턴·단계 재알림 금지 기간(일)
# ──────────────────────────────────────────────────────

ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)


def prior_drop(h: np.ndarray, i: int, low: float) -> float:
    return (h[max(0, i - 60):i + 1].max() / low - 1) * 100


def stage_for(c: np.ndarray, neck, start: int, n: int, low_ref: float):
    """neck(i) → 넥라인 값. (단계, 돌파 봉 번호, 현재 거리%) 또는 (None, 이유)"""
    bo = next((i for i in range(start + 1, n) if c[i] > neck(i)), None)
    pos = (c[-1] / neck(n - 1) - 1) * 100
    if bo is not None:
        if n - 1 - bo >= BREAKOUT_BARS:
            return None, f"이미 {n - 1 - bo}봉 전에 넥라인 돌파 (지난 신호)", pos
        if c[-1] < neck(n - 1) * 0.99:
            return None, "돌파 후 다시 넥라인 아래", pos
        if pos > MAX_ABOVE:
            return None, f"돌파 후 넥라인 위로 너무 멀리 감 ({pos:+.1f}%)", pos
        return ("돌파", bo), "", pos
    if pos >= -NEAR_PCT:
        return ("임박", None), "", pos
    if pos >= -FORMING_PCT and c[-1] >= low_ref * 1.03:
        return ("형성", None), "", pos
    return None, f"넥라인에서 멂 ({pos:+.1f}%)", pos


def find_ihs(df: pd.DataFrame) -> tuple[dict | None, str]:
    h, l, c, v = (df[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    n = len(c)
    P = [p for p in K.pivots(h, l, PIVOT_K, PIVOT_MIN_MOVE) if p[0] >= n - LOOKBACK]
    why = "스윙 저점 3개(어깨·머리·어깨) 구조 없음"
    for a in range(len(P) - 5, -1, -1):
        seq = P[a:a + 5]
        if [p[1] for p in seq] != ["L", "H", "L", "H", "L"]:
            continue
        (i1, _, l1), (j1, _, h1), (i2, _, l2), (j2, _, h2), (i3, _, l3) = seq
        if n - 1 - i3 > RECENT_LAST_LOW:
            why = f"오른쪽 어깨가 오래됨 ({n - 1 - i3}봉 전)"
            break                                                   # 더 이전 구조는 더 오래됨
        if l[i3 + 1:].min(initial=np.inf) < l3 * 0.97:
            why = "오른쪽 어깨 이후 더 낮은 저점이 나옴"
            continue
        sh_low = min(l1, l3)
        if (sh_low / l2 - 1) * 100 < HEAD_DEPTH:
            why = "머리가 어깨보다 충분히 낮지 않음"
            continue
        if abs(l1 / l3 - 1) * 100 > SHOULDER_TOL:
            why = f"두 어깨 높이 차이가 큼 ({abs(l1 / l3 - 1) * 100:.0f}%)"
            continue
        if abs(h2 / h1 - 1) * 100 > NECK_SLOPE_MAX:
            why = "넥라인 기울기가 너무 큼"
            continue
        dl, dr = i2 - i1, i3 - i2
        if max(dl, dr) / max(min(dl, dr), 1) > SYMMETRY_MAX:
            why = "왼쪽·오른쪽 기간이 너무 비대칭"
            continue
        neck = lambda i, j1=j1, h1=h1, j2=j2, h2=h2: K.line_at(j1, h1, j2, h2, i)
        height = (neck(i2) / l2 - 1) * 100
        if height < MIN_HEIGHT:
            why = f"패턴 높이 부족 ({height:.1f}%)"
            continue
        drop = prior_drop(h, i1, l1)
        if drop < PRIOR_DROP:
            why = f"패턴 전 하락 부족 ({drop:.0f}%)"
            continue
        if (c[j1:i3 + 1] > np.array([neck(i) for i in range(j1, i3 + 1)]) * 1.02).any():
            why = "어깨 사이에서 이미 넥라인을 크게 넘음"
            continue
        st, w, pos = stage_for(c, neck, i3, n, l3)
        if st is None:
            why = w
            continue
        stage, bo = st
        vavg = v[max(0, n - 51):n - 1].mean()
        target = neck(n - 1) + (neck(i2) - l2)
        return {"pattern": "IHS", "stage": stage, "neck": float(neck(n - 1)), "pos": pos, "target": float(target),
                "stop": float(l3 * 0.99), "height": height, "bars": i3 - i1, "drop": drop,
                "vol": float(v[bo] / vavg) if bo is not None and vavg > 0 else float(v[-1] / vavg) if vavg > 0 else np.nan,
                "pts": {"LS": (i1, l1), "H": (i2, l2), "RS": (i3, l3), "N1": (j1, h1), "N2": (j2, h2)},
                "start": i1, "key": df.index[i2].date().isoformat()}, ""
    return None, why


def find_db(df: pd.DataFrame) -> tuple[dict | None, str]:
    h, l, c, v = (df[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    n = len(c)
    P = [p for p in K.pivots(h, l, PIVOT_K, PIVOT_MIN_MOVE) if p[0] >= n - LOOKBACK]
    why = "스윙 저점 2개 + 가운데 고점 구조 없음"
    for a in range(len(P) - 3, -1, -1):
        seq = P[a:a + 3]
        if [p[1] for p in seq] != ["L", "H", "L"]:
            continue
        (i1, _, l1), (j1, _, h1), (i2, _, l2) = seq
        if n - 1 - i2 > RECENT_LAST_LOW:
            why = f"두 번째 바닥이 오래됨 ({n - 1 - i2}봉 전)"
            break
        if not (DB_MIN_BARS <= i2 - i1 <= DB_MAX_BARS):
            why = f"두 바닥 간격이 범위 밖 ({i2 - i1}봉)"
            continue
        if abs(l2 / l1 - 1) * 100 > DB_TOL:
            why = f"두 바닥 높이 차이가 큼 ({abs(l2 / l1 - 1) * 100:.1f}%)"
            continue
        if l[i2 + 1:].min(initial=np.inf) < l2 * 0.97:
            why = "두 번째 바닥 이후 더 낮은 저점이 나옴"
            continue
        low = min(l1, l2)
        height = (h1 / low - 1) * 100
        if height < MIN_HEIGHT:
            why = f"패턴 높이 부족 ({height:.1f}%)"
            continue
        drop = prior_drop(h, i1, l1)
        if drop < PRIOR_DROP:
            why = f"패턴 전 하락 부족 ({drop:.0f}%)"
            continue
        neck = lambda i, h1=h1: h1
        st, w, pos = stage_for(c, neck, i2, n, l2)
        if st is None:
            why = w
            continue
        stage, bo = st
        vavg = v[max(0, n - 51):n - 1].mean()
        return {"pattern": "DB", "stage": stage, "neck": float(h1), "pos": pos, "target": float(h1 + (h1 - low)),
                "stop": float(l2 * 0.99), "height": height, "bars": i2 - i1, "drop": drop,
                "vol": float(v[bo] / vavg) if bo is not None and vavg > 0 else float(v[-1] / vavg) if vavg > 0 else np.nan,
                "pts": {"B1": (i1, l1), "B2": (i2, l2), "N": (j1, h1)},
                "start": i1, "key": df.index[i2].date().isoformat()}, ""
    return None, why


# ───────────── 디스코드 ─────────────
NAMES = {"IHS": "역헤드앤숄더", "DB": "쌍바닥"}
STAGE = {"돌파": "🚀 돌파", "임박": "🎯 돌파 임박", "형성": "🔨 형성 중"}
STAGE_TXT = {"돌파": "돌파", "임박": "돌파 임박", "형성": "형성 중"}               # 그림 제목용 (이모지 없음)


def table(title: str, rows: list[dict]):
    head = (K.fit("TICKER", 7) + K.fit("넥라인", 9, True) + K.fit("종가", 9, True) + K.fit("거리%", 7, True)
            + K.fit("목표가", 9, True) + K.fit("목표%", 6, True) + K.fit("손절", 9, True) + K.fit("높이%", 6, True)
            + K.fit("봉", 5, True) + K.fit("거래량", 7, True) + "  이름")
    lines = [K.fit(r["t"], 7) + K.fit(K.px(r["neck"]), 9, True) + K.fit(K.px(r["close"]), 9, True) + f"{r['pos']:>+7.1f}"
             + K.fit(K.px(r["target"]), 9, True) + f"{(r['target'] / r['close'] - 1) * 100:>+6.0f}" + K.fit(K.px(r["stop"]), 9, True)
             + f"{r['height']:>6.0f}{r['bars']:>5}{r['vol']:>7.1f}  " + r["name"][:18]
             for r in rows]
    K.send_lines(title, head, lines)


def picture(t: str, name: str, df: pd.DataFrame, r: dict):
    import charts
    n = len(df)
    start = max(0, r["start"] - 30)
    p = r["pts"]
    if r["pattern"] == "IHS":
        (j1, h1), (j2, h2) = p["N1"], p["N2"]
        lines = [{"i1": j1, "v1": h1, "i2": j2, "v2": h2, "i3": n - 1, "color": charts.BLUE, "label": "넥라인"}]
        marks = [{"i": p[k][0], "v": p[k][1], "text": lab} for k, lab in (("LS", "왼어깨"), ("H", "머리"), ("RS", "오른어깨"))]
    else:
        lines = []
        marks = [{"i": p[k][0], "v": p[k][1], "text": lab} for k, lab in (("B1", "바닥1"), ("B2", "바닥2"))]
    hl = [{"v": r["neck"], "color": charts.BLUE, "ls": "--", "label": "넥라인"}] if r["pattern"] == "DB" else []
    hl += [{"v": r["target"], "color": charts.GREEN, "ls": ":", "label": f"목표 {K.px(r['target'])}"},
           {"v": r["stop"], "color": charts.RED, "ls": ":", "label": f"손절 {K.px(r['stop'])}"}]
    fig = K.candle_chart(df, start, f"{t} {name[:20]} · {NAMES[r['pattern']]} {STAGE_TXT[r['stage']]}",
                         f"넥라인 {K.px(r['neck'])} (현재가 대비 {r['pos']:+.1f}%) · 높이 {r['height']:.0f}% · 패턴 {r['bars']}봉",
                         lines=lines, marks=marks, hlines=hl)
    charts.send(fig, filename=f"{t}_{r['pattern']}.png")


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(K.NY)
    if now.weekday() >= 5 and not K.FORCE and not K.DIAGNOSE:
        print("주말. 종료.")
        return
    uni = K.load_all(ROOT, STATE)
    tickers = K.DIAGNOSE or list(uni)
    data = K.download(tickers)
    if not data:
        K.post("⚠️ 바닥 패턴: 시세를 받지 못했습니다.")
        return
    found, why = [], {}
    for t in tickers:
        if t not in data or len(data[t]) < 120:
            why[t] = {"all": "시세 없음 또는 기간 부족"}
            continue
        df = data[t]
        name, kind = uni.get(t, ("", "stock"))
        bad = K.liquid(df, kind)
        if bad and not K.DIAGNOSE:
            continue
        why[t] = {}
        for fn in (find_ihs, find_db):
            try:
                r, w = fn(df)
            except Exception as e:
                r, w = None, f"계산 오류: {e}"
            if r:
                found.append({"t": t, "name": name, "kind": kind, "close": float(df["Close"].iloc[-1]), **r})
            else:
                why[t][fn.__name__] = w
        if bad:
            why[t]["liquid"] = bad

    if K.DIAGNOSE:
        lines = []
        for t in K.DIAGNOSE:
            hits = [f for f in found if f["t"] == t]
            for f in hits:
                lines.append(f"**{t}** ✅ {NAMES[f['pattern']]} {STAGE[f['stage']]} · 넥라인 {K.px(f['neck'])} ({f['pos']:+.1f}%) · "
                             f"목표 {K.px(f['target'])} · 손절 {K.px(f['stop'])} · 높이 {f['height']:.0f}% · {f['bars']}봉")
            w = why.get(t, {})
            if "all" in w:
                lines.append(f"**{t}** ❌ {w['all']}")
            for fn, lab in (("find_ihs", "역헤드앤숄더"), ("find_db", "쌍바닥")):
                if fn in w:
                    lines.append(f"**{t}** ❌ {lab}: {w[fn]}")
            if "liquid" in w:
                lines.append(f"　(참고: {w['liquid']} → 자동 스캔에서는 제외됨)")
        K.post("**바닥 패턴 진단**\n" + "\n".join(lines))
        for f in found:
            try:
                picture(f["t"], f["name"], data[f["t"]], f)
            except Exception as e:
                print(f"[경고] 그림 실패: {e}")
        return

    sent = K.Sent(STATE, REALERT_DAYS)
    new = []
    for f in found:
        key = f"{f['t']}|{f['pattern']}|{f['stage']}|{f['key']}"
        if K.FORCE or sent.fresh(key):
            new.append(f)
            sent.mark(key)
    sent.save()
    session = max(d.index[-1] for d in data.values()).date()
    if not SHOW_FORMING:
        new = [f for f in new if f["stage"] != "형성"]
    cnt = lambda p, s: sum(1 for f in new if f["pattern"] == p and f["stage"] == s)
    fm = lambda p: f" · 🔨 {cnt(p, '형성')}" if SHOW_FORMING else ""
    K.post(f"**🔻➡🔺 바닥 반전 패턴 | {session}**\n"
           f"역헤드앤숄더: 🚀 {cnt('IHS', '돌파')} · 🎯 {cnt('IHS', '임박')}{fm('IHS')}  /  "
           f"쌍바닥: 🚀 {cnt('DB', '돌파')} · 🎯 {cnt('DB', '임박')}{fm('DB')}  (새로 잡힌 것만, 확인 {len(data)}종목)")
    stages = ["돌파", "임박"] + (["형성"] if SHOW_FORMING else [])
    for pat in ("IHS", "DB"):
        for st in stages:
            rows = sorted([f for f in new if f["pattern"] == pat and f["stage"] == st], key=lambda r: -r["height"])
            if rows:
                table(f"{NAMES[pat]} · {STAGE[st]}", rows)
        pics = sorted([f for f in new if f["pattern"] == pat and f["stage"] in ("돌파", "임박")],
                      key=lambda r: (r["stage"] != "돌파", -r["height"]))[:IMG_MAX]
        for f in pics:
            try:
                picture(f["t"], f["name"], data[f["t"]], f)
            except Exception as e:
                print(f"[경고] 그림 실패: {e}")
    if new:
        K.post("넥라인 = 오늘 기준 넥라인 값 · 거리% = 종가가 넥라인보다 몇 % 위/아래 · 목표가 = 넥라인 + 패턴 높이 · "
               "손절 = 오른쪽 어깨(두 번째 바닥) 저점 아래 · 높이% = 넥라인~최저점 · 봉 = 패턴 길이 · "
               "거래량 = 돌파일(임박·형성은 오늘) 거래량 ÷ 50일 평균 (돌파 때 1.5 이상이면 신뢰도↑)")


if __name__ == "__main__":
    sys.exit(main())
