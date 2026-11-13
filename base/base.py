"""
장기 베이스(2년 이상 바닥) 스캐너 — 컵앤핸들 · 박스
S&P 500 + 관심종목 + ETF, 일봉

찾는 것 2가지
  ① 🧱 베이스 진행 중 : 2년 이상 일정한 천장(저항) 아래에서 바닥을 다지고 있고, 지금 천장 근처까지 올라온 종목
  ② 🚀 베이스 돌파    : 2년 이상 베이스 천장을 일봉 종가로 뚫은 지 1달(21거래일)이 안 됐고,
                        돌파한 날(또는 그 뒤 2일 안) 거래량이 50일 평균의 1.5배 이상인 종목

판정 방식 (일봉의 고가·저가·종가 숫자로 계산)
  1) 베이스 기간 후보 = 504봉(2년) ~ 1260봉(5년), 가장 긴 기간을 고름
  2) 천장(저항) = 그 기간 가장 높은 일봉 종가 (윗꼬리는 무시)
  3) 조건
     - 천장에 2번 이상 닿음 (고가가 천장 -7% 안, 30봉 넘게 떨어진 것만 따로 셈). 그중 2번은 천장 -2% 안까지.
       첫 터치는 기간 앞쪽 40% 안, 첫·마지막 터치 간격 1년(252봉) 이상
     - 바닥 깊이 = 천장 대비 최저가 15~65% 아래 (너무 얕으면 그냥 횡보, 너무 깊으면 붕괴)
     - 최저점이 기간 맨 앞·맨 끝에 있지 않음 (하락 추세 한가운데가 아님)
     - 기간 중 고가가 천장 +15%를 넘은 적 없음 (중간에 크게 뚫고 간 적 없음)
  4) 모양 이름
     - 박스      : 천장 3번 이상 + 바닥 2번 이상 터치
     - 컵앤핸들  : 가운데가 깊은 U자 + 최근 100봉 안에서 천장 근처 찍고 5~30% 짧게 눌림 (손잡이)
     - 컵        : 가운데가 깊은 U자 (손잡이 없음)
     - 베이스    : 그 외 장기 횡보
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import patternkit as K

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
MIN_DAYS = 504             # 베이스 최소 길이(거래일). 504 = 약 2년
MAX_DAYS = 1260            # 베이스 최대 길이(거래일). 1260 = 약 5년
TOP_TOL = 7.0              # 천장 '터치'로 보는 범위: 천장 아래 이 % 안
TOUCH_GAP = 30             # 이 봉 수보다 떨어진 터치만 따로 셈
MIN_TOP_TOUCH = 2          # 천장 터치 최소 횟수
LID_TOL = 2.0              # 그중 최소 2번은 고가가 천장 -2% 안까지 와야 함 (천장이 진짜 저항인지 확인)
MIN_TOUCH_SPAN = 252       # 첫 터치 ~ 마지막 터치 최소 간격(거래일)
MIN_DEPTH = 15.0           # 바닥 깊이 최소 % (천장 대비 최저가)
MAX_DEPTH = 65.0           # 바닥 깊이 최대 %
MAX_SPIKE = 15.0           # 베이스 중 고가가 천장 위로 이 % 넘게 튄 적 있으면 제외
HANDLE_DAYS = 100          # 손잡이를 찾는 최근 봉 수
BO_MIN = 1.0               # 돌파 = 일봉 종가가 천장보다 이 % 이상 위
BO_DAYS = 21               # 돌파 후 이 거래일(약 1달) 안인 것만
BO_VOL_MIN = 1.5           # 돌파일 거래량 ÷ 직전 50일 평균 최소 (0 = 거래량 조건 없음)
BO_VOL_DAYS = 3            # 돌파일 포함 며칠 안에 거래량 조건을 만족하면 인정 (1 = 돌파 당일만)
NEAR_PCT = 15.0            # 베이스 진행 중: 현재가가 천장 아래 이 % 안인 것만 (클수록 결과 많아짐)
IMG_MAX_BO = 8             # 돌파 그림 최대 장 수
IMG_MAX_BASE = 5           # 진행 중 그림 최대 장 수 (천장에 가까운 순)
REALERT_BO = 35            # 같은 돌파 재알림 금지(일)
REALERT_BASE = 7           # 진행 중 목록 같은 종목 재알림 금지(일) → 1주에 한 번씩 다시 보여줌
STEP = 5                   # 베이스 길이 후보 간격(봉)
# ──────────────────────────────────────────────────────

ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)


def clusters(idx: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """가까운 봉 번호끼리 묶기 (gap봉 넘게 떨어지면 새 묶음) → [(시작, 끝)]"""
    out = []
    for i in idx:
        if out and i - out[-1][1] <= gap:
            out[-1] = (out[-1][0], int(i))
        else:
            out.append((int(i), int(i)))
    return out


def check_base(h: np.ndarray, l: np.ndarray, c: np.ndarray, s: int, e: int):
    """[s, e) 구간이 베이스인지. 맞으면 (정보, ""), 아니면 (None, 이유)"""
    h, l, c = h[s:e], l[s:e], c[s:e]
    L = e - s
    R = float(c.max())
    low_i = int(l.argmin())
    low = float(l[low_i])
    depth = (R - low) / R * 100
    if depth < MIN_DEPTH:
        return None, f"바닥이 너무 얕음 ({depth:.0f}% < {MIN_DEPTH:g}%) — 그냥 횡보"
    if depth > MAX_DEPTH:
        return None, f"바닥이 너무 깊음 ({depth:.0f}% > {MAX_DEPTH:g}%)"
    if h.max() > R * (1 + MAX_SPIKE / 100):
        return None, "베이스 중간에 천장 위로 크게 튄 적 있음"
    tops = clusters(np.where(h >= R * (1 - TOP_TOL / 100))[0], TOUCH_GAP)
    if len(tops) < MIN_TOP_TOUCH:
        return None, f"천장 터치 {len(tops)}번 (최소 {MIN_TOP_TOUCH})"
    lid = sum(1 for s0, e0 in tops if h[s0:e0 + 1].max() >= R * (1 - LID_TOL / 100))
    if lid < 2:
        return None, f"천장 가격(±{LID_TOL:g}%)까지 제대로 올라온 게 1번뿐 (하락 중 지나간 가격일 가능성)"
    if tops[0][0] > L * 0.4:
        return None, "천장 첫 터치가 기간 앞쪽에 없음 (왼쪽 벽 없음)"
    if tops[-1][1] - tops[0][0] < MIN_TOUCH_SPAN:
        return None, f"천장 터치 간격이 {MIN_TOUCH_SPAN}거래일 미만"
    if not (0.1 * L <= low_i <= 0.85 * L):
        return None, "최저점이 기간 맨 앞/맨 끝 (하락 추세 중)"
    sup = clusters(np.where(l <= low * 1.12)[0], TOUCH_GAP)
    shape = "베이스"
    if len(tops) >= 3 and len(sup) >= 2:
        shape = "박스"
    elif 0.2 * L <= low_i <= 0.8 * L:
        shape = "컵"
        tail = h[-HANDLE_DAYS:]
        j = int(tail.argmax())
        hh = float(tail[j])
        if hh >= R * 0.93 and j <= len(tail) - 5:
            dip = float(l[-HANDLE_DAYS:][j:].min())
            if hh * 0.70 <= dip <= hh * 0.95:
                shape = "컵앤핸들"
    return {"R": R, "low": low, "depth": depth, "days": L, "start": s, "end": e, "low_i": s + low_i,
            "touch": len(tops), "sup": len(sup), "shape": shape}, ""


def longest_base(h, l, c, e: int):
    """끝이 e(그날은 제외)인 가장 긴 베이스"""
    why = "기간 부족 (상장 2년 미만)"
    for L in range(min(MAX_DAYS, e), MIN_DAYS - 1, -STEP):
        b, why = check_base(h, l, c, e - L, e)
        if b:
            return b, ""
    return None, why


def analyze(df: pd.DataFrame):
    h, l, c, v = (df[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    n = len(c)
    if n < MIN_DAYS + 2:
        return None, "기간 부족 (상장 2년 미만)"
    last = float(c[-1])
    # ② 돌파: 최근 21거래일 중 처음으로 베이스 천장 위에 마감한 날. 그날~2일 안에 거래량 1.5배 이상이어야 함
    for b in range(max(MIN_DAYS, n - 1 - BO_DAYS), n):
        if c[b] < c[b - MIN_DAYS:b].max() * (1 + BO_MIN / 100):
            continue                                   # 최근 2년 최고 종가도 못 넘음 → 돌파 아님 (빠른 거르기)
        base, _ = longest_base(h, l, c, b)
        if not base or c[b] < base["R"] * (1 + BO_MIN / 100):
            continue
        prev, _ = longest_base(h, l, c, b - 42)          # 2달 전 기준으로 더 낮은 베이스를 이미 뚫고 올라온 거면 제외
        if prev and prev["R"] < base["R"] * 0.97 and c[b - 42:b].max() > prev["R"] * (1 + BO_MIN / 100):
            continue
        vavg = v[b - 50:b].mean()
        rv = [v[j] / vavg if vavg else 0.0 for j in range(b, min(b + BO_VOL_DAYS, n))]
        vol = max(rv)
        if BO_VOL_MIN and not (vol >= BO_VOL_MIN):
            return None, (f"{df.index[b]:%m/%d} 천장({K.px(base['R'])}) 돌파했지만 거래량 {vol:.1f}배 "
                          f"(돌파일~{BO_VOL_DAYS}일 안에 {BO_VOL_MIN:g}배 이상 필요)")
        return {"stage": "돌파", "bo_i": b, "bo_day": df.index[b], "since": n - 1 - b, "vol": vol,
                "close": last, "pos": (last / base["R"] - 1) * 100, **base}, ""
    # ① 진행 중: 오늘을 뺀 기간이 베이스이고, 지금 천장 아래 NEAR_PCT% 안
    base, why = longest_base(h, l, c, n - 1)
    if not base:
        return None, why
    pos = (last / base["R"] - 1) * 100
    if pos > BO_MIN:
        return None, "천장 위 (돌파 조건 미달)"
    if pos < -NEAR_PCT:
        return None, f"베이스는 맞지만 천장까지 {pos:.0f}% (기준 -{NEAR_PCT:g}% 안)"
    return {"stage": "진행", "close": last, "pos": pos, **base}, ""


# ───────────── 디스코드 ─────────────
def yrs(days: int) -> str:
    return f"{days / 252:.1f}년"


def table_bo(rows):
    head = (K.fit("TICKER", 7) + K.fit("모양", 9) + K.fit("천장", 9, True) + K.fit("종가", 9, True)
            + K.fit("천장대비", 9, True) + K.fit("돌파일", 8, True) + K.fit("경과", 5, True) + K.fit("기간", 6, True)
            + K.fit("깊이", 5, True) + K.fit("거래량", 7, True) + "  이름")
    lines = [K.fit(r["t"], 7) + K.fit(r["shape"], 9) + K.fit(K.px(r["R"]), 9, True) + K.fit(K.px(r["close"]), 9, True)
             + K.fit(f"{r['pos']:+.1f}%" + ("⚠" if r["pos"] < 0 else ""), 9, True)
             + K.fit(r["bo_day"].strftime("%m/%d"), 8, True) + K.fit(f"{r['since']}일", 5, True)
             + K.fit(yrs(r["days"]), 6, True) + K.fit(f"{r['depth']:.0f}%", 5, True) + K.fit(f"{r['vol']:.1f}x", 7, True)
             + "  " + r["name"][:18] for r in rows]
    K.send_lines(f"🚀 베이스 돌파 (1달 이내 · 거래량 {BO_VOL_MIN:g}배 이상)", head, lines)


def table_base(rows):
    head = (K.fit("TICKER", 7) + K.fit("모양", 9) + K.fit("천장", 9, True) + K.fit("종가", 9, True)
            + K.fit("천장까지", 9, True) + K.fit("기간", 6, True) + K.fit("깊이", 5, True) + K.fit("터치", 5, True) + "  이름")
    lines = [K.fit(r["t"], 7) + K.fit(r["shape"], 9) + K.fit(K.px(r["R"]), 9, True) + K.fit(K.px(r["close"]), 9, True)
             + K.fit(f"{r['pos']:+.1f}%", 9, True) + K.fit(yrs(r["days"]), 6, True) + K.fit(f"{r['depth']:.0f}%", 5, True)
             + K.fit(str(r["touch"]), 5, True) + "  " + r["name"][:18] for r in rows]
    K.send_lines("🧱 2년+ 베이스 진행 중 (천장 근처)", head, lines)


def picture(r: dict):
    import charts
    w = r["df"]
    start = max(0, r["start"] - 40)
    marks = [{"i": r["low_i"], "v": r["low"], "text": "바닥", "color": charts.GREEN}]
    if r["stage"] == "돌파":
        b = r["bo_i"]
        marks.append({"i": b, "v": float(w["High"].iloc[b]), "text": "돌파", "color": charts.YELLOW, "below": False})
        sub = (f"천장 {K.px(r['R'])} 돌파 {r['bo_day']:%Y-%m-%d} ({r['since']}거래일 전) · 현재 {r['pos']:+.1f}% · "
               f"베이스 {yrs(r['days'])} · 깊이 {r['depth']:.0f}% · 돌파일 거래량 {r['vol']:.1f}배")
        title = f"{r['t']} {r['name'][:20]} · {r['shape']} 돌파 (일봉)"
    else:
        sub = (f"천장 {K.px(r['R'])}까지 {r['pos']:+.1f}% · 베이스 {yrs(r['days'])} · 깊이 {r['depth']:.0f}% · "
               f"천장 터치 {r['touch']}번")
        title = f"{r['t']} {r['name'][:20]} · {r['shape']} 진행 중 (일봉)"
    hl = [{"v": r["R"], "color": charts.RED, "ls": "--", "label": f"천장 {K.px(r['R'])}"},
          {"v": r["low"], "color": charts.GREEN, "ls": ":", "label": f"바닥 {K.px(r['low'])}"}]
    marks.append({"i": r["start"], "v": r["R"], "text": "▼ 베이스 시작", "color": charts.BLUE, "below": False})
    fig = K.candle_chart(w, start, title, sub, marks=marks, hlines=hl)
    charts.send(fig, filename=f"{r['t']}_base.png")


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(K.NY)
    if now.weekday() >= 5 and not K.FORCE and not K.DIAGNOSE:
        print("주말. 종료.")
        return
    uni = K.load_all(ROOT, STATE)
    tickers = K.DIAGNOSE or list(uni)
    data = K.download(tickers, period="6y")
    if not data:
        K.post("⚠️ 장기 베이스: 시세를 받지 못했습니다.")
        return
    found, why = [], {}
    for t in tickers:
        if t not in data:
            why[t] = "시세 없음"
            continue
        df = data[t]
        name, kind = uni.get(t, ("", "stock"))
        bad = K.liquid(df, kind)
        if bad and not K.DIAGNOSE:
            continue
        try:
            r, w = analyze(df)
        except Exception as e:
            r, w = None, f"계산 오류: {e}"
        if r:
            found.append({"t": t, "name": name, "df": df, **r})
        why[t] = w + (f" (참고: {bad} → 자동 스캔에서는 제외됨)" if bad else "")

    if K.DIAGNOSE:
        out = []
        for t in K.DIAGNOSE:
            f = next((x for x in found if x["t"] == t), None)
            if f and f["stage"] == "돌파":
                out.append(f"**{t}** 🚀 {f['shape']} 돌파 · 천장 {K.px(f['R'])} · {f['bo_day']:%m/%d} ({f['since']}거래일 전) · "
                           f"현재 {f['pos']:+.1f}% · 베이스 {yrs(f['days'])} · 깊이 {f['depth']:.0f}%")
            elif f:
                out.append(f"**{t}** 🧱 {f['shape']} 진행 중 · 천장 {K.px(f['R'])}까지 {f['pos']:+.1f}% · "
                           f"베이스 {yrs(f['days'])} · 깊이 {f['depth']:.0f}%")
            else:
                out.append(f"**{t}** ❌ {why.get(t, '알 수 없음')}")
        K.post("**장기 베이스 진단**\n" + "\n".join(out))
        for f in found:
            charts_safe(f)
        return

    sent_bo = K.Sent(STATE, REALERT_BO)
    bo, base = [], []
    for f in found:
        if f["stage"] == "돌파":
            key = f"B|{f['t']}|{f['R']:.2f}"
            if K.FORCE or sent_bo.fresh(key):
                bo.append(f)
                sent_bo.mark(key)
        else:
            key = f"A|{f['t']}"
            d = sent_bo.db.get(key)
            if K.FORCE or d is None or (sent_bo.today - dt.date.fromisoformat(d)).days >= REALERT_BASE:
                base.append(f)
                sent_bo.mark(key)
    sent_bo.save()

    session = max(d.index[-1] for d in data.values()).date()
    K.post(f"**🏗️ 장기 베이스 (2년+ 컵앤핸들·박스) | {session}**\n"
           f"🚀 돌파 1달 이내 {len(bo)} · 🧱 진행 중(천장 -{NEAR_PCT:g}% 안) {len(base)} (새로 잡힌 것만, 확인 {len(data)}종목)")
    bo.sort(key=lambda r: r["since"])
    base.sort(key=lambda r: -r["pos"])
    if bo:
        table_bo(bo)
    if base:
        table_base(base)
    if not bo and not base:
        K.post("오늘 새로 잡힌 장기 베이스 종목 없음")
    for f in bo[:IMG_MAX_BO] + base[:IMG_MAX_BASE]:
        charts_safe(f)
    if bo or base:
        K.post("천장 = 베이스 기간 최고 일봉 종가 · 천장대비/천장까지 = 현재가가 천장보다 몇 % 위/아래 (⚠ = 돌파 후 다시 천장 아래) · "
               "경과 = 돌파일 이후 거래일 · 기간 = 베이스 길이 · 깊이 = 천장 대비 바닥 % · 거래량 = 돌파일 ÷ 직전 50일 평균 · "
               "터치 = 천장에 닿은 횟수")


def charts_safe(f):
    try:
        picture(f)
    except Exception as e:
        print(f"[경고] 그림 실패: {e}")


if __name__ == "__main__":
    sys.exit(main())
