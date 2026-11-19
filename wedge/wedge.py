"""
폴링 웻지(하락 쐐기) 스캐너
S&P 500 + 관심종목 + ETF, 일봉

판정 방식 (매일의 고가·저가·종가 숫자로 계산)
  1) 스윙 고점·저점 찾기 (앞뒤 3봉 기준, 2% 미만 잔물결 무시)
  2) 위쪽 선 = 고점들 위에 그은 하락 추세선, 아래쪽 선 = 저점들 아래에 그은 하락 추세선
     → 둘 다 내려가는데 위쪽 선이 더 가파르게 내려와서 간격이 좁아지는 모양 (30~180봉, 10봉 간격으로 모두 시도)
  3) 단계: 🚀 돌파 (최근 5봉 안에 위쪽 선 위로 마감) / 🎯 돌파 임박 (위쪽 선 아래 3% 이내) / 🔨 형성 중
  4) 목표가 = 웻지 시작점(위쪽 선의 출발 가격), 손절 = 최근 10봉 저점 아래
잡힌 종목은 표 + 두 선을 그린 캔들 차트 그림으로 보냅니다.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import patternkit as K

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
WINDOWS = list(range(30, 181, 10))   # 웻지 길이 후보(봉)
PIVOT_K = 3                # 스윙 고점·저점 판별 폭
PIVOT_MIN_MOVE = 2.0       # 이 % 미만 잔물결은 무시
TOL = 1.5                  # 선에 '닿음'으로 보는 오차(%)
MIN_UPPER_TOUCH = 3        # 위쪽 선 터치 최소
MIN_LOWER_TOUCH = 2        # 아래쪽 선 터치 최소
MIN_TOTAL_TOUCH = 6        # 합계 최소
MIN_DECLINE = 8.0          # 웻지 기간 위쪽 선 하락폭 최소 %
SQUEEZE = 0.65             # 지금 두 선 간격 ÷ 시작 간격 최대 (수렴)
MAX_BREAK_BELOW = 0.05     # 아래쪽 선을 2% 넘게 깨고 마감한 봉 비율 상한
BREAKOUT_BARS = 5          # 돌파로 볼 최근 봉 수
BO_VOL_MIN = 1.5           # 돌파일 거래량 ÷ 50일 평균 최소 (거래량 실린 돌파만)
BO_CLOSE_POS = 0.5         # 돌파일 종가가 그날 고가~저가 중 위쪽 이 비율 이상 (윗꼬리만 긴 가짜 돌파 제외)
NEAR_PCT = 3.0             # 위쪽 선 아래 이 % 이내 = 돌파 임박
MAX_ABOVE = 10.0           # 돌파 후 선 위 이 % 넘게 간 종목은 제외
SHOW_NEAR = False          # 돌파 임박(위쪽 선 아래 3% 이내)도 보내기. 기본은 돌파만
SHOW_FORMING = False       # 형성 중 단계도 표로 보내기 (켜면 결과가 많아짐)
IMG_MAX = 8                # 그림 최대 장 수 (돌파·임박만)
REALERT_DAYS = 10          # 같은 종목·단계 재알림 금지 기간(일)

# ── 🌊 소파동 (중파동 속 작은 웻지 돌파 + 5일선 상승 전환) ──
SMALL_ON = True            # 소파동 섹션 켜기/끄기
SMALL = {
    "WINDOWS": list(range(8, 31, 2)),   # 웻지 길이 후보(봉): 2주 ~ 1.5달
    "PIVOT_K": 2,                       # 앞뒤 2봉보다 높은/낮은 봉을 고점·저점으로
    "PIVOT_MIN_MOVE": 1.5,              # 1.5% 미만 잔물결 무시
    "TOL": 1.5,                         # 선에 닿음 오차(%)
    "MIN_UPPER_TOUCH": 2, "MIN_LOWER_TOUCH": 2, "MIN_TOTAL_TOUCH": 4,
    "MIN_DECLINE": 4.0,                 # 위쪽 선 하락폭 최소 %
    "SQUEEZE": 0.85,                    # 수렴 기준 (작은 파동은 느슨하게)
    "MAX_BREAK_BELOW": 0.10,
    "BREAKOUT_BARS": 2,                 # 최근 2봉 안에 돌파한 것만 (막 뚫은 순간)
    "BO_VOL_MIN": 1.2,                  # 돌파일 거래량 ÷ 50일 평균 최소
    "BO_CLOSE_POS": 0.5,
    "NEAR_PCT": 0.0,
    "MAX_ABOVE": 6.0,                   # 돌파 후 선 위 6% 넘게 간 건 제외
    "ONLY_BO": True,                    # 돌파한 웻지만 찾음
}
MA_LEN = 5                 # 꺾임을 볼 이동평균 (4일선이면 4)
TURN_DAYS = 3              # 이동평균이 하락 → 상승으로 바뀐 게 최근 며칠 안 (1 = 오늘만)
IMG_MAX_SMALL = 6          # 소파동 그림 최대 장 수
# ──────────────────────────────────────────────────────

BIG = {"WINDOWS": WINDOWS, "PIVOT_K": PIVOT_K, "PIVOT_MIN_MOVE": PIVOT_MIN_MOVE, "TOL": TOL,
       "MIN_UPPER_TOUCH": MIN_UPPER_TOUCH, "MIN_LOWER_TOUCH": MIN_LOWER_TOUCH, "MIN_TOTAL_TOUCH": MIN_TOTAL_TOUCH,
       "MIN_DECLINE": MIN_DECLINE, "SQUEEZE": SQUEEZE, "MAX_BREAK_BELOW": MAX_BREAK_BELOW, "BREAKOUT_BARS": BREAKOUT_BARS,
       "BO_VOL_MIN": BO_VOL_MIN, "BO_CLOSE_POS": BO_CLOSE_POS, "NEAR_PCT": NEAR_PCT, "MAX_ABOVE": MAX_ABOVE}

ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)


def best_line(pts: list[tuple[int, float]], upper: bool, start: int, tol: float = TOL):
    """pts = [(봉 번호, 가격)]. 모든 점이 선 아래(upper) 또는 위(lower)에 오도록 두 점을 잇는 선 중 터치가 가장 많은 것"""
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    best = None
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            lv = ys[a] + (ys[b] - ys[a]) * (xs - xs[a]) / (xs[b] - xs[a])
            r = ys / lv - 1
            if (upper and (r > tol / 100).any()) or (not upper and (r < -tol / 100).any()):
                continue
            touch = int((np.abs(r) <= tol / 100).sum())
            score = (touch, xs[b] - xs[a])
            if best is None or score > best[0]:
                best = (score, (int(xs[a]), float(ys[a]), int(xs[b]), float(ys[b])), touch)
    return best


def find_wedge(df: pd.DataFrame, P: dict = None) -> tuple[dict | None, str]:
    P = P or BIG
    h, l, c, v = (df[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    n = len(c)
    P_all = K.pivots(h, l, P['PIVOT_K'], P['PIVOT_MIN_MOVE'])
    why, best = "고점·저점이 하락 쐐기 모양으로 모이지 않음", None
    for W in P['WINDOWS']:
        s = n - W
        if s < 5:
            continue
        PP = [p for p in P_all if p[0] >= s]
        H = [(i, val) for i, t, val in PP if t == "H"]
        L = [(i, val) for i, t, val in PP if t == "L"]
        if len(H) < 2 or len(L) < 2:
            continue
        bu, bl = best_line(H, True, s, P['TOL']), best_line(L, False, s, P['TOL'])
        if not bu or not bl:
            continue
        (u1, uv1, u2, uv2), tu = bu[1], bu[2]
        (l1, lv1, l2, lv2), tl = bl[1], bl[2]
        U = lambda i: K.line_at(u1, uv1, u2, uv2, i)
        Lw = lambda i: K.line_at(l1, lv1, l2, lv2, i)
        su, sl = (U(n - 1) / U(s) - 1), (Lw(n - 1) / Lw(s) - 1)
        if not (su < 0 and sl < 0):
            why = "두 선이 모두 하락하지 않음"
            continue
        if not su < sl:
            why = "위쪽 선이 아래쪽 선보다 완만함 (하락 채널이거나 벌어지는 모양)"
            continue
        if -su * 100 < P['MIN_DECLINE']:
            why = f"웻지 하락폭 부족 ({-su * 100:.1f}%)"
            continue
        if tu < P['MIN_UPPER_TOUCH'] or tl < P['MIN_LOWER_TOUCH'] or tu + tl < P['MIN_TOTAL_TOUCH']:
            why = f"선 터치 부족 (위 {tu} · 아래 {tl})"
            continue
        start = min(u1, l1)
        g0 = (U(start) - Lw(start)) / U(start)
        g1 = (U(n - 1) - Lw(n - 1)) / U(n - 1)
        if not (-0.01 < g1 <= g0 * P['SQUEEZE']):
            why = f"수렴 부족 (간격 {g0 * 100:.1f}% → {g1 * 100:.1f}%)"
            continue
        body = range(start, max(start + 1, n - P['BREAKOUT_BARS']))
        if any(c[i] > U(i) * (1 + P['TOL'] / 100) for i in body):
            why = "웻지 중간에 이미 위쪽 선을 넘은 적 있음"
            continue
        below = np.mean([c[i] < Lw(i) * 0.98 for i in body])
        if below > P['MAX_BREAK_BELOW']:
            why = f"아래쪽 선 이탈이 잦음 ({below * 100:.0f}%)"
            continue
        bo = next((i for i in range(max(start, n - P['BREAKOUT_BARS']), n) if c[i] > U(i) * 1.005), None)
        if bo is None and P.get("ONLY_BO"):
            why = "최근 돌파 없음"
            continue
        pos = (c[-1] / U(n - 1) - 1) * 100
        if bo is not None:
            if c[-1] < U(n - 1) * 0.995:
                why = "돌파 후 다시 위쪽 선 아래"
                continue
            if pos > P['MAX_ABOVE']:
                why = f"돌파 후 선 위로 너무 멀리 감 ({pos:+.1f}%)"
                continue
            vavg_bo = v[max(0, bo - 50):bo].mean()
            bvol = v[bo] / vavg_bo if vavg_bo > 0 else 0
            if bvol < P['BO_VOL_MIN']:
                why = f"돌파했지만 거래량 부족 (평소의 {bvol:.1f}배 < {P['BO_VOL_MIN']:g}배)"
                continue
            rng = h[bo] - l[bo]
            if rng > 0 and (c[bo] - l[bo]) / rng < P['BO_CLOSE_POS']:
                why = "돌파일 종가가 그날 아래쪽에서 끝남 (윗꼬리 긴 약한 돌파)"
                continue
            stage = "돌파"
        elif pos >= -P['NEAR_PCT']:
            stage = "임박"
        elif c[-1] >= Lw(n - 1) * 0.98:
            stage = "형성"
        else:
            why = "아래쪽 선 아래로 이탈"
            continue
        vavg = v[max(0, n - 51):n - 1].mean()
        s200 = df["Close"].rolling(200).mean().iloc[-1]
        cand = {"stage": stage, "upper": float(U(n - 1)), "lower": float(Lw(n - 1)), "pos": pos,
                "target": float(max(uv1, U(start))), "stop": float(l[-10:].min() * 0.99), "bars": n - 1 - start,
                "touch": (tu, tl), "sqz": g1 / g0 * 100 if g0 else np.nan, "decline": -su * 100,
                "vol": float(v[bo] / v[max(0, bo - 50):bo].mean()) if bo is not None and v[max(0, bo - 50):bo].mean() > 0
                else float(v[-1] / vavg) if vavg > 0 else np.nan,
                "above200": bool(not pd.isna(s200) and c[-1] > s200),
                "lines": ((u1, uv1, u2, uv2), (l1, lv1, l2, lv2)), "start": start,
                "key": df.index[start].date().isoformat()}
        score = (tu + tl, cand["bars"])
        if best is None or score > best[0]:
            best = (score, cand)
    return (best[1], "") if best else (None, why)


def ma_turn(df: pd.DataFrame) -> tuple[int | None, float]:
    """MA_LEN일 이동평균이 하락 → 상승으로 바뀐 게 며칠 전인지 (0 = 오늘). 최근 TURN_DAYS일 안에 없으면 None.
    두 번째 값 = 오늘 기울기(%)"""
    ma = df["Close"].rolling(MA_LEN).mean().to_numpy(float)
    d = np.diff(ma)
    if len(d) < TURN_DAYS + 2 or np.isnan(d[-TURN_DAYS - 2:]).any() or d[-1] <= 0:
        return None, 0.0
    slope = float(d[-1] / ma[-2] * 100)
    for k in range(TURN_DAYS):                         # k일 전에 상승으로 바뀜: 그날 기울기 > 0, 전날 기울기 <= 0
        if d[-1 - k] > 0 and d[-2 - k] <= 0 and all(x > 0 for x in d[len(d) - 1 - k:]):
            return k, slope
    return None, slope


# ───────────── 디스코드 ─────────────
STAGE = {"돌파": "🚀 돌파", "임박": "🎯 돌파 임박", "형성": "🔨 형성 중"}
STAGE_TXT = {"돌파": "돌파", "임박": "돌파 임박", "형성": "형성 중"}


def table(title: str, rows: list[dict]):
    head = (K.fit("TICKER", 7) + K.fit("위쪽선", 9, True) + K.fit("종가", 9, True) + K.fit("거리%", 7, True)
            + K.fit("목표가", 9, True) + K.fit("목표%", 6, True) + K.fit("손절", 9, True) + K.fit("터치", 6, True)
            + K.fit("수렴%", 6, True) + K.fit("봉", 5, True) + K.fit("거래량", 7, True) + K.fit("200일", 6, True) + "  이름")
    lines = [K.fit(r["t"], 7) + K.fit(K.px(r["upper"]), 9, True) + K.fit(K.px(r["close"]), 9, True) + f"{r['pos']:>+7.1f}"
             + K.fit(K.px(r["target"]), 9, True) + f"{(r['target'] / r['close'] - 1) * 100:>+6.0f}" + K.fit(K.px(r["stop"]), 9, True)
             + K.fit(f"{r['touch'][0]}+{r['touch'][1]}", 6, True) + f"{r['sqz']:>6.0f}{r['bars']:>5}{r['vol']:>7.1f}"
             + K.fit("위" if r["above200"] else "아래", 6, True) + "  " + r["name"][:18]
             for r in rows]
    K.send_lines(title, head, lines)


def picture(t: str, name: str, df: pd.DataFrame, r: dict, small: bool = False):
    import charts
    n = len(df)
    (u1, uv1, u2, uv2), (l1, lv1, l2, lv2) = r["lines"]
    s = r["start"]
    lines = [{"i1": s, "v1": K.line_at(u1, uv1, u2, uv2, s), "i2": n - 1, "v2": r["upper"], "i3": n + 2,
              "color": charts.RED, "label": "위쪽 선 (저항)"},
             {"i1": s, "v1": K.line_at(l1, lv1, l2, lv2, s), "i2": n - 1, "v2": r["lower"], "i3": n + 2,
              "color": charts.GREEN, "label": "아래쪽 선 (지지)"}]
    hl = [{"v": r["target"], "color": charts.BLUE, "ls": ":", "label": f"목표 {K.px(r['target'])}"},
          {"v": r["stop"], "color": charts.YELLOW, "ls": ":", "label": f"손절 {K.px(r['stop'])}"}]
    st = max(0, s - (15 if small else 25))
    kind = "소파동 웻지" if small else "폴링 웻지"
    fig = K.candle_chart(df, st, f"{t} {name[:20]} · {kind} {STAGE_TXT[r['stage']]}",
                         f"위쪽 선 {K.px(r['upper'])} (현재가 대비 {r['pos']:+.1f}%) · 웻지 {r['bars']}봉 · 터치 {r['touch'][0]}+{r['touch'][1]} · "
                         f"간격 {r['sqz']:.0f}%로 수렴" + (f" · {MA_LEN}일선 상승 전환 {r['turn_txt']}" if small else ""),
                         lines=lines, hlines=hl)
    if small:                                          # 5일선 겹쳐 그리기
        ax = fig.axes[0]
        ma = df["Close"].rolling(MA_LEN).mean().to_numpy(float)[st:]
        ax.plot(np.arange(len(ma)), ma, color=charts.BLUE, lw=1.4, label=f"{MA_LEN}일선")
        ax.legend(loc="upper right", ncol=3, frameon=True, facecolor=charts.BG if hasattr(charts, "BG") else "#111",
                  edgecolor="none", framealpha=0.8, labelcolor=charts.TEXT, fontsize=8)
    charts.send(fig, filename=f"{t}_{'small' if small else 'wedge'}.png")


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
        K.post("⚠️ 폴링 웻지: 시세를 받지 못했습니다.")
        return
    found, why = [], {}
    small, why_s = [], {}
    for t in tickers:
        if t not in data or len(data[t]) < 120:
            why[t] = "시세 없음 또는 기간 부족"
            continue
        df = data[t]
        name, kind = uni.get(t, ("", "stock"))
        bad = K.liquid(df, kind)
        if bad and not K.DIAGNOSE:
            continue
        try:
            r, w = find_wedge(df)
        except Exception as e:
            r, w = None, f"계산 오류: {e}"
        if r:
            found.append({"t": t, "name": name, "kind": kind, "close": float(df["Close"].iloc[-1]), **r})
        why[t] = w + (f" (참고: {bad} → 자동 스캔에서는 제외됨)" if bad else "")
        if SMALL_ON:
            try:
                r2, w2 = find_wedge(df, SMALL)
                if r2 and r2["stage"] != "돌파":
                    r2, w2 = None, "작은 웻지는 있지만 최근 2봉 안 돌파 아님"
                if r2:
                    k, slope = ma_turn(df)
                    if k is None:
                        r2, w2 = None, f"작은 웻지 돌파는 했지만 {MA_LEN}일선이 최근 {TURN_DAYS}일 안에 상승 전환 안 함"
                    else:
                        r2["turn_txt"] = "오늘" if k == 0 else f"{k}일 전"
                        r2["slope"] = slope
            except Exception as e:
                r2, w2 = None, f"계산 오류: {e}"
            if r2:
                small.append({"t": t, "name": name, "kind": kind, "close": float(df["Close"].iloc[-1]), **r2})
            why_s[t] = w2

    if K.DIAGNOSE:
        lines = []
        for t in K.DIAGNOSE:
            f = next((x for x in found if x["t"] == t), None)
            if f:
                lines.append(f"**{t}** ✅ 폴링 웻지 {STAGE[f['stage']]} · 위쪽 선 {K.px(f['upper'])} ({f['pos']:+.1f}%) · "
                             f"목표 {K.px(f['target'])} · 손절 {K.px(f['stop'])} · {f['bars']}봉 · 터치 {f['touch'][0]}+{f['touch'][1]}")
            else:
                lines.append(f"**{t}** ❌ {why.get(t, '알 수 없음')}")
        if SMALL_ON:
            lines.append("")
            lines.append("**🌊 소파동**")
            for t in K.DIAGNOSE:
                f = next((x for x in small if x["t"] == t), None)
                if f:
                    lines.append(f"**{t}** ✅ 작은 웻지 돌파 · 위쪽 선 {K.px(f['upper'])} ({f['pos']:+.1f}%) · {f['bars']}봉 · "
                                 f"거래량 {f['vol']:.1f}배 · {MA_LEN}일선 상승 전환 {f['turn_txt']}")
                else:
                    lines.append(f"**{t}** ❌ {why_s.get(t, '알 수 없음')}")
        K.post("**폴링 웻지 진단**\n" + "\n".join(lines))
        for f in found:
            try:
                picture(f["t"], f["name"], data[f["t"]], f)
            except Exception as e:
                print(f"[경고] 그림 실패: {e}")
        for f in small:
            try:
                picture(f["t"], f["name"], data[f["t"]], f, small=True)
            except Exception as e:
                print(f"[경고] 그림 실패: {e}")
        return

    sent = K.Sent(STATE, REALERT_DAYS)
    new = []
    for f in found:
        key = f"{f['t']}|W|{f['stage']}|{f['key']}"
        if K.FORCE or sent.fresh(key):
            new.append(f)
            sent.mark(key)
    new_s = []
    for f in small:
        key = f"{f['t']}|S|{f['key']}"
        if K.FORCE or sent.fresh(key):
            new_s.append(f)
            sent.mark(key)
    sent.save()
    session = max(d.index[-1] for d in data.values()).date()
    new = [f for f in new if f["stage"] == "돌파" or (f["stage"] == "임박" and SHOW_NEAR)
           or (f["stage"] == "형성" and SHOW_FORMING)]
    cnt = lambda s: sum(1 for f in new if f["stage"] == s)
    K.post(f"**📐 폴링 웻지 돌파 (거래량 {BO_VOL_MIN:g}배 이상) | {session}**\n🚀 돌파 {cnt('돌파')}"
           + (f" · 🎯 돌파 임박 {cnt('임박')}" if SHOW_NEAR else "")
           + (f" · 🔨 형성 중 {cnt('형성')}" if SHOW_FORMING else "")
           + (f" · 🌊 소파동 {len(new_s)}" if SMALL_ON else "") +
           f" (새로 잡힌 것만, 확인 {len(data)}종목)")
    if not new:
        K.post(f"📐 중파동: 오늘 새로 거래량 실린 돌파(평소 {BO_VOL_MIN:g}배 이상)가 나온 폴링 웻지 없음")
    for st in ["돌파"] + (["임박"] if SHOW_NEAR else []) + (["형성"] if SHOW_FORMING else []):
        rows = sorted([f for f in new if f["stage"] == st], key=lambda r: (-(r["touch"][0] + r["touch"][1]), r["sqz"]))
        if rows:
            table(f"폴링 웻지 · {STAGE[st]}", rows)
    pics = sorted([f for f in new if f["stage"] in ("돌파", "임박")],
                  key=lambda r: (r["stage"] != "돌파", -(r["touch"][0] + r["touch"][1])))[:IMG_MAX]
    for f in pics:
        try:
            picture(f["t"], f["name"], data[f["t"]], f)
        except Exception as e:
            print(f"[경고] 그림 실패: {e}")
    if SMALL_ON:
        if new_s:
            rows = sorted(new_s, key=lambda r: (-r["vol"]))
            table(f"🌊 소파동 · 작은 웻지 돌파(최근 {SMALL['BREAKOUT_BARS']}봉, 거래량 {SMALL['BO_VOL_MIN']:g}배↑) + "
                  f"{MA_LEN}일선 상승 전환({TURN_DAYS}일 안)", rows)
            for f in rows[:IMG_MAX_SMALL]:
                try:
                    picture(f["t"], f["name"], data[f["t"]], f, small=True)
                except Exception as e:
                    print(f"[경고] 그림 실패: {e}")
        else:
            K.post(f"🌊 소파동: 오늘 새로 작은 웻지를 돌파하며 {MA_LEN}일선이 꺾여 올라온 종목 없음")
    if new or new_s:
        K.post("위쪽선 = 오늘 기준 위쪽(저항) 선 값 · 거리% = 종가가 위쪽 선보다 몇 % 위/아래 · 목표가 = 웻지 시작 가격 · "
               "손절 = 최근 10봉 저점 아래 · 터치 = 위+아래 선에 닿은 고점·저점 수 · 수렴% = 지금 간격 ÷ 시작 간격 · "
               "거래량 = 돌파일(그 외 오늘) ÷ 50일 평균 · 200일 = 200일선 위(상승 추세 속 눌림)/아래(하락 끝 반전 시도)")


if __name__ == "__main__":
    sys.exit(main())
