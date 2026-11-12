"""
데이터처리개론 과제 1·2 — JO-PT 채팅에서 호출하는 버전 (외부 라이브러리 없음)

원본 코드:
  과제1  github.com/jjohyun123-ctrl/Introduction-to-Data-Processing---Assignment-1-Code-
  과제2  github.com/jjohyun123-ctrl/Data-Processing-Assignment-2-ANN-Code

윈도우 '스마트 앱 컨트롤'이 pandas·numpy·scikit-learn 같은 라이브러리 파일을 차단해서,
같은 계산을 파이썬 기본 기능(csv, math, random)만으로 직접 구현했습니다.
  - 과제1: 최소제곱법(정규방정식)으로 선형 회귀 + 2차 다항 회귀  ← LinearRegression, PolynomialFeatures와 같은 계산
  - 과제2: 입력 3 → 은닉 6(시그모이드) → 출력 1 신경망을 순전파·오차역전파·Adam으로 직접 학습
           ← 원본 Keras 모델과 같은 구조·설정 (Adam 0.01, 배치 256, 60 epoch, MinMax 정규화, 8:2 분할)
그래프는 matplotlib 대신 SVG(웹 그림)로 직접 그립니다.
"""
import csv
import math
import os
import random
import re
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
A1_FILE = os.path.join(DATA_DIR, "a1_wind_wave.csv")
A2_FILE = os.path.join(DATA_DIR, "a2_ulsan_ships.csv")


# =====================================================================
# 공통 도구
# =====================================================================
def read_csv(path):
    """CSV 읽기 (공공데이터는 cp949인 경우가 많아서 인코딩 자동 처리)"""
    for enc in ["utf-8-sig", "cp949"]:
        try:
            with open(path, encoding=enc, newline="") as f:
                rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
            return rows[0], rows[1:]
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 읽을 수 없습니다.")


def read_xlsx(path):
    """엑셀(xlsx) 첫 번째 시트 읽기 — xlsx는 사실 zip 안의 xml 파일이라 직접 풀어서 읽음"""
    m = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(m + "si"):
                shared.append("".join(t.text or "" for t in si.iter(m + "t")))
        sheet = sorted(n for n in names if n.startswith("xl/worksheets/sheet"))[0]
        root = ET.fromstring(z.read(sheet))
    rows = []
    for row in root.iter(m + "row"):
        cells = {}
        for c in row.iter(m + "c"):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r", "A")).group():
                col = col * 26 + ord(ch) - 64
            t, v = c.get("t"), c.find(m + "v")
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(m + "t"))
            else:
                val = v.text if v is not None else ""
            cells[col - 1] = val
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    rows = [r for r in rows if any(str(c).strip() for c in r)]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return rows[0], rows[1:]


def read_table(path):
    if path.lower().endswith(".xlsx"):
        return read_xlsx(path)
    if path.lower().endswith(".xls"):
        raise ValueError("옛날 엑셀(.xls)은 읽을 수 없어요. 엑셀에서 '다른 이름으로 저장 → CSV 또는 xlsx'로 바꿔 주세요.")
    return read_csv(path)


def to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def numeric_columns(header, rows):
    """숫자로 된 열만 고르기 (원본 코드의 select_dtypes(include=[np.number])와 같은 역할)"""
    out = []
    for i, name in enumerate(header):
        vals = [r[i] for r in rows if i < len(r) and str(r[i]).strip() != ""]
        if vals and sum(to_float(v) is not None for v in vals) >= 0.9 * len(vals):
            out.append(name)
    return out


def aliases(col):
    """채팅에서 열 이름을 알아보기 위한 별명: '풍속(m/s)' → 풍속, '선박길이' → 선박길이·길이"""
    base = re.sub(r"\(.*?\)|\[.*?\]", "", col).strip()
    out = {base}
    if len(base) > 2:
        out.add(base[-2:])
    return [a for a in out if a]


def solve(A, b):
    """가우스 소거법으로 연립방정식 A·x = b 풀기 (정규방정식용)"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[c][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def least_squares(rows, y):
    """최소제곱법: (XᵀX)β = Xᵀy  → 회귀계수 β (rows 각 행 앞에 1이 들어 있어야 함)"""
    k = len(rows[0])
    XtX = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    Xty = [sum(r[i] * yy for r, yy in zip(rows, y)) for i in range(k)]
    return solve(XtX, Xty)


def r2(y_true, y_pred):
    mu = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mu) ** 2 for a in y_true)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------- SVG 그래프
def _fmt(v):
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"


def num(v):
    """보기 좋은 숫자: 12,345 / 123.4 / 1.234 / 0.01234"""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.3f}"
    return f"{v:.4g}"


class Plot:
    """아주 간단한 산점도/선 그래프 (SVG)"""
    W, H, L, R, T, B = 460, 320, 62, 28, 34, 46

    def __init__(self, xr, yr, title, xl, yl, logy=False):
        self.x0, self.x1 = xr
        self.logy = logy
        self.y0, self.y1 = (math.log10(yr[0]), math.log10(yr[1])) if logy else yr
        self.parts = [f'<text x="{self.W/2}" y="20" text-anchor="middle" font-size="14" font-weight="700" fill="#13253a">{title}</text>']
        pw, ph = self.W - self.L - self.R, self.H - self.T - self.B
        self.parts.append(f'<rect x="{self.L}" y="{self.T}" width="{pw}" height="{ph}" fill="#fff" stroke="#9fb3c8"/>')
        for i in range(5):                                   # 눈금
            fx = self.x0 + (self.x1 - self.x0) * i / 4
            fy = self.y0 + (self.y1 - self.y0) * i / 4
            X, Y = self.px(fx), self._py_raw(fy)
            ylab = _fmt(10 ** fy) if logy else _fmt(fy)
            self.parts += [f'<line x1="{X}" y1="{self.T}" x2="{X}" y2="{self.T+ph}" stroke="#e6eef5"/>',
                           f'<line x1="{self.L}" y1="{Y}" x2="{self.L+pw}" y2="{Y}" stroke="#e6eef5"/>',
                           f'<text x="{X}" y="{self.T+ph+16}" text-anchor="middle" font-size="10" fill="#4a6078">{_fmt(fx)}</text>',
                           f'<text x="{self.L-6}" y="{Y+3}" text-anchor="end" font-size="10" fill="#4a6078">{ylab}</text>']
        self.parts += [f'<text x="{self.L+pw/2}" y="{self.H-8}" text-anchor="middle" font-size="12" fill="#13253a">{xl}</text>',
                       f'<text x="14" y="{self.T+ph/2}" text-anchor="middle" font-size="12" fill="#13253a" transform="rotate(-90 14 {self.T+ph/2})">{yl}</text>']
        self.legend = []

    def px(self, x):
        return round(self.L + (x - self.x0) / (self.x1 - self.x0 or 1) * (self.W - self.L - self.R), 1)

    def _py_raw(self, y):
        return round(self.T + (1 - (y - self.y0) / (self.y1 - self.y0 or 1)) * (self.H - self.T - self.B), 1)

    def py(self, y):
        if self.logy:
            y = math.log10(max(y, 1e-12))
        y = min(max(y, self.y0), self.y1)
        return self._py_raw(y)

    def scatter(self, xs, ys, color, label, r=2.2, op=0.6):
        self.parts += [f'<circle cx="{self.px(x)}" cy="{self.py(y)}" r="{r}" fill="{color}" fill-opacity="{op}"/>'
                       for x, y in zip(xs, ys)]
        self.legend.append((color, label, "dot"))

    def line(self, xs, ys, color, label, dash=False, w=2):
        pts = " ".join(f"{self.px(x)},{self.py(y)}" for x, y in zip(xs, ys))
        d = ' stroke-dasharray="6 4"' if dash else ""
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}"{d}/>')
        if label:
            self.legend.append((color, label, "line"))

    def svg(self):
        y = self.T + 14
        for color, label, kind in self.legend:
            x = self.L + 10
            mark = (f'<circle cx="{x+6}" cy="{y-4}" r="4" fill="{color}"/>' if kind == "dot"
                    else f'<line x1="{x}" y1="{y-4}" x2="{x+14}" y2="{y-4}" stroke="{color}" stroke-width="2.5"/>')
            self.parts += [mark, f'<text x="{x+20}" y="{y}" font-size="11" fill="#13253a">{label}</text>']
            y += 16
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" '
                f'style="width:100%;max-width:{self.W}px;background:#fff;border-radius:10px">' + "".join(self.parts) + "</svg>")


def save_svg(name, svgs):
    path = os.path.join(DATA_DIR, name)
    html = ('<div style="display:flex;gap:10px;flex-wrap:wrap">' + "".join(svgs) + "</div>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# =====================================================================
# 과제 1 : 선형 회귀 + 비선형(2차 다항) 회귀
# =====================================================================
def train_assignment1(path=A1_FILE, x_col=None, y_col=None):
    header, rows = read_table(path)
    # 숫자 열 자동 감지 → 첫 번째 = X, 두 번째 = Y (원본 코드와 동일). 직접 고르면 그 열 사용
    nums = numeric_columns(header, rows)
    if len(nums) < 2:
        raise ValueError("숫자로 된 열이 2개 이상 있어야 회귀분석을 할 수 있어요.")
    x_col = x_col or nums[0]
    y_col = y_col or [c for c in nums if c != x_col][0]
    xi, yi = header.index(x_col), header.index(y_col)
    pts = [(to_float(r[xi]), to_float(r[yi])) for r in rows]
    pts = [(x, y) for x, y in pts if x is not None and y is not None]     # 결측치 제거
    X = [p[0] for p in pts]
    y = [p[1] for p in pts]

    # 선형 회귀: y = b0 + b1·x
    b0, b1 = least_squares([[1, x] for x in X], y)
    y_lin = [b0 + b1 * x for x in X]
    # 비선형 회귀 (2차 다항): y = c0 + c1·x + c2·x²
    c0, c1, c2 = least_squares([[1, x, x * x] for x in X], y)
    y_poly = [c0 + c1 * x + c2 * x * x for x in X]
    r2_lin, r2_poly = r2(y, y_lin), r2(y, y_poly)

    # 그래프 (원본과 같은 구성: 원본 데이터 + 선형 직선 + 다항 곡선)
    xmin, xmax = min(X), max(X)
    ymin, ymax = min(y + y_lin + y_poly), max(y + y_lin + y_poly)
    pad = (ymax - ymin) * 0.05
    pl = Plot((xmin, xmax), (ymin - pad, ymax + pad), f"Regression Analysis ({x_col} vs {y_col})", x_col, y_col)
    pl.scatter(X, y, "#222", "Original Data", r=3, op=0.75)
    pl.line([xmin, xmax], [b0 + b1 * xmin, b0 + b1 * xmax], "#1f5fd1", f"Linear (R²={r2_lin:.3f})")
    xs = [xmin + (xmax - xmin) * i / 60 for i in range(61)]
    pl.line(xs, [c0 + c1 * x + c2 * x * x for x in xs], "#d1342f", f"Poly deg2 (R²={r2_poly:.3f})", dash=True)
    img = save_svg("a1_result.html", [pl.svg()])

    summary = (
        f"**[과제1 회귀분석 결과]** `{os.path.basename(path)}` 데이터 {len(X)}개 · X = {x_col}, Y = {y_col}\n\n"
        f"| 모델 | 회귀식 | 결정계수 R² |\n|---|---|---|\n"
        f"| 선형 회귀 | {y_col} = {b1:.4f} × {x_col} {b0:+.4f} | {r2_lin:.4f} |\n"
        f"| 2차 다항 회귀 | {y_col} = {c2:.5f} × {x_col}² {c1:+.4f} × {x_col} {c0:+.4f} | {r2_poly:.4f} |\n\n"
        f"→ R²가 더 높은 모델: **{'2차 다항 회귀' if r2_poly > r2_lin else '선형 회귀'}**"
    )
    return {"summary": summary, "image": img, "x_col": x_col, "y_col": y_col,
            "lin": (b0, b1), "poly": (c0, c1, c2), "r2_lin": r2_lin, "r2_poly": r2_poly,
            "x_range": (xmin, xmax), "file": os.path.basename(path)}


def predict_assignment1(res, x):
    b0, b1 = res["lin"]
    c0, c1, c2 = res["poly"]
    lin, pol = b0 + b1 * x, c0 + c1 * x + c2 * x * x
    lo, hi = res["x_range"]
    warn = "" if lo <= x <= hi else f"\n\n※ 입력값이 학습 데이터 범위({lo:g}~{hi:g}) 밖이라 예측이 부정확할 수 있어요."
    return (f"**[과제1 예측]** {res['x_col']} = {x:g}\n\n"
            f"| 모델 | 예측 {res['y_col']} |\n|---|---|\n"
            f"| 선형 회귀 | {num(lin)} |\n| 2차 다항 회귀 | {num(pol)} |" + warn)


# =====================================================================
# 과제 2 : 인공신경망 선박 총톤수 예측 (입력: 길이·너비·깊이 → 출력: 총톤수)
# =====================================================================
# 기본 예시(선박) 설정 = ship_tonnage_ann.py,  내 데이터 설정 = general_ann.py
SHIP = {"path": A2_FILE, "x_cols": ["선박길이", "선박너비", "선박깊이"], "y_col": "총톤수",
        "hidden": 6, "epochs": 60, "batch": 256, "lr": 0.01}
GENERAL = {"hidden": 8, "epochs": 100, "batch": 64, "lr": 0.01}
MAX_WORK = 4_000_000      # (학습 데이터 수 × epoch) 상한 — 넘으면 epoch를 줄여서 너무 오래 걸리지 않게
MIN_STEPS = 8000          # 가중치 업데이트 최소 횟수 — 데이터가 적어도 충분히 학습하도록


class TinyANN:
    """입력 3 → 은닉 H(시그모이드) → 출력 1, Adam으로 학습하는 신경망
    (원본: keras.Sequential([Input(3), Dense(6, sigmoid), Dense(1)]), loss=mse, Adam(0.01))"""

    def __init__(self, n_in=3, n_hidden=6, seed=42):
        rnd = random.Random(seed)
        lim1 = math.sqrt(6 / (n_in + n_hidden))          # Glorot 초기화 (Keras 기본값과 같은 방식)
        lim2 = math.sqrt(6 / (n_hidden + 1))
        self.ni, self.nh = n_in, n_hidden
        # 가중치 한 줄로: W(입력×은닉) | b(은닉) | V(은닉→출력) | c(출력 바이어스)
        self.P = ([rnd.uniform(-lim1, lim1) for _ in range(n_in * n_hidden)] + [0.0] * n_hidden +
                  [rnd.uniform(-lim2, lim2) for _ in range(n_hidden)] + [0.0])
        self.loss_curve = []

    def forward(self, x):
        P, ni, nh = self.P, self.ni, self.nh
        ob, vb = ni * nh, ni * nh + nh
        h = []
        for j in range(nh):
            z = P[ob + j] + sum(P[ni * j + k] * x[k] for k in range(ni))   # Σ(입력×가중치)+바이어스
            h.append(1 / (1 + math.exp(-z)) if z > -60 else 0.0)           # 시그모이드
        return P[-1] + sum(P[vb + j] * h[j] for j in range(nh)), h

    def fit(self, X, Y, epochs=60, batch=256, lr=0.01, seed=0):
        P, ni, nh, n = self.P, self.ni, self.nh, len(self.P)
        ob, vb = ni * nh, ni * nh + nh
        m, v, t = [0.0] * n, [0.0] * n, 0
        b1, b2, eps = 0.9, 0.999, 1e-7
        rng, order = random.Random(seed), list(range(len(X)))
        exp = math.exp
        for _ in range(epochs):
            rng.shuffle(order)
            tot = 0.0
            for s in range(0, len(order), batch):
                ids = order[s:s + batch]
                g = [0.0] * n
                for i in ids:                                   # 순전파 + 오차 역전파
                    x = X[i]
                    out, h = P[-1], [0.0] * nh
                    for j in range(nh):
                        z = P[ob + j]
                        base = ni * j
                        for k in range(ni):
                            z += P[base + k] * x[k]             # Σ(입력×가중치)+바이어스
                        hj = 1 / (1 + exp(-z)) if z > -60 else 0.0   # 시그모이드
                        h[j] = hj
                        out += P[vb + j] * hj
                    d = out - Y[i]
                    tot += d * d
                    g[-1] += d
                    for j in range(nh):
                        hj = h[j]
                        g[vb + j] += d * hj
                        dz = d * P[vb + j] * hj * (1 - hj)      # 시그모이드 미분
                        g[ob + j] += dz
                        base = ni * j
                        for k in range(ni):
                            g[base + k] += dz * x[k]
                t += 1                                          # Adam으로 가중치 업데이트
                k2 = 2.0 / len(ids)                             # MSE 미분 = 2·오차 / 배치크기
                a = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                for q in range(n):
                    gq = g[q] * k2
                    m[q] = b1 * m[q] + (1 - b1) * gq
                    v[q] = b2 * v[q] + (1 - b2) * gq * gq
                    P[q] -= a * m[q] / (math.sqrt(v[q]) + eps)
            self.loss_curve.append(tot / len(X))                # epoch별 평균제곱오차
        return self

    def predict(self, x):
        return self.forward(x)[0]


def train_assignment2(path=A2_FILE, x_cols=None, y_col=None, hidden=None, epochs=None, batch=None, lr=None):
    """기본값 = 울산항 선박 예시(ship_tonnage_ann.py 설정). 다른 파일이면 general_ann.py 설정
    hidden·epochs·batch·lr를 직접 주면 그 값 그대로 학습 (원본 코드의 설정 셀과 같은 역할)"""
    user_set = any(v is not None for v in (hidden, epochs, batch, lr))
    is_ship = os.path.abspath(path) == os.path.abspath(A2_FILE) and not x_cols
    cfg = dict(SHIP) if is_ship else dict(GENERAL)
    header, rows = read_table(path)
    nums = numeric_columns(header, rows)
    y_col = y_col or (cfg.get("y_col") if is_ship else nums[-1])
    x_cols = x_cols or (cfg.get("x_cols") if is_ship else [c for c in nums if c != y_col])
    if not x_cols:
        raise ValueError("입력으로 쓸 숫자 열이 없어요.")
    hidden = hidden or cfg["hidden"]
    epochs = epochs or cfg["epochs"]
    batch, lr = batch or cfg["batch"], lr or cfg["lr"]

    idx = [header.index(c) for c in x_cols + [y_col]]
    data = []
    for r in rows:
        vals = [to_float(r[i]) if i < len(r) else None for i in idx]   # "1,234" 같은 쉼표 숫자도 숫자로
        if None not in vals:
            data.append(vals)                                         # 결측치 제거
    if len(data) < 10:
        raise ValueError("숫자가 모두 들어 있는 행이 10개 미만이라 학습할 수 없어요.")
    ni = len(x_cols)

    # 0~1 정규화 (MinMaxScaler와 같은 계산)
    cols = list(zip(*data))
    mn, mx = [min(c) for c in cols], [max(c) for c in cols]
    scale = lambda v, j: (v - mn[j]) / ((mx[j] - mn[j]) or 1)
    unscale_y = lambda v: v * (mx[ni] - mn[ni]) + mn[ni]
    Xs = [[scale(d[j], j) for j in range(ni)] for d in data]
    Ys = [scale(d[ni], ni) for d in data]

    # 학습 80% / 테스트 20%
    order = list(range(len(data)))
    random.Random(42).shuffle(order)
    cut_ = int(len(order) * 0.8)
    tr, te = order[:cut_], order[cut_:]

    note = ""
    steps = math.ceil(len(tr) / batch)                                # epoch당 가중치 업데이트 횟수
    if not user_set and steps * epochs < MIN_STEPS:                                    # 데이터가 적으면 학습 부족 → epoch 늘림
        new_ep = min(math.ceil(MIN_STEPS / steps), max(epochs, MAX_WORK // len(tr)))
        if new_ep > epochs:
            note = f" (데이터가 적어서 {epochs} → {new_ep} epoch로 늘림)"
            epochs = new_ep
    limit = MAX_WORK * (3 if user_set else 1)                         # 직접 정한 값은 더 넉넉하게 허용
    if len(tr) * epochs > limit:                                      # 너무 오래 걸리지 않게 조절
        new_ep = max(10, limit // len(tr))
        note = f" (데이터가 커서 {epochs} → {new_ep} epoch로 줄임)"
        epochs = new_ep
    model = TinyANN(ni, hidden).fit([Xs[i] for i in tr], [Ys[i] for i in tr], epochs, batch, lr)
    beta = least_squares([[1] + Xs[i] for i in tr], [Ys[i] for i in tr])      # 비교용 선형회귀

    y_true = [data[i][ni] for i in te]
    pred_nn = [unscale_y(model.predict(Xs[i])) for i in te]
    pred_lr = [unscale_y(beta[0] + sum(b * x for b, x in zip(beta[1:], Xs[i]))) for i in te]
    stats = {}
    for name, p in [("인공신경망", pred_nn), ("선형회귀", pred_lr)]:
        mae = sum(abs(a - b) for a, b in zip(y_true, p)) / len(p)
        neg = sum(1 for q in p if q < 0) / len(p) * 100
        stats[name] = (r2(y_true, p), mae, neg)

    # 그래프 1: 학습 오차 감소
    lc = model.loss_curve
    g1 = Plot((1, len(lc)), (min(lc) * 0.8, max(lc) * 1.2), "Loss during training", "epoch", "MSE (scaled)", logy=True)
    g1.line(list(range(1, len(lc) + 1)), lc, "#1b8a9a", "train loss")
    lab_nn = f"Neural network (R²={stats['인공신경망'][0]:.3f})"
    lab_lr = f"Linear regression (R²={stats['선형회귀'][0]:.3f})"
    if ni == 1:      # 그래프 2 (입력 1개): 데이터 위에 예측선 — general_ann.py와 같은 방식
        xs_all = [d[0] for d in data]
        ys_all = [d[1] for d in data]
        samp = random.Random(1).sample(range(len(data)), min(1500, len(data)))
        x0, x1 = min(xs_all), max(xs_all)
        grid = [x0 + (x1 - x0) * i / 80 for i in range(81)]
        yn = [unscale_y(model.predict([scale(g, 0)])) for g in grid]
        yl = [unscale_y(beta[0] + beta[1] * scale(g, 0)) for g in grid]
        allv = ys_all + yn + yl
        g2 = Plot((x0, x1), (min(allv), max(allv)), f"{x_cols[0]} vs {y_col}", x_cols[0], y_col)
        g2.scatter([xs_all[i] for i in samp], [ys_all[i] for i in samp], "#888", "data", r=2, op=0.5)
        g2.line(grid, yn, "#d1342f", lab_nn)
        g2.line(grid, yl, "#1f5fd1", lab_lr)
    else:            # 그래프 2 (입력 여러 개): 실제 vs 예측
        samp = random.Random(1).sample(range(len(te)), min(1500, len(te)))
        top = max(y_true + pred_nn)
        lo = min(0, min(y_true + pred_lr))
        g2 = Plot((lo, top), (lo, top), "Actual vs Predicted (test)", f"actual {y_col}", f"predicted {y_col}")
        g2.scatter([y_true[i] for i in samp], [pred_lr[i] for i in samp], "#1f77b4", lab_lr, r=1.8, op=0.45)
        g2.scatter([y_true[i] for i in samp], [pred_nn[i] for i in samp], "#ff7f0e", lab_nn, r=1.8, op=0.45)
        g2.line([lo, top], [lo, top], "#000", "perfect", dash=True, w=1.5)
    img = save_svg("a2_result.html", [g1.svg(), g2.svg()])

    rows_md = "\n".join(f"| {n} | {a:.3f} | {num(b)} | {c:.1f}% |" for n, (a, b, c) in stats.items())
    title = f"울산항 선박 {len(data):,}척" if is_ship else f"`{os.path.basename(path)}` 데이터 {len(data):,}개"
    summary = (
        f"**[과제2 인공신경망 결과]** {title} (학습 {len(tr):,} / 테스트 {len(te):,})\n\n"
        f"- 구조: 입력층 {ni} ({', '.join(x_cols)}) → 은닉층 {hidden}개 (시그모이드) → 출력층 1 ({y_col})\n"
        f"- 학습: Adam, 학습률 {lr}, 배치 {batch}, {epochs} epoch{note}, 0~1 정규화"
        f"{' (직접 설정한 값)' if user_set else ' (기본 설정)'}\n\n"
        f"| 모델 | 결정계수 R² | 평균절대오차 | 음수 예측 비율 |\n|---|---|---|---|\n{rows_md}\n\n"
        f"→ R²가 더 높은 모델: **{'인공신경망' if stats['인공신경망'][0] > stats['선형회귀'][0] else '선형회귀'}**"
    )
    return {"summary": summary, "image": img, "model": model, "beta": beta, "x_cols": x_cols,
            "hp": {"hidden": hidden, "epochs": epochs, "batch": batch, "lr": lr},
            "y_col": y_col, "scale": scale, "unscale_y": unscale_y, "stats": stats,
            "file": os.path.basename(path)}


def predict_assignment2(res, values):
    x = [res["scale"](v, j) for j, v in enumerate(values)]
    y_nn = res["unscale_y"](res["model"].predict(x))
    beta = res["beta"]
    y_lr = res["unscale_y"](beta[0] + sum(b * xx for b, xx in zip(beta[1:], x)))
    cond = ", ".join(f"{c} {v:g}" for c, v in zip(res["x_cols"], values))
    return (f"**[과제2 예측]** {cond}\n\n"
            f"| 모델 | 예측 {res['y_col']} |\n|---|---|\n"
            f"| 인공신경망 | {num(y_nn)} |\n| 선형회귀 | {num(y_lr)} |")
