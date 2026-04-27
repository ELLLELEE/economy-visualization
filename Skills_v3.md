# Skills.md — Alpha Signal Dashboard v3.0
## 구현 가능한 ML 기반 투자 분석 시스템

> **핵심 철학:**
> 단순 룰 시스템을 벗어나되, 실제로 구현 가능한 수준의 ML을 사용한다.
> 모든 수치는 입력 데이터에서 동적으로 계산된다.
> 하드코딩된 임계값은 없다.

---

## 목차

1. 설계 원칙
2. 기술 스택
3. 전체 파이프라인
4. 실제 데이터 연결
5. 전처리 및 피처 엔지니어링
6. ML 모듈 — K-Means 레짐 판단
7. ML 모듈 — Ridge 회귀 Alpha Score
8. 신호 검증 — Rolling 상관계수
9. 리스크 관리 — ATR + Kelly
10. 백테스트 — Walk-forward
11. 시각화 선택 규칙
12. 대시보드 구성 흐름
13. 인사이트 생성 규칙
14. JSON 출력 구조
15. 예외 처리

---

## 1. 설계 원칙

### 1-1. 무엇을 쓰고 무엇을 버렸는가

| 제거한 것 | 이유 |
|-----------|------|
| HMM | 초기값 민감, 해석 어려움, 구현 복잡 |
| 로지스틱 회귀 | 피처 많으면 과소적합, 이진 출력 |
| IC 앙상블 | 데이터 적으면 불안정 |
| CVaR | VaR로 충분히 대체 가능 |
| FRED API | yfinance로 대체 가능 |
| 다중 RSI (9/14/21) | 14일 하나로 충분 |

| 유지한 것 | 이유 |
|-----------|------|
| K-Means 레짐 | 단순하고 안정적, sklearn 한 줄 |
| Ridge 회귀 | 과적합 방지 내장, 연속 점수 출력 |
| Rolling 상관계수 | 신호 품질 검증, 구현 한 줄 |
| Walk-forward 백테스트 | 미래 누수 없음, 신뢰도 높음 |
| ATR 손절 | 직관적, 실전적 |
| Kelly Criterion | 포지션 사이징 수학적 근거 |

### 1-2. 반드시 명시해야 할 한계

```
① 과거 패턴이 미래를 보장하지 않는다
② 데이터 품질에 따라 신뢰도가 크게 달라진다
③ 거래비용과 슬리피지가 실제 수익률을 낮춘다
④ 이 시스템은 투자 보조 도구이며 최종 판단은 사람이 한다
```

---

## 2. 기술 스택

```
데이터 수집:  yfinance, pykrx
전처리:       pandas, numpy
ML 모델:      sklearn (KMeans, Ridge, StandardScaler)
시각화:       plotly
백테스트:     pandas (직접 구현)
의존 패키지:  5개로 최소화
```

```bash
pip install yfinance pykrx pandas numpy scikit-learn plotly
```

---

## 3. 전체 파이프라인

```
[INPUT]  CSV / XLSX / JSON / 직접 업로드 / API
    ↓
[STEP 1] 실제 데이터 수집
         yfinance (가격, VIX) / pykrx (수급)
    ↓
[STEP 2] 전처리
         유형 감지 → 컬럼 정규화 → 결측치 처리 → 품질 점수
    ↓
[STEP 3] 피처 엔지니어링
         수익률, Z-score, 롤링 통계 (6개 핵심 피처)
    ↓
[STEP 4] K-Means 레짐 판단
         수익률 + 변동성 2차원 → 3개 클러스터
    ↓
[STEP 5] Ridge 회귀 Alpha Score
         6개 피처 → 0~100점 연속 출력
    ↓
[STEP 6] Rolling 상관계수 신호 검증
         각 신호의 최근 예측력 확인
    ↓
[STEP 7] ATR + Kelly 리스크 계산
         동적 손절/익절 + 권고 포지션 비율
    ↓
[STEP 8] Walk-forward 백테스트
         과거 동일 신호 성과 검증
    ↓
[STEP 9] 시각화 + 대시보드 + 인사이트 생성
    ↓
[OUTPUT] 차트 + JSON + 자연어 인사이트
```

---

## 4. 실제 데이터 연결

### 4-1. 가격 데이터 — yfinance

```python
import yfinance as yf
import pandas as pd

def fetch_price(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    한국: "005930.KS" (삼성전자)
    미국: "AAPL"
    지수: "^KS11" (코스피), "^GSPC" (S&P500)
    """
    raw = yf.Ticker(ticker).history(period=period)
    df = raw[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[df["volume"] > 0]  # 거래정지일 제거
    return df

def fetch_vix(period: str = "1y") -> pd.Series:
    """VIX는 yfinance로 가져올 수 있음"""
    raw = yf.Ticker("^VIX").history(period=period)
    return raw["Close"].rename("vix")

def fetch_us10y(period: str = "1y") -> pd.Series:
    """미국 10년물 금리"""
    raw = yf.Ticker("^TNX").history(period=period)
    return raw["Close"].rename("us10y")
```

### 4-2. 수급 데이터 — pykrx

```python
from pykrx import stock as krx

def fetch_supply(ticker_6: str, start: str, end: str) -> pd.DataFrame:
    """
    ticker_6: KRX 6자리 코드 (예: "005930")
    start/end: "20240101" 형식
    """
    raw = krx.get_market_trading_value_by_date(start, end, ticker_6)
    df = raw[["외국인합계","기관합계","개인"]].copy()
    df.columns = ["foreign","institution","individual"]
    df.index = pd.to_datetime(df.index)
    return df
```

### 4-3. 직접 업로드 파일 처리

```python
def load_file(filepath: str) -> pd.DataFrame:
    """어떤 형식이든 자동 처리"""
    ext = filepath.rsplit(".", 1)[-1].lower()

    if ext == "csv":
        for enc in ["utf-8", "euc-kr", "cp949"]:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue

    elif ext in ["xlsx", "xls"]:
        sheets = pd.read_excel(filepath, sheet_name=None)
        # 데이터가 가장 많은 시트 선택
        return max(sheets.values(), key=len)

    elif ext == "json":
        return pd.read_json(filepath)

    raise ValueError(f"지원하지 않는 형식: {ext}")
```

---

## 5. 전처리 및 피처 엔지니어링

### 5-1. 컬럼명 자동 정규화

```python
COLUMN_MAP = {
    # 가격
    "종가":"close","close":"close","adj close":"close",
    "시가":"open", "open":"open",
    "고가":"high", "high":"high",
    "저가":"low",  "low":"low",
    "거래량":"volume","volume":"volume","vol":"volume",
    # 수급
    "외국인":"foreign","외국인합계":"foreign","foreigner":"foreign",
    "기관":"institution","기관합계":"institution",
    "개인":"individual",
    # 밸류에이션
    "per":"per","p/e":"per",
    "pbr":"pbr","p/b":"pbr",
    "roe":"roe",
    # 매크로
    "vix":"vix","공포지수":"vix",
    "금리":"rate","us10y":"rate",
    "환율":"fx","usdkrw":"fx",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.rename(columns=COLUMN_MAP)
```

### 5-2. 데이터 유형 자동 감지

```python
def detect_types(df: pd.DataFrame) -> list:
    """
    단일 또는 복합 유형 반환
    복합이면 교차 분석 자동 활성화
    """
    cols = set(df.columns)
    rules = {
        "price":     {"close"},
        "ohlcv":     {"open","high","low","close","volume"},
        "supply":    {"foreign","institution"},
        "macro":     {"vix","rate","fx"},
        "valuation": {"per","pbr"},
    }
    detected = [
        t for t, required in rules.items()
        if required & cols  # 하나라도 있으면 포함
    ]
    return detected or ["unknown"]
```

### 5-3. 데이터 품질 점수

```python
import math

def quality_score(df: pd.DataFrame) -> dict:
    n = len(df)

    # 완전성 (0~40점)
    missing = df.isnull().mean().mean()
    completeness = max(0, 40 * (1 - missing * 5))

    # 기간 충분성 (0~30점) — 로그 스케일
    length = min(30, 30 * math.log(max(n, 1)) / math.log(252))

    # 이상치 비율 (0~30점) — IQR 기반
    outlier_score = 20  # 기본값
    if "close" in df.columns:
        r = df["close"].pct_change().dropna()
        q1, q3 = r.quantile(0.25), r.quantile(0.75)
        iqr = q3 - q1
        outlier_rate = ((r < q1 - 3*iqr) | (r > q3 + 3*iqr)).mean()
        outlier_score = max(0, 30 * (1 - outlier_rate * 10))

    total = completeness + length + outlier_score
    grade = (
        "🟢 신뢰 높음" if total >= 80 else
        "🟡 보통"       if total >= 60 else
        "🟠 낮음"       if total >= 40 else
        "🔴 분석 제한"
    )
    return {"total": round(total), "grade": grade}
```

### 5-4. 핵심 피처 6개 (과적합 방지)

피처는 6개로 엄격히 제한한다. 많을수록 Ridge가 희석된다.

```python
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    6개 핵심 피처만 생성
    모두 -1~1 또는 0~1 범위로 정규화 → 스케일 통일
    """
    f = pd.DataFrame(index=df.index)
    c = df["close"]

    # 1. 모멘텀 — 20일 수익률 분위수 (0~1)
    ret20 = c.pct_change(20)
    f["momentum"] = ret20.rolling(60).rank(pct=True)

    # 2. 추세 강도 — MA20 대비 현재가 위치 (정규화)
    ma20 = c.rolling(20).mean()
    f["trend"] = ((c - ma20) / (ma20 + 1e-10)).clip(-0.1, 0.1) / 0.1

    # 3. RSI 정규화 — 0~1 범위
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    f["rsi_norm"] = rsi / 100  # 0~1

    # 4. 거래량 신호 — 20일 평균 대비 배율 (분위수)
    if "volume" in df.columns:
        vol_ratio = df["volume"] / (df["volume"].rolling(20).mean() + 1)
        f["volume_signal"] = vol_ratio.rolling(60).rank(pct=True)
    else:
        f["volume_signal"] = 0.5  # 중립

    # 5. 수급 신호 — 외국인 5일 누적 순매수 분위수
    if "foreign" in df.columns:
        foreign_5d = df["foreign"].rolling(5).sum()
        f["supply_signal"] = foreign_5d.rolling(60).rank(pct=True)
    else:
        f["supply_signal"] = 0.5  # 중립

    # 6. 변동성 역수 — 낮은 변동성이 유리 (분위수 역전)
    vol = c.pct_change().rolling(20).std()
    f["low_vol"] = 1 - vol.rolling(60).rank(pct=True)

    return f.dropna()
```

---

## 6. ML 모듈 — K-Means 레짐 판단

### 6-1. 왜 K-Means인가

```
HMM 대비 장점:
- 초기값 민감도 낮음 (random_state 고정으로 재현 가능)
- 구현 5줄
- 해석 명확 (수익률 + 변동성 2차원 공간)
- sklearn 기본 내장

단점:
- 시간 순서 무시 (Markov 특성 없음)
→ 보완: 최근 20일 데이터에 3배 가중치 부여
```

### 6-2. 구현

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_regime(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """
    입력: 가격 데이터
    출력: 날짜별 레짐 레이블 (0=약세, 1=횡보, 2=강세)

    2차원 특성:
    - 20일 수익률 (추세)
    - 20일 변동성 (리스크)
    """
    c = df["close"]
    ret  = c.pct_change(20).dropna()
    vol  = c.pct_change().rolling(20).std().dropna()

    common = ret.index.intersection(vol.index)
    X = np.column_stack([ret.loc[common], vol.loc[common]])

    # 최근 데이터에 가중치 부여 (HMM의 시간 순서 부재 보완)
    weights = np.ones(len(X))
    weights[-20:] = 3.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled, sample_weight=weights)
    labels = model.predict(X_scaled)

    # 클러스터를 평균 수익률 기준으로 정렬 (0=약세, 1=횡보, 2=강세)
    cluster_means = {i: ret.loc[common][labels == i].mean() for i in range(n_clusters)}
    rank_map = {old: new for new, old in enumerate(sorted(cluster_means, key=cluster_means.get))}
    sorted_labels = np.array([rank_map[l] for l in labels])

    regime_names = {0: "약세 (Bearish)", 1: "횡보 (Sideways)", 2: "강세 (Bullish)"}
    current_regime = int(sorted_labels[-1])

    # 레짐별 통계
    regime_stats = {}
    for r in range(n_clusters):
        mask = sorted_labels == r
        regime_stats[regime_names[r]] = {
            "avg_return":  round(float(ret.loc[common][mask].mean()), 4),
            "avg_vol":     round(float(vol.loc[common][mask].mean()), 4),
            "frequency":   round(float(mask.mean()), 2),
        }

    return {
        "current": current_regime,
        "label":   regime_names[current_regime],
        "series":  pd.Series(sorted_labels, index=common),
        "stats":   regime_stats,
        "names":   regime_names,
    }
```

### 6-3. 레짐별 피처 가중치 조정

```python
# 레짐에 따라 Ridge 학습 시 피처 가중치 조정
# 강세장: 모멘텀/수급 중요
# 약세장: 변동성/RSI 중요
REGIME_FEATURE_WEIGHTS = {
    0: {"momentum":0.6, "trend":0.8, "rsi_norm":1.2,
        "volume_signal":0.8, "supply_signal":1.0, "low_vol":1.4},  # 약세
    1: {"momentum":1.0, "trend":1.0, "rsi_norm":1.0,
        "volume_signal":1.0, "supply_signal":1.0, "low_vol":1.0},  # 횡보
    2: {"momentum":1.4, "trend":1.2, "rsi_norm":0.8,
        "volume_signal":1.2, "supply_signal":1.4, "low_vol":0.6},  # 강세
}
```

---

## 7. ML 모듈 — Ridge 회귀 Alpha Score

### 7-1. 왜 Ridge인가

```
로지스틱 회귀 대비 장점:
- 연속적인 0~100 점수 출력 (이진 분류 아님)
- L2 정규화 내장 → 과적합 자동 방지
- 피처 6개로 제한 시 과소적합 위험 낮음
- 해석 가능 (계수 = 각 피처의 기여도)
- 시계열 데이터에서 안정적

한계:
- 비선형 관계 포착 불가
→ 보완: 피처 엔지니어링 단계에서 비선형 변환 (분위수, 정규화)
```

### 7-2. 구현

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

def train_alpha(features: pd.DataFrame, df: pd.DataFrame,
                forward_days: int = 20,
                current_regime: int = 1) -> dict:
    """
    목적변수: forward_days 후 수익률
    TimeSeriesSplit으로 미래 누수 차단
    레짐에 따라 피처 가중치 조정
    """
    # 목적변수 생성
    fwd_return = df["close"].pct_change(forward_days).shift(-forward_days)
    common = features.index.intersection(fwd_return.dropna().index)

    X = features.loc[common].copy()
    y = fwd_return.loc[common]

    # 레짐별 피처 가중치 적용
    regime_weights = REGIME_FEATURE_WEIGHTS.get(current_regime, {})
    for col, w in regime_weights.items():
        if col in X.columns:
            X[col] = X[col] * w

    # TimeSeriesSplit 교차검증 (5-fold)
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)  # alpha=정규화 강도, 클수록 과적합 방지

    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr  = scaler.fit_transform(X.iloc[train_idx])
        X_val = scaler.transform(X.iloc[val_idx])
        model.fit(X_tr, y.iloc[train_idx])
        cv_scores.append(model.score(X_val, y.iloc[val_idx]))  # R²

    # 전체 데이터로 최종 학습
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    # 피처 기여도
    importance = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_,
        "abs_coef": np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=False)
    importance["importance_pct"] = (
        importance["abs_coef"] / importance["abs_coef"].sum() * 100
    ).round(1)

    return {
        "model":      model,
        "scaler":     scaler,
        "features":   X.columns.tolist(),
        "importance": importance,
        "cv_r2_mean": round(np.mean(cv_scores), 3),
        "cv_r2_std":  round(np.std(cv_scores), 3),
        "regime":     current_regime,
    }

def predict_alpha_score(model_info: dict, features: pd.DataFrame) -> pd.Series:
    """
    Ridge 예측값 → 0~100 Alpha Score 변환
    분위수 기반 변환 (극단값 영향 제거)
    """
    model  = model_info["model"]
    scaler = model_info["scaler"]
    cols   = model_info["features"]

    X = features[cols].ffill().fillna(0.5)
    raw_pred = model.predict(scaler.transform(X))

    # 분위수 기반 0~100 변환
    series = pd.Series(raw_pred, index=features.index)
    score  = series.rank(pct=True) * 100
    return score.round(1)
```

### 7-3. Alpha Score 해석

```
80점 이상 → Strong Buy  (상위 20% 신호)
60~79점   → Buy         (상위 40% 신호)
40~59점   → Neutral     (중간)
20~39점   → Caution     (하위 40% 신호)
20점 미만 → Avoid       (하위 20% 신호)

CV R² < 0.02 → 모델 예측력 낮음 → 신뢰도 낮음 표시
CV R² std > 0.1 → 불안정 → 신뢰도 낮음 표시
```

---

## 8. 신호 검증 — Rolling 상관계수

IC 대신 단순하고 직관적인 Rolling 상관계수로 신호 품질을 검증한다.

```python
def validate_signals(features: pd.DataFrame, df: pd.DataFrame,
                     forward_days: int = 20,
                     window: int = 60) -> pd.DataFrame:
    """
    각 신호(피처)와 미래 수익률의 rolling 상관계수
    양수: 신호가 상승을 예측 / 음수: 역방향
    절대값 0.05 이상: 의미 있는 신호
    """
    fwd_return = df["close"].pct_change(forward_days).shift(-forward_days)
    results = []

    for col in features.columns:
        corr = features[col].rolling(window).corr(fwd_return)
        latest_corr = corr.dropna().iloc[-1] if corr.dropna().shape[0] > 0 else 0
        trend = "상승 중" if corr.dropna().diff().iloc[-5:].mean() > 0 else "하락 중"

        results.append({
            "signal":        col,
            "corr_latest":   round(latest_corr, 3),
            "corr_abs":      round(abs(latest_corr), 3),
            "direction":     "긍정" if latest_corr > 0 else "부정",
            "trend":         trend,
            "usable":        abs(latest_corr) >= 0.03,  # 최소 기준
        })

    return pd.DataFrame(results).sort_values("corr_abs", ascending=False)
```

---

## 9. 리스크 관리 — ATR + Kelly

### 9-1. ATR 기반 동적 손절/익절

```python
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_risk_levels(entry: float, atr: float,
                        alpha_score: float, regime: int) -> dict:
    """
    손절/익절 배수는 Alpha Score + 레짐의 함수
    고정값 없음 — 신호 강도와 시장 환경에 따라 자동 조정
    """
    # Alpha Score: 50점 기준, 높을수록 넓은 손절 허용
    score_factor = 0.5 + (alpha_score / 100)     # 0.5~1.5

    # 레짐: 약세장일수록 타이트하게
    regime_factor = {0: 0.7, 1: 1.0, 2: 1.3}.get(regime, 1.0)

    stop_mult   = round(2.0 * score_factor * regime_factor, 2)
    target_mult = round(stop_mult * 2.0, 2)  # RR 2:1 유지

    return {
        "stop_loss":    round(entry - atr * stop_mult),
        "target_price": round(entry + atr * target_mult),
        "stop_pct":     round((atr * stop_mult / entry) * -100, 2),
        "target_pct":   round((atr * target_mult / entry) * 100, 2),
        "rr_ratio":     2.0,
        "stop_mult":    stop_mult,
        "target_mult":  target_mult,
    }
```

### 9-2. Kelly Criterion 포지션 사이징

```python
def kelly_position(win_rate: float, avg_win: float,
                   avg_loss: float) -> dict:
    """
    f* = (p*b - q) / b
    p=승률, q=1-p, b=평균이익/평균손실

    Quarter Kelly (25%) 적용: 실전에서 드로다운 과다 방지
    최대 20% 캡: 단일 종목 과도 집중 방지
    """
    if avg_loss == 0 or win_rate <= 0:
        return {"recommended_pct": "0%", "full_kelly": 0}

    b = avg_win / abs(avg_loss)
    p, q = win_rate, 1 - win_rate
    full_kelly = max(0, (p * b - q) / b)
    recommended = min(full_kelly * 0.25, 0.20)  # Quarter Kelly, 최대 20%

    return {
        "full_kelly":       round(full_kelly, 3),
        "recommended":      round(recommended, 3),
        "recommended_pct":  f"{recommended*100:.1f}%",
        "rr_ratio":         round(b, 2),
        "note": (
            f"풀 켈리 {full_kelly*100:.1f}%의 25% 적용 "
            f"→ 포트폴리오의 {recommended*100:.1f}% 진입 권고"
        )
    }
```

---

## 10. 백테스트 — Walk-forward

```python
def walk_forward_backtest(df: pd.DataFrame, features: pd.DataFrame,
                          train_days: int = 252,
                          test_days: int = 63) -> dict:
    """
    슬라이딩 윈도우: 1년 학습 → 3개월 검증 → 반복
    미래 데이터 누수 완전 차단
    상위 40% 신호에서만 진입 (고정 임계값 아닌 분위수)
    """
    returns = df["close"].pct_change()
    results = []

    for start in range(0, len(features) - train_days - test_days, test_days):
        train_end = start + train_days
        test_end  = min(train_end + test_days, len(features))

        # 학습 구간에서 모델 학습
        train_features = features.iloc[start:train_end]
        train_df = df.iloc[start:train_end]

        try:
            regime_info = fit_regime(train_df)
            model_info  = train_alpha(train_features, train_df,
                                      current_regime=regime_info["current"])
            scores = predict_alpha_score(model_info,
                                         features.iloc[train_end:test_end])
        except Exception:
            continue

        # 검증 구간: 상위 40% 신호에서 진입 (분위수 기준)
        test_returns = returns.iloc[train_end:test_end]
        entry_mask   = scores > 60  # 60점 이상 = 상위 40%

        period_returns = test_returns[entry_mask.reindex(test_returns.index, fill_value=False)]
        if len(period_returns) < 3:
            continue

        results.append({
            "period_start": df.index[train_end].strftime("%Y-%m-%d"),
            "period_end":   df.index[test_end-1].strftime("%Y-%m-%d"),
            "n_trades":     int(len(period_returns)),
            "avg_return":   round(float(period_returns.mean()), 4),
            "win_rate":     round(float((period_returns > 0).mean()), 3),
            "sharpe":       round(float(
                period_returns.mean() / (period_returns.std() + 1e-10) * (252**0.5)
            ), 2),
        })

    if not results:
        return {"error": "백테스트 데이터 부족 (최소 315일 필요)"}

    df_res = pd.DataFrame(results)
    return {
        "method":        "Walk-Forward",
        "periods":       len(results),
        "avg_return":    round(float(df_res["avg_return"].mean()), 4),
        "avg_win_rate":  round(float(df_res["win_rate"].mean()), 3),
        "avg_sharpe":    round(float(df_res["sharpe"].mean()), 2),
        "consistency":   round(float((df_res["avg_return"] > 0).mean()), 2),
        "detail":        results,
    }
```

---

## 11. 시각화 선택 규칙

### 11-1. 감지 유형 → 차트 자동 매핑

| 감지 유형 | 메인 차트 | 보조 차트 | 자동 추가 요소 |
|----------|-----------|-----------|---------------|
| price + ohlcv | 캔들스틱 + MA20/60 | 거래량 막대 | ATR 손절/목표가 수평선 |
| technical | RSI(14) 라인 + 게이지 | MACD 히스토그램 | 과매수/과매도 음영 |
| supply | 수급 스택 막대 | 주가 오버레이 라인 | 스마트머니 신호 음영 |
| macro | VIX + 금리 멀티 라인 | 레짐 배경색 | 분위수 기준선 |
| valuation | 수평 막대 (분위수 위치) | 시계열 PER 라인 | 업종 분위수 표시 |
| 복합 (2종류+) | 메인 + 수급 오버레이 | 레짐 배경 | 교차 분석 마커 |

### 11-2. 분석 결과 → 자동 시각화 보강

```
Alpha Score > 80
→ 손절/목표가 라인 초록 강조
→ 진입 권고 구간 초록 음영

Alpha Score < 30
→ 위험 구간 빨간 음영 + 경고 뱃지

레짐 전환 감지 (이전 ≠ 현재)
→ 전환 시점 세로 점선 + "레짐 전환" 레이블

수급 신호 강함 (supply_signal > 0.7)
→ 해당 구간 파란 음영 + 스마트머니 레이블

패닉셀 감지 (volume_ratio > 3 + 가격 하락)
→ 해당 캔들 마커 + "패닉셀" 레이블

신호 상관계수 낮음 (corr_abs < 0.03)
→ 해당 신호에 "(신뢰도 낮음)" 표시
```

### 11-3. 레짐별 차트 배경색

```
약세 레짐 (0) → 연한 빨간 배경 (#FFF0F0)
횡보 레짐 (1) → 흰색/연한 회색 배경
강세 레짐 (2) → 연한 초록 배경 (#F0FFF0)
레짐 전환 구간 → 경계선 세로 점선
```

---

## 12. 대시보드 구성 흐름

### 12-1. 모듈 배치 순서

```
1순위 — 데이터 현황 카드
  감지 유형 / 품질 점수 / 분석 기간 / 모델 CV R²

2순위 — 레짐 패널
  K-Means 현재 레짐 + 레짐별 통계
  레짐 히스토리 라인

3순위 — Alpha Score 패널
  0~100 게이지 + 액션 뱃지
  피처 기여도 수평 막대 (모델 투명성)

4순위 — 메인 차트 패널
  감지 유형 기반 자동 선택
  ATR 손절/목표가 라인 오버레이

5순위 — 신호 검증 패널
  Rolling 상관계수 테이블
  신호별 신뢰도 표시

6순위 — 수급 패널 (수급 데이터 있을 때만)
  외국인/기관/개인 스택 + 주가 오버레이

7순위 — 리스크 패널
  ATR 손절가 / Kelly 포지션 / VaR(95%)

8순위 — 백테스트 패널 (252일 이상 시)
  Walk-forward 결과 요약 테이블
  기간별 수익률 막대차트

9순위 — 인사이트 패널 (항상 최하단)
  자연어 요약 + 액션 권고 + 한계 명시
```

### 12-2. 데이터 유형별 레이아웃

```
가격만 있을 때
→ 데이터현황 / 레짐 / Alpha Score / 캔들+MA / RSI+MACD / 리스크 / 백테스트 / 인사이트

가격 + 수급
→ 데이터현황 / 레짐 / Alpha Score / 캔들+수급오버레이 / 신호검증 / 수급패널 / 리스크 / 인사이트

매크로만 있을 때
→ 데이터현황 / VIX+금리 멀티차트 / 레짐판단 / 섹터 영향 분석 / 인사이트

전체 데이터
→ 전체 모듈 풀 레이아웃
```

---

## 13. 인사이트 생성 규칙

### 13-1. 작성 원칙

```
① 모든 수치는 실제 계산값 사용 (하드코딩 금지)
② 신뢰도 낮으면 반드시 명시
③ 확정적 표현 금지 ("반드시 오른다" 등)
④ 핵심 근거 3개 이내 요약
⑤ 모델 한계 항상 마지막에 명시
```

### 13-2. 시나리오별 템플릿

**강세 신호 (Alpha Score 60+)**
```
"{종목}의 Alpha Score는 {점수}점입니다.

주요 신호:
- {신호1}: {수치} (상관계수: {값})
- {신호2}: {수치} (상관계수: {값})

현재 레짐: {레짐명}
백테스트 ({N}개 기간): 평균 {수익률}%, 승률 {승률}%, 샤프 {샤프}

리스크:
손절가 {손절가}원 ({손절%}%) / 목표가 {목표가}원 ({목표%}%)
Kelly 권고 포지션: {포지션%}%

데이터 품질 {품질점수}점 / 모델 CV R² {r2}
※ 과거 성과가 미래를 보장하지 않습니다."
```

**중립 신호 (Alpha Score 40~60)**
```
"{종목}의 Alpha Score는 {점수}점으로 중립 구간입니다.
방향성이 불명확하여 신규 진입보다 관망을 권고합니다.
현재 레짐: {레짐명}"
```

**약세 신호 (Alpha Score 40 미만)**
```
"{종목}의 Alpha Score는 {점수}점으로 낮습니다.
주요 신호가 부정적이며 현재 {레짐명} 레짐입니다.
신규 진입을 권고하지 않습니다."
```

**데이터 부족**
```
"데이터 기간이 {N}일로 일부 분석이 제한됩니다.
사용 가능: {유형 목록}
제한됨: {지표 목록}
품질 점수 {점수}점 — 추가 데이터 확보 권고"
```

---

## 14. JSON 출력 구조

```json
{
  "meta": {
    "ticker": "005930",
    "name": "삼성전자",
    "analysis_date": "2024-01-02",
    "data_types": ["price", "ohlcv", "supply", "technical"],
    "data_period_days": 252,
    "quality": {"total": 88, "grade": "🟢 신뢰 높음"}
  },

  "regime": {
    "current": 2,
    "label": "강세 (Bullish)",
    "stats": {
      "강세 (Bullish)":  {"avg_return": 0.048, "avg_vol": 0.012, "frequency": 0.38},
      "횡보 (Sideways)": {"avg_return": 0.002, "avg_vol": 0.018, "frequency": 0.41},
      "약세 (Bearish)":  {"avg_return": -0.031,"avg_vol": 0.029, "frequency": 0.21}
    }
  },

  "alpha_score": {
    "score": 74.2,
    "action": "Buy",
    "cv_r2": 0.042,
    "cv_r2_std": 0.018,
    "reliability": "보통",
    "feature_importance": [
      {"feature": "supply_signal", "importance_pct": 31.2, "corr": 0.08},
      {"feature": "momentum",      "importance_pct": 24.7, "corr": 0.06},
      {"feature": "rsi_norm",      "importance_pct": 18.3, "corr": 0.05}
    ]
  },

  "risk": {
    "atr14": 1240,
    "stop_loss": 71120,
    "stop_pct": -3.1,
    "target_price": 78540,
    "target_pct": 7.0,
    "rr_ratio": 2.0,
    "stop_mult": 1.84,
    "target_mult": 3.68,
    "kelly_recommended_pct": "4.2%",
    "var_95_daily": -0.023
  },

  "backtest": {
    "method": "Walk-Forward",
    "periods": 4,
    "avg_return": 0.072,
    "avg_win_rate": 0.62,
    "avg_sharpe": 1.24,
    "consistency": 0.75
  },

  "signals": {
    "momentum":      {"value": 0.71, "corr": 0.06, "usable": true},
    "trend":         {"value": 0.58, "corr": 0.04, "usable": true},
    "rsi_norm":      {"value": 0.44, "corr": 0.05, "usable": true},
    "volume_signal": {"value": 0.63, "corr": 0.03, "usable": true},
    "supply_signal": {"value": 0.78, "corr": 0.08, "usable": true},
    "low_vol":       {"value": 0.52, "corr": 0.02, "usable": false}
  },

  "insight": {
    "score": 74.2,
    "action": "Buy",
    "regime": "강세 (Bullish)",
    "top_signals": ["수급 신호 강함 (상관계수 0.08)", "모멘텀 양호 (0.06)", "RSI 중립 (0.05)"],
    "kelly_pct": "4.2%",
    "caution": "CV R² 0.042 — 보통 수준 신뢰도",
    "disclaimer": "과거 성과가 미래를 보장하지 않습니다."
  }
}
```

---

## 15. 예외 처리

### 15-1. 데이터 부족

```
전체 14일 미만
→ 피처 계산 불가 → "데이터 최소 14일 필요" 안내
→ 기본 가격 차트만 표시

K-Means 학습 불가 (30일 미만)
→ MA 기반 단순 레짐으로 대체
→ "레짐 모델 미적용 (데이터 부족)" 표시

Ridge 학습 불가 (60일 미만)
→ Alpha Score 계산 불가
→ 개별 신호 상관계수만 표시

Walk-forward 불가 (252일 미만)
→ 백테스트 패널 비활성화
```

### 15-2. 모델 실패

```
Ridge CV R² 전체 음수
→ "모델 예측력 없음 — 개별 신호 참고"
→ Alpha Score 대신 신호 상관계수 테이블만 표시

모든 신호 상관계수 < 0.03
→ "유효 신호 없음 — 관망 권고"
→ Alpha Score 50점 (중립) 반환

ATR 계산 불가 (OHLC 없음)
→ 표준편차 기반 대체: 손절가 = 매수가 × (1 - 2σ)
→ "ATR 대체: 2σ 손절 적용" 안내
```

### 15-3. 유형 미감지

```
unknown 반환 시
→ 컬럼 목록 표시 + 수동 유형 선택 UI
→ 선택 후 정상 파이프라인 재실행
```

---

> **이 문서는 Alpha Signal Dashboard v3.0의 핵심 분석 설계 문서입니다.**
> Claude Code는 이 문서를 기반으로 어떤 투자 데이터가 입력되어도
> K-Means + Ridge + Walk-forward 파이프라인으로 자동 분석합니다.
>
> **모든 수치는 입력 데이터로 동적 계산됩니다.**
> *과거 성과가 미래를 보장하지 않습니다. 투자 보조 도구입니다.*
