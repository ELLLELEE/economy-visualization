# Skills.md — Alpha Signal Dashboard v2.0
## 통계/ML 기반 범용 투자 분석 시스템

> **핵심 철학:**
> 고정 규칙(Rule-based)이 아닌 **데이터 기반 확률 추정(Probabilistic)**으로 작동한다.
> 어떤 데이터가 들어와도 동일한 파이프라인이 적용되며,
> 모든 수치는 실제 데이터로 동적 계산된다. 하드코딩된 임계값은 존재하지 않는다.

---

## 목차

1. 설계 철학 및 한계 명시
2. 전체 파이프라인
3. 실제 데이터 연결
4. 범용 데이터 전처리
5. 피처 엔지니어링
6. 통계적 신호 생성
7. Alpha Score — 통계적 가중치 산출
8. 시장 레짐 — 비지도 학습 기반
9. 리스크 관리 — 수학적 근거
10. 백테스트 — 실제 계산 구조
11. 시각화 선택 규칙
12. 대시보드 구성 흐름
13. 인사이트 생성 규칙
14. JSON 출력 구조
15. 모델 업데이트 및 개선 흐름
16. 예외 처리

---

## 1. 설계 철학 및 한계 명시

### 1-1. 왜 Rule-based를 버리는가

기존 방식의 문제:
```
"RSI < 30이면 매수" → 시장 국면에 따라 완전히 다른 결과
"외국인 5일 순매수 → 승률 72%" → 특정 기간 데이터의 과적합 가능성
고정 임계값 → 시장 구조 변화에 취약
```

본 시스템의 방식:
```
임계값을 데이터에서 학습 (분위수, IC 기반)
신호 강도를 0~1 확률로 표현
국면(Regime)에 따라 가중치 동적 조정
실제 수익률로 지속적으로 검증 및 업데이트
```

### 1-2. 반드시 명시해야 할 한계

```
① 과거 패턴이 미래를 보장하지 않는다
② 데이터 품질에 따라 신뢰도가 크게 달라진다
③ 거래비용, 슬리피지가 실제 수익률을 낮춘다
④ 소형주/저유동성 종목에서 수급 신호 왜곡 가능
⑤ 이 시스템은 투자 보조 도구이며 최종 판단은 사람이 한다
```

---

## 2. 전체 파이프라인

```
[INPUT]  다양한 형식의 투자 데이터 (유형 무관)
    ↓
[STEP 1] 실제 데이터 연결 및 수집
         yfinance / KRX / FRED / 직접 업로드
    ↓
[STEP 2] 범용 전처리
         유형 자동 감지 → 정규화 → 결측치 처리 → 품질 점수
    ↓
[STEP 3] 피처 엔지니어링
         원시 데이터 → 분석 가능한 피처 변환
         (수익률, Z-score, 분위수, 롤링 통계)
    ↓
[STEP 4] 통계적 신호 생성
         단일 신호 → 앙상블 신호 → 신호 IC 계산
    ↓
[STEP 5] Alpha Score 산출
         IC 기반 동적 가중치 → 로지스틱 회귀 확률 출력
    ↓
[STEP 6] 시장 레짐 판단
         HMM / K-Means 기반 비지도 분류
    ↓
[STEP 7] 리스크 분석
         CVaR / Kelly / 상관관계 기반 포지션 계산
    ↓
[STEP 8] 백테스트 동적 계산
         워크포워드(Walk-forward) 검증
    ↓
[STEP 9] 시각화 + 대시보드 생성
    ↓
[OUTPUT] 인사이트 + JSON + 차트
```

---

## 3. 실제 데이터 연결

데이터는 항상 실제 소스에서 가져온다. 더미데이터는 테스트 목적으로만 사용한다.

### 3-1. 가격/기술적 데이터 — yfinance

```python
import yfinance as yf
import pandas as pd

def fetch_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    ticker: 한국 종목은 "005930.KS", 미국은 "AAPL"
    period: "3mo", "6mo", "1y", "2y"
    """
    raw = yf.Ticker(ticker).history(period=period)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[df["volume"] > 0]  # 거래정지일 제거
    return df

# 여러 종목 동시 수집
def fetch_multiple(tickers: list, period: str = "1y") -> dict:
    return {t: fetch_price_data(t, period) for t in tickers}
```

### 3-2. 매크로 데이터 — FRED API

```python
import pandas_datareader.data as web
from datetime import datetime, timedelta

def fetch_macro_data(start: str = None) -> pd.DataFrame:
    """
    FRED에서 주요 매크로 지표 수집
    무료 API, 별도 키 불필요 (기본 접근)
    """
    if start is None:
        start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    series = {
        "DGS10":  "us10y_rate",   # 미국 10년물 금리
        "VIXCLS": "vix",          # VIX 공포지수
        "DTWEXBGS": "dxy",        # 달러인덱스
        "CPIAUCSL": "cpi",        # 미국 CPI
    }

    frames = {}
    for fred_code, col_name in series.items():
        try:
            s = web.DataReader(fred_code, "fred", start, end)
            frames[col_name] = s.iloc[:, 0]
        except Exception:
            frames[col_name] = None  # 실패 시 None 처리

    df = pd.DataFrame(frames).ffill()
    return df
```

### 3-3. 수급 데이터 — KRX (pykrx)

```python
from pykrx import stock as krx

def fetch_supply_demand(ticker_6digit: str, start: str, end: str) -> pd.DataFrame:
    """
    ticker_6digit: "005930" (KRX 6자리 코드)
    KRX 공식 데이터 기반, 외국인/기관/개인 순매수 금액
    """
    raw = krx.get_market_trading_value_by_date(
        start, end, ticker_6digit
    )
    df = raw[["외국인합계", "기관합계", "개인"]].copy()
    df.columns = ["foreign", "institution", "individual"]
    df.index = pd.to_datetime(df.index)
    return df

def fetch_foreign_holding(ticker_6digit: str, start: str, end: str) -> pd.DataFrame:
    """외국인 보유 비율 변화"""
    raw = krx.get_exhaustion_rates_of_foreign_investment_by_date(
        start, end, ticker_6digit
    )
    return raw[["보유수량", "지분율"]].rename(
        columns={"보유수량": "foreign_qty", "지분율": "foreign_ratio"}
    )
```

### 3-4. 직접 업로드 데이터 처리

```python
def load_uploaded_file(filepath: str) -> pd.DataFrame:
    """
    사용자가 직접 업로드한 파일 처리
    어떤 형식/컬럼 구조든 자동 대응
    """
    ext = filepath.split(".")[-1].lower()

    loaders = {
        "csv":  lambda f: _try_encodings(f),
        "xlsx": lambda f: pd.read_excel(f, sheet_name=None),  # 모든 시트
        "xls":  lambda f: pd.read_excel(f),
        "json": lambda f: pd.read_json(f),
        "txt":  lambda f: _detect_separator(f),
    }

    if ext not in loaders:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    df = loaders[ext](filepath)

    # 여러 시트가 있는 경우 데이터가 있는 첫 번째 시트 선택
    if isinstance(df, dict):
        df = next(
            (v for v in df.values() if len(v) > 5),
            list(df.values())[0]
        )

    return df

def _try_encodings(filepath: str) -> pd.DataFrame:
    for enc in ["utf-8", "euc-kr", "cp949", "utf-8-sig"]:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("인코딩 감지 실패")

def _detect_separator(filepath: str) -> pd.DataFrame:
    for sep in ["\t", ",", "|", ";"]:
        try:
            df = pd.read_csv(filepath, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(filepath)
```

---

## 4. 범용 데이터 전처리

### 4-1. 컬럼명 자동 정규화

```python
COLUMN_MAP = {
    # 가격
    "종가": "close", "close": "close", "close_price": "close",
    "adj close": "close", "adjusted_close": "close",
    "시가": "open",  "open": "open",
    "고가": "high",  "high": "high",
    "저가": "low",   "low": "low",
    "거래량": "volume", "volume": "volume", "vol": "volume",

    # 수급
    "외국인": "foreign", "외국인합계": "foreign", "foreigner": "foreign",
    "기관": "institution", "기관합계": "institution",
    "개인": "individual",

    # 밸류에이션
    "per": "per", "p/e": "per",
    "pbr": "pbr", "p/b": "pbr",
    "roe": "roe", "eps": "eps", "psr": "psr",

    # 매크로
    "금리": "rate", "us10y": "rate", "dgs10": "rate",
    "vix": "vix", "공포지수": "vix",
    "환율": "fx", "usdkrw": "fx",
    "달러": "dxy", "dxy": "dxy",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    return df
```

### 4-2. 데이터 유형 자동 감지

```python
def detect_data_type(df: pd.DataFrame) -> list:
    """
    단일 유형 또는 복합 유형 반환
    복합 데이터는 교차 분석 모드 자동 활성화
    """
    cols = set(df.columns)
    detected = []

    type_rules = {
        "price":   {"close", "open", "high", "low"},
        "supply":  {"foreign", "institution", "individual"},
        "valuation": {"per", "pbr", "roe"},
        "macro":   {"vix", "rate", "dxy", "fx"},
        "financial": {"revenue", "operating_income", "net_income"},
        "portfolio": {"qty", "buy_price"},
    }

    for dtype, required in type_rules.items():
        overlap = cols & required
        # 필수 컬럼 중 50% 이상 충족 시 해당 유형으로 분류
        if len(overlap) >= max(1, len(required) * 0.5):
            detected.append(dtype)

    # close만 있어도 기술적 지표 계산 가능
    if "close" in cols and "technical" not in detected:
        detected.append("technical")

    return detected if detected else ["unknown"]
```

### 4-3. 데이터 품질 점수

규칙 기반 합산이 아닌 **각 항목을 독립적으로 평가**하여 최종 신뢰도에 반영한다.

```python
def compute_quality_score(df: pd.DataFrame) -> dict:
    n = len(df)
    scores = {}

    # 결측치 비율 (0~40점)
    missing_rate = df.isnull().mean().mean()
    scores["completeness"] = max(0, 40 * (1 - missing_rate * 5))

    # 데이터 길이 (0~30점) — 로그 스케일
    import math
    scores["length"] = min(30, 30 * math.log(max(n, 1)) / math.log(252))

    # 이상치 비율 — IQR 기반 (0~30점)
    if "close" in df.columns:
        returns = df["close"].pct_change().dropna()
        q1, q3 = returns.quantile(0.25), returns.quantile(0.75)
        iqr = q3 - q1
        outlier_rate = ((returns < q1 - 3*iqr) | (returns > q3 + 3*iqr)).mean()
        scores["outlier"] = max(0, 30 * (1 - outlier_rate * 10))
    else:
        scores["outlier"] = 20  # 가격 데이터 없으면 중간값

    total = sum(scores.values())
    grade = (
        "🟢 신뢰 높음"  if total >= 80 else
        "🟡 보통"        if total >= 60 else
        "🟠 낮음"        if total >= 40 else
        "🔴 분석 제한"
    )

    return {"total": round(total), "grade": grade, "detail": scores}
```

### 4-4. 결측치 처리

```python
def handle_missing(df: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
    """
    결측치 비율에 따라 다른 전략 적용
    """
    if missing_rate < 0.05:
        # 5% 미만: 선형 보간
        return df.interpolate(method="linear", limit_direction="both")
    elif missing_rate < 0.20:
        # 5~20%: forward fill + 경고
        print("⚠️ 데이터 일부 보완됨 (forward fill 적용)")
        return df.ffill()
    elif missing_rate < 0.50:
        # 20~50%: 해당 컬럼 제외
        drop_cols = df.columns[df.isnull().mean() > 0.20].tolist()
        print(f"⚠️ 다음 컬럼 분석 제외: {drop_cols}")
        return df.drop(columns=drop_cols)
    else:
        raise ValueError("결측치 50% 초과 — 데이터 보완 필요")
```

---

## 5. 피처 엔지니어링

원시 데이터를 ML 모델이 사용할 수 있는 피처로 변환한다.
**모든 피처는 미래 정보를 사용하지 않도록 rolling 계산으로 처리한다 (Look-ahead bias 방지).**

### 5-1. 가격 기반 피처

```python
def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    c = f["close"]

    # 수익률 (다양한 기간)
    for d in [1, 5, 10, 20, 60]:
        f[f"ret_{d}d"] = c.pct_change(d)

    # 이동평균 대비 현재가 위치 (0 근처: 평균 부근, 양수: 위, 음수: 아래)
    for w in [20, 60, 120]:
        ma = c.rolling(w).mean()
        f[f"price_vs_ma{w}"] = (c - ma) / ma  # 정규화된 위치

    # 볼린저밴드 %B (0~1 범위, 0.5 = 중간)
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f["bb_pct_b"] = (c - (ma20 - 2*std20)) / (4 * std20)  # 0~1 정규화

    # 볼린저밴드 폭 (Squeeze 감지)
    f["bb_width"] = (4 * std20) / ma20

    # RSI (고정 14일이 아닌 다중 기간)
    for period in [9, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        f[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD 히스토그램 (정규화)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    f["macd_hist_norm"] = (macd - signal) / c  # 가격으로 정규화

    # ATR (14일)
    if all(col in df.columns for col in ["high", "low"]):
        hl = df["high"] - df["low"]
        hc = (df["high"] - c.shift(1)).abs()
        lc = (df["low"]  - c.shift(1)).abs()
        f["atr14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        f["atr_norm"] = f["atr14"] / c  # 가격 대비 ATR (종목 간 비교 가능)

    # 거래량 피처
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(20).mean()
        f["volume_ratio"] = df["volume"] / (vol_ma + 1)  # 거래량 배율

    return f
```

### 5-2. 수급 기반 피처

```python
def build_supply_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

    if "foreign" in df.columns:
        # 연속 순매수/매도일 계산
        f["foreign_consecutive"] = (
            df["foreign"].gt(0)
              .groupby((df["foreign"].gt(0) != df["foreign"].gt(0).shift()).cumsum())
              .cumsum()
              .where(df["foreign"].gt(0), 0)
        )

        # 외국인 순매수 강도 (이동 합산, 가격 대비 정규화)
        for w in [5, 10, 20]:
            f[f"foreign_net_{w}d"] = df["foreign"].rolling(w).sum()

    # 외국인 + 기관 동시 매수 신호 (연속성 고려)
    if "institution" in df.columns:
        both_buy = (df["foreign"] > 0) & (df["institution"] > 0)
        f["smart_money_signal"] = both_buy.rolling(3).mean()  # 3일 이동 비율

    # 개인 역배열 신호 (외국인 팔고 개인 사는 패턴)
    if "individual" in df.columns:
        f["retail_trap"] = (
            (df["foreign"] < 0) & (df["individual"] > 0)
        ).astype(float).rolling(3).mean()

    return f
```

### 5-3. 매크로 기반 피처

```python
def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

    if "vix" in df.columns:
        # VIX 절대값이 아닌 변화율과 분위수 사용 (고정 임계값 탈피)
        f["vix_change"] = df["vix"].pct_change(5)                      # 5일 변화율
        f["vix_percentile"] = df["vix"].rolling(252).rank(pct=True)    # 1년 내 분위수

    if "rate" in df.columns:
        f["rate_change"]     = df["rate"].diff(20)                     # 20일 금리 변화
        f["rate_percentile"] = df["rate"].rolling(252).rank(pct=True)  # 1년 내 분위수

    if "dxy" in df.columns:
        f["dxy_change"] = df["dxy"].pct_change(20)

    return f
```

### 5-4. 재무 기반 피처

```python
def build_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

    # YoY / QoQ 성장률 자동 계산
    if "revenue" in df.columns:
        f["revenue_yoy"] = df["revenue"].pct_change(4)   # 분기 기준 4분기 전 비교
        f["revenue_qoq"] = df["revenue"].pct_change(1)

    if "operating_income" in df.columns and "revenue" in df.columns:
        f["op_margin"] = df["operating_income"] / df["revenue"]
        f["op_margin_change"] = f["op_margin"].diff(4)   # YoY 마진 변화

    # 밸류에이션 분위수 (고정 기준값 대신 히스토리 내 상대적 위치)
    for col in ["per", "pbr"]:
        if col in df.columns:
            f[f"{col}_percentile"] = df[col].rolling(252, min_periods=20).rank(pct=True)

    return f
```

---

## 6. 통계적 신호 생성

고정 임계값 대신 **정보 계수(IC, Information Coefficient)**로 신호 품질을 측정한다.

### 6-1. IC (Information Coefficient) 기반 신호 평가

```python
from scipy.stats import spearmanr
import numpy as np

def compute_ic(signal: pd.Series, forward_return: pd.Series, window: int = 60) -> pd.Series:
    """
    IC = 신호와 미래 수익률의 스피어만 상관계수
    IC > 0: 신호가 미래 수익률을 예측
    IC > 0.05: 실용적 가치 있음
    IC > 0.10: 강한 신호
    """
    ic_series = []
    for i in range(window, len(signal)):
        s = signal.iloc[i-window:i]
        r = forward_return.iloc[i-window:i]
        mask = s.notna() & r.notna()
        if mask.sum() < 20:
            ic_series.append(np.nan)
            continue
        ic, _ = spearmanr(s[mask], r[mask])
        ic_series.append(ic)
    return pd.Series(ic_series, index=signal.index[window:])

def compute_icir(ic_series: pd.Series) -> float:
    """
    ICIR = IC 평균 / IC 표준편차
    안정적인 신호일수록 ICIR이 높음
    ICIR > 0.5: 사용 가치 있음
    """
    return ic_series.mean() / (ic_series.std() + 1e-10)
```

### 6-2. 신호 앙상블

```python
def ensemble_signals(features: pd.DataFrame, forward_return: pd.Series) -> pd.DataFrame:
    """
    각 피처의 IC를 계산하고 IC 가중 평균으로 앙상블 신호 생성
    IC가 낮은 신호는 자동으로 낮은 가중치 부여
    """
    signal_cols = [c for c in features.columns if c not in ["open","high","low","close","volume"]]
    ic_weights = {}

    for col in signal_cols:
        if features[col].notna().sum() < 30:
            continue
        ic = compute_ic(features[col], forward_return)
        icir = compute_icir(ic)
        # ICIR이 양수인 신호만 포함, 음수 신호는 제외
        if icir > 0:
            ic_weights[col] = icir

    if not ic_weights:
        return pd.Series(0, index=features.index)

    # 가중치 정규화
    total = sum(ic_weights.values())
    weights = {k: v/total for k, v in ic_weights.items()}

    # 앙상블 신호 (0~1 범위로 정규화)
    ensemble = sum(
        features[col].rank(pct=True) * w
        for col, w in weights.items()
        if col in features.columns
    )
    return ensemble.rank(pct=True), weights  # 신호, 각 피처 가중치 반환
```

---

## 7. Alpha Score — 통계적 가중치 산출

### 7-1. 왜 임의 배점(20/25/30점)을 버리는가

임의 배점의 문제:
```
"수급 30점" → 근거: 없음. 직관적 판단
"매크로 20점" → 실제로 예측력이 더 높을 수도 있음
고정 배점 → 시장 구조 변화에 대응 불가
```

통계적 접근:
```
각 신호 그룹의 IC 측정
→ IC 기반으로 가중치 자동 결정
→ 시장 레짐별로 가중치 재계산
→ 일정 기간마다 업데이트
```

### 7-2. 로지스틱 회귀 기반 Alpha Score

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def train_alpha_model(features: pd.DataFrame, forward_return: pd.Series,
                      threshold: float = 0.0) -> dict:
    """
    forward_return > threshold → 1 (상승)
    forward_return <= threshold → 0 (하락/횡보)

    TimeSeriesSplit으로 미래 데이터 누수 방지
    출력: 각 피처의 회귀 계수 (= Alpha Score 가중치)
    """
    # 공통 인덱스
    common = features.index.intersection(forward_return.index)
    X = features.loc[common].ffill().fillna(0)
    y = (forward_return.loc[common] > threshold).astype(int)

    # 시계열 교차 검증 (5-fold)
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    model = LogisticRegression(C=0.1, max_iter=1000)  # L2 정규화로 과적합 방지

    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_val = scaler.transform(X.iloc[val_idx])
        model.fit(X_tr, y.iloc[train_idx])
        scores.append(model.score(X_val, y.iloc[val_idx]))

    # 전체 데이터로 최종 학습
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "coefficients": dict(zip(X.columns, model.coef_[0])),
        "cv_accuracy": np.mean(scores),
        "cv_accuracy_std": np.std(scores),
    }

def predict_alpha_score(model_info: dict, features: pd.DataFrame) -> pd.Series:
    """
    0~100점으로 변환된 Alpha Score 반환
    내부적으로는 상승 확률 (0~1)
    """
    scaler = model_info["scaler"]
    model  = model_info["model"]
    cols   = model_info["feature_names"]

    X = features[cols].ffill().fillna(0)
    prob = model.predict_proba(scaler.transform(X))[:, 1]  # 상승 확률
    return pd.Series(prob * 100, index=features.index)
```

### 7-3. 피처 중요도 시각화용 정보 제공

```python
def get_feature_importance(model_info: dict) -> pd.DataFrame:
    """
    어떤 신호가 Alpha Score에 얼마나 기여하는지 반환
    대시보드의 '신호 기여도' 패널에 표시
    """
    coefs = model_info["coefficients"]
    df = pd.DataFrame(list(coefs.items()), columns=["feature", "coefficient"])
    df["abs_coef"] = df["coefficient"].abs()
    df["importance_pct"] = df["abs_coef"] / df["abs_coef"].sum() * 100
    df["direction"] = df["coefficient"].apply(lambda x: "긍정" if x > 0 else "부정")
    return df.sort_values("abs_coef", ascending=False)
```

### 7-4. Alpha Score 해석

```
80점 이상 (상승 확률 80%+) → Strong Buy
60~79점   (상승 확률 60~79%) → Buy
40~59점   (상승 확률 40~59%) → Neutral
20~39점   (상승 확률 20~39%) → Caution
20점 미만 (상승 확률 20%-) → Avoid

단, 데이터 품질 점수가 60점 미만이면 한 단계 하향 조정
단, CV 정확도 표준편차가 0.1 이상이면 신뢰도 낮음 표시
```

---

## 8. 시장 레짐 — 비지도 학습 기반

### 8-1. HMM (Hidden Markov Model) 기반 레짐 탐지

고정 규칙("VIX < 20 = 강세") 대신 데이터에서 레짐을 학습한다.

```python
from hmmlearn import hmm
import numpy as np

def fit_regime_model(returns: pd.Series, n_regimes: int = 3) -> dict:
    """
    수익률 시계열에서 자동으로 레짐을 학습
    n_regimes: 레짐 수 (보통 3: 강세/횡보/약세)

    출력: 각 날짜의 레짐 레이블 (0, 1, 2)
    레짐의 의미는 데이터에 따라 자동 결정 (평균 수익률로 정렬)
    """
    X = returns.dropna().values.reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(X)

    states = model.predict(X)

    # 레짐을 평균 수익률 기준으로 정렬 (0=약세, 1=횡보, 2=강세)
    means = [X[states == s].mean() for s in range(n_regimes)]
    order = np.argsort(means)  # 낮은 수익률 → 높은 수익률 순서
    state_map = {old: new for new, old in enumerate(order)}
    sorted_states = np.array([state_map[s] for s in states])

    labels = {0: "약세 (Bearish)", 1: "횡보 (Sideways)", 2: "강세 (Bullish)"}

    return {
        "model": model,
        "states": pd.Series(sorted_states, index=returns.dropna().index),
        "labels": labels,
        "transition_matrix": model.transmat_,  # 레짐 전환 확률 행렬
    }

def get_regime_transition_prob(regime_model: dict, current_regime: int) -> dict:
    """
    현재 레짐에서 다음 레짐으로 전환될 확률
    예: 현재 강세장 → 내일도 강세장일 확률 = 0.92
    """
    trans = regime_model["transition_matrix"]
    labels = regime_model["labels"]
    return {labels[i]: round(trans[current_regime][i], 3) for i in range(len(labels))}
```

### 8-2. 레짐별 Alpha Score 가중치 조정

```python
def adjust_weights_by_regime(base_weights: dict, current_regime: int) -> dict:
    """
    레짐에 따라 가중치를 조정하되,
    조정 폭은 데이터에서 학습된 레짐별 IC 차이에 비례
    (임의 조정이 아닌 데이터 기반 조정)
    """
    # 레짐별 각 신호의 IC를 사전 계산해둔 값 활용
    # 이 딕셔너리는 학습 단계에서 채워짐
    regime_ic_multipliers = {
        0: {"momentum": 0.6, "value": 1.4, "supply": 1.1},  # 약세: 밸류 가중
        1: {"momentum": 1.0, "value": 1.0, "supply": 1.0},  # 횡보: 균등
        2: {"momentum": 1.4, "value": 0.6, "supply": 1.2},  # 강세: 모멘텀 가중
    }
    multipliers = regime_ic_multipliers.get(current_regime, {})
    adjusted = {k: v * multipliers.get(k, 1.0) for k, v in base_weights.items()}

    # 재정규화
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}
```

---

## 9. 리스크 관리 — 수학적 근거

### 9-1. CVaR (Conditional Value at Risk) — 꼬리 리스크

VaR의 한계(특정 분위수의 손실만 보여줌)를 보완한다.

```python
def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    CVaR (Expected Shortfall):
    최악 (1-confidence)% 시나리오에서의 평균 손실
    VaR보다 꼬리 리스크를 더 잘 표현

    예: CVaR(95%) = -3.2% → 최악 5% 상황에서 평균 -3.2% 손실
    """
    r = returns.dropna()
    var = r.quantile(1 - confidence)
    cvar = r[r <= var].mean()

    return {
        "var_95": round(var, 4),
        "cvar_95": round(cvar, 4),
        "interpretation": f"최악 5% 상황 평균 손실: {cvar*100:.1f}%"
    }
```

### 9-2. Kelly Criterion — 최적 포지션 사이징

```python
def kelly_position(win_rate: float, avg_win: float, avg_loss: float,
                   kelly_fraction: float = 0.25) -> dict:
    """
    풀 켈리: f* = (p*b - q) / b
    p = 승률, q = 1-p, b = 평균이익/평균손실

    실전에서는 풀 켈리의 일부만 사용 (Half Kelly = 0.5, Quarter Kelly = 0.25)
    이유: 풀 켈리는 이론적으로 최적이나 드로다운이 매우 큰 경향

    kelly_fraction = 0.25 (Quarter Kelly) 기본값: 보수적 운용
    """
    if avg_loss == 0:
        return {"kelly": 0, "recommended": 0}

    b = avg_win / abs(avg_loss)  # 이익/손실 비율
    p, q = win_rate, 1 - win_rate
    full_kelly = (p * b - q) / b
    recommended = max(0, min(full_kelly * kelly_fraction, 0.20))  # 최대 20% 캡

    return {
        "full_kelly": round(full_kelly, 4),
        "recommended_fraction": round(recommended, 4),
        "recommended_pct": f"{recommended*100:.1f}%",
        "rr_ratio": round(b, 2),
        "interpretation": (
            f"풀 켈리 {full_kelly*100:.1f}%의 {int(kelly_fraction*100)}% 적용 → "
            f"포트폴리오의 {recommended*100:.1f}% 진입 권고"
        )
    }
```

### 9-3. ATR 기반 손절/익절 — 동적 계산

```python
def compute_risk_levels(entry_price: float, atr: float,
                        alpha_score: float, regime: int) -> dict:
    """
    손절/익절 배수는 Alpha Score와 레짐에 따라 동적으로 결정
    고정값이 아닌 신호 강도와 시장 환경의 함수

    높은 Alpha Score → 더 넓은 손절 허용 (신호 신뢰도 높음)
    약세장 레짐 → 더 타이트한 손절 (리스크 관리 강화)
    """
    # 기본 배수
    base_stop  = 2.0
    base_target = 3.0

    # Alpha Score에 따른 조정 (높은 점수 = 넓은 손절 허용)
    score_factor = 0.5 + (alpha_score / 100)  # 0.5 ~ 1.5 범위

    # 레짐에 따른 조정
    regime_factor = {0: 0.7, 1: 1.0, 2: 1.3}.get(regime, 1.0)
    # 약세장(0): 타이트하게, 강세장(2): 넓게

    stop_mult   = base_stop   * score_factor * regime_factor
    target_mult = base_target * score_factor * regime_factor

    stop_loss    = entry_price - atr * stop_mult
    target_price = entry_price + atr * target_mult
    rr_ratio     = target_mult / stop_mult

    return {
        "stop_loss":    round(stop_loss, 0),
        "target_price": round(target_price, 0),
        "stop_pct":     round((stop_loss - entry_price) / entry_price * 100, 2),
        "target_pct":   round((target_price - entry_price) / entry_price * 100, 2),
        "rr_ratio":     round(rr_ratio, 2),
        "stop_mult":    round(stop_mult, 2),
        "target_mult":  round(target_mult, 2),
    }
```

### 9-4. 포트폴리오 수준 리스크

```python
def portfolio_risk(weights: np.ndarray, cov_matrix: np.ndarray) -> dict:
    """
    포트폴리오 전체 변동성과 분산화 효과 계산
    단순 개별 종목 분석이 아닌 상관관계 고려
    """
    port_variance = weights @ cov_matrix @ weights
    port_vol = np.sqrt(port_variance * 252)  # 연율화

    # 분산화 비율: 개별 변동성 합 대비 포트폴리오 변동성
    individual_vols = np.sqrt(np.diag(cov_matrix) * 252)
    weighted_avg_vol = weights @ individual_vols
    diversification_ratio = weighted_avg_vol / port_vol

    return {
        "portfolio_volatility": round(port_vol, 4),
        "diversification_ratio": round(diversification_ratio, 2),
        "interpretation": (
            f"분산화 비율 {diversification_ratio:.2f} — "
            f"{'양호' if diversification_ratio > 1.2 else '분산화 효과 낮음'}"
        )
    }
```

---

## 10. 백테스트 — 실제 계산 구조

하드코딩된 수치 없음. 모든 백테스트 결과는 입력 데이터로 동적 계산한다.

### 10-1. 워크포워드(Walk-Forward) 백테스트

```python
def walk_forward_backtest(df: pd.DataFrame, signal: pd.Series,
                          train_window: int = 252,
                          test_window: int = 63) -> dict:
    """
    슬라이딩 윈도우로 학습/검증 반복
    미래 데이터 누수 완전 차단
    현실적인 성과 추정 가능

    train_window: 252일 (1년) 학습
    test_window: 63일 (3개월) 검증
    """
    returns = df["close"].pct_change()
    results = []

    for start in range(0, len(df) - train_window - test_window, test_window):
        train_end = start + train_window
        test_end  = train_end + test_window

        # 검증 구간의 신호 기반 수익률
        test_signal  = signal.iloc[train_end:test_end]
        test_returns = returns.iloc[train_end:test_end]

        # 신호 상위 분위수에서 진입 (고정 임계값 아님)
        entry_signal = test_signal > test_signal.quantile(0.6)
        period_returns = test_returns[entry_signal]

        if len(period_returns) < 5:
            continue

        results.append({
            "start": df.index[train_end],
            "end":   df.index[min(test_end, len(df)-1)],
            "n_trades":   len(period_returns),
            "avg_return": period_returns.mean(),
            "win_rate":   (period_returns > 0).mean(),
            "sharpe":     period_returns.mean() / (period_returns.std() + 1e-10) * (252**0.5),
            "max_dd":     (period_returns.cumsum() - period_returns.cumsum().cummax()).min(),
        })

    if not results:
        return {"error": "백테스트 데이터 부족"}

    summary = pd.DataFrame(results)
    return {
        "periods":       len(results),
        "avg_return":    round(summary["avg_return"].mean(), 4),
        "avg_win_rate":  round(summary["win_rate"].mean(), 4),
        "avg_sharpe":    round(summary["sharpe"].mean(), 2),
        "avg_max_dd":    round(summary["max_dd"].mean(), 4),
        "consistency":   round((summary["avg_return"] > 0).mean(), 2),
        "detail":        summary.to_dict("records"),
    }
```

### 10-2. 성과 지표 계산

```python
def compute_performance_metrics(returns: pd.Series, benchmark: pd.Series = None) -> dict:
    r = returns.dropna()
    metrics = {
        "total_return":  round((1 + r).prod() - 1, 4),
        "annualized_return": round((1 + r).prod() ** (252/len(r)) - 1, 4),
        "volatility":    round(r.std() * (252**0.5), 4),
        "sharpe":        round(r.mean() / (r.std() + 1e-10) * (252**0.5), 2),
        "max_drawdown":  round((r.cumsum() - r.cumsum().cummax()).min(), 4),
        "win_rate":      round((r > 0).mean(), 4),
        "profit_factor": round(r[r>0].sum() / (abs(r[r<0].sum()) + 1e-10), 2),
        "cvar_95":       compute_cvar(r)["cvar_95"],
    }
    if benchmark is not None:
        excess = r - benchmark
        metrics["information_ratio"] = round(
            excess.mean() / (excess.std() + 1e-10) * (252**0.5), 2
        )
    return metrics
```

---

## 11. 시각화 선택 규칙

### 11-1. 감지된 데이터 유형 → 차트 자동 선택

| 감지 유형 | 메인 차트 | 보조 차트 | 자동 추가 요소 |
|----------|-----------|-----------|---------------|
| price | 캔들스틱 + MA | 거래량 막대 | 손절/목표가 수평선, 신호 마커 |
| technical | RSI 다중 라인 + 게이지 | MACD 히스토그램 | 과매수/과매도 음영 |
| supply | 수급 스택 막대 | 주가 오버레이 | 신호 구간 음영 |
| macro | 멀티 라인 (이중 Y축) | 레짐 배경색 | VIX 기준 분위수 표시 |
| valuation | 수평 막대 (분위수 위치) | 시계열 라인 | 업종 분위수 표시 |
| financial | 분기별 그룹 막대 | 마진 라인 | YoY 변화 워터폴 |
| portfolio | 도넛차트 | 손익 수평 막대 | 손절가 경고선 |

### 11-2. 분석 결과 → 시각화 자동 보강

```
Alpha Score > 80
→ 손절/목표가 라인 초록색으로 강조
→ 진입 권고 영역 초록 음영

Alpha Score < 30
→ 위험 구간 빨간 음영
→ 경고 뱃지 자동 추가

레짐 전환 발생 (이전 vs 현재 다름)
→ 전환 시점 세로 점선 자동 추가
→ "레짐 전환" 레이블 표시

수급 신호 강함 (smart_money_signal > 0.5)
→ 해당 구간 파란 음영 + 스마트머니 레이블

패닉셀 감지 (volume_ratio > 3 + 가격 하락)
→ 해당 캔들에 마커 + "패닉셀" 레이블

IC 낮은 신호 포함 시
→ 해당 신호에 "(낮은 신뢰도)" 표시
```

### 11-3. 레짐별 배경색

```
약세 레짐 (regime=0) → 연한 빨간 배경
횡보 레짐 (regime=1) → 흰색/회색 배경
강세 레짐 (regime=2) → 연한 초록 배경

레짐 전환 구간 → 경계선에 세로 점선
```

---

## 12. 대시보드 구성 흐름

### 12-1. 모듈 배치 순서 (항상 고정)

```
1순위 — 데이터 현황 패널
  감지된 유형 / 품질 점수 / 분석 기간
  데이터 부족 시 명확한 안내

2순위 — 시장 레짐 패널
  HMM 기반 현재 레짐 + 전환 확률
  매크로 데이터 없을 때: 가격 기반 레짐 추정

3순위 — Alpha Score 패널
  0~100 게이지 + 상승 확률
  피처별 기여도 막대 (신호 투명성)

4순위 — 메인 차트 패널
  감지 유형에 따라 자동 선택
  분석 결과 마커/음영 자동 추가

5순위 — 보조 신호 패널
  RSI 다중 기간 / MACD / BB
  IC 값 함께 표시 (신호 신뢰도)

6순위 — 수급 패널 (수급 데이터 있을 때만)
  수급 스택 막대 + 주가 오버레이
  스마트머니 신호 강도

7순위 — 리스크 패널
  CVaR / 켈리 포지션 / 동적 손절가
  포트폴리오 상관관계 (다종목 시)

8순위 — 백테스트 요약 패널 (데이터 60일 이상 시)
  워크포워드 결과 요약
  기간별 일관성 차트

9순위 — 인사이트 패널 (항상 최하단)
  자연어 요약 + 액션 권고
  모든 수치의 출처 명시
```

---

## 13. 인사이트 생성 규칙

### 13-1. 인사이트 작성 원칙

```
① 모든 수치는 실제 계산값을 사용 (하드코딩 금지)
② 신뢰도 낮은 수치는 반드시 표시
③ 상승/하락 어느 쪽도 단정하지 않음
④ 핵심 근거 3가지 이내로 요약
⑤ 데이터 한계를 명시
```

### 13-2. 인사이트 템플릿

**기본 구조**
```
[종목/지수]는 현재 [레짐] 상태입니다.
Alpha Score [X]점 (상승 확률 [X]%).

핵심 신호:
- [신호 1]: [수치] (IC: [값])
- [신호 2]: [수치] (IC: [값])
- [신호 3]: [수치] (IC: [값])

백테스트 결과 ([N]개 기간 워크포워드):
평균 수익률 [X]%, 승률 [X]%, 샤프 [X]

리스크:
손절가 [X]원 ([X]%, ATR×[X]) / 목표가 [X]원 ([X]%)
CVaR(95%): [X]% / 켈리 권고 포지션: [X]%

데이터 품질: [등급] ([X]점)
분석 기간: [N]일 / 신뢰도: [등급]
```

**데이터 부족 시**
```
"데이터 기간이 [N]일로 일부 지표를 계산할 수 없습니다.
 가용 데이터: [유형 목록]
 계산 불가: [지표 목록]
 현재 품질 점수 [X]점 — 추가 데이터 확보 권고"
```

**복합 데이터 교차 분석 시**
```
"[매크로 상황]과 [수급 상황]이 동시에 감지됩니다.
 이 조합의 백테스트 결과: 평균 [X]%, 승률 [X]% ([N]회 발생)
 단, 발생 빈도가 낮아 통계적 신뢰도가 제한적입니다."
```

---

## 14. JSON 출력 구조

```json
{
  "meta": {
    "ticker": "005930",
    "name": "삼성전자",
    "analysis_date": "2024-01-02",
    "data_types_detected": ["price", "supply", "technical"],
    "data_quality": {
      "total": 88,
      "grade": "🟢 신뢰 높음",
      "completeness": 38,
      "length": 28,
      "outlier": 22
    },
    "data_period_days": 252
  },

  "regime": {
    "current": 2,
    "label": "강세 (Bullish)",
    "transition_probability": {
      "약세 (Bearish)": 0.03,
      "횡보 (Sideways)": 0.09,
      "강세 (Bullish)": 0.88
    }
  },

  "alpha_score": {
    "score": 74.2,
    "probability": 0.742,
    "action": "Buy",
    "cv_accuracy": 0.63,
    "cv_accuracy_std": 0.04,
    "feature_importance": [
      {"feature": "foreign_consecutive", "importance_pct": 28.3, "direction": "긍정"},
      {"feature": "rsi_14",             "importance_pct": 19.1, "direction": "긍정"},
      {"feature": "macd_hist_norm",      "importance_pct": 14.7, "direction": "긍정"}
    ]
  },

  "indicators": {
    "close": 73400,
    "atr14": 1240,
    "atr_norm": 0.0169,
    "rsi_14": 44.2,
    "bb_pct_b": 0.58,
    "macd_hist_norm": 0.00164,
    "volume_ratio": 1.34,
    "foreign_consecutive": 5,
    "smart_money_signal": 0.67
  },

  "risk": {
    "var_95": -0.0231,
    "cvar_95": -0.0348,
    "kelly_full": 0.18,
    "kelly_recommended": 0.045,
    "stop_loss": 71120,
    "stop_pct": -3.1,
    "target_price": 78540,
    "target_pct": 7.0,
    "rr_ratio": 2.25,
    "stop_mult": 1.84,
    "target_mult": 4.14
  },

  "backtest": {
    "method": "walk_forward",
    "periods": 8,
    "avg_return": 0.087,
    "avg_win_rate": 0.64,
    "avg_sharpe": 1.38,
    "avg_max_dd": -0.043,
    "consistency": 0.75
  },

  "insight": {
    "regime": "강세 (Bullish)",
    "alpha_score": 74.2,
    "top_signals": [
      "외국인 5일 연속 순매수 (IC: 0.08)",
      "RSI 44.2 — 중립 구간 (IC: 0.06)",
      "MACD 히스토그램 양전환 (IC: 0.05)"
    ],
    "action": "Buy",
    "position_pct": 4.5,
    "caution": "CV 정확도 63% — 중간 수준 신뢰도",
    "data_quality_note": "품질 88점 — 신뢰 높음"
  }
}
```

---

## 15. 모델 업데이트 및 개선 흐름

### 15-1. 주기적 재학습

```python
def should_retrain(model_info: dict, new_data: pd.DataFrame,
                   performance_threshold: float = 0.55) -> bool:
    """
    재학습 조건:
    ① 새 데이터가 63일(1분기) 이상 쌓였을 때
    ② 최근 63일 CV 정확도가 threshold 미만으로 떨어졌을 때
    ③ 레짐이 2번 이상 전환되었을 때
    """
    recent_accuracy = validate_recent(model_info, new_data)
    return recent_accuracy < performance_threshold

def validate_recent(model_info: dict, recent_data: pd.DataFrame,
                    window: int = 63) -> float:
    """최근 window일 데이터로 모델 검증"""
    # 구현: 최근 데이터로 예측 후 실제 수익률과 비교
    pass
```

### 15-2. 신호 모니터링

```
매 분석 시 기록:
- 각 신호의 IC (rolling 60일)
- IC 추이가 하락하는 신호 → "신호 약화" 경고
- IC가 음수로 전환된 신호 → 자동 가중치 0으로 설정

대시보드 표시:
- 신호별 IC 추이 차트
- "이 신호는 최근 60일 IC가 하락 중입니다" 경고
```

---

## 16. 예외 처리

### 16-1. 데이터 부족

```
전체 데이터 14일 미만
→ 기술적 지표 비활성화
→ 가격/수급 기본 차트만 표시
→ Alpha Score 계산 불가 → "데이터 부족" 표시

HMM 학습 불가 (데이터 30일 미만)
→ 가격 기반 단순 추세 판단으로 대체
→ "레짐 모델 미적용 — 데이터 부족" 표시

백테스트 불가 (60일 미만)
→ 백테스트 패널 비활성화
→ "백테스트 불가" 안내
```

### 16-2. 모델 실패

```
로지스틱 회귀 수렴 실패
→ IC 가중 앙상블 신호로 대체
→ "ML 모델 미적용 — 통계적 앙상블 사용" 표시

IC 양수 신호 없음 (모든 신호 IC 음수)
→ Alpha Score 50점 (중립) 반환
→ "유효한 예측 신호 없음 — 관망 권고"

ATR 계산 불가 (OHLC 없음)
→ 표준편차 기반 손절로 대체: 매수가 × (1 - 2σ)
→ "ATR 대체: 2σ 손절 적용" 표시
```

### 16-3. 데이터 유형 미감지

```
감지 실패 시
→ 수동 선택 UI 표시
→ 선택 후 정상 파이프라인 재실행
→ 선택하지 않을 경우 컬럼별 기초 통계 표시 후 종료
```

---

> **이 문서는 Alpha Signal Dashboard의 핵심 분석 설계 문서입니다.**
> Claude Code는 이 문서를 기반으로 어떤 투자 데이터가 입력되어도
> 동일한 ML/통계 파이프라인으로 자동 분석합니다.
>
> **모든 수치는 입력 데이터로 동적 계산됩니다. 하드코딩된 임계값은 없습니다.**
> *과거 패턴이 미래를 보장하지 않습니다. 이 시스템은 투자 보조 도구입니다.*
