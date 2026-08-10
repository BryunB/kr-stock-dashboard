"""기술적 지표 및 성과 지표.

모든 함수는 pandas Series/DataFrame을 받아 같은 인덱스의 결과를 돌려준다.
입력을 변형하지 않는다(부작용 없음).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS

# ---------------------------------------------------------------- 추세 지표


def sma(s: pd.Series, window: int = 20) -> pd.Series:
    """단순이동평균."""
    return s.rolling(window).mean()


def ema(s: pd.Series, span: int = 20) -> pd.Series:
    """지수이동평균."""
    return s.ewm(span=span, adjust=False).mean()


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 선, 시그널선, 히스토그램."""
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


# ---------------------------------------------------------------- 모멘텀/변동성


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI (0~100). 통상 70 이상 과매수, 30 이하 과매도로 읽는다."""
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder 평활 = alpha 1/window 인 EMA
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger(s: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """볼린저 밴드. %B는 밴드 내 위치(0=하단, 1=상단)."""
    mid = sma(s, window)
    std = s.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": (upper - lower) / mid,
            "pct_b": (s - lower) / (upper - lower),
        }
    )


# ---------------------------------------------------------------- 수익률/성과


def returns(s: pd.Series, log: bool = False) -> pd.Series:
    """일간 수익률. log=True면 로그수익률."""
    return np.log(s / s.shift(1)) if log else s.pct_change()


def cumulative_returns(s: pd.Series) -> pd.Series:
    """첫날 대비 누적 수익률 (0.0 = 원금)."""
    return s / s.iloc[0] - 1


def volatility(s: pd.Series, window: int | None = None) -> pd.Series | float:
    """연율화 변동성. window를 주면 rolling, 없으면 전체 기간 스칼라."""
    r = returns(s)
    if window:
        return r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return float(r.std() * np.sqrt(TRADING_DAYS))


def cagr(s: pd.Series) -> float:
    """연평균 성장률."""
    years = len(s) / TRADING_DAYS
    if years <= 0 or s.iloc[0] <= 0:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)


def drawdown(s: pd.Series) -> pd.Series:
    """전고점 대비 낙폭 (음수)."""
    return s / s.cummax() - 1


def max_drawdown(s: pd.Series) -> float:
    """최대낙폭(MDD)."""
    return float(drawdown(s).min())


def sharpe(s: pd.Series, risk_free: float = 0.03) -> float:
    """연율화 샤프지수. risk_free는 연 무위험수익률(예: 0.03 = 3%)."""
    r = returns(s).dropna()
    if r.empty or r.std() == 0:
        return float("nan")
    excess = r.mean() * TRADING_DAYS - risk_free
    return float(excess / (r.std() * np.sqrt(TRADING_DAYS)))


def summary(s: pd.Series) -> pd.Series:
    """한 종목의 성과 요약 지표 묶음."""
    return pd.Series(
        {
            "start": s.index[0].date(),
            "end": s.index[-1].date(),
            "total_return": float(s.iloc[-1] / s.iloc[0] - 1),
            "cagr": cagr(s),
            "volatility": volatility(s),
            "sharpe": sharpe(s),
            "max_drawdown": max_drawdown(s),
        }
    )


def add_all(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """OHLCV DataFrame에 주요 지표 열을 붙여 새 DataFrame으로 반환한다."""
    out = df.copy()
    s = out[col]
    out["sma20"] = sma(s, 20)
    out["sma60"] = sma(s, 60)
    out["sma120"] = sma(s, 120)
    out["rsi14"] = rsi(s, 14)
    out = out.join(macd(s))
    out = out.join(bollinger(s)[["upper", "lower", "pct_b"]])
    out["ret"] = returns(s)
    out["drawdown"] = drawdown(s)
    return out
