"""스크리닝 대시보드(screener, charts) 테스트.

screener는 GitHub 미러에서 실시간 스냅샷을 받아오므로 network 마커.
charts는 합성 데이터로 오프라인 검증 가능.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import charts, screener
from src import indicators as ind


def _fake_ohlcv(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1000, 100000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ----------------------------------------------------------- charts (오프라인)


def test_build_chart_minimal():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", sma_windows=(), show_volume=False)
    assert isinstance(fig, go.Figure)
    kinds = {type(t).__name__ for t in fig.data}
    assert "Candlestick" in kinds


def test_build_chart_all_panels():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(
        df, "테스트", sma_windows=(20, 60), show_volume=True, show_rsi=True, show_macd=True
    )
    kinds = [type(t).__name__ for t in fig.data]
    assert kinds.count("Candlestick") == 1
    assert kinds.count("Scatter") >= 2 + 1 + 2  # SMA20/60 + RSI + MACD/Signal
    assert kinds.count("Bar") == 2  # 거래량 + MACD 히스토그램


def test_build_chart_index_overlay_rebased():
    df = ind.add_all(_fake_ohlcv())
    other = _fake_ohlcv(seed=2)["Close"] * 1000  # 완전히 다른 가격 스케일
    fig = charts.build_chart(df, "테스트", index_overlays={"KOSPI": other})
    overlay = next(t for t in fig.data if t.name == "KOSPI (비교)")
    assert overlay.y[0] == pytest.approx(df["Close"].iloc[0], rel=1e-6)


def test_build_chart_bollinger_bands():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", sma_windows=(), show_bollinger=True, show_volume=False)
    names = {t.name for t in fig.data}
    assert {"BB 상단", "BB 하단"} <= names


def test_build_chart_missing_sma_column_skipped():
    """indicators.add_all은 sma5를 만들지 않는다 — 없는 컬럼은 조용히 건너뛴다."""
    df = ind.add_all(_fake_ohlcv())
    assert "sma5" not in df.columns
    fig = charts.build_chart(df, "테스트", sma_windows=(5, 20), show_volume=False)
    names = {t.name for t in fig.data}
    assert "SMA20" in names
    assert "SMA5" not in names


# ----------------------------------------------------------- screener (네트워크)


@pytest.mark.network
def test_screen_has_expected_columns():
    df = screener.screen(market="KOSDAQ", use_cache=False)
    assert {"Code", "Name", "DailyChangeRatio", "WeeklyChangeRatio", "Marcap"} <= set(df.columns)
    assert not df.empty
    assert set(df["Market"]) <= {"KOSDAQ"}


@pytest.mark.network
def test_top_movers_sorted_descending():
    df = screener.screen(market="KOSDAQ", use_cache=True)
    top = screener.top_movers(df, by="DailyChangeRatio", n=10)
    assert len(top) <= 10
    assert (top["DailyChangeRatio"].diff().dropna() <= 0).all()
