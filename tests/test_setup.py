"""환경 구축이 제대로 됐는지 확인하는 스모크 테스트.

네트워크가 필요한 테스트는 `-m network` 로 분리했다.
    전체:        pytest
    오프라인만:  pytest -m "not network"
"""

import numpy as np
import pandas as pd
import pytest

from src import config, data_loader, indicators, plotting


def _fake_prices(n=300, seed=0):
    """지표 계산 검증용 합성 가격 시계열."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    steps = rng.normal(0.0005, 0.015, n)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=idx, name="Close")


# ----------------------------------------------------------- 오프라인 테스트


def test_config_dirs_exist():
    assert config.CACHE_DIR.is_dir()
    assert config.OUTPUT_DIR.is_dir()


def test_korean_font_available():
    assert plotting.find_korean_font() is not None, "한글 폰트를 찾지 못했습니다"


def test_sma_matches_manual():
    s = _fake_prices()
    got = indicators.sma(s, 20)
    assert np.isnan(got.iloc[18])  # 20개 미만 구간은 NaN
    assert got.iloc[19] == pytest.approx(s.iloc[:20].mean())


def test_rsi_bounds():
    r = indicators.rsi(_fake_prices()).dropna()
    assert not r.empty
    assert r.between(0, 100).all()


def test_rsi_all_up_is_100():
    s = pd.Series(range(1, 60), index=pd.bdate_range("2024-01-01", periods=59)).astype(float)
    assert indicators.rsi(s).iloc[-1] == pytest.approx(100.0)


def test_bollinger_ordering():
    b = indicators.bollinger(_fake_prices()).dropna()
    assert (b["upper"] > b["mid"]).all()
    assert (b["mid"] > b["lower"]).all()


def test_drawdown_is_non_positive():
    dd = indicators.drawdown(_fake_prices())
    assert (dd <= 1e-12).all()
    assert indicators.max_drawdown(_fake_prices()) <= 0


def test_summary_keys():
    out = indicators.summary(_fake_prices())
    assert {"cagr", "volatility", "sharpe", "max_drawdown"} <= set(out.index)


def test_add_all_preserves_index():
    df = pd.DataFrame({"Close": _fake_prices()})
    out = indicators.add_all(df)
    assert out.index.equals(df.index)
    assert "rsi14" in out.columns and "macd" in out.columns
    assert list(df.columns) == ["Close"]  # 원본 불변


def test_find_symbol_matches_code_or_name(monkeypatch):
    """find_symbol이 종목코드로도 검색되는지 확인 (이름만 검색되던 버그의 회귀 테스트)."""
    fake_listing = pd.DataFrame(
        {
            "Code": ["005930", "000660", "005935"],
            "Name": ["삼성전자", "SK하이닉스", "삼성전자우"],
            "Market": ["KOSPI", "KOSPI", "KOSPI"],
        }
    )
    monkeypatch.setattr(data_loader, "get_listing", lambda market="KRX", use_cache=True: fake_listing)

    by_code = data_loader.find_symbol("005930")
    assert list(by_code["Code"]) == ["005930"]

    by_name = data_loader.find_symbol("삼성")
    assert set(by_name["Code"]) == {"005930", "005935"}


# ----------------------------------------------------------- 네트워크 테스트


@pytest.mark.network
def test_fetch_kospi():
    df = data_loader.get_price("KOSPI", "2025-01-01", "2025-06-30")
    assert not df.empty
    assert "Close" in df.columns


@pytest.mark.network
def test_fetch_nasdaq_and_krx_together():
    closes = data_loader.get_closes(["NASDAQ", "005930"], "2025-01-01", "2025-06-30")
    assert set(closes.columns) == {"NASDAQ", "005930"}
    assert not closes.dropna().empty
