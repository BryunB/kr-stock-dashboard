from __future__ import annotations

import pandas as pd
import pytest

from src import us_screener


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_ROWS = [
    {
        "symbol": "NVDA",
        "name": "NVIDIA Corporation Common Stock",
        "lastsale": "$225.30",
        "netchange": "1.21",
        "pctchange": "0.540%",
        "volume": "98604105",
        "marketCap": "5,452,260,000,000",
        "country": "United States",
        "ipoyear": "",
        "industry": "Semiconductors",
        "sector": "Technology",
        "url": "/market-activity/stocks/nvda",
    },
    {
        "symbol": "AACB",
        "name": "Artius II Acquisition Inc. Class A Ordinary Shares",
        "lastsale": "$10.575",
        "netchange": "0.045",
        "pctchange": "0.427%",
        "volume": "859609",
        "marketCap": "0.00",
        "country": "United States",
        "ipoyear": "2025",
        "industry": "Blank Checks",
        "sector": "Finance",
        "url": "/market-activity/stocks/aacb",
    },
    {
        "symbol": "AACG",
        "name": "ATA Creativity Global American Depositary Shares",
        "lastsale": "$1.11",
        "netchange": "0.20",
        "pctchange": "21.991%",
        "volume": "13159938",
        "marketCap": "75995062",
        "country": "China",
        "ipoyear": "",
        "industry": "Other Consumer Services",
        "sector": "Real Estate",
        "url": "/market-activity/stocks/aacg",
    },
    {
        # 신규 상장 등 전일 종가가 없어 pctchange가 빈 문자열인 실측 케이스
        "symbol": "GHXI",
        "name": "Some Newly Listed Co",
        "lastsale": "$9.95",
        "netchange": "9.95",
        "pctchange": "",
        "volume": "12024",
        "marketCap": "0.00",
        "country": "United States",
        "ipoyear": "2026",
        "industry": "",
        "sector": "",
        "url": "/market-activity/stocks/ghxi",
    },
]

_PAYLOAD = {
    "data": {"asOf": None, "headers": {}, "rows": _ROWS},
    "message": None,
    "status": {"rCode": 200},
}


def test_screen_parses_rows_and_columns(monkeypatch):
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(use_cache=False)

    assert list(df.columns) == us_screener._RESULT_COLS
    assert len(df) == 4
    assert set(df["Code"]) == {"NVDA", "AACB", "AACG", "GHXI"}
    assert (df["Market"] == "NASDAQ").all()


def test_screen_parses_close_and_daily_change(monkeypatch):
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(use_cache=False)
    nvda = df.loc[df["Code"] == "NVDA"].iloc[0]

    assert nvda["Close"] == pytest.approx(225.30)
    assert nvda["DailyChangeRatio"] == pytest.approx(0.540)  # 이미 퍼센트 값 — 100 곱하지 않음


def test_screen_weekly_change_ratio_always_nan(monkeypatch):
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(use_cache=False)

    assert df["WeeklyChangeRatio"].isna().all()  # 데이터 소스 한계 — 항상 NaN


def test_screen_zero_marketcap_becomes_nan(monkeypatch):
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(use_cache=False)

    aacb = df.loc[df["Code"] == "AACB"].iloc[0]
    aacg = df.loc[df["Code"] == "AACG"].iloc[0]
    assert pd.isna(aacb["Marcap"])  # "0.00" -> NaN
    assert aacg["Marcap"] == pytest.approx(75995062.0)  # 콤마 없는 숫자 문자열도 정상 파싱


def test_screen_amount_is_volume_times_close(monkeypatch):
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(use_cache=False)
    nvda = df.loc[df["Code"] == "NVDA"].iloc[0]

    assert nvda["Amount"] == pytest.approx(nvda["Volume"] * nvda["Close"])


def test_screen_handles_empty_pctchange(monkeypatch):
    """실측 케이스: 신규 상장 등으로 pctchange가 빈 문자열이면 NaN으로 처리하고 나머지
    행 파싱은 실패하지 않아야 한다."""
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(use_cache=False)
    ghxi = df.loc[df["Code"] == "GHXI"].iloc[0]

    assert pd.isna(ghxi["DailyChangeRatio"])
    assert ghxi["Close"] == pytest.approx(9.95)


def test_screen_min_marcap_filter_excludes_nan(monkeypatch):
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(min_marcap=1.0, use_cache=False)

    # AACB/GHXI는 Marcap이 NaN이라 min_marcap>0 필터에서 자연히 제외된다
    assert set(df["Code"]) == {"NVDA", "AACG"}


def test_screen_min_volume_filter(monkeypatch):
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(_PAYLOAD))

    df = us_screener.screen(min_volume=1_000_000, use_cache=False)

    assert set(df["Code"]) == {"NVDA", "AACG"}


def test_screen_raises_runtime_error_when_rows_empty(monkeypatch):
    empty_payload = {"data": {"asOf": None, "headers": {}, "rows": []}, "message": None, "status": {}}
    monkeypatch.setattr(us_screener.requests, "get", lambda *a, **kw: _FakeResponse(empty_payload))

    with pytest.raises(RuntimeError):
        us_screener.screen(use_cache=False)


def test_screen_raises_runtime_error_on_network_failure(monkeypatch):
    def _raise(*a, **kw):
        raise ConnectionError("boom")

    monkeypatch.setattr(us_screener.requests, "get", _raise)

    with pytest.raises(RuntimeError):
        us_screener.screen(use_cache=False)


@pytest.mark.network
def test_screen_live():
    df = us_screener.screen(use_cache=False)
    assert len(df) >= 1000
    assert set(us_screener._RESULT_COLS) <= set(df.columns)
    assert "NVDA" in df["Code"].values
