from __future__ import annotations

import pandas as pd
import pytest

from src import crypto_screener, screener


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_LISTING = pd.DataFrame(
    [
        {"Code": "KRW-BTC", "Name": "비트코인", "EnglishName": "Bitcoin"},
        {"Code": "KRW-ETH", "Name": "이더리움", "EnglishName": "Ethereum"},
    ]
)

_TICKER_PAYLOAD = [
    {
        "market": "KRW-BTC",
        "trade_price": 90000000.0,
        "signed_change_rate": 0.012,
        "acc_trade_volume_24h": 123.4,
        "acc_trade_price_24h": 5_000_000_000.0,
    },
    {
        "market": "KRW-ETH",
        "trade_price": 4000000.0,
        "signed_change_rate": -0.034,
        "acc_trade_volume_24h": 456.7,
        "acc_trade_price_24h": 2_000_000_000.0,
    },
]


def test_screen_merges_listing_and_ticker(monkeypatch):
    monkeypatch.setattr(crypto_screener, "get_listing", lambda use_cache=True: _LISTING)
    monkeypatch.setattr(crypto_screener.requests, "get", lambda *a, **kw: _FakeResponse(_TICKER_PAYLOAD))

    df = crypto_screener.screen(use_cache=False)

    assert list(df.columns) == crypto_screener._RESULT_COLS
    assert df.loc[df["Code"] == "KRW-BTC", "DailyChangeRatio"].iloc[0] == pytest.approx(1.2)
    assert df.loc[df["Code"] == "KRW-ETH", "DailyChangeRatio"].iloc[0] == pytest.approx(-3.4)
    assert df["WeeklyChangeRatio"].isna().all()  # 데이터 소스 한계 — 항상 NaN
    assert df["Marcap"].isna().all()
    assert (df["Market"] == "업비트").all()


def test_screen_raises_runtime_error_when_listing_empty(monkeypatch):
    monkeypatch.setattr(crypto_screener, "get_listing", lambda use_cache=True: pd.DataFrame())
    with pytest.raises(RuntimeError):
        crypto_screener.screen(use_cache=False)


def test_screen_raises_runtime_error_when_ticker_empty(monkeypatch):
    monkeypatch.setattr(crypto_screener, "get_listing", lambda use_cache=True: _LISTING)
    monkeypatch.setattr(crypto_screener.requests, "get", lambda *a, **kw: _FakeResponse([]))
    with pytest.raises(RuntimeError):
        crypto_screener.screen(use_cache=False)


def test_screener_top_movers_is_reused_for_crypto_columns(monkeypatch):
    """crypto_screener는 top_movers()를 따로 정의하지 않는다 — screener.top_movers()가
    순수 함수라 코인 스크리닝 결과에도 그대로 재사용 가능하다는 걸 확인한다."""
    monkeypatch.setattr(crypto_screener, "get_listing", lambda use_cache=True: _LISTING)
    monkeypatch.setattr(crypto_screener.requests, "get", lambda *a, **kw: _FakeResponse(_TICKER_PAYLOAD))

    df = crypto_screener.screen(use_cache=False)
    top = screener.top_movers(df, by="DailyChangeRatio", n=1)

    assert len(top) == 1
    assert top.iloc[0]["Code"] == "KRW-BTC"


@pytest.mark.network
def test_screen_live():
    df = crypto_screener.screen(use_cache=False)
    assert not df.empty
    assert "KRW-BTC" in df["Code"].values
