from __future__ import annotations

import pandas as pd
import pytest

from src import crypto_loader


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _make_batch(dates, price=100.0):
    return [
        {
            "market": "KRW-BTC",
            "candle_date_time_kst": d.strftime("%Y-%m-%dT09:00:00"),
            "opening_price": price,
            "high_price": price + 1,
            "low_price": price - 1,
            "trade_price": price + 0.5,
            "candle_acc_trade_volume": 1.0,
        }
        for d in dates
    ]


def test_get_listing_filters_krw_and_renames_columns(monkeypatch):
    payload = [
        {"market": "KRW-BTC", "korean_name": "비트코인", "english_name": "Bitcoin"},
        {"market": "BTC-ETH", "korean_name": "이더리움", "english_name": "Ethereum"},  # KRW 아님 -> 제외
        {"market": "KRW-ETH", "korean_name": "이더리움", "english_name": "Ethereum"},
    ]
    monkeypatch.setattr(crypto_loader.requests, "get", lambda *a, **kw: _FakeResponse(payload))

    df = crypto_loader.get_listing(use_cache=False)

    assert list(df.columns) == ["Code", "Name", "EnglishName"]
    assert set(df["Code"]) == {"KRW-BTC", "KRW-ETH"}


def test_find_symbol_matches_code_name_and_english(monkeypatch):
    listing = pd.DataFrame(
        [
            {"Code": "KRW-BTC", "Name": "비트코인", "EnglishName": "Bitcoin"},
            {"Code": "KRW-ETH", "Name": "이더리움", "EnglishName": "Ethereum"},
        ]
    )
    monkeypatch.setattr(crypto_loader, "get_listing", lambda use_cache=True: listing)

    assert crypto_loader.find_symbol("비트코인")["Code"].tolist() == ["KRW-BTC"]
    assert crypto_loader.find_symbol("ETH")["Code"].tolist() == ["KRW-ETH"]
    assert crypto_loader.find_symbol("KRW-BTC")["Code"].tolist() == ["KRW-BTC"]
    assert crypto_loader.find_symbol("없는코인").empty


def test_get_price_maps_columns_single_page(monkeypatch):
    dates = pd.date_range("2026-08-01", periods=5, freq="D")[::-1]
    batch = _make_batch(dates)
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        return _FakeResponse(batch)

    monkeypatch.setattr(crypto_loader.requests, "get", fake_get)

    df = crypto_loader.get_price("KRW-BTC", start="2026-08-01", use_cache=False)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 5
    assert len(calls) == 1  # 배치가 상한(200)보다 짧으면 한 번만 호출하고 멈춘다
    assert df.index.is_monotonic_increasing


def test_get_price_paginates_until_short_batch(monkeypatch):
    page1_dates = pd.date_range("2026-02-01", periods=200, freq="D")[::-1]
    page2_dates = pd.date_range("2026-01-01", periods=31, freq="D")[::-1]
    pages = [_make_batch(page1_dates), _make_batch(page2_dates)]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        return _FakeResponse(pages.pop(0))

    monkeypatch.setattr(crypto_loader.requests, "get", fake_get)
    monkeypatch.setattr(crypto_loader.time, "sleep", lambda _s: None)

    # end를 미래로 고정 — 기본값(오늘)을 쓰면 page1_dates가 실행 시점의 "오늘"을 넘어갈 수
    # 있어(예: 2026-02-01 + 200일), 실행 시점에 따라 결과 행 수가 달라지는 깨지기 쉬운
    # 테스트가 된다.
    df = crypto_loader.get_price("KRW-BTC", start="2020-01-01", end="2030-01-01", use_cache=False)

    assert len(calls) == 2
    assert "to" in calls[1]  # 두 번째 요청은 반드시 이전 배치의 가장 오래된 시각을 커서로 넘긴다
    assert len(df) == 200 + 31


def test_get_price_empty_response_returns_empty_df(monkeypatch):
    monkeypatch.setattr(crypto_loader.requests, "get", lambda *a, **kw: _FakeResponse([]))
    df = crypto_loader.get_price("KRW-NOPE", use_cache=False)
    assert df.empty


@pytest.mark.network
def test_get_price_live_bitcoin():
    df = crypto_loader.get_price("KRW-BTC", start="2026-07-01", use_cache=False)
    assert not df.empty
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(df.columns)


@pytest.mark.network
def test_get_listing_live():
    df = crypto_loader.get_listing(use_cache=False)
    assert not df.empty
    assert "KRW-BTC" in df["Code"].values
