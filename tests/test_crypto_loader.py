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
            # UTC = KST 09:00 - 9시간 = 같은 날 00:00. 실제 업비트 응답을 흉내내려면 두
            # 필드가 달라야 한다 — 테스트에서 둘을 같은 값으로 두면 "to에 kst를 잘못
            # 넘겨도" 우연히 통과해버려서 실제로 겪었던 버그(분봉에서 9시간 어긋난 커서)를
            # 못 잡는다.
            "candle_date_time_utc": d.strftime("%Y-%m-%dT00:00:00"),
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

    # 실제로 겪은 버그: candle_date_time_kst를 커서로 넘기면 업비트가 이를 UTC로 해석해
    # 9시간 어긋난(더 최신) 데이터를 돌려준다 — 반드시 candle_date_time_utc를 써야 한다.
    oldest_of_page1 = page1_dates.min()
    assert calls[1]["to"] == oldest_of_page1.strftime("%Y-%m-%dT00:00:00")  # UTC 필드 값
    assert calls[1]["to"] != oldest_of_page1.strftime("%Y-%m-%dT09:00:00")  # KST 필드 값이면 안 됨
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
def test_get_price_live_pagination_has_no_gap_at_page_boundary():
    """400일 조회(>200이라 실제 페이지네이션 발생)에서 200번째 근처에 빠지거나
    중복된 날짜가 없는지 확인 — UTC 커서 수정 전에는 이 경계에서 어긋났다."""
    start = (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    df = crypto_loader.get_price("KRW-BTC", start=start, use_cache=False)
    assert len(df) > 200  # 페이지네이션이 실제로 여러 번 일어났는지 확인
    assert df.index.is_unique
    gaps = df.index.to_series().diff().dropna()
    assert gaps.max() <= pd.Timedelta(days=2)  # 코인은 매일 거래되므로 이틀 이상 공백이면 버그


@pytest.mark.network
def test_get_listing_live():
    df = crypto_loader.get_listing(use_cache=False)
    assert not df.empty
    assert "KRW-BTC" in df["Code"].values


# ----------------------------------------------------------- get_minute_price


def _make_minute_batch(timestamps_kst):
    return [
        {
            "market": "KRW-BTC",
            "candle_date_time_kst": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "candle_date_time_utc": (ts - pd.Timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%S"),
            "opening_price": 100.0,
            "high_price": 101.0,
            "low_price": 99.0,
            "trade_price": 100.5,
            "candle_acc_trade_volume": 1.0,
        }
        for ts in timestamps_kst
    ]


def test_get_minute_price_rejects_unsupported_unit():
    with pytest.raises(ValueError):
        crypto_loader.get_minute_price("KRW-BTC", unit=7, use_cache=False)


def test_get_minute_price_does_not_normalize_time(monkeypatch):
    """일봉과 달리 분봉은 자정으로 절삭하면 안 된다 — 실제 분 단위 시각이 인덱스에 남아야 한다."""
    ts = pd.date_range("2026-08-14 10:00:00", periods=5, freq="5min")[::-1]
    batch = _make_minute_batch(ts)
    monkeypatch.setattr(crypto_loader.requests, "get", lambda *a, **kw: _FakeResponse(batch))

    df = crypto_loader.get_minute_price("KRW-BTC", unit=5, count=5, use_cache=False)

    assert len(df) == 5
    assert df.index[0].time() != pd.Timestamp("00:00:00").time()


def test_get_minute_price_paginates_with_utc_cursor(monkeypatch):
    page1_ts = pd.date_range("2026-08-14 10:00:00", periods=200, freq="1min")[::-1]
    page2_ts = pd.date_range("2026-08-14 06:30:00", periods=50, freq="1min")[::-1]
    pages = [_make_minute_batch(page1_ts), _make_minute_batch(page2_ts)]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        return _FakeResponse(pages.pop(0))

    monkeypatch.setattr(crypto_loader.requests, "get", fake_get)
    monkeypatch.setattr(crypto_loader.time, "sleep", lambda _s: None)

    df = crypto_loader.get_minute_price("KRW-BTC", unit=1, count=250, use_cache=False)

    assert len(calls) == 2
    oldest_of_page1 = page1_ts.min()
    assert calls[1]["to"] == (oldest_of_page1 - pd.Timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%S")
    assert len(df) == 250


def test_get_minute_price_count_capped_at_max(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        return _FakeResponse([])  # 상한 검증만 목적이라 빈 응답으로 바로 끝낸다

    monkeypatch.setattr(crypto_loader.requests, "get", fake_get)
    crypto_loader.get_minute_price("KRW-BTC", unit=1, count=999_999, use_cache=False)

    assert calls[0]["count"] == crypto_loader._MAX_CANDLES_PER_REQUEST


@pytest.mark.network
def test_get_minute_price_live_pagination_has_no_gap():
    """300개(>200이라 페이지네이션 발생) 5분봉에서 경계에 공백/중복이 없는지 확인."""
    df = crypto_loader.get_minute_price("KRW-BTC", unit=5, count=300, use_cache=False)
    assert len(df) == 300
    assert df.index.is_unique
    gaps = df.index.to_series().diff().dropna()
    assert gaps.max() == pd.Timedelta(minutes=5)
