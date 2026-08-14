"""업비트 공개 API 래퍼 — data_loader.py의 코인 버전.

업비트 REST API는 인증 없이 쓸 수 있고, 일봉 캔들 엔드포인트가 Open/High/Low/Close/Volume
구조라 indicators.py/charts.py/predictor.py가 기대하는 컬럼 계약을 그대로 만족한다 — 그
세 모듈은 이 사실을 몰라도 되고 실제로 수정하지 않았다. 코인은 24시간 연속거래라 주식과
달리 휴장일 캘린더를 신경 쓸 필요가 없다(매일 캔들이 존재).

**알려진 제약**: 업비트 일봉 캔들 엔드포인트는 요청당 최대 200개까지만 준다. 그보다 긴
구간을 조회하려면 `to` 파라미터로 과거 방향으로 페이지네이션해야 한다 — 이 모듈이
내부적으로 처리한다(get_price 호출부는 신경 쓸 필요 없음).
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from . import config
from .cache_utils import cache_path, is_fresh

_BASE_URL = "https://api.upbit.com/v1"
_MAX_CANDLES_PER_REQUEST = 200  # 업비트 캔들 API 1회 요청 상한 (일봉/분봉 공통)
_TIMEOUT_SEC = 10

_MINUTE_UNITS = (1, 3, 5, 10, 15, 30, 60, 240)
_MAX_MINUTE_CANDLES = 1000  # 그 이상은 요청 수가 과도해진다(5회+) — UI에서 쓸 일도 없음
_MINUTE_CACHE_TTL_SEC = 5 * 60  # 분봉은 일봉(config.CACHE_TTL_SEC=6시간)보다 훨씬 빨리 바뀐다

__all__ = ["get_price", "get_minute_price", "get_listing", "find_symbol"]


def get_listing(use_cache: bool = True) -> pd.DataFrame:
    """업비트 KRW 마켓 전체 종목 목록. columns: Code(마켓코드, 'KRW-BTC' 등), Name(한글명), EnglishName."""
    path = cache_path("crypto_listing", "KRW")
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    r = requests.get(f"{_BASE_URL}/market/all", params={"isDetails": "false"}, timeout=_TIMEOUT_SEC)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df[df["market"].str.startswith("KRW-")].reset_index(drop=True)
    df = df.rename(columns={"market": "Code", "korean_name": "Name", "english_name": "EnglishName"})

    if not df.empty and use_cache:
        df.to_parquet(path)
    return df


def find_symbol(keyword: str) -> pd.DataFrame:
    """코인명(한글/영문) 또는 마켓코드 일부로 검색. 예: find_symbol('비트코인'), find_symbol('BTC')."""
    listing = get_listing()
    if listing.empty:
        return listing

    hit_cols = ["Code", "Name", "EnglishName"]
    mask = False
    for col in hit_cols:
        mask = mask | listing[col].astype(str).str.contains(keyword, case=False, na=False)
    return listing[mask].drop_duplicates().reset_index(drop=True)


def get_price(
    market: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """단일 코인 마켓의 일봉 OHLCV를 DataFrame으로 반환한다.

    market: 'KRW-BTC'(비트코인), 'KRW-ETH'(이더리움) 등 업비트 마켓 코드.
    반환 컬럼은 data_loader.get_price()와 동일(Open/High/Low/Close/Volume, DatetimeIndex)해서
    indicators.add_all()/charts.build_chart()/predictor.train_and_predict()를 그대로 쓸 수 있다.
    """
    start = start or config.DEFAULT_START

    path = cache_path("crypto_price", f"{market}|{start}|{end}")
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()

    rows: list[dict] = []
    to_param: str | None = None
    while True:
        params = {"market": market, "count": _MAX_CANDLES_PER_REQUEST}
        if to_param:
            params["to"] = to_param
        r = requests.get(f"{_BASE_URL}/candles/days", params=params, timeout=_TIMEOUT_SEC)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)

        oldest = pd.Timestamp(batch[-1]["candle_date_time_kst"])
        if oldest <= start_ts or len(batch) < _MAX_CANDLES_PER_REQUEST:
            break
        # to 파라미터는 UTC 기준으로 해석된다(업비트 API 문서) — candle_date_time_kst를
        # 그대로 넘기면 9시간 어긋난 커서가 되어 실제로는 최신 데이터를 다시 받게 된다.
        # 일봉은 캔들 시각이 항상 KST 09:00:00으로 고정돼 있어 이 버그가 우연히 날짜
        # 경계를 벗어나지 않았을 뿐(실측 확인) — 분봉/시간봉에서는 실제로 어긋난다.
        to_param = batch[-1]["candle_date_time_utc"]
        time.sleep(0.1)  # 페이지네이션 연속 호출 시 업비트 서버 부담을 줄인다

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["candle_date_time_kst"]).dt.normalize()
    df = df.rename(
        columns={
            "opening_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "trade_price": "Close",
            "candle_acc_trade_volume": "Volume",
        }
    )
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    if not df.empty and use_cache:
        df.to_parquet(path)
    return df


def get_minute_price(market: str, unit: int = 1, count: int = 200, use_cache: bool = True) -> pd.DataFrame:
    """단일 코인 마켓의 분봉/시간봉 OHLCV(가장 최근 count개)를 반환한다.

    unit: 분 단위 — 1/3/5/10/15/30/60(1시간)/240(4시간)만 업비트가 지원한다.
    count: 가장 최근 캔들 몇 개를 가져올지(최대 _MAX_MINUTE_CANDLES). get_price()와 달리
    날짜 범위(start/end)가 아니라 "최근 N개" 방식이다 — 분봉은 구간이 길어지면 요청 수가
    급격히 늘어나서(200개/요청) 과거로 무한정 조회하는 UI는 지원하지 않기로 했다. 캐시
    TTL도 일봉보다 훨씬 짧다(분 단위로 계속 갱신되므로 오래 캐시하면 옛날 시세를 보여준다).
    """
    if unit not in _MINUTE_UNITS:
        raise ValueError(f"지원하지 않는 분봉 단위입니다: {unit} (지원: {_MINUTE_UNITS})")
    count = min(count, _MAX_MINUTE_CANDLES)

    path = cache_path("crypto_minute", f"{market}|{unit}|{count}")
    if use_cache and is_fresh(path, ttl_sec=_MINUTE_CACHE_TTL_SEC):
        return pd.read_parquet(path)

    rows: list[dict] = []
    to_param: str | None = None
    remaining = count
    while remaining > 0:
        batch_size = min(remaining, _MAX_CANDLES_PER_REQUEST)
        params = {"market": market, "count": batch_size}
        if to_param:
            params["to"] = to_param
        r = requests.get(f"{_BASE_URL}/candles/minutes/{unit}", params=params, timeout=_TIMEOUT_SEC)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        remaining -= len(batch)
        if len(batch) < batch_size:
            break
        to_param = batch[-1]["candle_date_time_utc"]  # get_price()와 동일한 이유로 UTC 커서를 쓴다
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["candle_date_time_kst"])  # 분봉은 정규화(자정 절삭)하면 안 된다
    df = df.rename(
        columns={
            "opening_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "trade_price": "Close",
            "candle_acc_trade_volume": "Volume",
        }
    )
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if not df.empty and use_cache:
        df.to_parquet(path)
    return df
