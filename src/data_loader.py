"""FinanceDataReader 래퍼 — 로컬 parquet 캐시를 얹어 반복 조회를 줄인다.

FDR은 매 호출마다 원격 서버를 때리므로, 노트북에서 셀을 여러 번 돌리면
느릴 뿐 아니라 상대 서버에 부담이 된다. 여기서는 (심볼, 기간) 단위로
parquet에 캐시하고 CACHE_TTL_SEC 이내면 로컬에서 읽는다.
"""

from __future__ import annotations

from collections.abc import Iterable

import FinanceDataReader as fdr
import pandas as pd

from . import config
from .cache_utils import cache_path, clear_cache, is_fresh

__all__ = [
    "get_price",
    "get_closes",
    "get_listing",
    "find_symbol",
    "clear_cache",
]


def get_price(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """단일 심볼의 OHLCV를 DataFrame으로 반환한다.

    symbol: '005930'(삼성전자), 'KS11'(코스피), 'AAPL', 'USD/KRW' 등
            config.INDICES / config.MACRO 의 키('KOSPI' 등)를 써도 된다.
    """
    symbol = config.INDICES.get(symbol, config.MACRO.get(symbol, symbol))
    start = start or config.DEFAULT_START

    path = cache_path("price", f"{symbol}|{start}|{end}")
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    df = fdr.DataReader(symbol, start, end)
    if not df.empty:
        df.to_parquet(path)
    return df


def get_closes(
    symbols: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """여러 심볼의 종가만 모아 wide 형태(열=심볼)로 반환한다.

    거래일이 다른 시장(KRX vs NASDAQ)을 섞으면 한쪽에 NaN이 생긴다.
    수익률 비교 시에는 .dropna() 로 공통 거래일만 남기고 쓰는 편이 안전하다.
    """
    series = {}
    for sym in symbols:
        df = get_price(sym, start, end, use_cache=use_cache)
        if df.empty:
            continue
        series[sym] = df["Close"]

    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def get_listing(market: str = "KRX", use_cache: bool = True) -> pd.DataFrame:
    """상장 종목 리스트.

    market: 'KRX', 'KOSPI', 'KOSDAQ', 'NASDAQ', 'NYSE', 'S&P500' 등
    """
    path = cache_path("listing", market)
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    df = fdr.StockListing(market)
    if not df.empty:
        df.to_parquet(path)
    return df


def find_symbol(keyword: str, market: str = "KRX") -> pd.DataFrame:
    """종목명 일부로 심볼을 찾는다. 예: find_symbol('삼성')"""
    listing = get_listing(market)
    name_col = next((c for c in ("Name", "Symbol") if c in listing.columns), listing.columns[0])
    hit = listing[listing[name_col].astype(str).str.contains(keyword, case=False, na=False)]
    keep = [c for c in ("Code", "Symbol", "Name", "Market", "Sector") if c in hit.columns]
    return hit[keep] if keep else hit
