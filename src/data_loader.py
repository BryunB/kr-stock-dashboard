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
    "get_usdkrw_rate",
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


def get_usdkrw_rate(use_cache: bool = True) -> float:
    """최근 USD/KRW 종가 1개. 해외증시 금액을 원화로 환산할 때 쓴다(모니터링 페이지의
    통화 토글, 모의투자의 해외증시 매매 체결 등) — 두 곳이 각자 fdr 심볼 문자열을
    들고 있지 않도록 여기 한 곳에만 둔다. 실패하면(네트워크 등) NaN을 돌려주고,
    호출부가 그에 맞게 폴백해야 한다(예: 달러 표기 유지, 해외증시 매매 보류)."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df = get_price("USD/KRW", start=start, use_cache=use_cache)
    return float(df["Close"].iloc[-1]) if not df.empty else float("nan")


def find_symbol(keyword: str, market: str = "KRX") -> pd.DataFrame:
    """종목코드 또는 종목명 일부로 심볼을 찾는다. 예: find_symbol('삼성'), find_symbol('005930')"""
    listing = get_listing(market)
    name_col = next((c for c in ("Name", "Symbol") if c in listing.columns), listing.columns[0])
    name_hit = listing[listing[name_col].astype(str).str.contains(keyword, case=False, na=False)]

    # 코드 컬럼(6자리 티커 등)도 부분 일치로 함께 검색한다 — 이게 없으면 "005930"처럼
    # 코드를 그대로 입력해도 검색 결과가 항상 비어버린다.
    code_col = next((c for c in ("Code", "Symbol") if c in listing.columns), None)
    if code_col is not None:
        code_hit = listing[listing[code_col].astype(str).str.contains(keyword, case=False, na=False)]
        hit = pd.concat([code_hit, name_hit]).drop_duplicates()
    else:
        hit = name_hit

    keep = [c for c in ("Code", "Symbol", "Name", "Market", "Sector") if c in hit.columns]
    return hit[keep] if keep else hit
