"""업비트 KRW 마켓 전종목 시세 스크리닝 — screener.py의 코인 버전.

업비트 ticker 엔드포인트는 마켓 코드를 콤마로 이어붙여 한 번에 여러 종목을 조회할 수 있어서,
KRW 마켓 전체(2026-08 기준 약 280여개)를 요청 1회로 받는다 — screener.py와 동일하게
"요청 수가 스크리닝 대상 종목 수와 무관하게 고정"된다.

반환 컬럼은 screener.screen()과 최대한 동일하게 맞춰서, screener.top_movers()를 그대로
재사용할 수 있게 했다(이 모듈은 top_movers를 따로 정의하지 않는다 — 순수 함수라 코인이든
주식이든 동일하게 동작한다).

**알려진 제약 (데이터 소스 자체의 한계 — 재조사해도 안 바뀐다)**:
- 업비트 API는 "N일 전 스냅샷"을 벌크로 제공하지 않는다. 283개 전 종목의 주간 등락률을
  구하려면 종목마다 개별 캔들 조회가 필요해 "벌크 조회" 원칙에 어긋난다 — 그래서
  WeeklyChangeRatio는 항상 NaN이다.
- 코인마다 발행량 개념이 다르고(무제한 발행, 소각 등) 업비트가 시가총액을 제공하지도
  않아 Marcap도 항상 NaN이다.
"""

from __future__ import annotations

import pandas as pd
import requests

from .cache_utils import cache_path, is_fresh
from .crypto_loader import get_listing

_BASE_URL = "https://api.upbit.com/v1"
_TIMEOUT_SEC = 10

_RESULT_COLS = [
    "Code",
    "Name",
    "Market",
    "Close",
    "DailyChangeRatio",
    "WeeklyChangeRatio",
    "Volume",
    "Amount",
    "Marcap",
]


def screen(use_cache: bool = True) -> pd.DataFrame:
    """업비트 KRW 마켓 전종목 시세 스크리닝 테이블 (컬럼 계약은 screener.screen()과 동일)."""
    path = cache_path("crypto_screen", "KRW")
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    listing = get_listing(use_cache=use_cache)
    if listing.empty:
        raise RuntimeError("업비트 마켓 목록을 가져오지 못했습니다 (네트워크 확인 필요)")

    markets = ",".join(listing["Code"])
    r = requests.get(f"{_BASE_URL}/ticker", params={"markets": markets}, timeout=_TIMEOUT_SEC)
    r.raise_for_status()
    ticker = pd.DataFrame(r.json())
    if ticker.empty:
        raise RuntimeError("업비트 시세를 가져오지 못했습니다 (네트워크 확인 필요)")

    merged = listing.merge(ticker.rename(columns={"market": "Code"}), on="Code", how="inner")
    merged["Market"] = "업비트"
    merged["Close"] = merged["trade_price"]
    merged["DailyChangeRatio"] = merged["signed_change_rate"] * 100
    merged["WeeklyChangeRatio"] = float("nan")
    merged["Volume"] = merged["acc_trade_volume_24h"]
    merged["Amount"] = merged["acc_trade_price_24h"]
    merged["Marcap"] = float("nan")

    result = merged[_RESULT_COLS].reset_index(drop=True)
    if use_cache:
        result.to_parquet(path)
    return result
