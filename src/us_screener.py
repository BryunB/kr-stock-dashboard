"""나스닥(등) 미국 증시 전종목 시세 스크리닝 — screener.py의 해외증시 버전.

나스닥 공개 스크리너 API는 거래소 전종목(NASDAQ 기준 약 4,100여개)을 요청 1회로
돌려줘서, screener.py(KRX 스냅샷)·crypto_screener.py(업비트 ticker)와 동일하게
"요청 수가 스크리닝 대상 종목 수와 무관하게 고정"된다.

반환 컬럼은 screener.screen()과 최대한 동일하게 맞춰서, screener.top_movers()를 그대로
재사용할 수 있게 했다(crypto_screener.py와 동일한 이유로 이 모듈도 top_movers를 따로
정의하지 않는다).

**알려진 제약 (데이터 소스 자체의 한계 — 재조사해도 안 바뀐다)**:
- 이 API는 오늘자 스냅샷만 주고 N일 전 히스토리를 벌크로 제공하지 않는다. 전종목의
  주간 등락률을 구하려면 종목마다 개별 조회가 필요해 "벌크 조회" 원칙에 어긋난다 —
  그래서 WeeklyChangeRatio는 항상 NaN이다(crypto_screener.py와 동일한 사정).
- 거래대금(Amount)에 해당하는 필드가 API에 없어 Volume * Close로 근사 계산한다 —
  당일 평균가가 아니라 마감가 기준이라 실제 거래대금과는 오차가 있는 근사치다.
- marketCap 필드는 스팩·워런트·라이츠 등 사업가치 개념이 없는 증권에서 "0.00"으로
  온다(실측: NASDAQ 4,118건 중 588건). 실제 시가총액이 0인 종목은 없으므로 0은
  "데이터 없음" 신호로 보고 NaN으로 바꾼다.
"""

from __future__ import annotations

import pandas as pd
import requests

from .cache_utils import cache_path, is_fresh

_BASE_URL = "https://api.nasdaq.com/api/screener/stocks"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_TIMEOUT_SEC = 15

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
    "Sector",
    "Industry",
]


def screen(
    exchange: str = "NASDAQ",
    min_marcap: float = 0.0,
    min_volume: float = 0.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """해외증시(나스닥 등) 전종목 스크리닝 테이블을 반환한다.

    exchange: 나스닥 스크리너 API의 exchange 파라미터 값 (예: 'NASDAQ'). 응답의 Market
        컬럼은 API가 돌려주는 값이 아니라 이 인자를 그대로 채운다.
    min_marcap: 이 시가총액(달러) 미만인 종목은 제외. Marcap이 NaN인 종목(위 "알려진
        제약" 참고)은 이 필터에 걸리지 않는다 — pandas 비교에서 NaN은 False가 되어
        자연히 걸러진다(screener.py와 동일한 패턴).
    min_volume: 이 거래량 미만인 종목은 제외.
    """
    path = cache_path("us_screen", exchange)
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    params = {
        "tableonly": "true",
        "exchange": exchange,
        "download": "true",
        "limit": "10000",
    }
    try:
        r = requests.get(_BASE_URL, params=params, headers=_HEADERS, timeout=_TIMEOUT_SEC)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        raise RuntimeError(f"나스닥 스크리너 응답을 가져오지 못했습니다 (네트워크 확인 필요): {e}") from e

    rows = (payload.get("data") or {}).get("rows")
    if not rows:
        raise RuntimeError("나스닥 스크리너 결과가 비어 있습니다 (네트워크 확인 필요)")

    raw = pd.DataFrame(rows)

    merged = pd.DataFrame()
    merged["Code"] = raw["symbol"]
    merged["Name"] = raw["name"]
    merged["Market"] = exchange
    # 스펙상 lastsale/pctchange/volume은 N/A 없이 항상 채워진다고 실측됐지만, 실제로는
    # (신규 상장 등 전일 종가가 없는) 극소수 행에서 빈 문자열이 섞여 있었다 — 문자열 치환
    # 후 astype(float) 대신 pd.to_numeric(errors="coerce")로 그런 값은 NaN으로 흘려보낸다.
    merged["Close"] = pd.to_numeric(raw["lastsale"].str.replace("$", "", regex=False), errors="coerce")
    merged["DailyChangeRatio"] = pd.to_numeric(
        raw["pctchange"].str.replace("%", "", regex=False), errors="coerce"
    )
    merged["WeeklyChangeRatio"] = float("nan")
    merged["Volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0).astype(int)
    merged["Amount"] = merged["Volume"] * merged["Close"]
    marcap = pd.to_numeric(raw["marketCap"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    merged["Marcap"] = marcap.replace(0.0, float("nan"))
    merged["Sector"] = raw["sector"]
    merged["Industry"] = raw["industry"]

    if min_marcap:
        merged = merged[merged["Marcap"] >= min_marcap]
    if min_volume:
        merged = merged[merged["Volume"] >= min_volume]

    result = merged[_RESULT_COLS].reset_index(drop=True)
    if use_cache:
        result.to_parquet(path)
    return result
