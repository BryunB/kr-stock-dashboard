"""KOSPI+KOSDAQ 전종목 일간·주간 등락률 스크리닝.

종목별로 개별 조회하면(~2,900회 호출) 너무 느리고 서버에도 부담이 크다.
대신 FinanceDataReader가 실제로 사용하는 KRX 일자별 스냅샷 미러
(GitHub: FinanceData/fdr_krx_data_cache)를 날짜를 지정해 두 번
(최근 거래일 / N영업일 전) 내려받아 Code 기준으로 조인하는 방식을 쓴다.
요청 수가 전종목 스크리닝 범위와 무관하게 항상 2회로 고정된다.
"""

from __future__ import annotations

import pandas as pd

from .cache_utils import cache_path, is_fresh

_GH_LISTING_URL = (
    "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache"
    "/refs/heads/master/data/listing/krx/{date}.csv"
)

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


def _snapshot_on(date: pd.Timestamp) -> pd.DataFrame | None:
    """해당 날짜의 KRX 전종목 스냅샷. 휴장일 등으로 없으면 None."""
    url = _GH_LISTING_URL.format(date=date.strftime("%Y-%m-%d"))
    try:
        df = pd.read_csv(url, index_col=0, dtype={"Code": str})
    except Exception:
        return None
    return df.reset_index(drop=True)


def _nearest_snapshot(around: pd.Timestamp, max_back_days: int = 10):
    """around 기준 가장 가까운 과거(포함) 거래일의 (날짜, 스냅샷)을 찾는다."""
    for i in range(max_back_days):
        d = around - pd.Timedelta(days=i)
        snap = _snapshot_on(d)
        if snap is not None and not snap.empty:
            return d, snap
    return None, None


_KRX_MARKETS = ("KOSPI", "KOSDAQ")  # KONEX(초소형 시장)는 기본 제외


def screen(
    market: str = "ALL",
    days_back: int = 7,
    min_marcap: float = 0.0,
    min_volume: float = 0.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """일간·주간 등락률이 포함된 전종목 스크리닝 테이블을 반환한다.

    market: 'ALL'(KOSPI+KOSDAQ) | 'KOSPI' | 'KOSDAQ'
    days_back: 비교 기준 며칠 전(달력 기준)인지. 기본 7일 = 1주일 전.
    min_marcap: 이 시가총액(원) 미만인 종목은 제외. 초소형주 노이즈 제거용.
    min_volume: 이 거래량 미만(거래정지 등 0거래량 포함)인 종목은 제외.
    """
    key = f"{market}|{days_back}|{min_marcap}|{min_volume}"
    path = cache_path("screen", key)
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    latest_date, latest = _nearest_snapshot(pd.Timestamp.today())
    if latest is None:
        raise RuntimeError("최근 거래일 스냅샷을 찾지 못했습니다 (네트워크 확인 필요)")

    _, past = _nearest_snapshot(latest_date - pd.Timedelta(days=days_back))
    if past is None:
        raise RuntimeError(f"{days_back}일 전 근처 스냅샷을 찾지 못했습니다")

    merged = latest.merge(
        past[["Code", "Close"]].rename(columns={"Close": "ClosePrev"}),
        on="Code",
        how="left",
    )
    merged["WeeklyChangeRatio"] = (merged["Close"] - merged["ClosePrev"]) / merged["ClosePrev"] * 100
    merged = merged.rename(columns={"ChagesRatio": "DailyChangeRatio"})

    markets = _KRX_MARKETS if market == "ALL" else (market,)
    merged = merged[merged["Market"].isin(markets)]
    if min_marcap:
        merged = merged[merged["Marcap"] >= min_marcap]
    if min_volume:
        merged = merged[merged["Volume"] >= min_volume]

    result = merged[_RESULT_COLS].reset_index(drop=True)

    if use_cache:
        result.to_parquet(path)
    return result


def top_movers(
    df: pd.DataFrame,
    by: str = "DailyChangeRatio",
    n: int = 30,
    ascending: bool = False,
) -> pd.DataFrame:
    """등락률 기준 상위 n개. ascending=True면 하락률 상위(급락주)."""
    return df.dropna(subset=[by]).sort_values(by, ascending=ascending).head(n).reset_index(drop=True)
