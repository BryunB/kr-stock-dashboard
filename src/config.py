"""프로젝트 전역 설정 — 경로, 기본값, 자주 쓰는 티커."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/config.py 기준 한 단계 위가 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")  # 있으면 로드, 없으면 조용히 무시

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (RAW_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 데이터 조회 기본 시작일 (지정 안 하면 여기서부터)
DEFAULT_START = "2015-01-01"

# 캐시 만료 시간(초). 장중 반복 조회를 막되 당일 데이터는 갱신되도록 6시간.
CACHE_TTL_SEC = 6 * 60 * 60

# 뉴스/공시는 가격보다 자주 바뀌므로 캐시를 짧게 둔다 (30분).
NEWS_CACHE_TTL_SEC = 30 * 60

# DART(전자공시시스템) OpenAPI 키. https://opendart.fss.or.kr 에서 무료 발급.
# 프로젝트 루트에 .env 파일을 만들어 DART_API_KEY=... 로 설정한다 (.env는 gitignore됨).
DART_API_KEY = os.environ.get("DART_API_KEY", "")

# 자주 쓰는 지수 심볼 (FinanceDataReader 표기)
INDICES = {
    "KOSPI": "KS11",
    "KOSDAQ": "KQ11",
    "KOSPI200": "KS200",
    "NASDAQ": "IXIC",
    "S&P500": "US500",
    "DOW": "DJI",
    "NIKKEI225": "N225",
    "VIX": "VIX",
}

# 자주 쓰는 환율/원자재
MACRO = {
    "USD/KRW": "USD/KRW",
    "US10YT": "US10YT=X",
    "WTI": "CL=F",
    "GOLD": "GC=F",
}

# 연간 거래일 수 — 연율화(annualization)에 사용
TRADING_DAYS = 252
