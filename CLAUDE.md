# CLAUDE.md

이 파일은 이 저장소에서 작업할 때 따라야 할 공통 규칙이다. 새 기능(스크리닝, 백테스트, 리포트 등)을 추가할 때마다 이 문서를 기준으로 삼고, 새로운 제약이나 결정이 생기면 이 문서를 업데이트한다.

## 프로젝트 개요

KOSPI/KOSDAQ/NASDAQ 등 주가 데이터를 수집·분석하는 작업환경. 데이터 계층(`src/`)과 그 위의 Streamlit 대시보드(`app.py`), 탐색용 노트북(`notebooks/`)으로 구성된다.

## 환경

- Python 3.14.7, Windows, `.venv` 기준. 패키지 관리는 **pip + requirements.txt** (uv/poetry 아님).
- pandas 3.0 — Copy-on-Write가 기본이고 `df.append()`, `fillna(method=...)` 같은 구 API가 제거됐다. 오래된 예제 코드를 그대로 쓰지 말 것.
- **새 의존성을 추가하기 전에 실제로 이 환경에서 동작하는지 스모크 테스트로 먼저 확인한다.** `pip install`이 성공했다고 끝난 게 아니다 — 실제 호출까지 해봐야 한다. (`pykrx`가 설치는 됐지만 실호출에서 로그인 요구로 막혔던 사례 참고 — 아래 "알려진 제약" 참고.)
- 새 라이브러리가 기존 pandas/numpy 버전을 강제로 바꾸면(`pip install`이 다운그레이드하면) 반드시 인지하고, 필요 없어지면 제거 후 버전을 원복한다.

## 아키텍처

```
src/
  config.py       경로, 기본값, 심볼 별칭(INDICES, MACRO) — 다른 모든 모듈이 여기서 상수를 가져온다
  cache_utils.py  parquet 캐시 공용 유틸 (cache_path, is_fresh)
  data_loader.py  개별 종목/지수 조회 (FDR 래퍼 + 캐시)
  screener.py     전종목 대량 스크리닝 (개별 조회가 아니라 벌크 소스 사용)
  indicators.py   기술적 지표 + 성과 지표 (순수 함수, pandas Series/DataFrame in-out)
  charts.py       Plotly 인터랙티브 차트 빌더 (Streamlit 비의존)
  news.py         네이버 금융 뉴스 스크래핑 + 요약(리드 문단 발췌, LLM 아님)
  sentiment.py    키워드 기반 감성 판정 (POSITIVE_WORDS/NEGATIVE_WORDS 매칭)
  dart.py         DART 전자공시 OpenAPI 연동 (API 키 필요, 없으면 DartKeyMissing)
  predictor.py    가격 예측 (릿지 회귀, 종목별 즉석 학습 + 홀드아웃 검증) — 투자 조언 아님
  plotting.py     matplotlib 한글 폰트 설정 (노트북 전용)
app.py            Streamlit 대시보드 진입점
notebooks/        탐색용. src를 import해서 재사용 (%autoreload 사용)
tests/            pytest
```

**규칙**: UI 코드(`app.py`)에서 `FinanceDataReader`나 외부 URL을 직접 호출하지 않는다. 항상 `src/data_loader.py` 또는 `src/screener.py`를 거친다. 데이터 로직과 화면 로직을 분리해야 나중에 노트북·다른 대시보드에서도 재사용 가능하다.

`app.py`가 계속 커지면(스크리닝 외 다른 화면이 추가되면) 한 파일에 다 넣지 말고 `pages/` 디렉토리 기반 멀티페이지 구조로 전환을 고려한다.

## 데이터 접근 규칙

- **캐시는 항상 쓴다.** 새 조회 함수를 추가할 때 `cache_utils.cache_path(kind, key)` + `is_fresh()` 패턴을 따르고, `use_cache: bool = True` 파라미터로 강제 갱신 여지를 남긴다. TTL은 `config.CACHE_TTL_SEC`.
- **전종목/대량 데이터가 필요하면 종목별 반복 호출을 절대 하지 않는다.** 수천 종목을 하나씩 조회하면 느리고 상대 서버에도 부담이다. 먼저 벌크로 가져올 방법을 찾는다 (예: `screener.py`가 KRX 일자별 스냅샷을 통째로 받아 조인하는 방식 — 요청 수가 종목 수와 무관하게 고정됨).
- 심볼 표기는 `config.INDICES` / `config.MACRO` 별칭을 따른다. 새 지수·매크로 지표를 자주 쓰게 되면 거기 추가한다.
- 네트워크 호출을 테스트/조사할 때도 반복문 규모를 작게 유지한다 (예: 과거 데이터 존재 여부 확인 시 400일치를 다 찔러보지 않고 40~50일 정도로 충분히 검증).

## 알려진 제약 (재조사 방지용)

- **pykrx는 못 쓴다.** 최신 버전의 벌크 조회 함수(`get_market_ohlcv_by_ticker` 등)가 `KRX_ID`/`KRX_PW` 환경변수 기반 로그인을 요구한다. 로그인 정보가 없으므로 배제.
- **`data.krx.co.kr`에 직접 POST로 스크래핑하는 것도 막혀 있다** (세션을 워밍업해도 `LOGOUT` 응답). KRX가 자체 방어를 강화한 상태.
- 대신 FinanceDataReader가 실제로 쓰는 **GitHub 미러** `FinanceData/fdr_krx_data_cache`의 일자별 스냅샷 CSV(`data/listing/krx/{YYYY-MM-DD}.csv`)를 직접 지정한 날짜로 내려받는 방식이 로그인 없이 동작한다. `screener.py`가 이 방식을 쓴다. 과거 스냅샷은 최소 45일 이상 존재하는 것을 확인했다(그보다 오래된 비교가 필요하면 존재 여부를 먼저 확인할 것).
- `pandas 3.0` + `streamlit`을 같이 쓰면 `streamlit`이 `pyarrow` 상한 버전을 요구해 최신 `pyarrow`보다 낮은 버전이 깔릴 수 있다 (현재 24.0.0). 문제는 없지만 버전 충돌 경고가 뜨면 이게 원인일 가능성이 높다.
- **네이버 금융 뉴스 스크래핑**(`news.py`)은 `Referer` 헤더(`https://finance.naver.com/item/news.naver?code={종목코드}`)가 없으면 빈 결과("검색된 뉴스가 없습니다")를 반환한다. 또한 응답 인코딩이 `euc-kr`이라 `r.encoding = "euc-kr"`을 명시해야 한글이 깨지지 않는다. 기사 원문은 목록 페이지가 아니라 `n.news.naver.com/mnews/article/{office_id}/{article_id}`에 있다 (목록의 `news_read.naver` 링크는 JS로 이 URL에 리다이렉트만 함 — office_id/article_id를 href에서 파싱해 바로 이 URL을 구성하면 리다이렉트 왕복을 생략할 수 있다).
- **DART API**는 무료지만 각자 https://opendart.fss.or.kr 에서 이메일 인증 후 키를 발급받아야 한다 — Claude가 대신 발급받을 수 없다. 키는 `.env`(gitignore됨)의 `DART_API_KEY`로 관리하고 `config.py`가 `python-dotenv`로 로드한다. 종목코드(6자리)와 DART의 corp_code(8자리)는 다른 체계라 `corpCode.xml`(zip) 전체를 내려받아 매핑해야 한다 — 이것도 상장사 전체를 한 번에 주므로 종목별 반복 호출이 아니다.
- **API 키는 절대 코드에 하드코딩하거나 커밋되는 파일에 넣지 않는다.** 항상 `.env` + `config.py`의 `os.environ.get(...)` 패턴을 따른다. 키가 없을 때는 조용히 실패하지 말고 (`DartKeyMissing`처럼) 무엇을 어디서 발급받아야 하는지 알려주는 예외/메시지를 낸다.
- **뉴스 감성의 과거 히스토리는 조회할 방법이 없다** (뉴스 소스가 최신 기사 목록만 제공). `predictor.py`가 뉴스 피처를 쓰려면 `news.log_daily_sentiment()`로 매 실행마다 그날의 평균 감성을 `data/raw/news_sentiment/{종목코드}.parquet`에 누적 기록하는 수밖에 없다 — 기록 시작 전 과거는 중립(0)으로 채운다. 이 로그는 `data/raw/`에 있어 `.gitignore`가 "재생성 가능한 캐시"로 취급하지만, 실제로는 한 번 지우면 복구 불가능한 유일한 데이터다(다른 data/raw·data/cache 파일과 성격이 다름).

## 가격 예측(ML) 기능 작성 규칙

- **투자 조언처럼 보이지 않게 한다.** "매수/매도 추천" 같은 문구를 쓰지 않는다. UI에 "참고용 통계 모델, 투자 조언 아님" 경고를 항상 표시한다.
- **정확도는 반드시 홀드아웃(학습에 안 쓴 구간)으로 검증한 값만 보여준다.** 학습 데이터 자체에 대한 적합도(in-sample)를 정확도인 것처럼 보여주지 않는다 — 실제보다 좋아 보이게 부풀리는 것이기 때문.
- 개별 종목의 일봉 데이터는 많아야 수백~수천 행이라 복잡한 모델(딥러닝 등)은 과적합 위험이 크다. 릿지 회귀처럼 단순한 모델을 우선한다.
- 종목마다 그때그때 새로 학습하고 모델을 저장해두지 않는다 — 데이터가 자주 바뀌고 종목별 모델을 다 저장/관리하는 비용이 이득보다 크다.

## UI/차트 컨벤션

- **언어: 코드의 docstring·주석·UI 라벨은 모두 한국어**로 작성한다 (변수/함수명은 영어 유지).
- **색상 관행**: 상승 = 빨강(`#ef4444`), 하락 = 파랑(`#3b82f6`) — 미국식(초록/빨강)과 반대이므로 새 차트를 만들 때 헷갈리지 않도록 주의.
- 대시보드(웹)는 **Plotly**로 인터랙티브하게, 노트북은 **matplotlib + `plotting.setup()`**(한글 폰트 적용)으로 그린다.
- 스케일이 다른 여러 시계열(개별주 vs 지수)을 한 축에 겹쳐 그릴 때는 절대값이 아니라 **시작값 기준 리베이스**(`charts._rebase` 참고)해서 비교한다.

## 테스트 규칙

- `pytest` 사용. 외부 네트워크가 필요한 테스트는 `@pytest.mark.network`로 표시한다 (`pytest -m "not network"`로 오프라인 실행 가능).
- 오프라인 테스트는 합성 데이터로 검증한다 (`tests/test_dashboard.py`의 `_fake_ohlcv` 같은 헬퍼 패턴 참고). 실제 API 응답 형태를 가정하지 말고 함수의 계산 로직 자체를 검증한다.
- **Streamlit 앱을 수정하면**: 먼저 `streamlit.testing.v1.AppTest`로 헤드리스 검증(위젯 조작 시 예외 없는지)을 하고, 그다음 실제로 `streamlit run app.py`를 띄워 브라우저에서 확인한다. 스크린샷이 안 되는 상황이면 `get_page_text` / 콘솔 로그로 대체 확인하되, 반드시 실제 실행 결과를 눈으로(혹은 텍스트로) 확인하고 나서 완료로 보고한다 — 코드가 "그럴듯해 보인다"는 이유로 완료 처리하지 않는다.
- **검증용 서버를 띄우기 전에 포트 8501이 이미 쓰이고 있는지 확인한다.** 사용자가 직접 `streamlit run app.py`를 실행해 둔 세션일 수 있다 — 그 프로세스를 죽이지 말고, 검증은 다른 포트(예: 8502, 8503)로 별도 실행한다. 검증이 끝나면 자신이 띄운 포트의 프로세스만 정리한다.
- 새 기능을 추가하면 관련 테스트도 같이 추가한다. 커밋 전 `pytest` 전체 통과를 확인한다.
- **린트/포맷은 ruff를 쓴다** (`ruff.toml` 설정, `notebooks/`는 제외 — `%autoreload` 매직이 import보다 먼저 오는 게 정상이라). 커밋 전 `ruff check .` / `ruff format .` 확인.

## 새 기능 추가 체크리스트

1. 필요한 데이터가 `data_loader.py`/`screener.py`에 이미 있는지 확인. 없으면 캐시 규칙을 지키며 추가.
2. 대량 조회가 필요하면 종목별 반복이 아니라 벌크 방법을 먼저 찾는다.
3. 계산/데이터 로직은 `src/` 모듈에, `app.py`는 그걸 얇게 호출만 한다.
4. 새 외부 라이브러리가 필요하면 추가 전에 스모크 테스트로 실동작을 확인한다.
5. 오프라인/네트워크 테스트를 분리해서 작성한다.
6. Streamlit 화면이면 AppTest → 실제 서버 실행 → 브라우저(또는 텍스트) 확인 순서로 검증한다.
7. `requirements.txt`와 `README.md`를 새 의존성/사용법 기준으로 갱신한다.
