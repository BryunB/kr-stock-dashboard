"""주가 데이터 수집·분석 패키지.

서브모듈을 여기서 미리 import하지 않는다 — 예전엔 전부 eager import했는데, 그러면
`from src import data_loader` 한 줄만 써도 `plotting.py`(matplotlib)·`charts.py`(plotly)까지
전부 로드돼서, 이 두 패키지가 없는 환경(예: GitHub Actions의 requirements-trading.txt,
대시보드/노트북 전용 패키지를 일부러 뺀 환경)에서 매매 스크립트가 아예 못 뜨는 문제가 있었다.
호출부는 이미 전부 `from src import data_loader` 같은 명시적 서브모듈 import를 쓰므로
(app.py, scripts/run_daily_trading.py, README 예제 전부 이 방식), 여기서 미리 로드해둘 필요가 없다.
"""
