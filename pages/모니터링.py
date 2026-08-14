"""모니터링 — 국내증시/코인 스크리닝·차트·예측·뉴스. app.py의 확장판.

app.py는 지금 지인에게 공유 중인 배포라 한 글자도 수정하지 않는다(CLAUDE.md 원칙). 이
페이지는 demo_app.py 전용 새 페이지로, 좌상단 시장 선택(국내증시/코인)에 따라 화면 내용을
다르게 그린다. 국내증시 모드는 app.py와 동일한 기능이고, 코인 모드는 src/crypto_*.py
(업비트 공개 API + 네이버 뉴스 검색)를 쓴다.

indicators.py/charts.py/predictor.py는 OHLCV 컬럼 계약만 맞으면 시장과 무관하게 그대로
동작하는 걸 이미 실측 확인했다 — 그래서 이 세 모듈은 분기가 없다. 분기가 필요한 곳은
유니버스 조회(screener vs crypto_screener), 종목 검색(data_loader vs crypto_loader),
뉴스(news vs crypto_news — 종목코드 대신 코인명 키워드), 공시(DART는 코인에 없어 탭 자체를
숨긴다), 그리고 코인은 시가총액·주간등락률 데이터가 없어 해당 탭/옵션을 뺀다.
"""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts, config, crypto_loader, crypto_news, crypto_screener, dart, news, predictor, screener
from src import data_loader as dl
from src import indicators as ind

DATE_RANGES = {"1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "2년": 730, "3년": 1095, "전체": None}
SMA_CHOICES = (5, 20, 60, 120)

UP_COLOR, DOWN_COLOR, FLAT_COLOR = "#ef4444", "#3b82f6", "#6b7280"  # 실측 시세용 (원색)
UP_SOFT, DOWN_SOFT, FLAT_SOFT = "#cc6666", "#668dcc", "#9ca3af"  # 예측값용 (파스텔)
_SENT_COLOR = {"긍정": UP_COLOR, "중립": FLAT_COLOR, "부정": DOWN_COLOR}

_CRYPTO_BENCHMARKS = {"비트코인": "KRW-BTC", "이더리움": "KRW-ETH", "리플": "KRW-XRP"}
_CRYPTO_DEFAULT = ("KRW-BTC", "비트코인")
_BAR_PERIODS = {"일봉": None, "주봉": "W", "월봉": "ME"}  # 차트 전용 — 시세 요약·예측은 항상 일봉 기준
_CHART_TYPES = ("캔들", "라인", "하이킨아시")
# 분봉/시간봉은 업비트 API 자체(코인)에만 있다 — 주식(FinanceDataReader/GitHub 스냅샷)은
# 일봉 이하 데이터가 아예 없어서 국내증시 모드에서는 이 옵션 자체를 보여주지 않는다.
_CRYPTO_MINUTE_PERIODS = {
    "1분": 1,
    "3분": 3,
    "5분": 5,
    "10분": 10,
    "15분": 15,
    "30분": 30,
    "1시간": 60,
    "4시간": 240,
}
_MINUTE_CANDLE_COUNT = 300  # 최근 N개만 (업비트 요청 상한 200개/회 대비 최대 2회 페이지네이션)
_STOCK_DEFAULT = ("005930", "삼성전자")


def _change_html(diff: float, pct: float, *, soft: bool = False) -> str:
    up, down, flat = (UP_SOFT, DOWN_SOFT, FLAT_SOFT) if soft else (UP_COLOR, DOWN_COLOR, FLAT_COLOR)
    color = up if diff > 0 else down if diff < 0 else flat
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "―"
    return (
        f"<span style='color:{color};font-weight:600;font-size:0.95rem'>"
        f"{arrow} {diff:+,.0f}원 ({pct:+.2f}%)</span>"
    )


def _section_title(text: str) -> None:
    """카드 제목을 스크롤 가능한 컨테이너 상단에 고정(sticky)해서 그린다.

    st.container(height=N)는 내용이 넘치면 내부 스크롤이 생기는데, 제목이 맨 위 콘텐츠로만
    있으면 스크롤할 때 같이 밀려 올라가 사라진다. position:sticky + top:0으로 컨테이너
    스크롤 영역 상단에 고정한다 — 배경은 스크롤되는 내용이 뒤로 비치지 않도록 불투명하게
    높이고 블러를 얹었다(연한 반투명이면 표/차트 내용이 제목 뒤로 겹쳐 보인다).
    """
    st.markdown(
        "<div style='background:rgba(127,127,127,0.55);backdrop-filter:blur(6px);"
        "-webkit-backdrop-filter:blur(6px);border-radius:6px;"
        "padding:0.4rem 0.7rem;margin-bottom:0.5rem;font-weight:700;"
        "font-size:1.05rem;line-height:1.3;position:sticky;top:0;z-index:5;"
        f"'>{text}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="전종목 시세 불러오는 중...")
def _screen(market: str, days_back: int, min_marcap: float, min_volume: float) -> pd.DataFrame:
    return screener.screen(market=market, days_back=days_back, min_marcap=min_marcap, min_volume=min_volume)


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="코인 시세 불러오는 중...")
def _crypto_screen() -> pd.DataFrame:
    return crypto_screener.screen()


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="가격 데이터 불러오는 중...")
def _price(symbol: str, start: str) -> pd.DataFrame:
    return dl.get_price(symbol, start=start)


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="코인 가격 데이터 불러오는 중...")
def _crypto_price(market: str, start: str) -> pd.DataFrame:
    return crypto_loader.get_price(market, start=start)


@st.cache_data(ttl=300, show_spinner="분봉/시간봉 불러오는 중...")  # 일봉보다 훨씬 짧은 캐시 — 자주 바뀐다
def _crypto_minute_price(market: str, unit: int, count: int) -> pd.DataFrame:
    return crypto_loader.get_minute_price(market, unit=unit, count=count)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="뉴스 불러오는 중...")
def _news(code: str, n: int) -> pd.DataFrame:
    return news.fetch_news_with_sentiment(code, n=n)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="뉴스 불러오는 중...")
def _crypto_news(keyword: str, n: int) -> pd.DataFrame:
    return crypto_news.fetch_news_with_sentiment(keyword, n=n)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="공시 불러오는 중...")
def _dart(code: str) -> pd.DataFrame:
    return dart.fetch_disclosures(code)


def _safe_predict_advanced(
    price_df: pd.DataFrame, horizon: int, sentiment_hist_full: pd.DataFrame | None
) -> dict:
    try:
        return predictor.train_and_predict_advanced(
            price_df, horizon=horizon, sentiment_hist_full=sentiment_hist_full
        )
    except Exception as e:
        return {"error": f"예측 중 오류가 발생했습니다: {e}"}


# ------------------------------------------------------------------ 컴팩트 레이아웃용 전역 CSS (app.py와 동일)
st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
h1, h2, h3, h4, h5 {margin-top: 0.1rem; margin-bottom: 0.3rem;}
div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
[data-testid="stMetricLabel"] {font-size: 0.88rem;}
[data-testid="stMetricDelta"] {font-size: 0.88rem;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {padding: 4px 10px; font-size: 0.92rem;}
div[data-testid="stWidgetLabel"] p {font-size: 0.88rem; margin-bottom: 0.1rem;}
.stButton button {padding: 0.25rem 0.7rem; font-size: 0.88rem;}
hr {margin: 0.5rem 0;}
p, .stCaption, .stMarkdown, label, span {font-size: 0.92rem;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("##### 🖥️ 모니터링")

# ==================================================================== 맨 위: 시장 선택
top_left, _top_rest = st.columns([2, 8])
with top_left:
    market_type = st.radio("시장 구분", ["국내증시", "코인"], horizontal=True, key="market_type")
is_crypto = market_type == "코인"
if is_crypto:
    st.caption(
        "⚠️ 코인 시세는 업비트 공개 API, 뉴스는 네이버 뉴스 키워드 검색 기반입니다. "
        "주간 등락률·시가총액은 데이터 소스 한계로 제공하지 않습니다."
    )

left_col, right_col = st.columns([3, 7], gap="medium")

_TOP_ROW_HEIGHT = 800
_BOTTOM_ROW_HEIGHT = 380
# 스크리닝 테이블 높이 상한 — _TOP_ROW_HEIGHT(800)는 우측 "종목 상세" 박스와 시작줄을
# 맞추려고 고정한 값이라, 좌측 "주가 요약"에는 필터·탭 등을 빼면 여유 공간이 남는다.
# 예전엔 260으로 낮게 고정해뒀더니 기본 표시개수(30개)에서도 테이블 아래로 빈 여백이
# 크게 남았다 — 그 위의 필터 행 개수가 시장별로 달라(국내증시가 코인보다 한 줄 더 많음)
# 상한도 다르게 잡는다. 브라우저로 직접 확인하며 미세조정한 값은 아니라, 실제로 보면서
# 더 조정이 필요할 수 있다.
_STOCK_TABLE_HEIGHT = 480
_CRYPTO_TABLE_HEIGHT = 520

# ==================================================================== 좌측 상단: 스크리닝
with left_col, st.container(key="summary", border=True, height=_TOP_ROW_HEIGHT):
    _section_title("🪙 코인 요약" if is_crypto else "📊 주가 요약")

    if is_crypto:
        top_n = st.selectbox("표시개수", [10, 20, 30, 50, 100], index=2)
        if st.button("🔄", help="새로고침 (캐시 초기화)"):
            st.cache_data.clear()
            st.rerun()

        try:
            universe = _crypto_screen()
        except Exception as e:
            st.error(f"코인 시세를 불러오지 못했습니다: {e}")
            st.stop()

        market_label = "업비트 KRW"

        def _render_table(ranked, key, value_col, value_label, fmt, scale=1.0):
            display = ranked.rename(columns={"Name": "종목명", "Close": "종가"}).copy()
            display[value_label] = ranked[value_col] / scale
            col_config = {
                "종가": st.column_config.NumberColumn(format="%,.0f원"),
                value_label: st.column_config.NumberColumn(format=fmt),
            }
            return st.dataframe(
                display[["종목명", "종가", value_label]],
                width="stretch",
                hide_index=True,
                height=min(30 * (len(display) + 1), _CRYPTO_TABLE_HEIGHT),
                column_config=col_config,
                on_select="rerun",
                selection_mode="single-row",
                key=key,
            )

        rise_tab, amount_tab = st.tabs(["📈 등락률", "💰 거래대금"])
        picks = []

        with rise_tab:
            direction = st.radio("방향", ["상승", "하락"], horizontal=True)
            ranked_rise = screener.top_movers(
                universe, by="DailyChangeRatio", n=top_n, ascending=(direction == "하락")
            )
            st.caption(
                f"{market_label} {len(universe):,}종목 중 24시간 {direction}률 상위 {len(ranked_rise)}개"
            )
            picks.append(
                (
                    "tbl_rise",
                    ranked_rise,
                    _render_table(ranked_rise, "tbl_rise", "DailyChangeRatio", "등락%", "%.2f%%"),
                )
            )

        with amount_tab:
            ranked_amount = screener.top_movers(universe, by="Amount", n=top_n, ascending=False)
            st.caption(f"{market_label} {len(universe):,}종목 중 24시간 거래대금 상위 {len(ranked_amount)}개")
            picks.append(
                (
                    "tbl_amount",
                    ranked_amount,
                    _render_table(ranked_amount, "tbl_amount", "Amount", "거래대금", "%,.0f억원", scale=1e8),
                )
            )
    else:
        fc1, fc2 = st.columns(2)
        with fc1:
            market = st.selectbox(
                "시장",
                ["ALL", "KOSPI", "KOSDAQ"],
                format_func=lambda m: {"ALL": "전체", "KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}[m],
            )
        with fc2:
            top_n = st.selectbox("표시개수", [10, 20, 30, 50, 100], index=2)

        fc3, fc4, fc5 = st.columns([1.3, 1.3, 0.6])
        with fc3:
            min_marcap_eok = st.selectbox(
                "최소 시총",
                [0, 100, 300, 500, 1000, 3000],
                index=2,
                format_func=lambda v: f"{v}억+" if v else "시총 전체",
            )
        with fc4:
            min_volume = st.selectbox(
                "최소 거래량",
                [0, 1000, 5000, 10000, 50000],
                index=1,
                format_func=lambda v: f"{v:,}주+" if v else "거래량 전체",
            )
        with fc5:
            st.markdown("<div style='height:1.55em'></div>", unsafe_allow_html=True)
            if st.button("🔄", help="새로고침 (캐시 초기화)"):
                st.cache_data.clear()
                dl.clear_cache()
                st.rerun()

        try:
            universe = _screen(market, 7, min_marcap_eok * 1e8, float(min_volume))
        except Exception as e:
            st.error(f"스크리닝 데이터를 불러오지 못했습니다: {e}")
            st.stop()

        market_label = "KOSPI+KOSDAQ" if market == "ALL" else market

        def _render_table(ranked, key, value_col, value_label, fmt, scale=1.0):
            display = ranked.rename(columns={"Name": "종목명", "Close": "종가"}).copy()
            display[value_label] = ranked[value_col] / scale
            col_config = {
                "종가": st.column_config.NumberColumn(format="%,d원"),
                value_label: st.column_config.NumberColumn(format=fmt),
            }
            return st.dataframe(
                display[["종목명", "종가", value_label]],
                width="stretch",
                hide_index=True,
                height=min(30 * (len(display) + 1), _STOCK_TABLE_HEIGHT),
                column_config=col_config,
                on_select="rerun",
                selection_mode="single-row",
                key=key,
            )

        rise_tab, amount_tab, marcap_tab = st.tabs(["📈 상승률", "💰 거래대금", "🏢 시가총액"])
        picks = []

        with rise_tab:
            c1, c2 = st.columns(2)
            with c1:
                basis = st.radio("기준", ["일간", "주간", "월간"], horizontal=True)
            with c2:
                direction = st.radio("방향", ["상승", "하락"], horizontal=True)
            basis_col = {
                "일간": "DailyChangeRatio",
                "주간": "WeeklyChangeRatio",
                "월간": "MonthlyChangeRatio",
            }[basis]
            ranked_rise = screener.top_movers(
                universe, by=basis_col, n=top_n, ascending=(direction == "하락")
            )
            st.caption(
                f"{market_label} {len(universe):,}종목 중 {basis} {direction}률 상위 {len(ranked_rise)}개"
            )
            picks.append(
                (
                    "tbl_rise",
                    ranked_rise,
                    _render_table(ranked_rise, "tbl_rise", basis_col, f"{basis}%", "%.2f%%"),
                )
            )

        with amount_tab:
            ranked_amount = screener.top_movers(universe, by="Amount", n=top_n, ascending=False)
            st.caption(f"{market_label} {len(universe):,}종목 중 거래대금 상위 {len(ranked_amount)}개")
            picks.append(
                (
                    "tbl_amount",
                    ranked_amount,
                    _render_table(ranked_amount, "tbl_amount", "Amount", "거래대금", "%,.0f억원", scale=1e8),
                )
            )

        with marcap_tab:
            ranked_marcap = screener.top_movers(universe, by="Marcap", n=top_n, ascending=False)
            st.caption(f"{market_label} {len(universe):,}종목 중 시가총액 상위 {len(ranked_marcap)}개")
            picks.append(
                (
                    "tbl_marcap",
                    ranked_marcap,
                    _render_table(ranked_marcap, "tbl_marcap", "Marcap", "시가총액", "%,.0f억원", scale=1e8),
                )
            )

    # 시장 전환 시 이전 선택이 다른 시장 코드로 남아있으면 안 되므로, 선택 상태를 시장별로 분리해 둔다.
    _sel_code_key = "selected_code_crypto" if is_crypto else "selected_code_stock"
    _sel_name_key = "selected_name_crypto" if is_crypto else "selected_name_stock"

    for tbl_key, ranked_df, event in picks:
        if not event.selection.rows:
            continue
        picked = ranked_df.iloc[event.selection.rows[0]]
        if st.session_state.get(f"_last_{tbl_key}_{market_type}") != picked["Code"]:
            st.session_state[f"_last_{tbl_key}_{market_type}"] = picked["Code"]
            st.session_state[_sel_code_key] = picked["Code"]
            st.session_state[_sel_name_key] = picked["Name"]

# ==================================================================== 우측 상단: 종목 상세
with right_col:
    with st.container(key="detail", border=True, height=_TOP_ROW_HEIGHT):
        selected_code = st.session_state.get(_sel_code_key)
        selected_name = st.session_state.get(_sel_name_key, "")

        _detail_title = "🪙 코인 상세" if is_crypto else "📈 종목 상세"
        if selected_code:
            _detail_title += f" · {selected_name}"
        _section_title(_detail_title)

        search_col, period_col, bar_col, idx_col = st.columns([2, 1, 1, 2])
        with search_col:
            placeholder = "예: KRW-BTC, 비트코인" if is_crypto else "예: 005930, 삼성전자"
            manual = st.text_input("코드/이름 검색", value="", placeholder=placeholder)
        with period_col:
            period_label = st.selectbox("조회 기간", list(DATE_RANGES.keys()), index=3)
        with bar_col:
            bar_options = (
                list(_CRYPTO_MINUTE_PERIODS.keys()) + list(_BAR_PERIODS.keys())
                if is_crypto
                else list(_BAR_PERIODS.keys())
            )
            bar_label = st.selectbox("봉 주기", bar_options, index=len(bar_options) - 3 if is_crypto else 0)
        with idx_col:
            if is_crypto:
                idx_sel = st.multiselect("코인 비교", list(_CRYPTO_BENCHMARKS.keys()), default=[])
            else:
                idx_sel = st.multiselect("지수 비교", list(config.INDICES.keys()), default=["KOSPI"])

        if manual.strip():
            hits = crypto_loader.find_symbol(manual.strip()) if is_crypto else dl.find_symbol(manual.strip())
            code_col = "Code" if "Code" in hits.columns else "Symbol"
            if not hits.empty:
                options = {
                    f"{row[code_col]} · {row['Name']}": row[code_col] for _, row in hits.head(20).iterrows()
                }
                pick = st.selectbox("검색 결과", list(options.keys()))
                selected_code = options[pick]
                selected_name = pick.split(" · ", 1)[1]
            else:
                st.warning("검색 결과가 없습니다.")

        if not selected_code:
            selected_code, selected_name = _CRYPTO_DEFAULT if is_crypto else _STOCK_DEFAULT

        # 예전엔 이 이름(코드)를 차트 자체의 Plotly title로 그렸는데, 매물대/레인지셀렉터가
        # 추가되면서 위쪽 공간이 빡빡해져 차트 상단(캔들/레인지셀렉터 버튼)과 겹쳐 보였다 —
        # 검색창 바로 아래에 별도 Streamlit 텍스트로 빼서 차트 영역과 아예 분리했다.
        st.markdown(f"##### {selected_name} ({selected_code})")

        days_back = DATE_RANGES[period_label]
        start_date = (
            config.DEFAULT_START
            if days_back is None
            else (pd.Timestamp.today() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
        )

        price_df = (
            _crypto_price(selected_code, start_date) if is_crypto else _price(selected_code, start_date)
        )
        if price_df.empty:
            st.error(f"'{selected_code}' 가격 데이터를 찾을 수 없습니다.")
            st.stop()

        # ------------------------------------------------------ 시세 요약 (원 단위)
        _last = price_df.iloc[-1]
        _last_date = price_df.index[-1]
        _close = float(_last["Close"])
        _prev_close = float(price_df["Close"].iloc[-2]) if len(price_df) >= 2 else None

        _urow = universe[universe["Code"] == selected_code]
        _amount_approx = _urow.empty
        _amount_stale = False
        _marcap = None
        if not _urow.empty:
            _amount = float(_urow.iloc[0]["Amount"])
            _marcap_raw = _urow.iloc[0]["Marcap"]
            _marcap = float(_marcap_raw) if pd.notna(_marcap_raw) else None
            if not is_crypto:
                _snap_vol = float(_urow.iloc[0]["Volume"])
                _live_vol = float(_last["Volume"])
                _amount_stale = _live_vol > 0 and abs(_snap_vol - _live_vol) / _live_vol > 0.01
        else:
            _amount = _close * float(_last["Volume"])

        _vol_unit = selected_code.replace("KRW-", "") if is_crypto else "주"
        _vol_fmt = f"{float(_last['Volume']):,.4f}" if is_crypto else f"{float(_last['Volume']):,.0f}"

        _r1 = st.columns(8)
        _r1[0].metric("현재가", f"{_close:,.0f}원")
        if _prev_close:
            _diff = _close - _prev_close
            _r1[0].markdown(_change_html(_diff, _diff / _prev_close * 100), unsafe_allow_html=True)
        _r1[1].metric("시가", f"{float(_last['Open']):,.0f}원")
        _r1[2].metric("고가", f"{float(_last['High']):,.0f}원")
        _r1[3].metric("저가", f"{float(_last['Low']):,.0f}원")
        _r1[4].metric("전일종가", f"{_prev_close:,.0f}원" if _prev_close else "—")
        _r1[5].metric("변동폭", f"{float(_last['High']) - float(_last['Low']):,.0f}원")
        _r1[6].metric("거래량", f"{_vol_fmt}{_vol_unit}")
        _r1[7].metric("거래대금" if not is_crypto else "24H 거래대금", f"{_amount / 1e8:,.0f}억원")

        _notes = [f"{_last_date:%Y-%m-%d (%a)} 기준"]
        if _marcap:
            _notes.append(f"시총 {_marcap / 1e8:,.0f}억원")
        if is_crypto:
            _notes.append("거래대금은 최근 24시간 누적(업비트 기준)")
        elif _amount_approx:
            _notes.append("거래대금 추정치")
        elif _amount_stale:
            _notes.append("거래대금·시총은 KRX 스냅샷 기준")
        _notes.append("상승=빨강 / 하락=파랑")
        st.caption(" · ".join(_notes))

        # 봉 주기: 시세 요약·예측은 항상 price_df(일봉)를 쓰고, 차트만 바꾼다 —
        # 업비트도 상단 현재가 티커는 실시간 그대로 두고 캔들 굵기만 바꾼다.
        if bar_label in _CRYPTO_MINUTE_PERIODS:
            # 분봉/시간봉은 일봉을 리샘플링해서 만들 수 없다(더 잘게 쪼개는 건 원본에
            # 없던 정보를 만드는 것) — 업비트 분봉 API를 따로 호출한다. 코인 모드에서만
            # 이 옵션이 보이므로 is_crypto 분기가 필요 없다.
            chart_price_df = _crypto_minute_price(
                selected_code, _CRYPTO_MINUTE_PERIODS[bar_label], _MINUTE_CANDLE_COUNT
            )
            st.caption(f"분봉/시간봉은 최근 {_MINUTE_CANDLE_COUNT}개까지만 제공됩니다(업비트 API 한계).")
        else:
            bar_rule = _BAR_PERIODS[bar_label]
            chart_price_df = ind.resample_ohlcv(price_df, bar_rule) if bar_rule else price_df
        if chart_price_df.empty:  # 데이터가 비면(너무 짧은 상장 이력 등) 안전하게 일봉으로 폴백
            chart_price_df = price_df

        enriched = ind.add_all(chart_price_df)

        # 컬럼을 잘게 쪼갤수록(예전엔 체크박스 10개를 한 줄에) 좁은 화면에서 Streamlit이
        # 모바일 스택 레이아웃으로 전환해버려서 PC 화면 비율이 세로로 길게 깨졌다 —
        # 개별 체크박스 대신 멀티셀렉트로 묶어서 한 줄당 컬럼 수를 4개로 줄였다.
        ma_col, ind_col, ctype_col, log_col = st.columns([1.5, 2.1, 1.3, 0.9])
        with ma_col:
            ma_sel = st.multiselect("이동평균", SMA_CHOICES, default=[20, 60])
        with ind_col:
            ind_sel = st.multiselect(
                "보조지표",
                ["볼린저밴드", "거래량", "RSI", "MACD", "스토캐스틱", "일목균형표", "매물대"],
                default=["거래량"],
            )
        with ctype_col:
            chart_type_label = st.selectbox("차트 유형", _CHART_TYPES, index=0)
        with log_col:
            st.markdown("<div style='height:1.55em'></div>", unsafe_allow_html=True)
            log_y = st.checkbox("로그축", value=False)

        show_bb = "볼린저밴드" in ind_sel
        show_vol = "거래량" in ind_sel
        show_rsi = "RSI" in ind_sel
        show_macd = "MACD" in ind_sel
        show_stoch = "스토캐스틱" in ind_sel
        show_ichimoku = "일목균형표" in ind_sel
        show_vp = "매물대" in ind_sel

        if show_bb:
            bbcol1, bbcol2 = st.columns(2)
            bb_window = bbcol1.number_input("볼밴드 기간", min_value=5, max_value=120, value=20, step=1)
            bb_std = bbcol2.number_input("볼밴드 표준편차", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
            enriched = enriched.drop(columns=["upper", "lower", "pct_b"], errors="ignore").join(
                ind.bollinger(enriched["Close"], window=int(bb_window), num_std=bb_std)[
                    ["upper", "lower", "pct_b"]
                ]
            )

        sma_windows = list(ma_sel)
        for w in sma_windows:
            col = f"sma{w}"
            if col not in enriched.columns:
                enriched[col] = ind.sma(enriched["Close"], w)

        if show_stoch:
            enriched = enriched.join(ind.stochastic(enriched))
        if show_ichimoku:
            enriched = enriched.join(ind.ichimoku(enriched))

        chart_source = ind.heikin_ashi(enriched) if chart_type_label == "하이킨아시" else enriched
        chart_type = "line" if chart_type_label == "라인" else "candle"
        vp_df = ind.volume_profile(chart_source) if show_vp else None

        overlays = {}
        for label in idx_sel:
            ov_symbol = _CRYPTO_BENCHMARKS[label] if is_crypto else label
            idx_df = _crypto_price(ov_symbol, start_date) if is_crypto else _price(ov_symbol, start_date)
            if not idx_df.empty:
                overlays[label] = idx_df["Close"]

        fig = charts.build_chart(
            chart_source,
            title="",
            sma_windows=tuple(sma_windows),
            show_bollinger=show_bb,
            show_volume=show_vol,
            show_rsi=show_rsi,
            show_macd=show_macd,
            index_overlays=overlays or None,
            base_height=320,
            panel_height=85,
            chart_type=chart_type,
            show_rangeselector=True,
            log_y=log_y,
            show_stochastic=show_stoch,
            show_ichimoku=show_ichimoku,
            volume_profile=vp_df,
            crosshair=True,
            drag_pan=True,
            drawing_tools=True,
        )
        # scrollZoom은 Figure 속성이 아니라 렌더링 옵션이라 여기서 켠다 — 업비트처럼
        # 마우스 휠로 확대/축소, 드래그로 이동(위 drag_pan=True)하는 조작감을 맞춘다.
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

        summary = ind.summary(price_df["Close"])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("수익률", f"{summary['total_return'] * 100:.1f}%")
        m2.metric("CAGR", f"{summary['cagr'] * 100:.1f}%")
        m3.metric("변동성", f"{summary['volatility'] * 100:.1f}%")
        m4.metric("샤프", f"{summary['sharpe']:.2f}")
        m5.metric("MDD", f"{summary['max_drawdown'] * 100:.1f}%")

_HORIZON_PRESETS = (5, 15, 30)


def _apply_horizon_preset(preset: int) -> None:
    """체크박스가 방금 체크됐을 때만(체크 해제 시엔 무시) 입력값을 그 값으로 맞추고,
    나머지 프리셋 체크박스는 꺼서 라디오처럼 하나만 선택된 것처럼 보이게 한다."""
    if st.session_state.get(f"horizon_preset_{preset}"):
        st.session_state["custom_horizon"] = preset
        for other in _HORIZON_PRESETS:
            if other != preset:
                st.session_state[f"horizon_preset_{other}"] = False


# ==================================================================== 좌측 하단: 가격 예측
with left_col, st.container(key="predict", border=True, height=_BOTTOM_ROW_HEIGHT):
    _section_title("💹 가격 예측")
    st.caption(
        "⚠️ 참고용 추정치이며 투자 조언이 아닙니다. Ridge/RandomForest/GradientBoosting을 "
        "시계열 교차검증으로 비교한 모델로 예측합니다(보유한 전체 기간 데이터 사용)."
    )

    try:
        if is_crypto:
            news.log_sentiment_from_news(selected_code, _crypto_news(selected_name, 10))
        else:
            news.log_sentiment_from_news(selected_code, _news(selected_code, 10))
    except Exception:
        pass

    st.markdown("**몇 거래일 후를 예측할까요?**")
    pcols = st.columns(len(_HORIZON_PRESETS))
    for col, preset in zip(pcols, _HORIZON_PRESETS, strict=True):
        with col:
            st.checkbox(
                f"{preset}일",
                key=f"horizon_preset_{preset}",
                on_change=_apply_horizon_preset,
                args=(preset,),
            )

    st.session_state.setdefault("custom_horizon", 5)
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.number_input("거래일 수", min_value=1, max_value=60, step=1, key="custom_horizon")
    with hcol2:
        st.markdown("<div style='height:1.55em'></div>", unsafe_allow_html=True)
        if st.button("조회", key="custom_horizon_query"):
            horizon = int(st.session_state["custom_horizon"])
            with st.spinner("여러 모델을 교차검증하며 심층 학습하는 중..."):
                full_price_df = (
                    _crypto_price(selected_code, config.DEFAULT_START)
                    if is_crypto
                    else _price(selected_code, config.DEFAULT_START)
                )
                sentiment_hist_full = news.sentiment_history_full(selected_code)
                st.session_state["adv_pred"] = {
                    "code": selected_code,
                    "horizon": horizon,
                    "result": _safe_predict_advanced(
                        full_price_df, horizon=horizon, sentiment_hist_full=sentiment_hist_full
                    ),
                }

    adv_pred = st.session_state.get("adv_pred")
    if adv_pred and adv_pred["code"] == selected_code:
        horizon = adv_pred["horizon"]
        pred = adv_pred["result"]
        st.markdown(f"**{horizon}거래일 후 예측**")
        if "error" in pred:
            st.info(pred["error"])
        else:
            pred_col, info_col = st.columns([1, 1])
            with pred_col:
                st.metric(pred["target_date"].strftime("%m-%d(%a)"), f"{pred['predicted_price']:,.0f}원")
            with info_col:
                st.markdown("<div style='height:0.35em'></div>", unsafe_allow_html=True)
                st.markdown(
                    _change_html(
                        pred["predicted_price"] - pred["last_close"],
                        pred["predicted_return"] * 100,
                        soft=True,
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{pred['best_model']} · 검증{pred['n_holdout']}일 · MAE {pred['mae']:,.0f}원 · "
                    f"MAPE {pred['mape'] * 100:.1f}% · 방향적중 {pred['directional_accuracy'] * 100:.0f}%"
                )

            # ---------------------------------------------- 실제 추이(실선) + 예측 추이(대시선)
            lookback = max(30, horizon * 5)  # 예측 기간에 비례해 과거 구간도 넉넉히 보여준다
            hist = price_df.tail(lookback)
            pred_color = (
                UP_SOFT
                if pred["predicted_price"] > pred["last_close"]
                else DOWN_SOFT
                if pred["predicted_price"] < pred["last_close"]
                else FLAT_SOFT
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=hist.index, y=hist["Close"], name="실제", line=dict(width=1.8, color="#0ea5e9"))
            )
            fig.add_trace(
                go.Scatter(
                    x=[hist.index[-1], pred["target_date"]],
                    y=[hist["Close"].iloc[-1], pred["predicted_price"]],
                    name=f"{horizon}거래일 후 예측",
                    line=dict(width=2, color=pred_color, dash="dash"),
                )
            )
            fig.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "예측선은 마지막 실제 종가와 예측값을 직선으로 이은 것으로, 그 사이의 실제 경로를 뜻하지 않습니다."
            )

            with st.expander("모델 상세 (교차검증 비교 · 피처 영향도)"):
                st.markdown(
                    f"선정: {pred['best_model']} · 학습 {pred['n_train']}행 / "
                    f"홀드아웃 {pred['n_holdout']}행 · 뉴스 감성 히스토리 {pred['news_days']}일 누적"
                )
                cv_df = pd.DataFrame(
                    {
                        "모델": list(pred["cv_scores"].keys()),
                        "교차검증 MAE(수익률)": list(pred["cv_scores"].values()),
                    }
                )
                st.dataframe(cv_df, hide_index=True, width="stretch")
                st.dataframe(
                    pred["feature_importance"].rename(columns={"label": "설명", "coef": "중요도"})[
                        ["설명", "중요도"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "교차검증 점수는 모델을 고르는 데만 쓰였고, 위 정확도는 모델 선정에 관여하지 않은 "
                    "마지막 홀드아웃 구간 기준입니다. 중요도 값은 모델별로 계산 방식이 다릅니다"
                    "(회귀계수 / 트리 중요도 / 순열 중요도)."
                )
    else:
        st.caption("거래일 수를 정하고 조회를 누르면 예측 결과가 여기 표시됩니다.")

# ==================================================================== 우측 하단: 뉴스 & 공시
with right_col:
    with st.container(key="news", border=True, height=_BOTTOM_ROW_HEIGHT):
        _section_title("📰 뉴스" if is_crypto else "📰 뉴스 & 공시")

        tabs_col, count_col = st.columns([4, 1])
        with tabs_col:
            tab_labels = ["📰 뉴스"] if is_crypto else ["📰 뉴스", "📋 공시 (DART)"]
            tabs = st.tabs(tab_labels)
            news_tab = tabs[0]
            dart_tab = tabs[1] if len(tabs) > 1 else None
        with count_col:
            news_n = st.selectbox(
                "표시개수",
                [5, 10, 15, 20],
                index=1,
                key="news_n",
                label_visibility="collapsed",
                help="뉴스 표시 개수",
            )

        with news_tab:
            news_df = _crypto_news(selected_name, news_n) if is_crypto else _news(selected_code, news_n)

            if news_df.empty:
                st.info("최근 뉴스를 찾지 못했습니다.")
            else:
                with st.container(height=260):
                    for _, row in news_df.iterrows():
                        with st.container(border=True):
                            left, right = st.columns([6, 1])
                            with left:
                                title_safe = html.escape(row["title"])
                                url_safe = html.escape(row["url"], quote=True)
                                meta_safe = html.escape(f"{row['press']} · {row['date']}")
                                summary_safe = html.escape(row["summary"])
                                st.markdown(
                                    "<div style='display:flex;justify-content:space-between;"
                                    "align-items:baseline;gap:0.6em;'>"
                                    f"<a href='{url_safe}' target='_blank'>{title_safe}</a>"
                                    f"<span style='font-size:0.8em;opacity:0.65;white-space:nowrap;'>{meta_safe}</span>"
                                    "</div>"
                                    "<div style='white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                                    f"font-size:0.88em;opacity:0.85;' title='{summary_safe}'>{summary_safe}</div>",
                                    unsafe_allow_html=True,
                                )
                            with right:
                                color = _SENT_COLOR[row["sentiment_label"]]
                                st.markdown(
                                    "<div style='text-align:center;padding-top:0.3em'>"
                                    f"<span style='color:{color};font-weight:700;font-size:0.95em'>{row['sentiment_label']}</span><br>"
                                    f"<span style='color:{color};font-size:0.78em'>{row['sentiment_score']:+d}</span>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )

        if dart_tab is not None:
            with dart_tab:
                try:
                    dart_df = _dart(selected_code)
                except dart.DartKeyMissing as e:
                    st.info(str(e))
                else:
                    if dart_df.empty:
                        st.info("최근 90일 내 공시가 없습니다.")
                    else:
                        st.dataframe(
                            dart_df.rename(
                                columns={"rcept_dt": "접수일", "report_nm": "보고서명", "flr_nm": "제출인"}
                            ),
                            column_config={"url": st.column_config.LinkColumn("링크", display_text="열기")},
                            hide_index=True,
                            height=260,
                            width="stretch",
                        )
