"""KOSPI/KOSDAQ 상승률 스크리닝 대시보드.

실행: .venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts, config, dart, news, predictor, screener
from src import data_loader as dl
from src import indicators as ind

st.set_page_config(page_title="주가 스크리닝 대시보드", layout="wide")

DATE_RANGES = {"1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "2년": 730, "3년": 1095, "전체": None}
SMA_CHOICES = (5, 20, 60, 120)


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="전종목 시세 불러오는 중...")
def _screen(market: str, days_back: int, min_marcap: float, min_volume: float) -> pd.DataFrame:
    return screener.screen(market=market, days_back=days_back, min_marcap=min_marcap, min_volume=min_volume)


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="가격 데이터 불러오는 중...")
def _price(symbol: str, start: str) -> pd.DataFrame:
    return dl.get_price(symbol, start=start)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="뉴스 불러오는 중...")
def _news(code: str, n: int) -> pd.DataFrame:
    return news.fetch_news_with_sentiment(code, n=n)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="공시 불러오는 중...")
def _dart(code: str) -> pd.DataFrame:
    return dart.fetch_disclosures(code)


st.title("📈 KOSPI · KOSDAQ 상승률 스크리닝")

# ------------------------------------------------------------------ 사이드바: 스크리닝 조건
with st.sidebar:
    st.header("스크리닝 조건")
    market = st.selectbox(
        "시장",
        ["ALL", "KOSPI", "KOSDAQ"],
        format_func=lambda m: {"ALL": "전체 (KOSPI+KOSDAQ)", "KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}[m],
    )
    basis = st.radio("정렬 기준", ["일간", "주간"], horizontal=True)
    direction = st.radio("방향", ["상승률 높은순", "하락률 큰순"], horizontal=True)
    top_n = st.slider("표시 개수", 10, 100, 30, step=10)
    min_marcap_eok = st.slider("최소 시가총액 (억원)", 0, 5000, 300, step=50)
    min_volume = st.number_input("최소 거래량(주)", min_value=0, value=1000, step=1000)

    if st.button("🔄 새로고침 (캐시 초기화)"):
        st.cache_data.clear()
        dl.clear_cache()
        st.rerun()

# ------------------------------------------------------------------ 스크리닝 테이블
basis_col = "DailyChangeRatio" if basis == "일간" else "WeeklyChangeRatio"
ascending = direction == "하락률 큰순"

try:
    universe = _screen(market, 7, min_marcap_eok * 1e8, float(min_volume))
except Exception as e:
    st.error(f"스크리닝 데이터를 불러오지 못했습니다: {e}")
    st.stop()

ranked = screener.top_movers(universe, by=basis_col, n=top_n, ascending=ascending)

market_label = "KOSPI+KOSDAQ" if market == "ALL" else market
st.caption(f"{market_label} 전체 {len(universe):,}종목 중 {basis} {direction} 상위 {len(ranked)}개")

display = ranked.rename(
    columns={
        "Name": "종목명",
        "Market": "시장",
        "Close": "종가",
        "DailyChangeRatio": "일간%",
        "WeeklyChangeRatio": "주간%",
        "Volume": "거래량",
        "Marcap": "시가총액",
    }
).copy()
display["시가총액"] = display["시가총액"] / 1e8

event = st.dataframe(
    display[["Code", "종목명", "시장", "종가", "일간%", "주간%", "거래량", "시가총액"]],
    width="stretch",
    hide_index=True,
    height=min(38 * (len(display) + 1), 640),
    column_config={
        "종가": st.column_config.NumberColumn(format="%,d원"),
        "일간%": st.column_config.NumberColumn(format="%.2f%%"),
        "주간%": st.column_config.NumberColumn(format="%.2f%%"),
        "거래량": st.column_config.NumberColumn(format="%,d"),
        "시가총액": st.column_config.NumberColumn(format="%,.0f억원"),
    },
    on_select="rerun",
    selection_mode="single-row",
    key="gainers_table",
)

if event.selection.rows:
    picked = ranked.iloc[event.selection.rows[0]]
    st.session_state["selected_code"] = picked["Code"]
    st.session_state["selected_name"] = picked["Name"]

st.divider()

# ------------------------------------------------------------------ 종목 상세 차트
st.subheader("종목 상세")
st.caption("위 표에서 종목을 클릭하면 아래 차트가 해당 종목으로 바뀝니다.")

search_col, period_col, idx_col = st.columns([2, 1, 2])
with search_col:
    manual = st.text_input("종목코드 또는 이름으로 검색 (예: 005930, 삼성전자)", value="")
with period_col:
    period_label = st.selectbox("조회 기간", list(DATE_RANGES.keys()), index=3)
with idx_col:
    idx_sel = st.multiselect("지수 비교", list(config.INDICES.keys()), default=["KOSPI"])

selected_code = st.session_state.get("selected_code")
selected_name = st.session_state.get("selected_name", "")

if manual.strip():
    hits = dl.find_symbol(manual.strip())
    code_col = "Code" if "Code" in hits.columns else "Symbol"
    if not hits.empty:
        options = {f"{row[code_col]} · {row['Name']}": row[code_col] for _, row in hits.head(20).iterrows()}
        pick = st.selectbox("검색 결과", list(options.keys()))
        selected_code = options[pick]
        selected_name = pick.split(" · ", 1)[1]
    else:
        st.warning("검색 결과가 없습니다.")

if not selected_code:
    selected_code, selected_name = "005930", "삼성전자"

days_back = DATE_RANGES[period_label]
start_date = (
    config.DEFAULT_START
    if days_back is None
    else (pd.Timestamp.today() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
)

price_df = _price(selected_code, start_date)
if price_df.empty:
    st.error(f"'{selected_code}' 가격 데이터를 찾을 수 없습니다.")
    st.stop()

enriched = ind.add_all(price_df)

# 차트(좌, 넓게) + 지표 체크박스(우측 하단)
chart_col, indicator_col = st.columns([4, 1])

with indicator_col:
    st.markdown("**지표 선택**")
    st.container(height=260, border=False)  # 체크박스를 우측 하단쯤에 오도록 밀어내는 여백
    show_sma5 = st.checkbox("SMA5", value=False)
    show_sma20 = st.checkbox("SMA20", value=True)
    show_sma60 = st.checkbox("SMA60", value=True)
    show_sma120 = st.checkbox("SMA120", value=False)
    show_bb = st.checkbox("볼린저밴드", value=False)
    show_vol = st.checkbox("거래량", value=True)
    show_rsi = st.checkbox("RSI", value=False)
    show_macd = st.checkbox("MACD", value=False)

sma_windows = [w for w, on in [(5, show_sma5), (20, show_sma20), (60, show_sma60), (120, show_sma120)] if on]
for w in sma_windows:
    col = f"sma{w}"
    if col not in enriched.columns:
        enriched[col] = ind.sma(enriched["Close"], w)

overlays = {}
for label in idx_sel:
    idx_df = _price(label, start_date)
    if not idx_df.empty:
        overlays[label] = idx_df["Close"]

fig = charts.build_chart(
    enriched,
    title=f"{selected_name} ({selected_code})",
    sma_windows=tuple(sma_windows),
    show_bollinger=show_bb,
    show_volume=show_vol,
    show_rsi=show_rsi,
    show_macd=show_macd,
    index_overlays=overlays or None,
)
with chart_col:
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ 성과 요약
summary = ind.summary(price_df["Close"])
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("기간 수익률", f"{summary['total_return'] * 100:.1f}%")
m2.metric("CAGR", f"{summary['cagr'] * 100:.1f}%")
m3.metric("연변동성", f"{summary['volatility'] * 100:.1f}%")
m4.metric("샤프", f"{summary['sharpe']:.2f}")
m5.metric("MDD", f"{summary['max_drawdown'] * 100:.1f}%")

# ------------------------------------------------------------------ 가격 예측
st.divider()
st.subheader("가격 예측")
st.caption(
    "⚠️ 과거 가격·기술지표(+뉴스 감성)로 그때그때 학습한 통계 모델의 참고용 추정치입니다. "
    "투자 조언이 아니며 실제 가격과 다를 수 있습니다."
)

# 예측 피처로 쓸 뉴스 감성 히스토리를 먼저 갱신한다 (조회 실패해도 페이지 전체가 죽지 않도록 방어).
try:
    _news_for_log = _news(selected_code, 10)
    if not _news_for_log.empty:
        news.log_daily_sentiment(selected_code, float(_news_for_log["sentiment_score"].mean()))
except Exception:
    pass
sentiment_hist = news.sentiment_history(selected_code)

pred_1d = predictor.train_and_predict(price_df, horizon=1, sentiment_hist=sentiment_hist)
pred_5d = predictor.train_and_predict(price_df, horizon=5, sentiment_hist=sentiment_hist)

pcol1, pcol2 = st.columns(2)
for col, pred, label in [
    (pcol1, pred_1d, "다음 거래일 종가"),
    (pcol2, pred_5d, "5거래일 후 (약 1주일) 종가"),
]:
    with col:
        st.markdown(f"**{label}**")
        if "error" in pred:
            st.info(pred["error"])
            continue
        delta = pred["predicted_price"] - pred["last_close"]
        st.metric(
            pred["target_date"].strftime("%Y-%m-%d (%a)"),
            f"{pred['predicted_price']:,.0f}원",
            f"{delta:+,.0f}원 ({pred['predicted_return'] * 100:+.2f}%)",
        )
        st.caption(
            f"홀드아웃 검증 {pred['n_test']}거래일 기준 · "
            f"MAE {pred['mae']:,.0f}원 · MAPE {pred['mape'] * 100:.2f}% · "
            f"방향 적중률 {pred['directional_accuracy'] * 100:.0f}%"
        )

if st.button(
    "🎯 정확한 예측 (심층 학습)",
    help=(
        "Ridge/RandomForest/GradientBoosting을 시계열 교차검증으로 비교해 가장 좋은 모델로 다시 "
        "예측합니다. 조회 기간과 무관하게 보유한 전체 기간 데이터를 쓰며, 기본 예측보다 시간이 더 걸립니다."
    ),
):
    with st.spinner("여러 모델을 교차검증하며 심층 학습하는 중... (시간이 다소 걸립니다)"):
        full_price_df = _price(selected_code, config.DEFAULT_START)
        sentiment_hist_full = news.sentiment_history_full(selected_code)
        st.session_state["adv_pred"] = {
            "code": selected_code,
            "1d": predictor.train_and_predict_advanced(
                full_price_df, horizon=1, sentiment_hist_full=sentiment_hist_full
            ),
            "5d": predictor.train_and_predict_advanced(
                full_price_df, horizon=5, sentiment_hist_full=sentiment_hist_full
            ),
        }

adv_pred = st.session_state.get("adv_pred")
if adv_pred and adv_pred["code"] == selected_code:
    st.markdown("**🎯 정확한 예측 (심층 학습) 결과**")
    acol1, acol2 = st.columns(2)
    for col, pred, label in [
        (acol1, adv_pred["1d"], "다음 거래일 종가"),
        (acol2, adv_pred["5d"], "5거래일 후 (약 1주일) 종가"),
    ]:
        with col:
            st.markdown(f"**{label}**")
            if "error" in pred:
                st.info(pred["error"])
                continue
            delta = pred["predicted_price"] - pred["last_close"]
            st.metric(
                pred["target_date"].strftime("%Y-%m-%d (%a)"),
                f"{pred['predicted_price']:,.0f}원",
                f"{delta:+,.0f}원 ({pred['predicted_return'] * 100:+.2f}%)",
            )
            st.caption(
                f"선정 모델: {pred['best_model']} · 홀드아웃 검증 {pred['n_holdout']}거래일 기준 · "
                f"MAE {pred['mae']:,.0f}원 · MAPE {pred['mape'] * 100:.2f}% · "
                f"방향 적중률 {pred['directional_accuracy'] * 100:.0f}%"
            )

    with st.expander("심층 모델 상세 (교차검증 비교 · 피처 영향도)"):
        for pred, label in [(adv_pred["1d"], "다음 거래일 모델"), (adv_pred["5d"], "5거래일 후 모델")]:
            if "error" in pred:
                continue
            st.markdown(
                f"**{label}** — 선정: {pred['best_model']} · 학습 {pred['n_train']}행 / "
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
    st.caption(
        "기본 예측보다 느리지만 여러 모델을 교차검증으로 비교해 더 정교하게 다시 예측합니다. "
        "클릭 시 보유한 전체 기간 데이터로 새로 학습합니다."
    )

with st.expander("모델 상세 (피처 영향도)"):
    for pred, label in [(pred_1d, "다음 거래일 모델"), (pred_5d, "5거래일 후 모델")]:
        if "error" in pred:
            continue
        st.markdown(
            f"**{label}** — 학습 {pred['n_train']}행 / 검증 {pred['n_test']}행 · "
            f"뉴스 감성 히스토리 {pred['news_days']}일 누적"
        )
        st.dataframe(
            pred["feature_importance"].rename(columns={"label": "설명", "coef": "회귀계수"})[
                ["설명", "회귀계수"]
            ],
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "회귀계수가 클수록(절대값 기준) 예측에 미치는 영향이 큽니다. 뉴스 감성은 히스토리가 쌓일수록 값이 유의미해집니다."
    )

# ------------------------------------------------------------------ 뉴스 & 공시
st.divider()
st.subheader("뉴스 & 공시")

_SENT_COLOR = {"긍정": "#ef4444", "중립": "#6b7280", "부정": "#3b82f6"}  # 상승=빨강, 하락=파랑

news_tab, dart_tab = st.tabs(["📰 뉴스", "📋 공시 (DART)"])

with news_tab:
    news_n = st.slider("표시 개수", 5, 20, 10, step=5, key="news_n")
    news_df = _news(selected_code, news_n)

    if news_df.empty:
        st.info("최근 뉴스를 찾지 못했습니다.")
    else:
        st.caption(
            "우측의 상승지표는 기사 속 긍정/부정 키워드 빈도로 매긴 것으로, 문맥은 이해하지 못하는 참고용 보조지표입니다."
        )
        for _, row in news_df.iterrows():
            with st.container(border=True):
                left, right = st.columns([6, 1])
                with left:
                    st.markdown(f"[{row['title']}]({row['url']})")
                    st.caption(f"{row['press']} · {row['date']}")
                    st.write(row["summary"])
                with right:
                    color = _SENT_COLOR[row["sentiment_label"]]
                    st.markdown(
                        "<div style='text-align:center;padding-top:0.5em'>"
                        f"<span style='color:{color};font-weight:700;font-size:1.1em'>{row['sentiment_label']}</span><br>"
                        f"<span style='color:{color};font-size:0.85em'>{row['sentiment_score']:+d}</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

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
                dart_df.rename(columns={"rcept_dt": "접수일", "report_nm": "보고서명", "flr_nm": "제출인"}),
                column_config={"url": st.column_config.LinkColumn("링크", display_text="열기")},
                hide_index=True,
                width="stretch",
            )
