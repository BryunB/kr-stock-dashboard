"""모의투자 성과 리포팅 — 원장을 읽기만 하는 화면. PRD.md 5.6 참고.

**읽기 전용.** 매매를 유발하는 코드를 단 한 줄도 포함하지 않는다 — src.trading_agent나
portfolio의 쓰기 함수(apply_trade/save_daily_result)는 import조차 하지 않는다. 실행(매매)은
scripts/run_daily_trading.py + GitHub Actions가 매일 1회 전담하고, 여기는 그 결과만 보여준다
(4장 실행/조회 분리 원칙).

demo_app.py의 st.navigation을 통해서만 로드되므로 st.set_page_config는 호출하지 않는다
(app.py가 이미 호출하고, 같은 세션에서 두 번 호출하면 에러).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, portfolio, screener

PORTFOLIO_COLOR = "#0ea5e9"  # 포트폴리오 자산 라인 (하늘색)
KOSPI_COLOR = "#f59e0b"  # 코스피 비교 라인 — charts.py의 지수 오버레이 색(amber)과 통일
UP_COLOR, DOWN_COLOR, FLAT_COLOR = "#ef4444", "#3b82f6", "#6b7280"  # app.py와 동일한 국내 관행

st.title("💰 모의투자")

st.warning(
    "⚠️ **실제 금전 거래가 아닌 시뮬레이션입니다. 투자 조언이 아닙니다.** "
    "가상 현금 1억원으로 시작해 규칙 기반 엔진이 자동으로 매매한 결과를 보여줄 뿐, "
    "특정 종목의 매수·매도를 권유하는 것이 아닙니다.",
    icon="⚠️",
)
st.caption(
    "장중 실시간이 아니라 **하루 1회, 장 마감 후** 자동 매매된 결과입니다 "
    "(GitHub Actions가 평일 19:30 KST 전후 실행). 방문 시점과 매매 실행 시점은 무관합니다."
)


def _change_html(diff: float, pct: float) -> str:
    color = UP_COLOR if diff > 0 else DOWN_COLOR if diff < 0 else FLAT_COLOR
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "―"
    return f"<span style='color:{color};font-weight:600'>{arrow} {diff:+,.0f}원 ({pct:+.2f}%)</span>"


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="보유종목 현재가 불러오는 중...")
def _current_prices() -> dict[str, float]:
    snapshot = screener.screen(market="ALL")
    return dict(zip(snapshot["Code"], snapshot["Close"], strict=True))


state = portfolio.get_state()
holdings = portfolio.get_holdings()
trades = portfolio.get_trades()
equity_hist = portfolio.get_equity_history()

if holdings.empty:
    holdings_mtm, holdings_value = (
        holdings.assign(
            current_price=pd.Series(dtype=float),
            market_value=pd.Series(dtype=float),
            unrealized_pnl=pd.Series(dtype=float),
            price_is_stale=pd.Series(dtype=bool),
        ),
        0.0,
    )
else:
    try:
        prices = _current_prices()
    except Exception:
        prices = {}
    holdings_mtm, holdings_value = portfolio.mark_to_market(holdings, prices)

total_equity = state["cash"] + holdings_value
last_run = state["last_run_date"]

# ==================================================================== 상단 요약
m1, m2, m3, m4 = st.columns(4)
m1.metric("총자산", f"{total_equity:,.0f}원")

cum_return_pct = (total_equity / portfolio.INITIAL_CASH - 1) * 100
m2.metric("누적 수익률 (원금 대비)", f"{cum_return_pct:+.2f}%")

if len(equity_hist) >= 2 and equity_hist["kospi_close"].notna().sum() >= 2:
    kospi_valid = equity_hist.dropna(subset=["kospi_close"])
    port_return = equity_hist["total_equity"].iloc[-1] / equity_hist["total_equity"].iloc[0] - 1
    kospi_return = kospi_valid["kospi_close"].iloc[-1] / kospi_valid["kospi_close"].iloc[0] - 1
    excess_pct = (port_return - kospi_return) * 100
    m3.metric("코스피 대비 초과수익률", f"{excess_pct:+.2f}%")
else:
    m3.metric("코스피 대비 초과수익률", "―")

m4.metric("마지막 매매 기준일", last_run if last_run else "아직 없음")

if last_run is None:
    st.info(
        "아직 첫 매매가 실행되지 않았습니다. GitHub Actions가 처음 실행되면 "
        "이 화면에 자산 추이·보유종목·거래내역이 표시됩니다."
    )

st.divider()

# ==================================================================== 자산 추이 차트
st.subheader("📈 자산 추이 (vs 코스피)")
if len(equity_hist) < 2:
    st.caption("아직 비교할 만큼 자산 추이 기록이 쌓이지 않았습니다 (최소 2거래일 필요).")
else:
    eh = equity_hist.copy()
    eh["date"] = pd.to_datetime(eh["date"])

    port_base = eh["total_equity"].iloc[0]
    port_idx = eh["total_equity"] / port_base * 100

    kospi_filled = eh["kospi_close"].ffill().bfill()
    kospi_base = kospi_filled.iloc[0]
    kospi_idx = kospi_filled / kospi_base * 100 if kospi_base and pd.notna(kospi_base) else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=eh["date"], y=port_idx, name="포트폴리오", line=dict(width=2.4, color=PORTFOLIO_COLOR))
    )
    if kospi_idx is not None:
        fig.add_trace(
            go.Scatter(
                x=eh["date"],
                y=kospi_idx,
                name="코스피",
                line=dict(width=1.6, color=KOSPI_COLOR, dash="dot"),
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="시작일=100 기준 지수",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("두 선 모두 최초 기록일 값을 100으로 맞춘 상대 비교(리베이스)입니다.")

st.divider()

# ==================================================================== 보유종목
st.subheader("📦 보유종목")
if holdings_mtm.empty:
    st.caption("보유 중인 종목이 없습니다.")
else:
    display = holdings_mtm.copy()
    display["평가손익률"] = (display["current_price"] / display["avg_price"] - 1) * 100
    display["비중"] = display["market_value"] / total_equity * 100 if total_equity else 0.0
    if display["price_is_stale"].any():
        st.caption("⚠️ 표시에 ※가 붙은 종목은 오늘 종가 조회에 실패해 평단가로 대체 평가한 값입니다.")
    display["종목명"] = display["name"] + display["price_is_stale"].map({True: " ※", False: ""})

    show = display[["종목명", "code", "quantity", "avg_price", "current_price", "평가손익률", "비중"]].rename(
        columns={"code": "종목코드", "quantity": "수량", "avg_price": "평단가", "current_price": "현재가"}
    )
    st.dataframe(
        show,
        column_config={
            "평단가": st.column_config.NumberColumn(format="%,.0f원"),
            "현재가": st.column_config.NumberColumn(format="%,.0f원"),
            "평가손익률": st.column_config.NumberColumn(format="%+.2f%%"),
            "비중": st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True,
        width="stretch",
    )

st.divider()

# ==================================================================== 거래 내역
st.subheader("📜 거래 내역")
st.caption("체결가는 그날(장 마감 후 확정된) 종가 기준으로 가정한 시뮬레이션 값입니다.")
if trades.empty:
    st.caption("거래 이력이 없습니다.")
else:
    recent_trades = trades.sort_values("date", ascending=False).head(50).copy()
    recent_trades["action"] = recent_trades["action"].map({"buy": "매수", "sell": "매도"})
    show_trades = recent_trades[["date", "name", "code", "action", "quantity", "price", "reason"]].rename(
        columns={
            "date": "날짜",
            "name": "종목명",
            "code": "종목코드",
            "action": "구분",
            "quantity": "수량",
            "price": "체결가",
            "reason": "판단 근거",
        }
    )
    st.dataframe(
        show_trades,
        column_config={"체결가": st.column_config.NumberColumn(format="%,.0f원")},
        hide_index=True,
        width="stretch",
    )
    st.caption("판단 근거는 정해진 규칙(임계값)에 따른 자동 판단 서술일 뿐, 종목 추천이 아닙니다.")

st.divider()

# ==================================================================== 오늘의 판단 요약
st.subheader("🗒️ 오늘의 판단 요약")
if last_run is None or trades.empty:
    st.caption("아직 매매 이력이 없습니다.")
else:
    today_trades = trades[trades["date"] == last_run]
    n_buy = (today_trades["action"] == "buy").sum()
    n_sell = (today_trades["action"] == "sell").sum()

    if len(equity_hist) >= 2:
        prev_equity = equity_hist["total_equity"].iloc[-2]
        diff = total_equity - prev_equity
        pct = diff / prev_equity * 100 if prev_equity else 0.0
        change_html = _change_html(diff, pct)
    else:
        change_html = ""

    st.markdown(
        f"**{last_run}** 기준 매수 {n_buy}건, 매도 {n_sell}건. 총자산 {total_equity:,.0f}원 {change_html}",
        unsafe_allow_html=True,
    )
