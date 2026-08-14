"""스크리닝 대시보드(screener, charts) 테스트.

screener는 GitHub 미러에서 실시간 스냅샷을 받아오므로 network 마커.
charts는 합성 데이터로 오프라인 검증 가능.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import charts, screener
from src import indicators as ind


def _fake_ohlcv(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1000, 100000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ----------------------------------------------------------- charts (오프라인)


def test_build_chart_minimal():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", sma_windows=(), show_volume=False)
    assert isinstance(fig, go.Figure)
    kinds = {type(t).__name__ for t in fig.data}
    assert "Candlestick" in kinds


def test_build_chart_all_panels():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(
        df, "테스트", sma_windows=(20, 60), show_volume=True, show_rsi=True, show_macd=True
    )
    kinds = [type(t).__name__ for t in fig.data]
    assert kinds.count("Candlestick") == 1
    assert kinds.count("Scatter") >= 2 + 1 + 2  # SMA20/60 + RSI + MACD/Signal
    assert kinds.count("Bar") == 2  # 거래량 + MACD 히스토그램


def test_build_chart_index_overlay_rebased():
    df = ind.add_all(_fake_ohlcv())
    other = _fake_ohlcv(seed=2)["Close"] * 1000  # 완전히 다른 가격 스케일
    fig = charts.build_chart(df, "테스트", index_overlays={"KOSPI": other})
    overlay = next(t for t in fig.data if t.name == "KOSPI (비교)")
    assert overlay.y[0] == pytest.approx(df["Close"].iloc[0], rel=1e-6)


def test_build_chart_index_overlay_survives_weekly_resample():
    """차트 인덱스가 주봉처럼 원본 일봉 지수와 날짜가 거의 안 겹치면, 예전 _rebase는
    전부 NaN을 만들었다(reindex 후 ffill은 원본과의 근접매칭을 못 함) — 회귀 테스트."""
    weekly = ind.add_all(ind.resample_ohlcv(_fake_ohlcv(), "W"))
    other = _fake_ohlcv(seed=2)["Close"] * 1000
    fig = charts.build_chart(weekly, "테스트", index_overlays={"KOSPI": other}, sma_windows=())
    overlay = next(t for t in fig.data if t.name == "KOSPI (비교)")
    assert not pd.isna(overlay.y).all()
    assert not pd.isna(overlay.y[-1])


def test_build_chart_bollinger_bands():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", sma_windows=(), show_bollinger=True, show_volume=False)
    names = {t.name for t in fig.data}
    assert {"BB 상단", "BB 하단"} <= names


def test_build_chart_missing_sma_column_skipped():
    """indicators.add_all은 sma5를 만들지 않는다 — 없는 컬럼은 조용히 건너뛴다."""
    df = ind.add_all(_fake_ohlcv())
    assert "sma5" not in df.columns
    fig = charts.build_chart(df, "테스트", sma_windows=(5, 20), show_volume=False)
    names = {t.name for t in fig.data}
    assert "SMA20" in names
    assert "SMA5" not in names


def test_build_chart_line_type_uses_scatter_not_candlestick():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", chart_type="line", sma_windows=(), show_volume=False)
    kinds = {type(t).__name__ for t in fig.data}
    assert "Candlestick" not in kinds
    assert any(t.name == "종가" for t in fig.data)


def test_build_chart_stochastic_panel():
    df = ind.add_all(_fake_ohlcv()).join(ind.stochastic(_fake_ohlcv()))
    fig = charts.build_chart(df, "테스트", show_stochastic=True, show_volume=False, sma_windows=())
    names = {t.name for t in fig.data}
    assert {"%K", "%D"} <= names


def test_build_chart_ichimoku_overlay():
    raw = _fake_ohlcv()
    df = ind.add_all(raw).join(
        ind.ichimoku(raw)[["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b"]]
    )
    fig = charts.build_chart(df, "테스트", show_ichimoku=True, show_volume=False, sma_windows=())
    names = {t.name for t in fig.data}
    assert {"전환선", "기준선", "선행스팬A", "선행스팬B"} <= names


def test_build_chart_ichimoku_skipped_when_columns_missing():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", show_ichimoku=True, show_volume=False, sma_windows=())
    names = {t.name for t in fig.data}
    assert "전환선" not in names


def test_build_chart_rangeselector_and_log_y_do_not_error():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(
        df, "테스트", show_rangeselector=True, log_y=True, show_volume=False, sma_windows=()
    )
    assert fig.layout.xaxis.rangeselector.buttons
    assert fig.layout.yaxis.type == "log"


def test_build_chart_crosshair_and_drag_pan():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", crosshair=True, drag_pan=True, show_volume=True, sma_windows=())
    assert fig.layout.xaxis.showspikes
    assert fig.layout.yaxis.showspikes
    assert fig.layout.dragmode == "pan"


def test_build_chart_crosshair_off_by_default_matches_app_py_usage():
    """app.py는 crosshair/drag_pan을 안 넘긴다 — 기본값이 꺼져 있어야 기존 화면이 그대로다."""
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", show_volume=False, sma_windows=())
    assert not fig.layout.xaxis.showspikes
    assert fig.layout.dragmode is None


def test_build_chart_drawing_tools_adds_modebar_buttons_without_changing_dragmode():
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(
        df, "테스트", drag_pan=True, drawing_tools=True, show_volume=False, sma_windows=()
    )
    assert list(fig.layout.modebar.add) == ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"]
    # drag_pan이 켠 dragmode="pan"을 drawing_tools가 덮어쓰면 안 된다 — 그리기는 모드바
    # 버튼을 눌러야만 활성화되고, 기본 조작감(팬)은 그대로 유지돼야 한다.
    assert fig.layout.dragmode == "pan"
    assert fig.layout.newshape.line.color == charts._UP


def test_build_chart_drawing_tools_off_by_default_matches_app_py_usage():
    """app.py는 drawing_tools를 안 넘긴다 — 기본값이 꺼져 있어야 모드바가 기존과 같다."""
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", show_volume=False, sma_windows=())
    assert fig.layout.modebar.add is None


def test_build_chart_volume_profile_adds_second_column():
    df = ind.add_all(_fake_ohlcv())
    vp = ind.volume_profile(df, bins=12)
    fig = charts.build_chart(df, "테스트", volume_profile=vp, show_volume=True, show_rsi=True, sma_windows=())
    vp_trace = next(t for t in fig.data if t.name == "매물대")
    assert vp_trace.orientation == "h"
    assert len(vp_trace.x) == 12
    # 매물대(col2)는 시간축이 아니라 거래량 크기 축을 쓰므로 가격 패널(col1)과 x축이 달라야 한다
    price_trace = next(t for t in fig.data if t.name == "가격")
    assert vp_trace.xaxis != price_trace.xaxis
    # 하지만 y축(가격 스케일)은 같이 움직여야 매물대 막대 높이가 실제 가격과 맞는다
    vp_yaxis_name = "yaxis" + vp_trace.yaxis[1:] if vp_trace.yaxis != "y" else "yaxis"
    assert fig.layout[vp_yaxis_name].matches == price_trace.yaxis
    # 거래량/RSI 등 다른 행(colspan)은 반드시 가격 패널과 같은 시간축으로 묶여 확대·이동이
    # 같이 움직여야 한다 — colspan+shared_xaxes 조합에서 Plotly가 이 연결을 안 만드는
    # 버그를 실제로 겪어서 수동으로 matches를 걸었다. 회귀 확인.
    vol_trace = next(t for t in fig.data if t.name == "거래량")
    vol_xaxis_name = "xaxis" + vol_trace.xaxis[1:] if vol_trace.xaxis != "x" else "xaxis"
    assert fig.layout[vol_xaxis_name].matches == price_trace.xaxis


def test_build_chart_no_volume_profile_keeps_single_column_layout():
    """volume_profile을 안 넘기면(app.py 등) 예전과 똑같이 1열 레이아웃이어야 한다."""
    df = ind.add_all(_fake_ohlcv())
    fig = charts.build_chart(df, "테스트", show_volume=False, sma_windows=())
    assert not any(t.name == "매물대" for t in fig.data)
    assert not any(getattr(a, "domain", None) and a.domain[0] > 0.5 for a in fig.select_xaxes())


def test_build_chart_volume_profile_ignored_when_empty():
    df = ind.add_all(_fake_ohlcv())
    empty_vp = pd.DataFrame(columns=["price_low", "price_high", "price_mid", "volume"])
    fig = charts.build_chart(df, "테스트", volume_profile=empty_vp, show_volume=False, sma_windows=())
    assert not any(t.name == "매물대" for t in fig.data)


# ----------------------------------------------------------- indicators (오프라인)


def test_stochastic_range_and_columns():
    df = _fake_ohlcv()
    out = ind.stochastic(df)
    assert {"stoch_k", "stoch_d"} == set(out.columns)
    valid = out.dropna()
    assert (valid["stoch_k"].between(0, 100)).all()
    assert (valid["stoch_d"].between(0, 100)).all()


def test_ichimoku_columns_and_shift_direction():
    df = _fake_ohlcv()
    out = ind.ichimoku(df)
    assert {
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "ichimoku_chikou",
    } == set(out.columns)
    # 선행스팬은 미래로(끝부분에 결측이 생기지 않고 시작부분에 더 많은 결측) 밀려있어야 한다
    assert out["ichimoku_senkou_a"].isna().sum() > df["Close"].rolling(9).mean().isna().sum()
    # 후행스팬은 과거로 밀려있어 마지막 26개가 결측이어야 한다
    assert out["ichimoku_chikou"].iloc[-26:].isna().all()


def test_heikin_ashi_close_matches_formula_and_volume_unchanged():
    df = _fake_ohlcv()
    ha = ind.heikin_ashi(df)
    expected_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    pd.testing.assert_series_equal(ha["Close"], expected_close, check_names=False)
    pd.testing.assert_series_equal(ha["Volume"], df["Volume"])
    # 하이킨아시 첫 시가는 원본의 (시가+종가)/2
    assert ha["Open"].iloc[0] == pytest.approx((df["Open"].iloc[0] + df["Close"].iloc[0]) / 2)
    # 고가/저가는 항상 그날 시가·종가·원본 고저를 포함하는 범위여야 한다
    assert (ha["High"] >= ha[["Open", "Close"]].max(axis=1)).all()
    assert (ha["Low"] <= ha[["Open", "Close"]].min(axis=1)).all()


def test_resample_ohlcv_weekly_aggregation():
    df = _fake_ohlcv(n=30)  # 영업일 30개 -> 약 6주
    weekly = ind.resample_ohlcv(df, "W")
    assert len(weekly) < len(df)
    assert list(weekly.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # 마지막 주 구간(pandas가 실제로 나눈 bin)에 속한 원본 행들로부터 정확히 집계됐는지 확인 —
    # bin 경계를 직접 계산(예: "마지막 날짜 - 6일")하면 마지막 주가 부분 주(휴장일 등으로
    # 7일이 안 채워진 주)일 때 이전 주 데이터까지 잘못 포함될 수 있어, pandas의 groupby로
    # 같은 방식으로 나눈 그룹을 그대로 가져와 비교한다.
    last_bin_rows = df.groupby(pd.Grouper(freq="W")).get_group(weekly.index[-1])
    assert weekly["Close"].iloc[-1] == last_bin_rows["Close"].iloc[-1]
    assert weekly["Open"].iloc[-1] == last_bin_rows["Open"].iloc[0]
    assert weekly["Volume"].iloc[-1] == pytest.approx(last_bin_rows["Volume"].sum())
    assert weekly["Volume"].sum() == pytest.approx(df["Volume"].sum())  # 총 거래량은 보존돼야 한다


def test_volume_profile_conserves_total_volume_and_covers_price_range():
    df = _fake_ohlcv()
    vp = ind.volume_profile(df, bins=24)
    assert len(vp) == 24
    assert vp["volume"].sum() == pytest.approx(df["Volume"].sum())
    assert vp["price_low"].iloc[0] == pytest.approx(df["Low"].min())
    assert vp["price_high"].iloc[-1] == pytest.approx(df["High"].max())
    assert (vp["volume"] >= 0).all()


def test_volume_profile_distributes_across_overlapping_bins():
    """봉 하나가 여러 구간에 걸쳐 있으면(저가~고가 폭이 넓으면) 거래량이 한 구간에
    몰리지 않고 겹치는 구간들에 나뉘어 들어가야 한다."""
    df = pd.DataFrame(
        {"Open": [100.0], "High": [110.0], "Low": [90.0], "Close": [105.0], "Volume": [1000.0]},
        index=pd.bdate_range("2025-01-01", periods=1),
    )
    vp = ind.volume_profile(df, bins=4)  # 90~110을 4구간(각 5폭)으로 -> 이 봉이 전 구간에 걸침
    assert (vp["volume"] > 0).all()
    assert vp["volume"].sum() == pytest.approx(1000.0)


def test_volume_profile_empty_input_returns_empty_df():
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    vp = ind.volume_profile(df)
    assert vp.empty
    assert list(vp.columns) == ["price_low", "price_high", "price_mid", "volume"]


def test_volume_profile_flat_price_returns_empty_df():
    """전 구간 고가==저가(예: 거래정지로 하루치만 있고 변동이 전혀 없는 경우)면 구간을
    나눌 수 없으니 빈 결과를 반환한다 — 0으로 나누기 등을 방지."""
    df = pd.DataFrame(
        {
            "Open": [100.0] * 3,
            "High": [100.0] * 3,
            "Low": [100.0] * 3,
            "Close": [100.0] * 3,
            "Volume": [1.0] * 3,
        },
        index=pd.bdate_range("2025-01-01", periods=3),
    )
    vp = ind.volume_profile(df)
    assert vp.empty


# ----------------------------------------------------------- screener (네트워크)


@pytest.mark.network
def test_screen_has_expected_columns():
    df = screener.screen(market="KOSDAQ", use_cache=False)
    assert {"Code", "Name", "DailyChangeRatio", "WeeklyChangeRatio", "Marcap"} <= set(df.columns)
    assert not df.empty
    assert set(df["Market"]) <= {"KOSDAQ"}


@pytest.mark.network
def test_top_movers_sorted_descending():
    df = screener.screen(market="KOSDAQ", use_cache=True)
    top = screener.top_movers(df, by="DailyChangeRatio", n=10)
    assert len(top) <= 10
    assert (top["DailyChangeRatio"].diff().dropna() <= 0).all()
