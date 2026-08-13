"""인터랙티브 Plotly 차트 — 캔들스틱 + 이동평균 + 지수 오버레이 + 거래량/RSI/MACD.

Streamlit 대시보드에서 쓰지만, 노트북에서 fig.show() 로도 그대로 쓸 수 있게
Streamlit에 의존하지 않는다.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_INDEX_COLORS = ["#f59e0b", "#10b981", "#8b5cf6", "#06b6d4", "#ec4899"]
_UP, _DOWN = "#ef4444", "#3b82f6"  # 국내 관행: 상승=빨강, 하락=파랑


def _rebase(series: pd.Series, to_index: pd.DatetimeIndex, base_value: float) -> pd.Series:
    """series를 to_index에 맞춰 정렬(ffill)하고, 첫 값이 base_value가 되도록 비례 스케일한다.

    가격 스케일이 전혀 다른 지수(예: 코스피 3,200 vs 개별주 5만원)를
    같은 y축에 겹쳐 그리기 위한 리베이스. 절대값이 아니라 '같은 출발점에서
    상대적으로 얼마나 움직였는가'를 비교하는 용도.
    """
    aligned = series.reindex(to_index).ffill().bfill()
    first = aligned.iloc[0] if len(aligned) else None
    if not aligned.empty and first not in (0, None) and pd.notna(first):
        aligned = aligned / first * base_value
    return aligned


def build_chart(
    df: pd.DataFrame,
    title: str,
    sma_windows: tuple[int, ...] = (20, 60),
    show_bollinger: bool = False,
    show_volume: bool = True,
    show_rsi: bool = False,
    show_macd: bool = False,
    index_overlays: dict[str, pd.Series] | None = None,
    base_height: int = 520,
    panel_height: int = 140,
) -> go.Figure:
    """df: indicators.add_all()을 거친 OHLCV+지표 DataFrame (컬럼: Open/High/Low/Close/Volume/sma*/rsi14/macd/signal).

    index_overlays: {"KOSPI": 가격시계열, ...} — df의 시작 종가에 맞춰 리베이스해 가격 패널에 겹쳐 그린다.
    """
    panels = ["가격"]
    if show_volume:
        panels.append("거래량")
    if show_rsi:
        panels.append("RSI(14)")
    if show_macd:
        panels.append("MACD")

    rows = len(panels)
    raw_heights = [0.55] + [0.45 / (rows - 1)] * (rows - 1) if rows > 1 else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=raw_heights,
        subplot_titles=panels,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
            increasing_line_color=_UP,
            decreasing_line_color=_DOWN,
        ),
        row=1,
        col=1,
    )

    for w in sma_windows:
        col = f"sma{w}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=f"SMA{w}", line=dict(width=1.3)),
                row=1,
                col=1,
            )

    if show_bollinger and {"upper", "lower"} <= set(df.columns):
        band_color = "rgba(148,163,184,0.9)"
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["upper"], name="BB 상단", line=dict(width=1, color=band_color, dash="dot")
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["lower"],
                name="BB 하단",
                line=dict(width=1, color=band_color, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(148,163,184,0.12)",
            ),
            row=1,
            col=1,
        )

    if index_overlays:
        base = float(df["Close"].iloc[0])
        for i, (label, series) in enumerate(index_overlays.items()):
            rebased = _rebase(series, df.index, base)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rebased,
                    name=f"{label} (비교)",
                    line=dict(width=1.2, dash="dot", color=_INDEX_COLORS[i % len(_INDEX_COLORS)]),
                ),
                row=1,
                col=1,
            )

    row = 1
    if show_volume:
        row += 1
        colors = [_UP if c >= o else _DOWN for o, c in zip(df["Open"], df["Close"], strict=True)]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=colors, showlegend=False),
            row=row,
            col=1,
        )

    if show_rsi:
        row += 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi14"],
                name="RSI14",
                line=dict(color="#7c3aed", width=1.2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="gray", line_width=0.8, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray", line_width=0.8, row=row, col=1)

    if show_macd:
        row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#0ea5e9", width=1.2)),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["signal"], name="Signal", line=dict(color="#f97316", width=1.2)),
            row=row,
            col=1,
        )
        hist_colors = [_UP if v >= 0 else _DOWN for v in df["hist"]]
        fig.add_trace(
            go.Bar(x=df.index, y=df["hist"], name="Hist", marker_color=hist_colors, showlegend=False),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=base_height + panel_height * (rows - 1),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=70, b=30),
        hovermode="x unified",
    )
    return fig
