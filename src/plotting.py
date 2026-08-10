"""matplotlib 환경 설정 — 한글 폰트 및 기본 스타일.

차트를 그리기 전에 setup() 을 한 번 호출하면 된다. 한글 폰트를 지정하지
않으면 종목명이 네모(tofu)로 깨지고, 기본 설정에서는 음수 부호도 깨진다.

주: 실제 차트 함수는 여기 두지 않았다. 분석 코드를 만들 때 목적에 맞게
추가할 예정.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 우선순위 순. 설치된 첫 번째 폰트를 쓴다.
_KOREAN_FONTS = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans KR", "Gulim"]


def find_korean_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in _KOREAN_FONTS if f in available), None)


def setup(figsize=(12, 6), dpi=110, style: str = "seaborn-v0_8-whitegrid") -> str | None:
    """한글 폰트 + 기본 rcParams를 적용하고, 사용된 폰트 이름을 반환한다."""
    if style in plt.style.available:
        plt.style.use(style)

    font = find_korean_font()
    if font:
        mpl.rcParams["font.family"] = font
    # 한글 폰트는 유니코드 마이너스 글리프가 없어 축 음수가 깨진다
    mpl.rcParams["axes.unicode_minus"] = False

    mpl.rcParams.update(
        {
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )
    return font
