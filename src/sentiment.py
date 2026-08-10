"""키워드 기반 금융 뉴스 감성(상승지표) 판정.

문맥을 이해하지 못하는 단순 키워드 매칭이다. "흑자 전환 실패"처럼 부정어가
긍정 키워드를 뒤집는 문장이나 반어법은 놓친다 — 참고용 보조지표로만 쓸 것.
"""

from __future__ import annotations

POSITIVE_WORDS = [
    "상승",
    "급등",
    "강세",
    "호조",
    "개선",
    "최고",
    "사상 최대",
    "최대 실적",
    "흑자",
    "성장",
    "확대",
    "상향",
    "매수",
    "호실적",
    "수주",
    "체결",
    "청신호",
    "신고가",
    "반등",
    "훈풍",
    "기대감",
    "돌파",
    "역대급",
    "잭팟",
    "수혜",
]

NEGATIVE_WORDS = [
    "하락",
    "급락",
    "약세",
    "부진",
    "우려",
    "최저",
    "적자",
    "감소",
    "축소",
    "하향",
    "매도",
    "리스크",
    "손실",
    "소송",
    "조사",
    "제재",
    "결함",
    "리콜",
    "신저가",
    "경고",
    "위기",
    "쇼크",
    "논란",
    "차질",
    "철수",
    "구조조정",
    "감원",
]


def score(text: str) -> dict:
    """text 안의 긍정/부정 키워드 출현 횟수로 상승지표를 매긴다.

    반환: {"label": "긍정"|"중립"|"부정", "score": int, "positive": [...], "negative": [...]}
    score = 긍정 키워드 매칭 수 - 부정 키워드 매칭 수 (단어별 최대 1회 집계).
    """
    hits_pos = [w for w in POSITIVE_WORDS if w in text]
    hits_neg = [w for w in NEGATIVE_WORDS if w in text]
    net = len(hits_pos) - len(hits_neg)

    if net > 0:
        label = "긍정"
    elif net < 0:
        label = "부정"
    else:
        label = "중립"

    return {"label": label, "score": net, "positive": hits_pos, "negative": hits_neg}
