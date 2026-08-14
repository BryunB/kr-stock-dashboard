from __future__ import annotations

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src import crypto_news

# 실제 네이버 뉴스 검색 결과 카드 구조를 간소화해 재현한 fixture (2026-08-14 실측).
# 카드 하나에 (a) 헤드라인 -> 원문 언론사 링크, (b) 언론사명 -> media.naver.com 링크,
# (c) "N시간 전 네이버뉴스" 텍스트를 담은 subtexts div, (d) 네이버뉴스 배지 링크가 들어있다.
_SEARCH_HTML = """
<html><body>
<div class="sds-comps-vertical-layout fds-news-item-list-tab">
  <div class="sds-comps-full-layout">
    <a href="https://www.example-press.com/article/1">첫 번째 코인 뉴스 제목</a>
    <div class="sds-comps-horizontal-layout sds-comps-profile-source">
      <a href="https://media.naver.com/press/003">뉴시스새 창 열림</a>
      <div class="sds-comps-horizontal-layout sds-comps-profile-info-subtexts">
        2시간 전 <a href="https://n.news.naver.com/mnews/article/003/0000000001?sid=101">네이버뉴스새 창 열림</a>
      </div>
    </div>
  </div>
  <div class="sds-comps-full-layout">
    <a href="https://www.example-press2.com/article/2">두 번째 코인 뉴스 제목</a>
    <div class="sds-comps-horizontal-layout sds-comps-profile-source">
      <a href="https://media.naver.com/press/018">헤럴드경제새 창 열림</a>
      <div class="sds-comps-horizontal-layout sds-comps-profile-info-subtexts">
        5시간 전 <a href="https://n.news.naver.com/mnews/article/018/0000000002?sid=101">네이버뉴스새 창 열림</a>
      </div>
    </div>
  </div>
  <div class="sds-comps-full-layout">
    <a href="#">Keep에 저장</a>
    <div class="sds-comps-horizontal-layout sds-comps-profile-info-subtexts">
      1일 전 <a href="https://n.news.naver.com/mnews/article/003/0000000001?sid=101">네이버뉴스새 창 열림</a>
    </div>
  </div>
</div>
</body></html>
"""


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def test_extract_meta_finds_title_press_date():
    soup = BeautifulSoup(_SEARCH_HTML, "html.parser")
    badge = soup.select('a[href*="n.news.naver.com/mnews/article/003/0000000001"]')[0]

    title, press, date = crypto_news._extract_meta(badge)

    assert title == "첫 번째 코인 뉴스 제목"
    assert press == "뉴시스"
    assert "2시간 전" in date


def test_extract_meta_ignores_placeholder_href():
    """href="#"인 "Keep에 저장" 링크를 헤드라인으로 착각하면 안 된다 (실제로 겪었던 버그)."""
    soup = BeautifulSoup(_SEARCH_HTML, "html.parser")
    # 세 번째 카드는 헤드라인 링크가 없고(placeholder만 있음) 배지만 있다 —
    # 하지만 첫 번째 카드와 article_id가 같아 fetch_news_list의 중복 제거로 걸러진다.
    # 여기서는 _extract_meta 단독 동작만 검증: badge를 가져와도 "Keep에 저장"을 title로
    # 삼지 않는지 확인.
    badges = soup.select('a[href*="n.news.naver.com/mnews/article"]')
    for badge in badges:
        title, _press, _date = crypto_news._extract_meta(badge)
        assert title != "Keep에 저장"


def test_fetch_news_list_parses_cards_and_dedupes(monkeypatch):
    monkeypatch.setattr(crypto_news.requests, "get", lambda *a, **kw: _FakeResponse(_SEARCH_HTML))

    df = crypto_news.fetch_news_list("테스트코인", n=10, use_cache=False)

    assert list(df.columns) == ["article_id", "office_id", "title", "press", "date", "url"]
    assert len(df) == 2  # 중복 article_id(0000000001)는 한 번만 남는다
    assert set(df["title"]) == {"첫 번째 코인 뉴스 제목", "두 번째 코인 뉴스 제목"}
    assert df.loc[df["article_id"] == "0000000001", "url"].iloc[0] == (
        "https://n.news.naver.com/mnews/article/003/0000000001"
    )


def test_fetch_news_list_respects_n_limit(monkeypatch):
    monkeypatch.setattr(crypto_news.requests, "get", lambda *a, **kw: _FakeResponse(_SEARCH_HTML))
    df = crypto_news.fetch_news_list("테스트코인", n=1, use_cache=False)
    assert len(df) == 1


def test_fetch_news_list_empty_page_returns_empty_df(monkeypatch):
    monkeypatch.setattr(crypto_news.requests, "get", lambda *a, **kw: _FakeResponse("<html></html>"))
    df = crypto_news.fetch_news_list("없는코인이름", use_cache=False)
    assert df.empty


def test_fetch_news_with_sentiment_reuses_news_pipeline(monkeypatch):
    listing = pd.DataFrame(
        [
            {
                "article_id": "1",
                "office_id": "003",
                "title": "긍정적인 상승 호재 코인 뉴스",
                "press": "뉴시스",
                "date": "1시간 전",
                "url": "https://n.news.naver.com/mnews/article/003/1",
            }
        ]
    )
    monkeypatch.setattr(crypto_news, "fetch_news_list", lambda kw, n=10, use_cache=True: listing)
    monkeypatch.setattr(
        crypto_news.news, "fetch_article_body", lambda office_id, article_id, use_cache=True: "본문"
    )

    df = crypto_news.fetch_news_with_sentiment("코인", n=10, use_cache=False)

    assert "summary" in df.columns
    assert "sentiment_label" in df.columns
    assert "sentiment_score" in df.columns
    assert len(df) == 1


def test_fetch_news_with_sentiment_empty_listing_short_circuits(monkeypatch):
    monkeypatch.setattr(crypto_news, "fetch_news_list", lambda kw, n=10, use_cache=True: pd.DataFrame())
    df = crypto_news.fetch_news_with_sentiment("없는코인", use_cache=False)
    assert df.empty


@pytest.mark.network
def test_fetch_news_with_sentiment_live_bitcoin():
    df = crypto_news.fetch_news_with_sentiment("비트코인", n=3)
    assert not df.empty
    assert {"title", "summary", "sentiment_label", "sentiment_score"} <= set(df.columns)
