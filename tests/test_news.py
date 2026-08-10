"""뉴스/감성/공시 모듈 테스트.

sentiment는 순수 함수라 오프라인. news/dart는 외부 사이트·API를 치므로 network 마커.
dart는 DART_API_KEY가 없으면 자동으로 skip한다 (키 발급은 사용자 몫).
"""

import pandas as pd
import pytest

from src import config, dart, news, sentiment

# ----------------------------------------------------------- sentiment (오프라인)


def test_sentiment_positive():
    result = sentiment.score("실적 개선에 목표주가 상향, 강세 지속 기대감")
    assert result["label"] == "긍정"
    assert result["score"] > 0


def test_sentiment_negative():
    result = sentiment.score("실적 부진에 목표주가 하향, 급락 우려 확산")
    assert result["label"] == "부정"
    assert result["score"] < 0


def test_sentiment_neutral_on_no_keywords():
    result = sentiment.score("오늘 회사는 정기 이사회를 개최했다")
    assert result["label"] == "중립"
    assert result["score"] == 0


# ----------------------------------------------------------- 감성 히스토리 로그 (오프라인)
# data/raw/news_sentiment는 재생성 불가능한 유일한 데이터이므로 테스트는 tmp_path로 격리한다.


def test_log_daily_sentiment_and_history_full(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    code = "TESTCODE"
    news.log_daily_sentiment(
        code, 0.5, positive_count=3, negative_count=1, article_count=4, date=pd.Timestamp("2026-01-01")
    )
    news.log_daily_sentiment(
        code, -0.2, positive_count=1, negative_count=2, article_count=3, date=pd.Timestamp("2026-01-02")
    )

    hist = news.sentiment_history(code)
    assert list(hist.values) == [0.5, -0.2]

    full = news.sentiment_history_full(code)
    assert list(full["positive_count"]) == [3, 1]
    assert list(full["negative_count"]) == [1, 2]
    assert list(full["article_count"]) == [4, 3]


def test_sentiment_history_full_empty_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    full = news.sentiment_history_full("NOPE")
    assert full.empty
    assert list(full.columns) == ["avg_sentiment", "positive_count", "negative_count", "article_count"]


def test_log_daily_sentiment_overwrites_same_date(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    code = "TESTCODE2"
    d = pd.Timestamp("2026-01-01")
    news.log_daily_sentiment(code, 0.1, date=d)
    news.log_daily_sentiment(code, 0.9, date=d)
    hist = news.sentiment_history(code)
    assert len(hist) == 1
    assert hist.iloc[0] == pytest.approx(0.9)


def test_summarize_truncates_and_splits_sentences():
    body = "첫 문장입니다. 두 번째 문장입니다. 세 번째는 안 보여야 합니다."
    summary = news.summarize(body, max_sentences=2, max_chars=200)
    assert "첫 문장" in summary
    assert "두 번째" in summary
    assert "세 번째" not in summary


def test_summarize_empty_body():
    assert news.summarize("") == ""


# ----------------------------------------------------------- news (네트워크)


@pytest.mark.network
def test_fetch_news_list_has_expected_columns():
    df = news.fetch_news_list("005930", use_cache=False)
    assert not df.empty
    assert {"title", "press", "date", "url", "article_id"} <= set(df.columns)
    assert df["article_id"].is_unique  # 중복 게재 기사 제거 확인


@pytest.mark.network
def test_fetch_news_with_sentiment_end_to_end():
    df = news.fetch_news_with_sentiment("005930", n=3, use_cache=False)
    assert len(df) <= 3
    assert set(df["sentiment_label"]) <= {"긍정", "중립", "부정"}
    assert (df["summary"].str.len() > 0).all()


# ----------------------------------------------------------- dart (네트워크 + 키 필요)


def test_dart_raises_without_key(monkeypatch):
    monkeypatch.setattr(config, "DART_API_KEY", "")
    monkeypatch.setattr(dart.config, "DART_API_KEY", "")
    with pytest.raises(dart.DartKeyMissing):
        dart.fetch_disclosures("005930")


@pytest.mark.network
@pytest.mark.skipif(not config.DART_API_KEY, reason="DART_API_KEY가 설정되지 않음")
def test_dart_fetch_disclosures_with_key():
    df = dart.fetch_disclosures("005930", days_back=180)
    assert {"rcept_dt", "report_nm", "flr_nm", "url"} <= set(df.columns)
