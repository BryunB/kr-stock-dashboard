"""predictor.py 테스트 — 합성 데이터로 오프라인 검증.

실제 예측 정확도를 주장하는 게 아니라, 파이프라인이 깨지지 않고 형태가
맞는 결과를 내는지(에러 처리 포함)를 확인한다.
"""

import numpy as np
import pandas as pd
import pytest

from src import predictor


def _fake_price_df(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1000, 100000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_train_and_predict_shape():
    result = predictor.train_and_predict(_fake_price_df(), horizon=1)
    assert "error" not in result
    assert result["predicted_price"] > 0
    assert result["target_date"] > pd.Timestamp("2023-01-01")
    assert 0 <= result["directional_accuracy"] <= 1
    assert result["mae"] >= 0
    assert set(result["feature_importance"]["feature"]) == set(predictor.FEATURE_LABELS)


def test_train_and_predict_5day_horizon():
    result = predictor.train_and_predict(_fake_price_df(), horizon=5)
    assert "error" not in result
    # 5일 예측 target_date가 1일 예측보다 미래여야 함
    result1 = predictor.train_and_predict(_fake_price_df(), horizon=1)
    assert result["target_date"] > result1["target_date"]


def test_insufficient_data_returns_error():
    result = predictor.train_and_predict(_fake_price_df(n=50), horizon=1)
    assert "error" in result


def test_news_sentiment_feature_neutral_when_no_history():
    result = predictor.train_and_predict(_fake_price_df(), horizon=1, sentiment_hist=None)
    assert "error" not in result
    row = result["feature_importance"]
    news_row = row[row["feature"] == "news_sentiment"]
    assert not news_row.empty
    assert news_row["coef"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_news_sentiment_history_recognized():
    df = _fake_price_df()
    hist = pd.Series(
        [1.0, -1.0, 2.0],
        index=[df.index[-3], df.index[-2], df.index[-1]],
        name="avg_sentiment",
    )
    result = predictor.train_and_predict(df, horizon=1, sentiment_hist=hist)
    assert "error" not in result
    assert result["news_days"] == 3


# ----------------------------------------------------------- "정확한 예측" (심층 학습)


def test_train_and_predict_advanced_shape():
    result = predictor.train_and_predict_advanced(_fake_price_df(), horizon=1)
    assert "error" not in result
    assert result["predicted_price"] > 0
    assert result["target_date"] > pd.Timestamp("2023-01-01")
    assert result["best_model"] in {"Ridge", "RandomForest", "GradientBoosting"}
    assert set(result["cv_scores"]) == {"Ridge", "RandomForest", "GradientBoosting"}
    assert 0 <= result["directional_accuracy"] <= 1
    assert result["mae"] >= 0
    assert result["n_holdout"] > 0
    assert set(result["feature_importance"]["feature"]) == set(predictor.FEATURE_LABELS_ADV)


def test_train_and_predict_advanced_5day_horizon():
    result1 = predictor.train_and_predict_advanced(_fake_price_df(), horizon=1)
    result5 = predictor.train_and_predict_advanced(_fake_price_df(), horizon=5)
    assert "error" not in result5
    assert result5["target_date"] > result1["target_date"]


def test_train_and_predict_advanced_insufficient_data_returns_error():
    result = predictor.train_and_predict_advanced(_fake_price_df(n=100), horizon=1)
    assert "error" in result


def test_train_and_predict_advanced_uses_news_features():
    df = _fake_price_df()
    hist_full = pd.DataFrame(
        {
            "avg_sentiment": [1.0, -1.0, 2.0],
            "positive_count": [2, 0, 3],
            "negative_count": [0, 2, 0],
            "article_count": [2, 2, 3],
        },
        index=[df.index[-3], df.index[-2], df.index[-1]],
    )
    result = predictor.train_and_predict_advanced(df, horizon=1, sentiment_hist_full=hist_full)
    assert "error" not in result
    assert result["news_days"] == 3
