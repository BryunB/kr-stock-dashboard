"""가격 예측 — 과거 가격/기술지표(+뉴스 감성)로 그때그때 학습하는 경량 회귀 모델.

⚠️ 참고용 통계 모델이다. 투자 조언이 아니며, 정확도 지표를 반드시 함께 보여줘야 한다.

종목마다 매번 새로 학습한다(사전 학습 모델을 저장해두지 않음). 개별 종목의
일봉 데이터는 많아야 수백~수천 행이라 복잡한 모델은 과적합 위험이 크므로
릿지 회귀 정도의 단순한 선형 모델을 쓴다.

뉴스 감성은 `news.sentiment_history()`가 주는 날짜별 히스토리를 피처로 조인한다.
이 히스토리는 앱을 실행한 날부터 쌓이기 시작하므로, 기록 이전 구간은 중립(0)으로
채워진다 — 초기에는 이 피처의 영향력이 거의 없다가 히스토리가 쌓일수록 유의미해진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import indicators as ind

MIN_ROWS = 120  # 지표 워밍업 + 학습/검증 분할에 필요한 최소 거래일 수

FEATURE_LABELS = {
    "ret1": "1일 수익률",
    "ret5": "5일 수익률",
    "ret20": "20일 수익률",
    "sma5_gap": "SMA5 대비 괴리율",
    "sma20_gap": "SMA20 대비 괴리율",
    "sma60_gap": "SMA60 대비 괴리율",
    "rsi14": "RSI(14)",
    "vol20": "20일 변동성",
    "vol_chg": "거래량 변화율",
    "macd_hist": "MACD 히스토그램",
    "bb_pctb": "볼린저 %B",
    "news_sentiment": "뉴스 감성(누적 히스토리)",
}


def _build_features(price_df: pd.DataFrame, sentiment_hist: pd.Series | None) -> pd.DataFrame:
    close = price_df["Close"]
    feat = pd.DataFrame(index=price_df.index)
    feat["ret1"] = close.pct_change(1)
    feat["ret5"] = close.pct_change(5)
    feat["ret20"] = close.pct_change(20)
    feat["sma5_gap"] = close / ind.sma(close, 5) - 1
    feat["sma20_gap"] = close / ind.sma(close, 20) - 1
    feat["sma60_gap"] = close / ind.sma(close, 60) - 1
    feat["rsi14"] = ind.rsi(close, 14)
    feat["vol20"] = ind.volatility(close, 20)
    feat["vol_chg"] = price_df["Volume"].pct_change(5)
    feat["macd_hist"] = ind.macd(close)["hist"]
    feat["bb_pctb"] = ind.bollinger(close)["pct_b"]

    if sentiment_hist is not None and not sentiment_hist.empty:
        feat["news_sentiment"] = sentiment_hist.reindex(feat.index).fillna(0.0)
    else:
        feat["news_sentiment"] = 0.0

    return feat


def _target_date(last_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    """last_date 기준 horizon 거래일 뒤 날짜 (공휴일은 무시한 근사치, 표시용)."""
    future = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    return future[-1]


def train_and_predict(
    price_df: pd.DataFrame,
    horizon: int,
    sentiment_hist: pd.Series | None = None,
    test_ratio: float = 0.2,
) -> dict:
    """horizon 거래일 뒤 종가를 예측하고, 홀드아웃 구간에서의 검증 지표를 함께 반환한다.

    반환 딕셔너리:
      predicted_price, predicted_return, last_close, target_date,
      mae(원), mape(비율), directional_accuracy(비율),
      n_train, n_test, news_days(뉴스 감성 히스토리 축적 일수), feature_importance(DataFrame)
    학습이 불가능하면 {"error": str} 만 반환한다.
    """
    close = price_df["Close"]
    if len(close) < MIN_ROWS:
        return {
            "error": f"학습에는 최소 {MIN_ROWS}거래일 데이터가 필요합니다 (현재 {len(close)}일). 조회 기간을 늘려주세요."
        }

    feat = _build_features(price_df, sentiment_hist)
    target = close.shift(-horizon) / close - 1  # horizon일 뒤 수익률

    feature_cols = list(feat.columns)
    tech_cols = [c for c in feature_cols if c != "news_sentiment"]

    data = feat.copy()
    data["target"] = target
    data = data.dropna(subset=tech_cols)  # 지표 워밍업 구간 제거 (news_sentiment는 0으로 이미 채워짐)

    usable = data.dropna(subset=["target"])  # 미래 target이 없는 마지막 horizon행 제외
    if len(usable) < 40:
        return {"error": "유효한 학습 샘플이 부족합니다. 조회 기간을 늘려주세요."}

    split = max(int(len(usable) * (1 - test_ratio)), len(usable) - 60)
    split = min(split, len(usable) - 5)  # 검증 구간 최소 5행 보장
    train, test = usable.iloc[:split], usable.iloc[split:]

    model = Ridge(alpha=1.0)
    model.fit(train[feature_cols], train["target"])

    test_pred_ret = model.predict(test[feature_cols])
    test_actual_close = close.loc[test.index] * (1 + test["target"])
    test_pred_close = close.loc[test.index] * (1 + test_pred_ret)

    mae = mean_absolute_error(test_actual_close, test_pred_close)
    mape = mean_absolute_percentage_error(test_actual_close, test_pred_close)
    directional_accuracy = float(np.mean(np.sign(test_pred_ret) == np.sign(test["target"])))

    # 실제 예측: 가장 최근(마지막) 완비된 피처 행을 사용
    latest_feat = feat.iloc[[-1]][feature_cols]
    predicted_return = float(model.predict(latest_feat)[0])
    last_close = float(close.iloc[-1])
    predicted_price = last_close * (1 + predicted_return)

    importance = (
        pd.DataFrame({"feature": feature_cols, "coef": model.coef_})
        .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    news_days = int(sentiment_hist.notna().sum()) if sentiment_hist is not None else 0

    return {
        "predicted_price": predicted_price,
        "predicted_return": predicted_return,
        "last_close": last_close,
        "target_date": _target_date(close.index[-1], horizon),
        "mae": float(mae),
        "mape": float(mape),
        "directional_accuracy": directional_accuracy,
        "n_train": len(train),
        "n_test": len(test),
        "news_days": news_days,
        "feature_importance": importance,
    }


# ============================================================================
# "정확한 예측" — 더 깊게 학습하는 버전
#
# 기본 train_and_predict()는 그대로 둔 채, 사용자가 명시적으로 버튼을 눌렀을 때만
# 도는 무거운 버전이다. 차이:
#   - 항상 보유한 전체 기간으로 학습한다 (차트에 선택된 조회 기간과 무관).
#   - 피처를 늘린다 (EMA, 볼린저 밴드폭, 일중 변동폭, 요일, 뉴스 기사 수 등).
#   - 모델 하나를 바로 쓰지 않고 Ridge/RandomForest/GradientBoosting을
#     TimeSeriesSplit 교차검증으로 비교해 가장 좋은 것을 고른다.
#   - 정확도는 모델 선정에 쓰지 않은 '진짜' 마지막 홀드아웃 구간으로만 보고한다
#     (교차검증 점수는 모델을 고르는 데 썼으므로 그 자체를 정확도로 보여주지 않는다).
# ============================================================================

ADV_MIN_ROWS = 200  # 피처가 더 많고 워밍업 구간도 길어서 기본 모델보다 넉넉히 잡는다

FEATURE_LABELS_ADV = {
    **FEATURE_LABELS,
    "ema12_gap": "EMA12 대비 괴리율",
    "ema26_gap": "EMA26 대비 괴리율",
    "bb_bandwidth": "볼린저 밴드폭",
    "hl_range": "일중 변동폭",
    "co_gap": "종가-시가 갭",
    "vol_sma_ratio": "거래량 20일 평균 대비",
    "rsi_slope": "RSI 기울기",
    "dow_sin": "요일(주기 성분 1)",
    "dow_cos": "요일(주기 성분 2)",
    "news_pos_count": "뉴스 긍정 기사 수",
    "news_neg_count": "뉴스 부정 기사 수",
    "news_article_count": "뉴스 기사 수",
}


def _model_specs() -> dict:
    """모델 이름 -> 새 인스턴스를 만드는 factory. 폴드마다 새로 학습해야 하므로 함수로 둔다."""
    return {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=150, max_depth=4, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=100, max_depth=2, learning_rate=0.05, random_state=42
        ),
    }


def _build_features_advanced(
    price_df: pd.DataFrame, sentiment_hist_full: pd.DataFrame | None
) -> pd.DataFrame:
    feat = _build_features(price_df, None)  # 기본 피처 재사용, news_sentiment는 아래서 다시 채움
    close = price_df["Close"]

    feat["ema12_gap"] = close / ind.ema(close, 12) - 1
    feat["ema26_gap"] = close / ind.ema(close, 26) - 1
    feat["bb_bandwidth"] = ind.bollinger(close)["bandwidth"]
    feat["hl_range"] = (price_df["High"] - price_df["Low"]) / close
    feat["co_gap"] = (close - price_df["Open"]) / price_df["Open"]
    feat["vol_sma_ratio"] = price_df["Volume"] / price_df["Volume"].rolling(20).mean() - 1
    feat["rsi_slope"] = ind.rsi(close, 14).diff()

    dow = price_df.index.dayofweek
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    if sentiment_hist_full is not None and not sentiment_hist_full.empty:
        joined = sentiment_hist_full.reindex(feat.index)
        feat["news_sentiment"] = joined.get("avg_sentiment", pd.Series(dtype=float)).fillna(0.0)
        feat["news_pos_count"] = joined.get("positive_count", pd.Series(dtype=float)).fillna(0.0)
        feat["news_neg_count"] = joined.get("negative_count", pd.Series(dtype=float)).fillna(0.0)
        feat["news_article_count"] = joined.get("article_count", pd.Series(dtype=float)).fillna(0.0)
    else:
        feat["news_sentiment"] = 0.0
        feat["news_pos_count"] = 0.0
        feat["news_neg_count"] = 0.0
        feat["news_article_count"] = 0.0

    return feat


def _extract_importance(model, feature_cols: list[str], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    inner = model.steps[-1][1] if hasattr(model, "steps") else model  # Pipeline이면 마지막 단계
    if hasattr(inner, "coef_"):
        values = inner.coef_
    elif hasattr(inner, "feature_importances_"):
        values = inner.feature_importances_
    else:
        # HistGradientBoosting처럼 둘 다 없는 모델은 순열 중요도(permutation importance)로 대체.
        # 해당 피처를 무작위로 섞었을 때 성능이 얼마나 나빠지는지로 중요도를 추정한다.
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error"
        )
        values = result.importances_mean

    return (
        pd.DataFrame({"feature": feature_cols, "coef": values})
        .assign(label=lambda d: d["feature"].map(lambda f: FEATURE_LABELS_ADV.get(f, f)))
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )


def train_and_predict_advanced(
    price_df: pd.DataFrame,
    horizon: int,
    sentiment_hist_full: pd.DataFrame | None = None,
    n_splits: int = 5,
) -> dict:
    """여러 모델을 시계열 교차검증으로 비교해 고른 뒤, 진짜 홀드아웃으로 최종 검증한다.

    train_and_predict()와 반환 스키마는 대체로 같고 추가로 best_model, cv_scores,
    n_holdout이 붙는다. 학습 불가능하면 {"error": str}만 반환한다.
    """
    close = price_df["Close"]
    if len(close) < ADV_MIN_ROWS:
        return {
            "error": f"심층 학습에는 최소 {ADV_MIN_ROWS}거래일 데이터가 필요합니다 (현재 {len(close)}일)."
        }

    feat = _build_features_advanced(price_df, sentiment_hist_full)
    target = close.shift(-horizon) / close - 1

    feature_cols = list(feat.columns)
    tech_cols = [c for c in feature_cols if not c.startswith("news_")]

    data = feat.copy()
    data["target"] = target
    # 거래정지 등으로 거래량이 0인 날엔 vol_sma_ratio 등의 비율 피처가 inf가 될 수 있다 —
    # NaN으로 바꿔서 아래 dropna에 같이 걸리게 한다 (기본 모델보다 나눗셈 피처가 많아 필요).
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=tech_cols)  # 지표 워밍업 구간 제거 (뉴스 피처는 이미 0으로 채워짐)

    usable = data.dropna(subset=["target"])
    if len(usable) < 100:
        return {"error": "유효한 학습 샘플이 부족합니다."}

    X, y = usable[feature_cols], usable["target"]

    # 마지막 구간은 모델 선정에 전혀 관여하지 않는 '진짜' 홀드아웃으로 떼어둔다.
    final_split = max(len(usable) - 40, int(len(usable) * 0.8))
    final_split = min(final_split, len(usable) - 5)
    dev_X, dev_y = X.iloc[:final_split], y.iloc[:final_split]
    holdout_X, holdout_y = X.iloc[final_split:], y.iloc[final_split:]

    if len(dev_X) < 60:
        return {"error": "교차검증에 필요한 학습 구간이 부족합니다. 데이터가 더 쌓이면 다시 시도해주세요."}

    n_splits = max(2, min(n_splits, len(dev_X) // 40))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = {}
    for name, make_model in _model_specs().items():
        fold_errors = []
        for train_idx, val_idx in tscv.split(dev_X):
            model = make_model()
            model.fit(dev_X.iloc[train_idx], dev_y.iloc[train_idx])
            pred = model.predict(dev_X.iloc[val_idx])
            fold_errors.append(mean_absolute_error(dev_y.iloc[val_idx], pred))
        cv_scores[name] = float(np.mean(fold_errors))

    best_name = min(cv_scores, key=cv_scores.get)

    # 선정된 모델을 dev 구간(=홀드아웃 제외)으로 학습해 홀드아웃 성능을 정직하게 잰다.
    holdout_model = _model_specs()[best_name]()
    holdout_model.fit(dev_X, dev_y)
    holdout_pred_ret = holdout_model.predict(holdout_X)
    holdout_actual_close = close.loc[holdout_X.index] * (1 + holdout_y)
    holdout_pred_close = close.loc[holdout_X.index] * (1 + holdout_pred_ret)

    mae = mean_absolute_error(holdout_actual_close, holdout_pred_close)
    mape = mean_absolute_percentage_error(holdout_actual_close, holdout_pred_close)
    directional_accuracy = float(np.mean(np.sign(holdout_pred_ret) == np.sign(holdout_y)))

    # 실제 서비스용 예측: 보유한 데이터 전체(dev+holdout)로 재학습해 가장 최근 정보까지 반영.
    final_model = _model_specs()[best_name]()
    final_model.fit(X, y)
    latest_feat = feat.iloc[[-1]][feature_cols]
    predicted_return = float(final_model.predict(latest_feat)[0])
    last_close = float(close.iloc[-1])
    predicted_price = last_close * (1 + predicted_return)

    news_days = int(len(sentiment_hist_full)) if sentiment_hist_full is not None else 0

    return {
        "predicted_price": predicted_price,
        "predicted_return": predicted_return,
        "last_close": last_close,
        "target_date": _target_date(close.index[-1], horizon),
        "mae": float(mae),
        "mape": float(mape),
        "directional_accuracy": directional_accuracy,
        "n_train": len(X),
        "n_holdout": len(holdout_X),
        "news_days": news_days,
        "best_model": best_name,
        "cv_scores": cv_scores,
        "feature_importance": _extract_importance(final_model, feature_cols, holdout_X, holdout_y),
    }
