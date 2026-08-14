"""모의투자 일일 매매 — GitHub Actions가 매 평일 19:30 KST에 실행하는 단일 진입점.

PRD.md 5.5·10장 3단계 참고. ①(가격예측 신호 생성) → ②(매매 규칙 엔진) → ③(리스크
가드레일) → ④(포트폴리오 원장 반영)를 순서대로 묶는다. git 커밋은 이 스크립트가 아니라
`.github/workflows/daily_trading.yml`이 한다 — 여기는 `data/portfolio/` 로컬 파일까지만
책임진다.

로컬 실행: .venv\\Scripts\\python.exe scripts\\run_daily_trading.py [--dry-run]
--dry-run은 원장 파일을 건드리지 않고 판단 결과만 출력한다(디버깅용, PRD 5.5
"로컬 수동 실행과의 충돌" 참고 — 원장을 실제로 갱신하는 실행은 GitHub Actions로 일원화한다).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import data_loader as dl
from src import indicators as ind
from src import news, portfolio, predictor, screener
from src.trading_agent import apply_risk_guardrail, decide_trades

WATCHLIST_SIZE = 40  # PRD 5.1 — 종목별 개별 학습이 필요해 전종목이 아니라 워치리스트로 제한
PREDICT_HORIZON = 5  # 매매 규칙(TRADING_RULES)이 5거래일 예측 수익률 기준이므로 고정


def _build_signal(code: str, name: str) -> dict | None:
    """종목 하나의 매매 신호(예측 5일 수익률·RSI·뉴스감성·방향적중률)를 만든다.

    가격 이력 부족·예측 실패 등으로 신호를 못 만들면 None — 호출부가 그 종목을
    이번 판단에서 스킵한다(워치리스트 종목이면 매수 후보에서 빠지고, 보유 종목이면
    가격 기반 손절/익절만 평가된다 — decide_trades() docstring 참고).
    """
    price_df = dl.get_price(code)
    if price_df.empty:
        return None

    # app.py와 동일한 순서: 오늘자 뉴스를 먼저 히스토리에 기록한 뒤, 그 히스토리를
    # 학습 피처로 읽는다 (CLAUDE.md 규칙 — log_daily_sentiment 직접 호출 금지, 반드시
    # log_sentiment_from_news를 거쳐 긍정/부정/기사 수 피처까지 채운다).
    news_df = news.fetch_news_with_sentiment(code, n=10)
    news.log_sentiment_from_news(code, news_df)
    sentiment_hist = news.sentiment_history(code)

    result = predictor.train_and_predict(price_df, horizon=PREDICT_HORIZON, sentiment_hist=sentiment_hist)
    if "error" in result:
        return None

    rsi14 = ind.add_all(price_df)["rsi14"].iloc[-1]
    if pd.isna(rsi14):
        return None

    news_sentiment = float(news_df["sentiment_score"].mean()) if not news_df.empty else 0.0

    return {
        "code": code,
        "name": name,
        "predicted_return_5d": result["predicted_return"],
        "rsi14": float(rsi14),
        "news_sentiment": news_sentiment,
        "directional_accuracy": result["directional_accuracy"],
    }


def run(dry_run: bool = False) -> None:
    today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    today_str = today.strftime("%Y-%m-%d")

    state = portfolio.get_state()
    if state["last_run_date"] == today_str:
        print(f"[{today_str}] 오늘은 이미 실행했습니다(last_run_date={state['last_run_date']}) — 종료.")
        return

    try:
        snapshot = screener.screen(market="ALL", days_back=7)
    except RuntimeError as e:
        print(f"[{today_str}] 오늘 KRX 스냅샷을 아직 가져올 수 없습니다 ({e}) — 매매 없이 종료.")
        return

    current_prices = dict(zip(snapshot["Code"], snapshot["Close"], strict=True))
    name_by_code = dict(zip(snapshot["Code"], snapshot["Name"], strict=True))

    holdings = portfolio.get_holdings()
    cash = state["cash"]
    held_names = dict(zip(holdings["code"], holdings["name"], strict=True))

    watchlist = screener.top_movers(snapshot, by="Amount", n=WATCHLIST_SIZE)
    candidate_codes = sorted(set(watchlist["Code"]) | set(holdings["code"]))

    signals = []
    skipped = []
    for code in candidate_codes:
        name = name_by_code.get(code, held_names.get(code, code))
        try:
            sig = _build_signal(code, name)
        except Exception as e:
            print(f"  {code}({name}) 신호 생성 중 오류: {e}")
            sig = None
        if sig is None:
            skipped.append(code)
        else:
            signals.append(sig)

    print(
        f"[{today_str}] 워치리스트 {len(watchlist)}종목 + 보유 {len(holdings)}종목 중 "
        f"신호 {len(signals)}개 생성, {len(skipped)}개 스킵"
    )

    actions = decide_trades(signals, holdings, current_prices, cash)
    approved, rejected = apply_risk_guardrail(actions, holdings, cash, current_prices)

    for r in rejected:
        print(f"  거부: {r['action']} {r['code']} x{r['quantity']} — {r['reason_rejected']}")

    new_trades = []
    for a in approved:
        price = current_prices[a.code]
        name = name_by_code.get(a.code, held_names.get(a.code, a.code))
        holdings, cash, trade = portfolio.apply_trade(
            holdings,
            cash,
            date=today_str,
            code=a.code,
            name=name,
            action=a.action,
            quantity=a.quantity,
            price=price,
            reason=a.reason,
        )
        new_trades.append(trade)
        print(f"  체결: {a.action} {a.code}({name}) {a.quantity}주 @ {price:,.0f}원 — {a.reason}")

    holdings_mtm, holdings_value = portfolio.mark_to_market(holdings, current_prices)
    stale = holdings_mtm[holdings_mtm["price_is_stale"]] if not holdings_mtm.empty else holdings_mtm
    for _, row in stale.iterrows():
        print(f"  주의: {row['code']}({row['name']}) 오늘 종가를 못 가져와 평단가로 대체 평가")

    kospi_df = dl.get_price("KOSPI")
    kospi_close = float(kospi_df["Close"].iloc[-1]) if not kospi_df.empty else None

    total_equity = cash + holdings_value
    equity_row = {
        "date": today_str,
        "cash": cash,
        "holdings_value": holdings_value,
        "total_equity": total_equity,
        "kospi_close": kospi_close,
    }

    n_buy = sum(1 for a in approved if a.action == "buy")
    n_sell = sum(1 for a in approved if a.action == "sell")
    print(
        f"[{today_str}] 매수 {n_buy}건, 매도 {n_sell}건. "
        f"총자산 {total_equity:,.0f}원 (현금 {cash:,.0f} + 평가금액 {holdings_value:,.0f})"
    )

    if dry_run:
        print("[dry-run] 원장을 갱신하지 않았습니다.")
        return

    portfolio.save_daily_result(today_str, cash, holdings, new_trades, equity_row)
    print(f"[{today_str}] 원장 갱신 완료.")


def main() -> None:
    parser = argparse.ArgumentParser(description="모의투자 일일 매매 실행")
    parser.add_argument(
        "--dry-run", action="store_true", help="원장 파일을 건드리지 않고 판단 결과만 출력한다"
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
