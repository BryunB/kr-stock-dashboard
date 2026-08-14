"""portfolio.py 테스트 — 모의투자 원장 읽기/쓰기.

원장 경로를 tmp_path로 격리한다(news.py 감성 로그 테스트와 같은 패턴) — 실제
data/portfolio/를 절대 건드리지 않는다. 전부 오프라인, 네트워크 없음.
"""

import json

import pandas as pd
import pytest

from src import portfolio


@pytest.fixture(autouse=True)
def _isolate_portfolio_dir(tmp_path, monkeypatch):
    """모든 테스트에서 원장 경로를 tmp_path 아래로 돌린다."""
    monkeypatch.setattr(portfolio, "_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(portfolio, "_HOLDINGS_PATH", tmp_path / "holdings.csv")
    monkeypatch.setattr(portfolio, "_TRADES_PATH", tmp_path / "trades.csv")
    monkeypatch.setattr(portfolio, "_EQUITY_PATH", tmp_path / "equity_history.csv")


# ----------------------------------------------------------- 초기 상태


def test_initial_state_is_100m_cash_no_holdings():
    state = portfolio.get_state()
    assert state == {"cash": 100_000_000.0, "last_run_date": None}
    assert portfolio.get_holdings().empty
    assert portfolio.get_trades().empty
    assert portfolio.get_equity_history().empty


def test_initial_holdings_have_expected_columns():
    holdings = portfolio.get_holdings()
    assert list(holdings.columns) == ["code", "name", "quantity", "avg_price"]


# ----------------------------------------------------------- apply_trade — 매수


def test_apply_trade_buy_new_position():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH

    holdings, cash, record = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="테스트 매수",
    )

    assert cash == pytest.approx(100_000_000 - 700_000)
    row = holdings[holdings["code"] == "005930"].iloc[0]
    assert row["quantity"] == 10
    assert row["avg_price"] == pytest.approx(70_000)
    assert record == {
        "date": "2026-08-13",
        "code": "005930",
        "name": "삼성전자",
        "action": "buy",
        "quantity": 10,
        "price": 70_000,
        "amount": 700_000,
        "reason": "테스트 매수",
    }


def test_apply_trade_buy_existing_position_weighted_average():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH

    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=10_000,
        reason="1차 매수",
    )
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-14",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=12_000,
        reason="2차 매수",
    )

    row = holdings[holdings["code"] == "005930"].iloc[0]
    assert row["quantity"] == 20
    assert row["avg_price"] == pytest.approx(11_000)  # (10*10000 + 10*12000) / 20
    assert cash == pytest.approx(100_000_000 - 100_000 - 120_000)


def test_apply_trade_buy_insufficient_cash_raises():
    holdings = portfolio.get_holdings()
    with pytest.raises(ValueError, match="현금 부족"):
        portfolio.apply_trade(
            holdings,
            1_000,
            date="2026-08-13",
            code="005930",
            name="삼성전자",
            action="buy",
            quantity=1,
            price=70_000,
            reason="현금 부족 테스트",
        )


# ----------------------------------------------------------- apply_trade — 매도


def test_apply_trade_sell_partial_keeps_avg_price():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="매수",
    )

    holdings, cash, record = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-14",
        code="005930",
        name="삼성전자",
        action="sell",
        quantity=4,
        price=80_000,
        reason="일부 매도",
    )

    row = holdings[holdings["code"] == "005930"].iloc[0]
    assert row["quantity"] == 6
    assert row["avg_price"] == pytest.approx(70_000)  # 매도는 평단가를 바꾸지 않는다
    assert cash == pytest.approx(100_000_000 - 700_000 + 320_000)
    assert record["amount"] == 320_000


def test_apply_trade_sell_full_removes_holding():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="매수",
    )

    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-14",
        code="005930",
        name="삼성전자",
        action="sell",
        quantity=10,
        price=75_000,
        reason="전량 매도",
    )

    assert holdings.empty
    assert cash == pytest.approx(100_000_000 - 700_000 + 750_000)


def test_apply_trade_sell_more_than_held_raises():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=5,
        price=70_000,
        reason="매수",
    )
    with pytest.raises(ValueError, match="보유 수량"):
        portfolio.apply_trade(
            holdings,
            cash,
            date="2026-08-14",
            code="005930",
            name="삼성전자",
            action="sell",
            quantity=10,
            price=70_000,
            reason="초과 매도",
        )


def test_apply_trade_sell_unheld_stock_raises():
    holdings = portfolio.get_holdings()
    with pytest.raises(ValueError, match="보유하지 않은"):
        portfolio.apply_trade(
            holdings,
            portfolio.INITIAL_CASH,
            date="2026-08-13",
            code="005930",
            name="삼성전자",
            action="sell",
            quantity=1,
            price=70_000,
            reason="미보유 매도",
        )


def test_apply_trade_invalid_action_raises():
    holdings = portfolio.get_holdings()
    with pytest.raises(ValueError, match="action"):
        portfolio.apply_trade(
            holdings,
            portfolio.INITIAL_CASH,
            date="2026-08-13",
            code="005930",
            name="삼성전자",
            action="hold",
            quantity=1,
            price=70_000,
            reason="잘못된 액션",
        )


# ----------------------------------------------------------- mark_to_market


def test_mark_to_market_all_prices_available():
    holdings = pd.DataFrame(
        [
            {"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000},
            {"code": "000660", "name": "SK하이닉스", "quantity": 5, "avg_price": 100_000},
        ]
    )
    valued, total = portfolio.mark_to_market(holdings, {"005930": 75_000, "000660": 90_000})

    assert total == pytest.approx(75_000 * 10 + 90_000 * 5)
    row = valued[valued["code"] == "005930"].iloc[0]
    assert row["current_price"] == 75_000
    assert row["market_value"] == 750_000
    assert row["unrealized_pnl"] == pytest.approx((75_000 - 70_000) * 10)
    assert not row["price_is_stale"]


def test_mark_to_market_missing_price_falls_back_to_avg_price():
    holdings = pd.DataFrame([{"code": "999999", "name": "거래정지종목", "quantity": 3, "avg_price": 50_000}])
    valued, total = portfolio.mark_to_market(holdings, {})  # 시세 없음 (거래정지 등)

    row = valued.iloc[0]
    assert row["current_price"] == 50_000  # 평단가로 대체
    assert row["price_is_stale"]
    assert total == pytest.approx(150_000)


def test_mark_to_market_empty_holdings():
    valued, total = portfolio.mark_to_market(portfolio.get_holdings(), {})
    assert valued.empty
    assert total == 0.0


# ----------------------------------------------------------- save_daily_result


def test_save_daily_result_writes_state_last_and_records_trades():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, trade = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="테스트",
    )

    portfolio.save_daily_result(
        date="2026-08-13",
        cash=cash,
        holdings=holdings,
        new_trades=[trade],
        equity_row={
            "date": "2026-08-13",
            "cash": cash,
            "holdings_value": 700_000,
            "total_equity": cash + 700_000,
            "kospi_close": 3000.0,
        },
    )

    state = portfolio.get_state()
    assert state["last_run_date"] == "2026-08-13"
    assert state["cash"] == pytest.approx(cash)

    trades = portfolio.get_trades()
    assert len(trades) == 1
    assert trades.iloc[0]["code"] == "005930"

    equity = portfolio.get_equity_history()
    assert len(equity) == 1


def test_save_daily_result_no_trades_still_adds_one_equity_row():
    """매매 0건(관망)인 날도 시가평가 행은 반드시 하나 쌓인다."""
    holdings = portfolio.get_holdings()
    portfolio.save_daily_result(
        date="2026-08-13",
        cash=portfolio.INITIAL_CASH,
        holdings=holdings,
        new_trades=[],
        equity_row={
            "date": "2026-08-13",
            "cash": portfolio.INITIAL_CASH,
            "holdings_value": 0.0,
            "total_equity": portfolio.INITIAL_CASH,
            "kospi_close": 3000.0,
        },
    )
    assert len(portfolio.get_trades()) == 0
    assert len(portfolio.get_equity_history()) == 1


def test_save_daily_result_appends_across_multiple_days():
    for day, price in [("2026-08-13", 70_000), ("2026-08-14", 71_000)]:
        holdings = portfolio.get_holdings()
        state = portfolio.get_state()
        holdings, cash, trade = portfolio.apply_trade(
            holdings,
            state["cash"],
            date=day,
            code="005930",
            name="삼성전자",
            action="buy",
            quantity=1,
            price=price,
            reason="분할 매수",
        )
        portfolio.save_daily_result(
            date=day,
            cash=cash,
            holdings=holdings,
            new_trades=[trade],
            equity_row={
                "date": day,
                "cash": cash,
                "holdings_value": price,
                "total_equity": cash + price,
                "kospi_close": 3000.0,
            },
        )

    assert len(portfolio.get_trades()) == 2
    assert len(portfolio.get_equity_history()) == 2
    holdings = portfolio.get_holdings()
    assert holdings.iloc[0]["quantity"] == 2


# ----------------------------------------------------------- CSV 안정성 (핵심 설계 검증)
# 이 원장에 parquet 대신 CSV를 쓰기로 한 이유가 실제로 성립하는지 직접 검증한다:
# 같은 내용을 다시 써도(읽었다가 그대로 재작성해도) 바이트가 완전히 같아야 한다.


def test_holdings_csv_round_trip_is_byte_identical():
    holdings = pd.DataFrame(
        [
            {"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000.0},
            {"code": "000660", "name": "SK하이닉스", "quantity": 5, "avg_price": 123_456.789},
        ]
    )
    portfolio.save_daily_result(
        date="2026-08-13",
        cash=1_000_000.0,
        holdings=holdings,
        new_trades=[],
        equity_row={
            "date": "2026-08-13",
            "cash": 1_000_000.0,
            "holdings_value": 0.0,
            "total_equity": 1_000_000.0,
            "kospi_close": 3000.0,
        },
    )
    bytes_first = portfolio._HOLDINGS_PATH.read_bytes()

    # 읽어서 그대로 다시 쓴다 — 매매가 없는 날 save_daily_result가 실제로 하는 일과 같다.
    reread = portfolio.get_holdings()
    portfolio.save_daily_result(
        date="2026-08-14",
        cash=1_000_000.0,
        holdings=reread,
        new_trades=[],
        equity_row={
            "date": "2026-08-14",
            "cash": 1_000_000.0,
            "holdings_value": 0.0,
            "total_equity": 1_000_000.0,
            "kospi_close": 3000.0,
        },
    )
    bytes_second = portfolio._HOLDINGS_PATH.read_bytes()

    assert bytes_first == bytes_second, (
        "holdings.csv 내용이 안 바뀌었는데 바이트가 달라졌다 (parquet과 같은 문제 재발)"
    )


def test_state_json_last_run_date_updates_correctly():
    """state.json은 순수 텍스트이므로 같은 내용이면 항상 같은 바이트 — 그리고 무엇보다
    last_run_date가 매번 정확히 그날 날짜로 갱신되는지 확인한다(원자적 쓰기 순서의 전제)."""
    holdings = portfolio.get_holdings()
    portfolio.save_daily_result(
        date="2026-08-13",
        cash=portfolio.INITIAL_CASH,
        holdings=holdings,
        new_trades=[],
        equity_row={
            "date": "2026-08-13",
            "cash": portfolio.INITIAL_CASH,
            "holdings_value": 0.0,
            "total_equity": portfolio.INITIAL_CASH,
            "kospi_close": 3000.0,
        },
    )
    saved = json.loads(portfolio._STATE_PATH.read_text(encoding="utf-8"))
    assert saved["last_run_date"] == "2026-08-13"
