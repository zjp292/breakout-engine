"""
paper trading — integrates with the breakout scanner via main.py

runs automatically at the end of every main.py scan in three phases:
  1. sync_fills   — update db from alpaca order fills; place stop orders
                    for positions that just filled their buy
  2. check_exits  — close positions whose close crossed the stop or sma_10
  3. place_entries — submit moo buy orders for new high-score signals

entry:  market on open (moo), next trading day
stop:   gtc stop-loss order placed on alpaca once the buy fill is confirmed
exit:   close < sma_{exit_ma_period}  OR  close <= scan_stop_level

requires in .env:
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv


class AlpacaClient:
    """thin wrapper around alpaca-py; all methods return plain dicts"""

    def __init__(self):
        load_dotenv()
        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        from alpaca.trading.client import TradingClient
        self._client = TradingClient(api_key, secret_key, paper=True)

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "equity":        float(acct.equity),
            "cash":          float(acct.cash),
            "buying_power":  float(acct.buying_power),
        }

    def place_moo_buy(self, symbol: str, qty: int) -> str:
        """market buy (day); returns order id"""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums   import OrderSide, TimeInForce
        req   = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        return str(order.id)

    def place_gtc_stop(self, symbol: str, qty: int, stop_price: float) -> str:
        """gtc stop-loss sell; returns order id"""
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums   import OrderSide, TimeInForce
        req   = StopOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
        )
        order = self._client.submit_order(req)
        return str(order.id)

    def place_moo_sell(self, symbol: str, qty: int) -> str:
        """market sell (day); returns order id"""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums   import OrderSide, TimeInForce
        req   = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        return str(order.id)

    def cancel_order(self, order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception:
            pass  # already filled or cancelled — safe to ignore

    def get_order(self, order_id: str) -> Optional[dict]:
        try:
            o = self._client.get_order_by_id(order_id)
            status = o.status.value if hasattr(o.status, "value") else str(o.status)
            return {
                "id":                str(o.id),
                "status":            status,
                "filled_avg_price":  float(o.filled_avg_price) if o.filled_avg_price else None,
                "filled_qty":        int(o.filled_qty)          if o.filled_qty        else 0,
            }
        except Exception:
            return None

    def get_positions(self) -> dict:
        """returns {symbol: {qty, avg_entry_price, unrealized_plpc}}"""
        result = {}
        for p in self._client.get_all_positions():
            result[p.symbol] = {
                "qty":              int(p.qty),
                "avg_entry_price":  float(p.avg_entry_price),
                "unrealized_plpc":  float(p.unrealized_plpc),
            }
        return result


class PaperTradeManager:
    """
    orchestrates the three-phase paper trading lifecycle.
    called once per main.py run after the scanner saves to the db.
    """

    def __init__(self, alpaca: AlpacaClient, db, config: dict):
        self.alpaca    = alpaca
        self.db        = db
        self.cfg       = config.get("paper_trading", {})
        self.data_dir  = Path("data")

    def run(
        self,
        scan_date: str,
        scored_dfs: dict,
        market_condition=None,
        macro_result=None,
    ) -> None:
        """full paper trading cycle for a scan date"""
        _header("PAPER TRADING")
        self.sync_fills()
        self.check_exits(scored_dfs)
        self.place_entries(scan_date, scored_dfs, market_condition, macro_result)
        self._print_status()

    # ── phase 1 ───────────────────────────────────────────────────────────────

    def sync_fills(self) -> int:
        """
        poll alpaca for order status changes and update the db.

        buy fills:  transition submitted → open; place the gtc stop order now
                    (alpaca rejects stop-sell orders for positions not yet held,
                    so the stop must be placed after the buy confirms)
        sell fills: transition exiting → closed; compute p&l
        stop fills: detect if alpaca's stop-loss order fired between runs
        """
        updated = 0

        # ── buy fills
        for _, trade in self.db.get_paper_trades_by_status("submitted").iterrows():
            order = self.alpaca.get_order(trade["alpaca_order_id"])
            if order is None:
                continue

            if order["status"] in ("filled", "partially_filled"):
                fill_price   = order["filled_avg_price"]
                stop_level   = trade["scan_stop_level"]
                stop_order_id = None

                if stop_level and stop_level > 0 and fill_price and fill_price > stop_level:
                    try:
                        stop_order_id = self.alpaca.place_gtc_stop(
                            trade["symbol"], int(trade["shares"]), stop_level
                        )
                    except Exception as exc:
                        print(f"  ⚠ stop order for {trade['symbol']}: {exc}")

                self.db.update_paper_trade(
                    trade["id"],
                    status="open",
                    entry_price=fill_price,
                    entry_date=_today(),
                    position_value=fill_price * int(trade["shares"]) if fill_price else None,
                    alpaca_stop_order_id=stop_order_id,
                )
                updated += 1
                print(f"  ✓ filled   {trade['symbol']:6s}  {trade['shares']}sh @ ${fill_price:.2f}")

            elif order["status"] in ("cancelled", "expired", "rejected"):
                self.db.update_paper_trade(trade["id"], status="cancelled")
                updated += 1

        # ── sell fills
        for _, trade in self.db.get_paper_trades_by_status("exiting").iterrows():
            if not trade.get("exit_order_id"):
                continue
            order = self.alpaca.get_order(trade["exit_order_id"])
            if order is None:
                continue

            if order["status"] in ("filled", "partially_filled"):
                updated += 1
                self.db.update_paper_trade(
                    trade["id"],
                    **_close_fields(trade, order["filled_avg_price"]),
                )
                _print_close(trade)

        # ── alpaca stop orders that fired between runs (no sell order from us)
        for _, trade in self.db.get_paper_trades_by_status("open").iterrows():
            stop_oid = trade.get("alpaca_stop_order_id")
            if not stop_oid:
                continue
            order = self.alpaca.get_order(stop_oid)
            if order and order["status"] in ("filled", "partially_filled"):
                updated += 1
                fields = _close_fields(trade, order["filled_avg_price"])
                fields["exit_reason"] = "stop_order"
                self.db.update_paper_trade(trade["id"], **fields)
                _print_close(trade)

        return updated

    # ── phase 2 ───────────────────────────────────────────────────────────────

    def check_exits(self, scored_dfs: dict) -> int:
        """
        evaluate exit conditions for all open positions using today's close.
        places moo sell orders for positions that breach stop or sma.
        cancels the open gtc stop order before placing the moo sell so alpaca
        doesn't attempt two sells for the same position.
        """
        open_trades = self.db.get_paper_trades_by_status("open")
        if open_trades.empty:
            return 0

        exits = 0
        ma_period = self.cfg.get("exit_ma_period", 10)
        sma_col   = f"sma_{ma_period}"

        for _, trade in open_trades.iterrows():
            symbol     = trade["symbol"]
            stop_level = trade["scan_stop_level"]

            row = self._latest_row(symbol, scored_dfs, ma_period)
            if row is None:
                continue

            close  = row.get("close")
            sma_ma = row.get(sma_col)

            if close is None:
                continue

            reason = None
            if stop_level and close <= stop_level:
                reason = "stop_cross"
            elif sma_ma and not pd.isna(sma_ma) and close < sma_ma:
                reason = "ma_cross"

            if reason:
                stop_oid = trade.get("alpaca_stop_order_id")
                if stop_oid:
                    self.alpaca.cancel_order(stop_oid)

                try:
                    exit_oid = self.alpaca.place_moo_sell(symbol, int(trade["shares"]))
                except Exception as exc:
                    print(f"  ⚠ sell order for {symbol}: {exc}")
                    continue

                self.db.update_paper_trade(
                    trade["id"],
                    status="exiting",
                    exit_reason=reason,
                    exit_order_id=exit_oid,
                    alpaca_stop_order_id=None,  # stop cancelled
                )
                exits += 1
                ref = f"stop=${stop_level:.2f}" if reason == "stop_cross" else f"sma{ma_period}=${sma_ma:.2f}"
                print(f"  → exit     {symbol:6s}  close=${close:.2f}  {ref}  [{reason}]")

        return exits

    # ── phase 3 ───────────────────────────────────────────────────────────────

    def place_entries(
        self,
        scan_date: str,
        scored_dfs: dict,
        market_condition=None,
        macro_result=None,
    ) -> int:
        """
        read today's scan signals and submit moo buy orders for new positions.
        skips symbols already held, regimes that are blocked, and signals where
        position sizing would require less than min_shares.
        """
        if not self._regime_allows_entries(market_condition, macro_result):
            daily_regime = getattr(market_condition, "regime", "?") if market_condition else "?"
            macro_label  = getattr(macro_result, "regime_label", "?") if macro_result else "?"
            mom_21d      = getattr(macro_result, "mom_21d", None) if macro_result else None
            mom_str      = f"  21d_mom={mom_21d:+.1%}" if mom_21d is not None else ""
            print(f"  ⚑ entries blocked — daily={daily_regime}  macro={macro_label}{mom_str}")
            return 0

        max_pos          = self.cfg.get("max_open_positions", 10)
        blocked_regimes  = self.cfg.get("blocked_regimes", ["CAUTION", "DOWNTREND"])
        min_raw_score    = self.cfg.get("min_raw_score", 75)

        active  = (
            len(self.db.get_paper_trades_by_status("submitted"))
            + len(self.db.get_paper_trades_by_status("open"))
            + len(self.db.get_paper_trades_by_status("exiting"))
        )
        slots = max_pos - active
        if slots <= 0:
            print(f"  position cap reached ({max_pos}), no new entries")
            return 0

        equity          = self._live_equity()
        existing_syms   = set(self.db.get_open_symbols())

        signals = (
            self.db
            .load_scans(from_date=scan_date, to_date=scan_date, passed_only=True, min_raw_score=min_raw_score)
            .sort_values("raw_score", ascending=False)
        )

        placed = 0
        skipped_regime = []
        for _, sig in signals.iterrows():
            if placed >= slots:
                break

            symbol = sig["symbol"]
            if symbol in existing_syms:
                continue

            regime = sig.get("market_regime") or ""
            if regime in blocked_regimes:
                skipped_regime.append(symbol)
                continue

            price       = sig.get("price")
            stop_level  = sig.get("stop_level")

            if not price or price <= 0:
                continue
            if not stop_level or stop_level <= 0 or stop_level >= price:
                continue

            shares = self._position_size(price, stop_level, equity)
            if shares <= 0:
                continue

            try:
                order_id = self.alpaca.place_moo_buy(symbol, shares)
            except Exception as exc:
                print(f"  ⚠ buy order for {symbol}: {exc}")
                continue

            self.db.save_paper_trade({
                "symbol":               symbol,
                "scan_date":            scan_date,
                "signal_score":         sig.get("score"),
                "signal_raw_score":     sig.get("raw_score"),
                "signal_grade":         sig.get("grade"),
                "market_regime":        regime,
                "regime_multiplier":    sig.get("regime_multiplier"),
                "shares":               shares,
                "scan_stop_level":      stop_level,
                "scan_breakout_level":  sig.get("breakout_level"),
                "scan_price":           price,
                "alpaca_order_id":      order_id,
                "alpaca_stop_order_id": None,
                "status":               "submitted",
            })

            risk_dollars = shares * (price - stop_level)
            existing_syms.add(symbol)
            placed += 1
            print(
                f"  ↑ buy      {symbol:6s}  {shares}sh  "
                f"raw={sig.get('raw_score', 0):.0f}  "
                f"stop=${stop_level:.2f}  risk=${risk_dollars:.0f}"
            )

        if skipped_regime:
            print(f"  ⚠ skipped (regime={skipped_regime[0] if len(skipped_regime)==1 else blocked_regimes}): {', '.join(skipped_regime)}")
        if placed == 0 and not skipped_regime and signals.empty:
            print(f"  no signals with raw_score >= {min_raw_score}")

        return placed

    # ── helpers ───────────────────────────────────────────────────────────────

    def _regime_allows_entries(self, market_condition=None, macro_result=None) -> bool:
        """return False if the current market regime is unfavorable for new longs"""
        blocked_daily = set(self.cfg.get("blocked_regimes", ["CAUTION", "DOWNTREND"]))
        blocked_macro = set(self.cfg.get(
            "blocked_macro_regimes",
            ["BEAR_TRANSITION", "DOWNTREND", "BEAR_CHOP"],
        ))
        block_on_neg_mom = self.cfg.get("block_on_negative_21d_momentum", True)

        if market_condition is not None:
            if getattr(market_condition, "regime", None) in blocked_daily:
                return False

        if macro_result is not None:
            if getattr(macro_result, "regime_label", None) in blocked_macro:
                return False
            if block_on_neg_mom and getattr(macro_result, "mom_21d", 0) < 0:
                return False

        return True

    def _latest_row(
        self, symbol: str, scored_dfs: dict, ma_period: int
    ) -> Optional[dict]:
        """
        return the most recent ohlcv+feature row for a symbol.
        first tries scored_dfs (already computed by the scanner);
        falls back to the most recent pickle and computes sma manually.
        """
        if symbol in scored_dfs and not scored_dfs[symbol].empty:
            row = scored_dfs[symbol].iloc[-1]
            return row.to_dict() if hasattr(row, "to_dict") else dict(row)

        df = self._load_latest_pickle(symbol)
        if df is None or df.empty:
            return None

        result = df.iloc[-1].to_dict()
        sma_col = f"sma_{ma_period}"
        if sma_col not in result or pd.isna(result.get(sma_col)):
            result[sma_col] = float(df["close"].rolling(ma_period).mean().iloc[-1])
        return result

    def _load_latest_pickle(self, symbol: str) -> Optional[pd.DataFrame]:
        """most-recent pickle for symbol across all data/<date>/ dirs"""
        candidates = sorted(
            self.data_dir.glob(f"*/{symbol}-*.pkl"),
            key=lambda p: p.parent.name,
            reverse=True,
        )
        if not candidates:
            return None

        with open(candidates[0], "rb") as fh:
            df = pickle.load(fh)

        if "datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
            df = df.set_index("datetime")
            df.index = df.index.normalize()

        return df

    def _position_size(self, entry: float, stop: float, equity: float) -> int:
        risk_pct   = self.cfg.get("risk_per_trade", 0.01)
        max_pct    = self.cfg.get("max_position_pct", 0.20)
        min_shares = self.cfg.get("min_shares", 1)

        stop_dist = entry - stop
        if stop_dist <= 0:
            return 0

        shares = int((equity * risk_pct) / stop_dist)
        shares = min(shares, int((equity * max_pct) / entry))
        return max(shares, min_shares) if shares >= min_shares else 0

    def _live_equity(self) -> float:
        """fetch live equity from alpaca; fall back to config if unavailable"""
        try:
            return self.alpaca.get_account()["equity"]
        except Exception:
            return self.cfg.get("account_equity", 100_000)

    def _print_status(self) -> None:
        open_trades = self.db.get_paper_trades_by_status("open")
        submitted   = self.db.get_paper_trades_by_status("submitted")

        total_active = len(open_trades) + len(submitted)
        if total_active == 0 and self.db.load_paper_trades(status="closed").empty:
            return

        print(f"\n  positions open={len(open_trades)}  pending={len(submitted)}")

        if not open_trades.empty:
            try:
                live = self.alpaca.get_positions()
            except Exception:
                live = {}

            for _, t in open_trades.iterrows():
                sym  = t["symbol"]
                plpc = live.get(sym, {}).get("unrealized_plpc", 0.0) * 100
                sign = "+" if plpc >= 0 else ""
                print(
                    f"    {sym:6s}  {int(t['shares'])}sh @ ${t['entry_price']:.2f}"
                    f"  stop=${t['scan_stop_level']:.2f}  unrealized={sign}{plpc:.1f}%"
                )

        closed = self.db.load_paper_trades(status="closed")
        if not closed.empty:
            wins  = int((closed["pnl_pct"] > 0).sum())
            total = len(closed)
            avg   = closed["pnl_pct"].mean() * 100
            print(f"\n  closed trades: {total}  wins: {wins}  avg pnl: {avg:+.1f}%")


# ── module-level helpers ──────────────────────────────────────────────────────

def _today() -> str:
    from datetime import date
    return date.today().isoformat()


def _close_fields(trade: pd.Series, exit_price: Optional[float]) -> dict:
    entry_price = trade.get("entry_price")
    shares      = int(trade.get("shares", 0))
    pnl_pct     = (exit_price - entry_price) / entry_price if entry_price and exit_price else None
    pnl_dollars = (exit_price - entry_price) * shares      if entry_price and exit_price else None
    return {
        "status":      "closed",
        "exit_price":  exit_price,
        "exit_date":   _today(),
        "pnl_pct":     pnl_pct,
        "pnl_dollars": pnl_dollars,
    }


def _print_close(trade: pd.Series) -> None:
    pnl  = (trade.get("pnl_pct") or 0) * 100
    sign = "+" if pnl >= 0 else ""
    print(f"  ✗ closed   {trade['symbol']:6s}  {sign}{pnl:.1f}%  ({trade.get('exit_reason', '?')})")


def _header(title: str) -> None:
    w = 54
    print(f"\n{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}")
