"""
Portfolio Manager for the hedge fund simulation system.
Responsible for capital allocation, trade execution, and position management.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List
import config

# Import models and utilities
from src.models.pydantic_models import (
    CapitalAllocation,
    PortfolioAllocation,
    StockPosition,
    OptionPosition,
    Trade,
    Portfolio,
    MarketRegime,
)
from src.utils.financial_utils import sanitize_option_price
from src.utils.llm_utils import get_llm_response, build_portfolio_allocation_prompt


class PortfolioManager:
    """
    Manages portfolio allocation, trade execution, and position tracking.
    """

    def __init__(self, data_manager, initial_capital: float = 1000000.0, config_obj=None):
        """
        Initialize the PortfolioManager.

        Args:
            data_manager: DataManager instance for accessing stock data.
            initial_capital: Initial capital for the portfolio.
        """
        self.data_manager = data_manager
        self.config = config_obj
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.stock_positions = {}  # ticker -> StockPosition
        self.option_positions = []  # List of OptionPosition
        self.trades_history = []  # List of Trade
        self.portfolio_history = []  # List of Portfolio snapshots
        self.cumulative_realized_pnl = 0.0  # Track actual realized P&L from closed positions
        self.max_leverage = 1.5
        self.margin_rate = 0.06

    def _equity_fee(self, trade_value: float) -> float:
        rate = getattr(self.config, "equity_transaction_cost_bps", 10.0) / 10_000.0
        return abs(trade_value) * rate

    def _option_fee(self, trade_value: float) -> float:
        rate = getattr(self.config, "options_transaction_cost_bps", 20.0) / 10_000.0
        return abs(trade_value) * rate

    def _buying_power(self, nav: float, cash: float) -> float:
        return cash + (self.max_leverage - 1.0) * nav

    def allocate_capital(
        self,
        assets: List[str],
        market_analysis: Dict[str, MarketRegime],
        current_date: date,
        use_llm: bool = True,
    ) -> PortfolioAllocation:
        """
        Allocate capital across multiple assets.

        Args:
            assets: List of asset ticker symbols.
            market_analysis: Dictionary mapping tickers to MarketRegime objects.
            current_date: Current simulation date.
            use_llm: Whether to use LLM for allocation decisions.

        Returns:
            PortfolioAllocation object containing allocation decisions.
        """
        if use_llm:
            # Convert current positions to dictionary format for prompt
            current_positions = {}
            for ticker, position in self.stock_positions.items():
                current_positions[ticker] = {
                    "shares": position.shares,
                    "average_price": position.average_price,
                    "market_value": position.market_value,
                }

            # Build prompt for LLM
            prompt = build_portfolio_allocation_prompt(
                available_capital=self.cash,
                assets=assets,
                market_analysis=market_analysis,
                current_positions=current_positions,
            )

            # Get response from LLM
            response = get_llm_response(prompt, PortfolioAllocation)
            # Check if response is valid
            if isinstance(response, PortfolioAllocation):
                return response
            else:
                # Fall back to rule-based approach if LLM fails
                print(f"LLM allocation failed, falling back to rule-based approach")
                return self._rule_based_allocation(
                    assets, market_analysis, current_date
                )
        else:
            # Use rule-based approach
            return self._rule_based_allocation(assets, market_analysis, current_date)

    def _rule_based_allocation(
        self,
        assets: List[str],
        market_analysis: Dict[str, MarketRegime],
        current_date: date,
    ) -> PortfolioAllocation:
        """
        Perform rule-based capital allocation without using LLM.

        Args:
            assets: List of asset ticker symbols.
            market_analysis: Dictionary mapping tickers to MarketRegime objects.
            current_date: Current simulation date.

        Returns:
            PortfolioAllocation object containing allocation decisions.
        """
        # Calculate allocation weights based on market regime and sentiment
        weights = {}
        total_weight = 0.0

        for ticker in assets:
            analysis = market_analysis.get(ticker)
            if not analysis:
                weights[ticker] = 0.0
                continue

            # Base weight
            weight = 1.0

            # Adjust based on regime
            if analysis.regime == "bullish":
                weight *= 1.5
            elif analysis.regime == "bearish":
                weight *= 0.5
            elif analysis.regime == "volatile":
                weight *= 0.7

            # Adjust based on sentiment
            if analysis.sentiment > 0.25:
                weight *= 1.3
            elif analysis.sentiment < -0.25:
                weight *= 0.7

            # Adjust based on confidence
            weight *= analysis.confidence

            weights[ticker] = weight
            total_weight += weight

        # Normalize weights and calculate allocations
        allocations = []
        total_allocated = 0.0

        for ticker in assets:
            if total_weight > 0:
                normalized_weight = weights[ticker] / total_weight
            else:
                normalized_weight = 1.0 / len(assets)

            # Calculate allocation amount
            allocation_amount = self.cash * normalized_weight
            total_allocated += allocation_amount

            # Create allocation object
            allocation = CapitalAllocation(
                asset=ticker,
                amount=allocation_amount,
                percentage=normalized_weight,
                action="buy",
                rationale=(
                    f"Rule-based allocation based on {market_analysis.get(ticker).regime} regime "
                    f"and {market_analysis.get(ticker).sentiment_label} sentiment score "
                    f"({market_analysis.get(ticker).sentiment:.2f})."
                ),
            )

            allocations.append(allocation)

        # Create portfolio allocation object
        portfolio_allocation = PortfolioAllocation(
            total_capital=self.cash,
            allocations=allocations,
            cash_reserve=self.cash - total_allocated,
            allocation_date=current_date,
            rationale="Rule-based allocation without LLM.",
        )

        return portfolio_allocation

    def execute_trades(
        self, allocation: PortfolioAllocation, current_date: date
    ) -> List[Trade]:
        """
        Execute trades based on capital allocation, including short-selling.

        Args:
            allocation: PortfolioAllocation object containing allocation decisions.
            current_date: Current simulation date.

        Returns:
            List of executed Trade objects.
        """
        executed_trades = []

        for asset_alloc in allocation.allocations:
            ticker = asset_alloc.asset
            target_amount = asset_alloc.amount

            # If the user says "sell", we want a negative target value
            if asset_alloc.action == "sell":
                target_amount = -target_amount

            # Get current position value (long positive, short negative)
            current_pos = self.stock_positions.get(ticker)
            current_amount = current_pos.market_value if current_pos else 0.0

            # How much $ to trade today
            trade_amount = target_amount - current_amount

            # Skip if too small
            if abs(trade_amount) < 100:
                continue

            # Get price
            price = self.data_manager.get_stock_price(ticker, current_date)
            if price is None:
                print(
                    f"Could not get price for {ticker} on {current_date}, skipping trade"
                )
                continue

            # Determine how many shares (always positive quantity)
            shares = int(abs(trade_amount) / price)
            if shares == 0:
                continue

            # Use the allocation’s action, not inferred
            action = asset_alloc.action

            # Compute trade value and fees
            trade_value = shares * price
            fee = self._equity_fee(trade_value)

            # Build Trade object
            trade = Trade(
                asset=ticker,
                trade_date=current_date,
                trade_type="stock",
                action=action,
                quantity=shares,
                price=price,
                total_value=trade_value,
                fees=fee,
            )

            # Cash impact: buying *debits* cash, selling *credits* cash
            if action == "buy":
                self.cash -= trade_value + fee
            else:  # sell (including opening or adding to a short)
                self.cash += trade_value - fee

            # Update your position manager; it must handle short positions when action=="sell"
            self._update_stock_position(ticker, shares, price, action)

            executed_trades.append(trade)
            self.trades_history.append(trade)

        return executed_trades

    def execute_option_trade(self, option: OptionPosition, current_date: date) -> Trade:
        """
        Execute a single option trade at the current market price.

        Args:
            option: OptionPosition object representing the option trade to execute.
            current_date: The date of execution.

        Returns:
            Trade object representing the executed option trade.
        """
        # Get current option data
        option_data = self.data_manager.get_option_data_by_id(
            option.option_id, option.asset, current_date
        )
        if not option_data or "LAST" not in option_data:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Option ID '{option.option_id}' not found in available chain for {option.asset} on {current_date}. "
                f"This option may have been hallucinated by the LLM. Skipping trade."
            )
            raise ValueError(
                f"Could not retrieve current price for option {option.option_id} on {current_date}."
            )

        current_price = self._get_sanitized_option_price(
            option, option_data, current_date
        )
        if np.isnan(current_price):
            current_price = option_data["LAST"]
        # Calculate total trade value (each contract covers 100 shares)
        trade_value = current_price * option.contracts * 100
        # Apply transaction fee (adjust as needed)
        fee = self._option_fee(trade_value)

        if option.action == "buy":
            # For a buy, ensure sufficient cash and then add the option position normally.
            if self.cash < trade_value + fee:
                raise ValueError(
                    f"Not enough cash to execute option buy trade for {option.asset}."
                )
            self.cash -= trade_value + fee
            option.paid_amount = trade_value + fee
            option.market_value = trade_value
            option.unrealized_pnl = option.market_value - option.paid_amount
            # Add the option to the portfolio (as a long position) if not already present.
            # (Assuming each buy is a new position; you may also aggregate if desired.)
            self.option_positions.append(option)
        elif option.action == "sell":
            # For a sell, we want to record a short position rather than simply remove an existing position.
            # Check if a short position for this option already exists.
            short_position = None
            for pos in self.option_positions:
                if pos.option_id == option.option_id and pos.action == "sell":
                    short_position = pos
                    break
            if short_position is None:
                # Create a new short option position.
                # Here we store paid_amount as a negative number (the credit received).
                trade_value = -trade_value
                short_position = OptionPosition(
                    asset=option.asset,
                    option_type=option.option_type,
                    option_id=option.option_id,
                    strike=option.strike,
                    expiration_date=option.expiration_date,
                    contracts=option.contracts,
                    premium_per_contract=option.premium_per_contract,
                    paid_amount=(trade_value + fee),  # Negative cost (credit)
                    market_value=(trade_value + fee),  # Initially equal to the credit
                    action="sell",
                    strategy=option.strategy,
                )
                # Calculate unrealized P&L: for short positions, this will be negative if option value increased
                short_position.unrealized_pnl = short_position.market_value - short_position.paid_amount
                self.option_positions.append(short_position)
            else:
                # Update the existing short position.
                short_position.contracts += option.contracts
                short_position.paid_amount -= trade_value - fee
                short_position.market_value -= trade_value - fee
            # Add the cash received.
            self.cash -= trade_value - fee
        else:
            raise ValueError("Invalid option action. Must be 'buy' or 'sell'.")

        # Create a trade record.
        trade = Trade(
            asset=option.asset,
            trade_date=current_date,
            trade_type="option",
            action=option.action,
            quantity=option.contracts,
            price=current_price,
            total_value=trade_value,
            fees=fee,
            option_position=option,
        )
        self.trades_history.append(trade)
        return trade

    def _get_sanitized_option_price(
        self, position: OptionPosition, option_data: Dict, quote_date: date
    ) -> float:
        """Return a market price adjusted by a Black–Scholes sanity check."""
        market_price = option_data.get("LAST")
        try:
            market_price = float(market_price)
        except (TypeError, ValueError):
            return float("nan")

        underlying_price = (
            option_data.get("UNDERLYING_LAST")
            or option_data.get("UNDERLYINGLAST")
            or option_data.get("UNDERLYING_PRICE")
        )
        try:
            underlying_price = (
                float(underlying_price)
                if underlying_price is not None
                else float("nan")
            )
        except (TypeError, ValueError):
            underlying_price = float("nan")

        if np.isnan(underlying_price):
            fallback_price = self.data_manager.get_stock_price(
                position.asset, quote_date
            )
            try:
                underlying_price = float(fallback_price)
            except (TypeError, ValueError):
                underlying_price = float("nan")

        if np.isnan(underlying_price) or underlying_price <= 0:
            return market_price

        days_to_expiry = (position.expiration_date - quote_date).days
        if days_to_expiry < 0:
            return market_price

        time_to_expiry = max(days_to_expiry / 365.0, 1.0 / 365.0)

        sigma = option_data.get("IMPLIED_VOLATILITY")
        if sigma is None:
            sigma = option_data.get("IV")
        if sigma is None:
            sigma = option_data.get("iv")
        try:
            sigma = float(sigma) if sigma is not None else 0.25
        except (TypeError, ValueError):
            sigma = 0.25
        if sigma > 1.0:
            sigma /= 100.0
        if sigma <= 0:
            sigma = 0.25

        sanitized_price = sanitize_option_price(
            option_type=position.option_type,
            market_price=market_price,
            S=underlying_price,
            K=position.strike,
            T=time_to_expiry,
            sigma=sigma,
        )

        return float(sanitized_price)

    def _update_stock_position(
        self, ticker: str, shares: int, price: float, action: str
    ) -> None:
        """
        Update a stock position after a trade, including short selling.

        Args:
            ticker: Stock ticker symbol.
            shares: Number of shares traded (always positive).
            price: Trade price per share.
            action: 'buy' or 'sell'.
        """
        position = self.stock_positions.get(ticker)

        if action == "buy":
            if position:
                # If we were short (shares < 0), buying reduces the short
                new_shares = position.shares + shares
                if position.shares < 0 and new_shares > 0:
                    # closed shorts and opened a long
                    # Realize P&L from the short portion
                    closed_shares = abs(position.shares)
                    realized_pnl = (position.average_price - price) * closed_shares
                    self.cumulative_realized_pnl += realized_pnl
                    position.average_price = price
                elif position.shares < 0 and new_shares == 0:
                    # just closed out the short—no position left
                    closed_shares = abs(position.shares)
                    realized_pnl = (position.average_price - price) * closed_shares
                    self.cumulative_realized_pnl += realized_pnl
                    del self.stock_positions[ticker]
                    return
                elif position.shares < 0 and new_shares < 0:
                    # Partially covering a short position
                    # Realize P&L on the covered portion
                    realized_pnl = (position.average_price - price) * shares
                    self.cumulative_realized_pnl += realized_pnl
                    # Average price stays the same for shorts when covering
                # else adding to an existing long
                total_cost = (
                    position.average_price * abs(position.shares) + price * shares
                )
                total_shares = abs(position.shares) + shares
                position.average_price = total_cost / total_shares
                position.shares = new_shares

            else:
                # brand‑new long
                position = StockPosition(
                    asset=ticker,
                    shares=shares,
                    average_price=price,
                    current_price=price,
                    market_value=shares * price,
                    unrealized_pnl=0.0,
                )
                self.stock_positions[ticker] = position

        else:  # action == "sell"
            if position:
                new_shares = position.shares - shares

                if position.shares > 0 and new_shares < 0:
                    # Would flip from long → short
                    # Check if short selling is allowed
                    allow_short = getattr(config, "ALLOW_SHORT_SELLING", False)
                    if not allow_short:
                        # Only close the long position, don't go short
                        realized_pnl = (price - position.average_price) * position.shares
                        self.cumulative_realized_pnl += realized_pnl
                        del self.stock_positions[ticker]
                        # Refund the excess shares that would have gone short
                        excess_shares = abs(new_shares)
                        refund = excess_shares * price
                        self.cash += refund
                        print(f"Short selling disabled: Closed {position.shares} shares of {ticker}, refunded ${refund:.2f} for {excess_shares} excess shares")
                        return

                    # Short selling allowed - flip from long → short
                    # Realize P&L from closing the long portion
                    realized_pnl = (price - position.average_price) * position.shares
                    self.cumulative_realized_pnl += realized_pnl
                    short_shares = abs(new_shares)
                    position.shares = -short_shares
                    position.average_price = price

                elif position.shares > 0 and new_shares == 0:
                    # just closed the long
                    realized_pnl = (price - position.average_price) * position.shares
                    self.cumulative_realized_pnl += realized_pnl
                    del self.stock_positions[ticker]
                    return

                elif position.shares > 0 and new_shares > 0:
                    # Partially closing a long position
                    realized_pnl = (price - position.average_price) * shares
                    self.cumulative_realized_pnl += realized_pnl
                    position.shares = new_shares
                    # average_price stays the same for reducing a long

                elif position.shares < 0:
                    # adding to an existing short
                    total_shorts = abs(position.shares) + shares
                    total_proceeds = (
                        position.average_price * abs(position.shares) + price * shares
                    )
                    position.average_price = total_proceeds / total_shorts
                    position.shares = -total_shorts

            else:
                # Would open a brand‑new short
                allow_short = getattr(config, "ALLOW_SHORT_SELLING", False)
                if not allow_short:
                    # Refund the cash and skip this trade
                    refund = shares * price
                    self.cash += refund
                    print(f"Short selling disabled: Skipped opening {shares} share short position in {ticker}, refunded ${refund:.2f}")
                    return

                # Short selling allowed - open brand‑new short
                position = StockPosition(
                    asset=ticker,
                    shares=-shares,
                    average_price=price,
                    current_price=price,
                    market_value=-shares * price,
                    unrealized_pnl=0.0,
                )
                self.stock_positions[ticker] = position

        # Update mark‑to‑market
        if ticker in self.stock_positions:
            position.current_price = price
            position.market_value = position.shares * price
            position.unrealized_pnl = position.shares * (price - position.average_price)

    def update_positions(self, current_date: date) -> None:
        """
        Update all positions with current market prices.

        Args:
            current_date: Current simulation date.
        """
        # Update stock positions
        for ticker, position in list(self.stock_positions.items()):
            current_price = self.data_manager.get_stock_price(ticker, current_date)
            if current_price:
                position.current_price = current_price
                position.market_value = position.shares * current_price
                position.unrealized_pnl = position.shares * (
                    current_price - position.average_price
                )

        # Update option positions (simplified)
        for position in list(self.option_positions):
            # Remove expired options
            position.expiration_date = pd.to_datetime(position.expiration_date).date()
            if position.expiration_date <= current_date:
                # Calculate payoff at expiration
                # underlying_price = self.data_manager.get_stock_price(position.asset, current_date)
                # if underlying_price:
                #     if position.option_type == "call":
                #         intrinsic_value = max(0, underlying_price - position.strike)
                #     else:  # put
                #         intrinsic_value = max(0, position.strike - underlying_price)
                # else:
                #     intrinsic_value = 0.0
                # Calculate total payoff value (each contract covers 100 shares)
                current_option_price = (
                    position.market_value / (position.contracts * 100)
                    if position.contracts > 0
                    else 0.0
                )

                # For a long option (action "buy"), closing the position is equivalent to selling.
                # For a short option (action "sell"), closing the position is equivalent to buying (covering).
                if position.action == "buy":
                    # Long option expiring - realize the P&L
                    realized_pnl = position.market_value - position.paid_amount
                    self.cash += position.market_value
                    trade_action = "sell"
                else:
                    # Short option expiring - realize the P&L
                    # paid_amount is negative (credit), market_value is negative (liability)
                    realized_pnl = position.paid_amount - abs(position.market_value)
                    self.cash -= abs(position.market_value)
                    trade_action = "buy"

                # Track realized P&L
                self.cumulative_realized_pnl += realized_pnl

                # Create a trade record for the expiration event
                trade = Trade(
                    asset=position.asset,
                    trade_date=current_date,
                    trade_type="option",
                    action=trade_action,
                    quantity=position.contracts,
                    price=current_option_price,
                    total_value=position.market_value,
                    fees=0.0,  # Adjust fee calculation as needed
                    option_position=position,
                )
                self.trades_history.append(trade)

                # Remove the expired option position
                self.option_positions.remove(position)
                continue
            self._update_option_position_with_future_check(position, current_date)

    def get_portfolio_value(self) -> float:
        """
        Calculate the total portfolio value.

        Returns:
            Total portfolio value (cash + positions).
        """
        # Sum up stock position values
        stock_value = sum(
            position.market_value for position in self.stock_positions.values()
        )

        # Sum up option position values (simplified)
        option_value = sum(position.market_value for position in self.option_positions)

        # Total value
        total_value = self.cash + stock_value + option_value

        return total_value

    def get_portfolio_snapshot(self, current_date: date) -> Portfolio:
        """
        Create a snapshot of the current portfolio state.

        Args:
            current_date: Current simulation date.

        Returns:
            Portfolio object representing the current state.
        """
        # Calculate total value and PnL
        total_value = self.get_portfolio_value()
        unrealized_pnl = sum(
            position.unrealized_pnl for position in self.stock_positions.values()
        )
        # Use tracked realized P&L instead of calculating from formula
        realized_pnl = self.cumulative_realized_pnl

        # Create portfolio snapshot
        portfolio = Portfolio(
            date=current_date,
            cash=self.cash,
            stock_positions=list(self.stock_positions.values()),
            option_positions=self.option_positions,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )

        # Add to history
        self.portfolio_history.append(portfolio)

        return portfolio

    def get_portfolio_summary(self) -> str:
        """
        Generate a summary of the current portfolio.

        Returns:
            Summary string.
        """
        total_value = self.get_portfolio_value()

        summary = f"Portfolio Summary:\n"
        summary += f"Total Value: ${total_value:,.2f}\n"
        summary += f"Cash: ${self.cash:,.2f} ({self.cash/total_value*100:.1f}%)\n\n"

        if self.stock_positions:
            summary += "Stock Positions:\n"
            for ticker, position in self.stock_positions.items():
                summary += f"- {ticker}: {position.shares} shares @ ${position.average_price:.2f}, "
                summary += f"Current: ${position.current_price:.2f}, "
                summary += f"Value: ${position.market_value:,.2f}, "
                summary += f"P&L: ${position.unrealized_pnl:,.2f} "
                summary += (
                    f"({position.unrealized_pnl/position.market_value*100:.1f}%)\n"
                )

        if self.option_positions:
            summary += "\nOption Positions:\n"
            for position in self.option_positions:
                summary += (
                    f"- {position.asset} {position.option_type} {position.strike} "
                )
                summary += f"exp {position.expiration_date}, "
                summary += f"{position.contracts} contracts @ ${position.premium_per_contract:.2f}, "
                summary += f"Total: ${position.market_value:,.2f}\n"

        return summary

    def sell_all_option_positions(self, current_date: date) -> None:
        """
        Sells (or covers) all held option contracts at the current market price.
        For long positions (action "buy"), the sale proceeds are added to cash.
        For short positions (action "sell"), covering the short deducts cash.
        All option positions are then removed from the portfolio.
        """
        for option in list(self.option_positions):
            # Get current option data
            option_data = self.data_manager.get_option_data_by_id(
                option.option_id, option.asset, current_date
            )
            if option_data and "LAST" in option_data:
                current_price = option_data["LAST"]
            else:
                print(
                    f"Could not retrieve current price for option {option.option_id} on {current_date}. Using last known price."
                )
                # Fallback: derive last known price from market_value if available.
                if option.contracts > 0:
                    current_price = option.market_value / (option.contracts * 100)
                else:
                    current_price = option.premium_per_contract

            trade_value = current_price * option.contracts * 100
            fee = self._option_fee(trade_value)

            if option.action == "buy":
                # Selling a long option: add cash
                # Realized P&L = proceeds - what we paid
                realized_pnl = trade_value - fee - option.paid_amount
                print(
                    f"Selling long option {option.option_id} at ${current_price:.2f} (bought at ${option.premium_per_contract:.2f}) for a total of ${trade_value:.2f}"
                )
                self.cash += trade_value - fee
                trade_action = "sell"
            elif option.action == "sell":
                # Covering a short option: buy back, so deduct cash
                # Realized P&L = what we received initially - buyback cost
                # paid_amount is negative (credit received), so: paid_amount - (trade_value + fee)
                realized_pnl = option.paid_amount - (trade_value + fee)
                print(
                    f"Covering short option {option.option_id} at ${current_price:.2f} for a total of ${trade_value:.2f}"
                )
                self.cash -= trade_value + fee
                trade_action = "buy"
            else:
                raise ValueError("Invalid option action. Must be 'buy' or 'sell'.")

            # Track realized P&L
            self.cumulative_realized_pnl += realized_pnl

            # Create trade record
            trade = Trade(
                asset=option.asset,
                trade_date=current_date,
                trade_type="option",
                action=trade_action,
                quantity=option.contracts,
                price=current_price,
                total_value=trade_value,
                fees=fee,
                option_position=option,
            )
            self.trades_history.append(trade)
            # Remove the option position from the portfolio
            self.option_positions.remove(option)

    def sell_all_stocks_position(self, current_date: date) -> None:
        """
        Sells all held stock positions at the current market price.
        The sale proceeds are added to the portfolio's cash and the stock positions are cleared.

        Args:
            current_date: Current simulation date.
        """
        # Make a copy of the tickers since we'll modify the dictionary during iteration
        tickers = list(self.stock_positions.keys())

        for ticker in tickers:
            position = self.stock_positions[ticker]
            current_price = self.data_manager.get_stock_price(ticker, current_date)

            if current_price:
                shares = position.shares

                # For long positions: selling positive shares
                # For short positions: buying back negative shares (closing the short)
                if shares > 0:
                    # Closing long position
                    sale_value = current_price * shares
                    fee = self._equity_fee(sale_value)
                    realized_pnl = (current_price - position.average_price) * shares - fee

                    trade = Trade(
                        asset=ticker,
                        trade_date=current_date,
                        trade_type="stock",
                        action="sell",
                        quantity=shares,
                        price=current_price,
                        total_value=sale_value,
                        fees=fee,
                    )

                    self.cash += sale_value - fee
                    print(f"Closing long: Sold {shares} shares of {ticker} at ${current_price:.2f} for ${sale_value:.2f}")

                elif shares < 0:
                    # Closing short position (buying back)
                    abs_shares = abs(shares)
                    buy_value = current_price * abs_shares
                    fee = self._equity_fee(buy_value)
                    # For shorts: profit if current price < average price (we sold high, buying back low)
                    realized_pnl = (position.average_price - current_price) * abs_shares - fee

                    trade = Trade(
                        asset=ticker,
                        trade_date=current_date,
                        trade_type="stock",
                        action="buy",  # Closing a short is a buy action
                        quantity=abs_shares,
                        price=current_price,
                        total_value=buy_value,
                        fees=fee,
                    )

                    self.cash -= buy_value + fee
                    print(f"Closing short: Bought back {abs_shares} shares of {ticker} at ${current_price:.2f} for ${buy_value:.2f}")

                # Track cumulative realized P&L
                self.cumulative_realized_pnl += realized_pnl
                self.trades_history.append(trade)

                # Remove the stock position
                del self.stock_positions[ticker]
            else:
                print(
                    f"Could not retrieve current price for {ticker} on {current_date}."
                )

    def apply_unused_options_capital(
        self,
        ticker: str,
        allocated_options_capital: float,
        used_options_capital: float,
        current_date: date,
    ) -> None:
        """
        Uses any unused options capital to buy additional stocks for the given asset.

        Args:
            ticker: The asset ticker symbol.
            allocated_options_capital: The total capital allocated to options for the asset.
            used_options_capital: The capital actually used by the options hedging strategy.
            current_date: The current date for pricing.
        """
        unused_options_capital = allocated_options_capital - used_options_capital
        if unused_options_capital > 0:
            current_price = self.data_manager.get_stock_price(ticker, current_date)
            if current_price and current_price > 0:
                additional_shares = int(unused_options_capital / current_price)
                if additional_shares > 0:
                    additional_purchase_value = additional_shares * current_price
                    # Deduct the cost from cash
                    self.cash -= additional_purchase_value
                    # Create a trade record for the additional stock purchase
                    additional_trade = Trade(
                        asset=ticker,
                        trade_date=current_date,
                        trade_type="stock",
                        action="buy",
                        quantity=additional_shares,
                        price=current_price,
                        total_value=additional_purchase_value,
                        fees=0.0,  # Adjust fee calculation as needed
                    )
                    self.trades_history.append(additional_trade)
                    # Update the stock position with the additional shares
                    self._update_stock_position(
                        ticker, additional_shares, current_price, "buy"
                    )
                    print(
                        f"Additional stock purchase for {ticker}: Bought {additional_shares} shares for ${additional_purchase_value:.2f} using unused options capital (${unused_options_capital:.2f})."
                    )

    def get_profit_loss_breakdown(
        self, if_print=True, add_realized_pnl=False
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate the profit or loss for each asset, broken down by stock and option trades.

        Returns:
            A dictionary keyed by asset ticker symbol, where each value is a dictionary with keys
            'stock' and 'option' representing the net P&L for each category.
        """
        pnl_breakdown = {}

        for trade in self.trades_history:
            asset = trade.asset
            if asset not in pnl_breakdown:
                pnl_breakdown[asset] = {"stock": 0.0, "option": {}}

            # For stock trades, add sell proceeds and subtract buy costs (including fees)
            if trade.trade_type == "stock":
                if trade.action == "sell":
                    pnl_breakdown[asset]["stock"] += trade.total_value - trade.fees
                else:  # buy
                    pnl_breakdown[asset]["stock"] -= trade.total_value + trade.fees

            # Similarly for options trades
            elif trade.trade_type == "option":
                strategy = (
                    trade.option_position.strategy.lower()
                    if trade.option_position
                    and hasattr(trade.option_position, "strategy")
                    else "unknown"
                )
                if not hasattr(trade.option_position, "strategy"):
                    print("")
                if strategy not in pnl_breakdown[asset]["option"]:
                    pnl_breakdown[asset]["option"][strategy] = 0.0

                if trade.action == "sell":
                    pnl_breakdown[asset]["option"][strategy] += (
                        abs(trade.total_value) - trade.fees
                    )
                else:  # buy (cover or long option)
                    pnl_breakdown[asset]["option"][strategy] -= abs(
                        trade.total_value + trade.fees
                    )
        if add_realized_pnl:
            # Now, add the unrealized P&L from open stock positions.
            for asset, position in self.stock_positions.items():
                if asset not in pnl_breakdown:
                    pnl_breakdown[asset] = {"stock": 0.0, "option": {}}
                # Assuming your position.unrealized_pnl is already computed correctly.
                pnl_breakdown[asset]["stock"] += position.market_value

            # Also, add unrealized P&L from open option positions.
            for option in self.option_positions:
                asset = option.asset
                if asset not in pnl_breakdown:
                    pnl_breakdown[asset] = {"stock": 0.0, "option": {}}
                strategy = (
                    option.strategy.lower()
                    if (hasattr(option, "strategy") and option.strategy)
                    else "unknown"
                )
                if strategy not in pnl_breakdown[asset]["option"]:
                    pnl_breakdown[asset]["option"][strategy] = 0.0
                pnl_breakdown[asset]["option"][strategy] += option.market_value
        if if_print:
            print("Profit/Loss Breakdown per Asset:")
            # Compute overall P&L per asset as sum of stock P&L and option sub-P&Ls.
            for asset, values in pnl_breakdown.items():
                option_pnl = sum(values["option"].values())
                overall = values["stock"] + option_pnl
                values["overall"] = overall
                option_sum = sum(values["option"].values())
                print(
                    f"{asset}: Overall: {overall:+,.2f} | Stock: {values['stock']:+,.2f} | Options: {option_sum:+,.2f} | {values['option']}"
                )

        return pnl_breakdown

    def _update_option_position_with_future_check(
        self, position: OptionPosition, current_date: date
    ) -> None:
        """
        Check for valid option data for the given option position within the next 5 days.
        If found, update the position's market value and unrealized PnL using the average of the
        last known price and the first valid future price.
        If not found, sell the option at the last known price.

        Args:
            position: The OptionPosition to update.
            current_date: The current simulation date.
        """
        # Try to get current option data
        option_data = self.data_manager.get_option_data_by_id(
            position.option_id, position.asset, current_date
        )
        if option_data and "LAST" in option_data:
            # If data is available today, update as usual
            current_price = self._get_sanitized_option_price(
                position, option_data, current_date
            )
            if np.isnan(current_price):
                current_price = option_data["LAST"]

            option_value = current_price * position.contracts * 100
            if np.isnan(option_value):
                option_value = 0.0
            position.market_value = option_value
            position.unrealized_pnl = position.market_value - abs(position.paid_amount)
            if position.action == "sell":
                position.market_value = -position.market_value
                position.unrealized_pnl = -position.unrealized_pnl
            return

        # No valid data today: Look ahead up to 5 days
        future_price = None
        for i in range(1, 6):
            future_date = current_date + timedelta(days=i)
            future_data = self.data_manager.get_option_data_by_id(
                position.option_id, position.asset, future_date
            )
            if future_data and "LAST" in future_data:
                future_price = self._get_sanitized_option_price(
                    position, future_data, future_date
                )
                if np.isnan(future_price):
                    future_price = future_data["LAST"]
                break

        if future_price is not None:
            # Compute last known price from stored market value (if contracts > 0)
            if position.contracts > 0:
                last_known_price = abs(position.market_value) / (position.contracts * 100)
            else:
                last_known_price = (
                    position.premium_per_contract
                )  # fallback if contracts is zero
            # Average the last known price and the future price
            avg_price = (last_known_price + future_price) / 2.0
            option_value = avg_price * position.contracts * 100
            if np.isnan(option_value):
                option_value = 0.0
            position.market_value = option_value
            position.unrealized_pnl = position.market_value - abs(position.paid_amount)
            # Handle short positions
            if position.action == "sell":
                position.market_value = -position.market_value
                position.unrealized_pnl = -position.unrealized_pnl
        else:
            # No valid future data found; sell the option at the last known price
            if position.contracts > 0:
                last_known_price = position.market_value / (position.contracts * 100)
            else:
                last_known_price = position.premium_per_contract
            sale_value = last_known_price * position.contracts * 100
            print(
                f"Option data for {position.option_id} not found in next 5 days. Selling at last known price ${last_known_price:.2f} for total ${sale_value:.2f}."
            )
            # Update cash (assuming no fees for simplicity; adjust if needed)
            self.cash += sale_value
            # Record the trade
            trade = Trade(
                asset=position.asset,
                trade_date=current_date,
                trade_type="option",
                action="sell",
                quantity=position.contracts,
                price=last_known_price,
                total_value=sale_value,
                fees=0.0,
                option_position=position,
            )
            self.trades_history.append(trade)
            # Remove the position
            self.option_positions.remove(position)
