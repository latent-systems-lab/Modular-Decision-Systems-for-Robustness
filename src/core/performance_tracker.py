"""
Performance Tracker for the hedge fund simulation system.
Responsible for tracking and reporting portfolio performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Optional
import os
import json

# Import models and utilities

import config
from src.models.pydantic_models import Portfolio, DailyPerformance, SimulationResult
from src.utils.financial_utils import calculate_sharpe_ratio


class PerformanceTracker:
    """
    Tracks and reports portfolio performance metrics.
    """

    def __init__(self, initial_capital: float, start_date: date):
        """
        Initialize the PerformanceTracker.

        Args:
            initial_capital: Initial portfolio capital.
            start_date: Start date of the simulation.
        """
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.daily_performances = []  # List of DailyPerformance objects
        self.previous_portfolio_value = initial_capital
        self.peak_portfolio_value = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_date = None

        # Determine a unique file to store the portfolio history for this run to
        # avoid overwriting previous simulation results.
        default_history_file = os.path.join(
            config.DEFAULT_OUTPUT_DIR, "portfolio_history.jsonl"
        )
        self.history_file = self._get_unique_file_path(default_history_file)

    def update(
        self, portfolio: Portfolio, daily_breakdown: Dict[str, Dict[str, float]]
    ) -> DailyPerformance:
        """
        Update performance metrics based on current portfolio state.

        Args:
            portfolio: Current portfolio state.

        Returns:
            DailyPerformance object with updated metrics.
        """
        # Calculate daily return
        daily_return = (
            (portfolio.total_value / self.previous_portfolio_value) - 1
            if self.previous_portfolio_value > 0
            else 0
        )

        # Calculate daily PnL
        daily_pnl = portfolio.total_value - self.previous_portfolio_value

        # Calculate cumulative return
        cumulative_return = (portfolio.total_value / self.initial_capital) - 1

        # Calculate cumulative PnL
        cumulative_pnl = portfolio.total_value - self.initial_capital

        # Calculate drawdown
        if portfolio.total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio.total_value

        current_drawdown = 0.0
        if self.peak_portfolio_value > 0:
            current_drawdown = (
                self.peak_portfolio_value - portfolio.total_value
            ) / self.peak_portfolio_value

        # Update max drawdown
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            self.max_drawdown_date = portfolio.date

        # Create daily performance object
        performance = DailyPerformance(
            date=portfolio.date,
            portfolio_value=portfolio.total_value,
            daily_return=daily_return,
            daily_pnl=daily_pnl,
            cumulative_return=cumulative_return,
            cumulative_pnl=cumulative_pnl,
            drawdown=current_drawdown,
            asset_pnl_breakdown=daily_breakdown or {},
        )

        # Add to history
        self.daily_performances.append(performance)

        # Update previous portfolio value for next calculation
        self.previous_portfolio_value = portfolio.total_value

        return performance

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Generate a summary of performance metrics.

        Returns:
            Dictionary of performance metrics.
        """
        if not self.daily_performances:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0,
                "average_daily_return": 0.0,
            }

        # Calculate metrics
        total_return = self.daily_performances[-1].cumulative_return

        # Calculate annualized return
        days = (self.daily_performances[-1].date - self.start_date).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = 0.0

        # Calculate volatility and Sharpe ratio
        daily_returns = [perf.daily_return for perf in self.daily_performances]
        volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0.0
        annualized_volatility = volatility * np.sqrt(252) if volatility > 0 else 0.0

        risk_free_rate = 0.0  # Simplified assumption
        sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)

        # Calculate average daily return
        average_daily_return = np.mean(daily_returns) if daily_returns else 0.0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "volatility": annualized_volatility,
            "average_daily_return": average_daily_return,
        }

    def get_performance_dataframe(self) -> pd.DataFrame:
        """
        Convert daily performances to a pandas DataFrame.

        Returns:
            DataFrame containing performance metrics.
        """
        if not self.daily_performances:
            return pd.DataFrame()

        # Convert to dictionaries
        data = [perf.dict() for perf in self.daily_performances]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Set date as index
        if "date" in df.columns:
            df.set_index("date", inplace=True)

        return df

    def generate_simulation_result(
        self, config, final_portfolio: Portfolio, coordinator
    ) -> SimulationResult:
        # Get performance summary
        summary = self.get_performance_summary()

        # Calculate additional metrics
        trades_executed = len(coordinator.portfolio_manager.trades_history)

        # Get a detailed profit/loss breakdown per asset from the Portfolio Manager
        profit_loss_breakdown = (
            coordinator.portfolio_manager.get_profit_loss_breakdown()
        )

        # Get a full portfolio summary (stocks, options, and their individual details)
        portfolio_summary = coordinator.portfolio_manager.get_portfolio_summary()

        # Determine best and worst performing assets based on stock positions
        asset_returns = {}
        for position in final_portfolio.stock_positions:
            if position.average_price > 0:
                asset_return = (position.current_price / position.average_price) - 1
                asset_returns[position.asset] = asset_return

        best_performing_asset = (
            max(asset_returns.items(), key=lambda x: x[1])[0] if asset_returns else ""
        )
        worst_performing_asset = (
            min(asset_returns.items(), key=lambda x: x[1])[0] if asset_returns else ""
        )

        # Create and return the simulation result with extra details
        result = SimulationResult(
            config=config,
            final_portfolio=final_portfolio,
            daily_performances=self.daily_performances,
            total_return=summary["total_return"],
            annualized_return=summary["annualized_return"],
            max_drawdown=summary["max_drawdown"],
            sharpe_ratio=summary["sharpe_ratio"],
            trades_executed=trades_executed,
            average_position_duration=30.0,  # Simplified assumption
            best_performing_asset=best_performing_asset,
            worst_performing_asset=worst_performing_asset,
            profit_loss_breakdown=profit_loss_breakdown,  # New field with detailed P&L per asset
            portfolio_summary=portfolio_summary,  # New field with detailed portfolio snapshot
        )

        return result

    def report(self) -> str:
        """
        Generate a performance report.

        Returns:
            Report string.
        """
        summary = self.get_performance_summary()

        report = "Performance Report\n"
        report += "=================\n\n"

        report += f"Initial Capital: ${self.initial_capital:,.2f}\n"
        if self.daily_performances:
            final_value = self.daily_performances[-1].portfolio_value
            report += f"Final Portfolio Value: ${final_value:,.2f}\n"
            report += f"Total Return: {summary['total_return']:.2%}\n"
            report += f"Annualized Return: {summary['annualized_return']:.2%}\n"
            report += f"Maximum Drawdown: {summary['max_drawdown']:.2%}"
            if self.max_drawdown_date:
                report += f" (on {self.max_drawdown_date})\n"
            else:
                report += "\n"
            report += f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}\n"
            report += f"Annualized Volatility: {summary['volatility']:.2%}\n"
            report += f"Average Daily Return: {summary['average_daily_return']:.2%}\n"
        else:
            report += "No performance data available.\n"

        return report

    def plot_performance(self, output_file: str = None) -> None:
        """
        Plot performance metrics.

        Args:
            output_file: Optional file path to save the plot.
        """
        try:
            import matplotlib.pyplot as plt

            # Convert to DataFrame
            df = self.get_performance_dataframe()

            if df.empty:
                print("No performance data to plot.")
                return

            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

            # Plot portfolio value
            df["portfolio_value"].plot(
                ax=axes[0], title="Portfolio Value", color="blue"
            )
            axes[0].set_ylabel("Value ($)")
            axes[0].grid(True)

            # Plot cumulative return
            df["cumulative_return"].plot(
                ax=axes[1], title="Cumulative Return", color="green"
            )
            axes[1].set_ylabel("Return (%)")
            axes[1].grid(True)
            axes[1].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.1%}")
            )

            # Plot drawdown
            df["drawdown"].plot(ax=axes[2], title="Drawdown", color="red")
            axes[2].set_ylabel("Drawdown (%)")
            axes[2].grid(True)
            axes[2].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.1%}")
            )
            axes[2].invert_yaxis()  # Invert y-axis for drawdown

            # Format x-axis
            plt.xlabel("Date")
            plt.tight_layout()

            # Save or show
            if output_file:
                plt.savefig(output_file)
                print(f"Performance plot saved to {output_file}")
            else:
                plt.show()

        except ImportError:
            print(
                "Matplotlib is required for plotting. Install with: pip install matplotlib"
            )
        except Exception as e:
            print(f"Error plotting performance: {str(e)}")

    def append_portfolio_snapshot(self, portfolio):
        """
        Append a portfolio snapshot to a JSON Lines file with additional computed fields.
        Date and total_value appear first for quick scanning.

        Args:
            portfolio: The Portfolio object (a pydantic model) from get_portfolio_snapshot.
        """

        # Use the unique history file path determined during initialization.
        history_file = self.history_file
        # Ensure that the directory exists.
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        # Convert the portfolio to a dictionary.
        snapshot = portfolio.dict()

        # Compute the total value of stock positions.
        snapshot["stock_value"] = sum(
            pos["market_value"] for pos in snapshot.get("stock_positions", [])
        )
        # Compute the total value of option positions.
        snapshot["option_value"] = sum(
            pos["market_value"] for pos in snapshot.get("option_positions", [])
        )
        # Recompute the total portfolio value as cash plus stock and option values.
        snapshot["total_value"] = (
            snapshot["cash"] + snapshot["stock_value"] + snapshot["option_value"]
        )

        # Reorder keys to put date and total_value at the top for quick scanning
        ordered_snapshot = {}
        ordered_snapshot["date"] = snapshot.pop("date")
        ordered_snapshot["total_value"] = snapshot.pop("total_value")
        # Add all remaining fields
        ordered_snapshot.update(snapshot)

        # Open the file in append mode and write the updated snapshot as a new line.
        with open(history_file, "a") as f:
            json_snapshot = json.dumps(ordered_snapshot, default=str)
            f.write(json_snapshot + "\n")

        print(f"Appended portfolio snapshot for {portfolio.date} to {history_file}")

    def convert_jsonl_to_json(
        self, jsonl_file: Optional[str] = None, json_file: Optional[str] = None
    ):
        """
        Convert a JSON Lines file into a single JSON file containing a list of objects.
        If the target json_file already exists, append an incrementing number before the extension.

        Args:
            jsonl_file: Path to the input .jsonl file. Defaults to the tracker history file.
            json_file: Path to the output .json file. Defaults to match the history file name.
        """
        if jsonl_file is None:
            jsonl_file = self.history_file

        if json_file is None:
            base_name = os.path.splitext(os.path.basename(jsonl_file))[0]
            json_file = os.path.join(config.DEFAULT_OUTPUT_DIR, f"{base_name}.json")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        # If target file exists, create a new filename with an incrementing suffix
        base, ext = os.path.splitext(json_file)
        candidate = json_file
        counter = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{counter}{ext}"
            counter += 1
        output_file = candidate

        data = []
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    # Each line is a valid JSON object; strip newline and parse.
                    try:
                        record = json.loads(line.strip())
                        data.append(record)
                    except json.JSONDecodeError:
                        print("Warning: Could not decode a line. Skipping.")
        except FileNotFoundError:
            print(f"Input file not found: {jsonl_file}")
            return

        # Write the complete list as a JSON array to the output file.
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Converted {jsonl_file} to {output_file}")

    @staticmethod
    def _get_unique_file_path(file_path: str) -> str:
        """
        Return a unique file path by appending an incrementing suffix to the output
        folder name (e.g., openai-1, openai-2, ...) if a file with the same name
        already exists in the configured folder.
        """
        dir_path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        # If no directory part, fall back to file-based unique naming
        if not dir_path:
            base, ext = os.path.splitext(file_path)
            candidate = file_path
            counter = 1
            while os.path.exists(candidate):
                candidate = f"{base}_{counter}{ext}"
                counter += 1
            return candidate

        parent_dir = os.path.dirname(dir_path)
        folder_name = os.path.basename(dir_path)

        # Start with the configured directory; if the target file exists there,
        # create numbered folders like folder_name-1, folder_name-2, ...
        candidate_dir = dir_path
        candidate_file = os.path.join(candidate_dir, filename)
        counter = 1
        while os.path.exists(candidate_file):
            candidate_dir = os.path.join(parent_dir, f"{folder_name}-{counter}")
            candidate_file = os.path.join(candidate_dir, filename)
            counter += 1

        return candidate_file
