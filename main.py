"""
Main entry point for the hedge fund simulation system.
Provides a command-line interface for running simulations.
"""

import os
import sys
import argparse
import json
import random
import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure deterministic simulations
np.random.seed(42)
random.seed(42)

# Configure logging once at import
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import configuration and models
import config
from src.models.pydantic_models import SimulationConfig
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig
from src.core.coordinator import Coordinator


def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hedge Fund Simulation System")

    # Simulation parameters
    parser.add_argument("--start-date", type=parse_date, default=config.START_DATE,
                        help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=parse_date, default=config.END_DATE,
                        help="End date for simulation (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=config.INITIAL_CAPITAL,
                        help="Initial capital for simulation")
    parser.add_argument("--rebalance-frequency", type=int, default=config.REBALANCE_FREQUENCY,
                        help="Rebalance frequency in days")
    parser.add_argument("--emergency-threshold", type=float, default=config.EMERGENCY_THRESHOLD,
                        help="Emergency rebalance threshold (drawdown percentage)")

    parser.add_argument("--use-memory-rl", type=bool, default=config.use_memory_rl,
                        help="Use memory-based RL for decision making")
    parser.add_argument("--controller-mode",
                        type=str,
                        default=getattr(config, "CONTROLLER_MODE", "full_llm"),
                        choices=["full_llm", "retrieval_only", "random_retrieval", "no_memory"],
                        help="Controller variant to run.")
    # Assets
    parser.add_argument("--assets", type=str, nargs="+", default=config.ASSETS,
                        help="Assets to include in simulation")

    # Hedging strategies
    parser.add_argument("--hedging-strategies", type=str, nargs="+", default=config.HEDGING_STRATEGIES,
                        choices=["collar", "straddle", "delta-neutral", "rules-only"],
                        help="Hedging strategies to use")

    parser.add_argument("--equal-options-strategy-split",
                        action="store_true",
                        dest="equal_options_strategy_split",
                        default=config.EQUAL_OPTIONS_STRATEGY_SPLIT,
                        help="Divide any options allocation equally across the configured hedging strategies.")
    parser.add_argument("--no-equal-options-strategy-split",
                        action="store_false",
                        dest="equal_options_strategy_split",
                        help="Use strategy weights returned by the agent instead of forcing equal splits.")
    parser.add_argument("--max-options-allocation-ratio",
                        type=float,
                        default=config.MAX_OPTIONS_ALLOCATION_RATIO,
                        help="Maximum fraction of each asset allocation that can go to options.")
    parser.add_argument("--max-collar-options-weight",
                        type=float,
                        default=config.MAX_COLLAR_OPTIONS_WEIGHT,
                        help="Maximum fraction of the options sleeve that can go to collars.")
    parser.add_argument("--allow-short-selling",
                        action="store_true",
                        default=config.ALLOW_SHORT_SELLING,
                        help="Allow stock allocations to open or increase short positions.")
    parser.add_argument("--no-short-selling",
                        action="store_false",
                        dest="allow_short_selling",
                        help="Force all stock allocations to use buy actions.")

    # LLM settings
    parser.add_argument("--use-llm", action="store_true", default=config.USE_LLM,
                        help="Use LLM for decision making")
    parser.add_argument("--no-llm", action="store_false", dest="use_llm",
                        help="Don't use LLM for decision making")

    # Output settings
    default_output_dir = config.DEFAULT_OUTPUT_DIR
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Directory for output files")
    parser.add_argument("--equity-transaction-cost-bps", type=float, default=10.0,
                        help="Equity transaction cost in basis points.")
    parser.add_argument("--options-transaction-cost-bps", type=float, default=20.0,
                        help="Options transaction cost in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=2.0,
                        help="Execution slippage in basis points.")
    parser.add_argument("--generate-plots", action="store_true", default=config.GENERATE_PLOTS,
                        help="Generate performance plots")
    parser.add_argument("--save-results", action="store_true", default=config.SAVE_RESULTS,
                        help="Save simulation results to file")

    return parser.parse_args()


def create_simulation_config(args):
    """Create simulation configuration from arguments."""
    return ExtendedSimulationConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        assets=args.assets,
        rebalance_frequency=args.rebalance_frequency,
        emergency_threshold=args.emergency_threshold,
        use_llm=args.use_llm,
        controller_mode=args.controller_mode,
        hedging_strategies=args.hedging_strategies,
        use_memory_rl=args.use_memory_rl,
        equal_options_strategy_split=args.equal_options_strategy_split,
        max_options_allocation_ratio=args.max_options_allocation_ratio,
        max_collar_options_weight=args.max_collar_options_weight,
        allow_short_selling=args.allow_short_selling,
        equity_transaction_cost_bps=args.equity_transaction_cost_bps,
        options_transaction_cost_bps=args.options_transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )


def ensure_output_directory(output_dir):
    """Ensure output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Created output directory: %s", output_dir)

def save_simulation_results(result, output_dir):
    """Save simulation results to files."""
    # Ensure output directory exists
    ensure_output_directory(output_dir)

    # Save performance data to CSV (daily performance metrics)
    performance_df = pd.DataFrame([p.dict() for p in result.daily_performances])
    performance_file = os.path.join(output_dir, "performance.csv")
    performance_df.to_csv(performance_file, index=False)
    logger.info("Performance data saved to %s", performance_file)

    # Save summary to JSON with additional details:
    # - profit_loss_breakdown: a detailed per-asset P&L from stock and options trades.
    # - portfolio_summary: a textual snapshot of the portfolio state.
    summary = {
        "initial_capital": result.config.initial_capital,
        "final_value": result.final_portfolio.total_value,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "trades_executed": result.trades_executed,
        "best_performing_asset": result.best_performing_asset,
        "worst_performing_asset": result.worst_performing_asset,
        "profit_loss_breakdown": result.profit_loss_breakdown,  # New field with detailed P&L per asset
        "portfolio_summary": result.portfolio_summary,          # New field with a detailed portfolio snapshot
        "simulation_period": f"{result.config.start_date} to {result.config.end_date}",
        "assets": result.config.assets
    }

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4, default=str)
    logger.info("Summary saved to %s", summary_file)

    # Save final portfolio state
    portfolio_data = {
        "date": str(result.final_portfolio.date),
        "cash": result.final_portfolio.cash,
        "total_value": result.final_portfolio.total_value,
        "stock_positions": [p.dict() for p in result.final_portfolio.stock_positions],
        "option_positions": [p.dict() for p in result.final_portfolio.option_positions]
    }

    portfolio_file = os.path.join(output_dir, "final_portfolio.json")
    with open(portfolio_file, "w") as f:
        json.dump(portfolio_data, f, indent=4, default=str)
    logger.info("Final portfolio saved to %s", portfolio_file)



def main():
    """Main function to run the simulation."""
    # Parse command line arguments
    args = parse_args()

    # Create simulation configuration
    sim_config = create_simulation_config(args)

    logger.info("=" * 80)
    logger.info("Hedge Fund Simulation System")
    logger.info("=" * 80)
    logger.info(
        "Simulation period: %s to %s", sim_config.start_date, sim_config.end_date
    )
    logger.info("Initial capital: $%s", f"{sim_config.initial_capital:,.2f}")
    logger.info("Assets: %s", ", ".join(sim_config.assets))
    logger.info("Using LLM: %s", "Yes" if sim_config.use_llm else "No")
    logger.info("Controller mode: %s", sim_config.controller_mode)
    logger.info("Hedging strategies: %s", ", ".join(sim_config.hedging_strategies))
    if getattr(sim_config, "equal_options_strategy_split", False):
        logger.info(
            "Options allocations will be split equally across hedging strategies."
        )
    logger.info("=" * 80)

    # Create coordinator and run simulation
    coordinator = Coordinator(sim_config)

    logger.info("Starting simulation...")
    result = coordinator.run_simulation()

    # Print performance report
    logger.info("\n%s", coordinator.get_performance_report())

    # Generate plots if requested
    if args.generate_plots:
        ensure_output_directory(args.output_dir)
        plot_file = os.path.join(args.output_dir, "performance_plot.png")
        coordinator.plot_performance(plot_file)

    # Save results if requested
    if args.save_results:
        save_simulation_results(result, args.output_dir)

    logger.info("Simulation completed successfully.")
    logger.info(
        "Output folder '%s' located at: %s",
        config.OUTPUT_FOLDER_NAME,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
