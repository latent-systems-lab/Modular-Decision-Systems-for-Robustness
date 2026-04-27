"""Run the (H2) rules-only selector hedging simulation."""

import os
import random

import numpy as np

import config
from main import ensure_output_directory, save_simulation_results
from src.core.coordinator import Coordinator
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig

# Ensure deterministic simulations
np.random.seed(42)
random.seed(42)


def build_h2_config() -> ExtendedSimulationConfig:
    """Create a simulation configuration tailored to the rules-only selector."""
    return ExtendedSimulationConfig(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        initial_capital=config.INITIAL_CAPITAL,
        assets=config.ASSETS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        emergency_threshold=config.EMERGENCY_THRESHOLD,
        hedging_strategies=config.HEDGING_STRATEGIES,
        use_llm=True,
        use_memory_rl=False,
        rule_based_strategy_split=False,
    )


def main() -> None:
    """Run the coordinator with the rules-only selector configuration."""
    simulation_config = build_h2_config()
    coordinator = Coordinator(simulation_config)

    print("=" * 80)
    print("Running (H2) Rules-Only Selector Simulation")
    print("=" * 80)
    print(f"Simulation period: {simulation_config.start_date} to {simulation_config.end_date}")
    print(f"Initial capital: ${simulation_config.initial_capital:,.2f}")
    print(f"Assets: {', '.join(simulation_config.assets)}")
    print("Hedging strategy: Rules-Only Selector")
    print("=" * 80)

    result = coordinator.run_simulation()

    print("\nPerformance Summary:")
    for metric, value in coordinator.get_performance_summary().items():
        print(f"{metric}: {value}")

    if result:
        print("\nFinal portfolio value:", f"${result.final_portfolio.total_value:,.2f}")

    output_dir = config.DEFAULT_OUTPUT_DIR
    ensure_output_directory(output_dir)

    performance_plot = os.path.join(output_dir, "performance_plot.png")
    coordinator.plot_performance(performance_plot)

    if result:
        save_simulation_results(result, output_dir)
        coordinator.performance_tracker.convert_jsonl_to_json()

    print("Simulation completed successfully.")
    print(f"Output folder '{config.OUTPUT_FOLDER_NAME}' located at: {output_dir}")


if __name__ == "__main__":
    main()
