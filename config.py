"""
Configuration file for the hedge fund simulation system.
Contains global configuration constants and settings.
"""

import os
from datetime import date

# Simulation settings
START_DATE = date(2021, 1, 1)
END_DATE = date(2023, 12, 31)
INITIAL_CAPITAL = 1000000.0  # $1M initial capital
REBALANCE_FREQUENCY = 90
EMERGENCY_THRESHOLD = 0.1  # 10% drawdown triggers emergency rebalance

# Assets to include in simulation
ASSETS = ["AAPL", "NVDA", "TSLA"]
use_memory_rl = True  # Use memory-based RL for decision making
# Hedging strategies to use
# Options: "collar", "straddle", "delta-neutral", "protective-put", "rules-only"
HEDGING_STRATEGIES = ["collar", "straddle", "delta-neutral"]

# Convenience script entry points for packaged strategy runs
RULES_ONLY_SELECTOR_SCRIPT = os.path.join("experiments", "run_h2_rules_only_selector.py")

# Options allocation settings
EQUAL_OPTIONS_STRATEGY_SPLIT = False
MAX_OPTIONS_ALLOCATION_RATIO = 0.03
MAX_COLLAR_OPTIONS_WEIGHT = 0.05

# Option contract sizing
OPTION_CONTRACT_SIZE = 100
DEFAULT_OPTION_PREMIUM = 10.0

# Maximum fraction of available capital to deploy per individual option leg
MAX_OPTION_LEG_CAPITAL_RATIO = 1.0

# Short selling settings
ALLOW_SHORT_SELLING = False

# LLM settings
USE_LLM = True  # Whether to use LLM for decision making
CONTROLLER_MODE = "full_llm"
# Supported providers: "gemini" or "openai". Defaults can be overridden via environment variables.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash-lite" if LLM_PROVIDER.lower() == "gemini" else "gpt-5",
)
DEFAULT_GPT_MODEL = os.getenv("DEFAULT_GPT_MODEL", "gpt-5")

# Data settings
DATA_DIR = "data"
STOCK_DATA_FORMAT = "csv"
OPTIONS_DATA_FORMAT = "sqlite"

# Performance settings
RISK_FREE_RATE = 0.03  # 3% annual risk-free rate for Sharpe ratio calculation

# Reporting settings
GENERATE_PLOTS = True
SAVE_RESULTS = True
RESULTS_DIR = "results"
OUTPUT_FOLDER_NAME = "experiments/results"
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, OUTPUT_FOLDER_NAME)
PAKDD_RESULTS_SUBDIR = os.getenv("PAKDD_RESULTS_SUBDIR", "experiments")
PAKDD_OUTPUT_DIR = os.path.join(RESULTS_DIR, PAKDD_RESULTS_SUBDIR)

# Strategy parameterization
STRADDLE_MIN_DTE = 38
STRADDLE_MAX_DTE = 60
STRADDLE_ATM_TOLERANCE = 0.05

COLLAR_MIN_DTE = 30
COLLAR_MAX_DTE = 60
COLLAR_PUT_STRIKE_LOWER_RATIO = 0.85
COLLAR_PUT_STRIKE_UPPER_RATIO = 0.90
COLLAR_CALL_STRIKE_LOWER_RATIO = 1.10
COLLAR_CALL_STRIKE_UPPER_RATIO = 1.20
