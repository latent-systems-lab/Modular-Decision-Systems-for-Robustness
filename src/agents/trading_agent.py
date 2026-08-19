from src.models.pydantic_models import TradingAllocationModel
from src.utils.llm_utils import get_llm_response


class TradingAgent:
    """
    Determines the optimal split between stock and options allocation for an asset.
    """

    def __init__(self, asset_name: str):
        self.asset_name = asset_name
        self.allocation = 0
        self.stock_percentage = 0

    def determine_stock_allocation(self, asset_data, total_allocation: float) -> dict:
        """
        Uses an LLM to determine what percentage of total_allocation should go to stock.
        Returns a dictionary with 'stock_allocation' and 'options_allocation'.
        """
        self.allocation = total_allocation
        prompt = self._build_trading_prompt(asset_data)
        result = get_llm_response(prompt, TradingAllocationModel)

        reasoning = ""
        if isinstance(result, TradingAllocationModel):
            self.stock_percentage = result.stock_percentage
            reasoning = result.reasoning
        elif isinstance(result, dict):
            self.stock_percentage = float(result.get("stock_percentage", 0.7))
            reasoning = result.get("reasoning", "")
        else:
            self.stock_percentage = 0.7

        stock_allocation = self.allocation * self.stock_percentage
        options_allocation = self.allocation - stock_allocation
        print(f"{self.asset_name} Trading Reasoning:", reasoning)
        return {
            "stock_allocation": stock_allocation,
            "options_allocation": options_allocation,
        }

    def _build_trading_prompt(self, asset_data) -> str:
        """
        Constructs a prompt with key technical indicators for the asset.
        Uses only the last available RSI value from the rolling series.
        """
        latest_price = asset_data["Close"].iloc[-1]

        # Safe rolling means
        ma5 = (
            asset_data["Close"].rolling(5).mean().iloc[-1]
            if len(asset_data) >= 5
            else latest_price
        )
        ma20 = (
            asset_data["Close"].rolling(20).mean().iloc[-1]
            if len(asset_data) >= 20
            else latest_price
        )

        # Compute RSI from the last data point
        if len(asset_data) >= 14:
            delta = asset_data["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()

            # If total loss is zero, we can't do gain/loss ratio
            if loss.sum() == 0:
                # Fallback: assume RSI=50 if there's no negative movement
                rsi_val = 50
            else:
                rs_series = gain / loss
                # Use the last value of rs_series
                if not rs_series.empty:
                    last_rs = rs_series.iloc[-1]
                    rsi_val = 100 - (100 / (1 + last_rs)) if last_rs != 0 else 50
                else:
                    rsi_val = 50
        else:
            rsi_val = 50  # Fallback if not enough data for 14-day RSI

        trend = "Bullish" if ma5 > ma20 else "Bearish"

        prompt = f"""
        For asset {self.asset_name}, based on the following data, determine the optimal stock allocation percentage (between 0.6 and 0.8).
        Data:
        - Price: {latest_price:.2f}
        - 5-day MA: {ma5:.2f}
        - 20-day MA: {ma20:.2f}
        - RSI: {rsi_val:.2f}
        - Trend: {trend}
        Also, provide a brief reasoning.
        Output as JSON in the format: {{"stock_percentage": 0.7, "reasoning": "your explanation here"}}
        """
        return prompt.strip()
