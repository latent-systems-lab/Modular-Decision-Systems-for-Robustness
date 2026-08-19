"""
Market Analyzer for the hedge fund simulation system.
Responsible for analyzing market conditions and determining regimes and sentiment.
"""

import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, List
import json

# Import models and utilities
from src.core.data_manager import DataManager
from src.models.pydantic_models import (
    BatchMarketAnalysisResponse,
    MarketRegime,
    StockData,
)
from src.utils.llm_utils import get_llm_response, build_market_analysis_prompt


class MarketAnalyzer:
    """
    Analyzes market conditions to determine regimes and sentiment.
    """

    def __init__(self, data_manager):
        """
        Initialize the MarketAnalyzer.

        Args:
            data_manager: DataManager instance for accessing stock data.
        """
        self.data_manager: DataManager = data_manager

    def analyze_asset(
        self, ticker: str, current_date: date, use_llm: bool = True
    ) -> MarketRegime:
        """
        Analyze a single asset to determine market regime and sentiment.

        Args:
            ticker: Stock ticker symbol.
            current_date: Date for which to perform the analysis.
            use_llm: Whether to use LLM for analysis (if False, uses rule-based approach).

        Returns:
            MarketRegime object containing the analysis results.
        """
        # Get historical data and technical indicators
        stock_data = self.data_manager.load_stock_data(ticker, end_date=current_date)
        technical_indicators = self.data_manager.get_technical_indicators(
            ticker, current_date
        )

        if use_llm:
            # Convert stock data to dictionary format for prompt
            stock_data_dicts = [
                data.dict() for data in stock_data[-30:]
            ]  # Last 30 days

            # Build prompt for LLM
            prompt = build_market_analysis_prompt(
                ticker, stock_data_dicts, technical_indicators
            )

            # Get response from LLM
            response = get_llm_response(prompt, MarketRegime)

            # Check if response is valid
            if isinstance(response, MarketRegime):
                # Ensure asset field is set correctly
                response.asset = ticker
                return response
            else:
                # Fall back to rule-based approach if LLM fails
                print(
                    f"LLM analysis failed for {ticker}, falling back to rule-based approach"
                )
                return self._rule_based_analysis(
                    ticker, stock_data, technical_indicators
                )
        else:
            # Use rule-based approach
            return self._rule_based_analysis(ticker, stock_data, technical_indicators)

    def _rule_based_analysis(
        self,
        ticker: str,
        stock_data: List[StockData],
        technical_indicators: Dict[str, float],
    ) -> MarketRegime:
        """
        Perform rule-based market analysis without using LLM.

        Args:
            ticker: Stock ticker symbol.
            stock_data: List of StockData objects.
            technical_indicators: Dictionary of technical indicators.

        Returns:
            MarketRegime object containing the analysis results.
        """
        # Extract key indicators
        rsi = technical_indicators.get("rsi_14", 50)
        price_vs_sma_50 = technical_indicators.get("price_vs_sma_50", 0)
        price_vs_sma_200 = technical_indicators.get("price_vs_sma_200", 0)
        macd_histogram = technical_indicators.get("macd_histogram", 0)

        # Determine regime using price momentum and volatility
        regime = "neutral"
        if price_vs_sma_50 > 0.05 and price_vs_sma_200 > 0.05 and rsi > 60:
            regime = "bullish"
        elif price_vs_sma_50 < -0.05 and price_vs_sma_200 < -0.05 and rsi < 40:
            regime = "bearish"
        elif abs(price_vs_sma_50 - price_vs_sma_200) > 0.12:
            regime = "volatile"

        # Compute continuous sentiment score
        rsi_component = (rsi - 50.0) / 25.0  # map RSI into approximately [-1, 1]
        trend_component = np.clip(price_vs_sma_50 + 0.5 * price_vs_sma_200, -1.5, 1.5)
        momentum_component = np.tanh(macd_histogram * 5)
        sentiment_score = (
            0.5 * rsi_component + 0.3 * trend_component + 0.2 * momentum_component
        )
        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))

        # Compute outlook score emphasising forward momentum and volatility regime
        outlook_trend = np.clip(
            0.6 * price_vs_sma_50 + 0.4 * price_vs_sma_200, -1.5, 1.5
        )
        volatility_penalty = -0.2 if regime == "volatile" else 0.0
        outlook_score = 0.6 * sentiment_score + 0.4 * outlook_trend + volatility_penalty
        outlook_score = float(np.clip(outlook_score, -1.0, 1.0))

        # Generate narrative based on scores
        if outlook_score >= 0.25:
            outlook_narrative = "Momentum remains constructive; maintain exposure while monitoring risk levels."
        elif outlook_score <= -0.25:
            outlook_narrative = "Expect defensive conditions—prioritise capital preservation and hedging."
        elif regime == "volatile":
            outlook_narrative = "Volatile regime detected; reduce position size and emphasise hedging flexibility."
        else:
            outlook_narrative = "Range-bound signals dominate; await clearer breakout cues before repositioning."

        # Calculate confidence based on indicator alignment
        indicators_aligned = sum(
            [
                1 if (regime == "bullish" and price_vs_sma_50 > 0) else 0,
                1 if (regime == "bullish" and price_vs_sma_200 > 0) else 0,
                1 if (regime == "bullish" and rsi > 50) else 0,
                1 if (regime == "bullish" and macd_histogram > 0) else 0,
                1 if (regime == "bearish" and price_vs_sma_50 < 0) else 0,
                1 if (regime == "bearish" and price_vs_sma_200 < 0) else 0,
                1 if (regime == "bearish" and rsi < 50) else 0,
                1 if (regime == "bearish" and macd_histogram < 0) else 0,
            ]
        )
        confidence = min(
            0.5 + (indicators_aligned / 16), 0.9
        )  # Max confidence of 0.9 for rule-based

        return MarketRegime(
            asset=ticker,
            regime=regime,
            sentiment=sentiment_score,
            outlook=outlook_score,
            confidence=confidence,
            narrative=outlook_narrative,
            days_to_next_earnings=None,
            revenue_growth=None,
        )

    def analyze_assets(
        self, tickers: List[str], current_date: date, use_llm: bool = True
    ) -> Dict[str, MarketRegime]:
        """
        Analyze multiple assets to determine market regimes and sentiment.

        Args:
            tickers: List of stock ticker symbols.
            current_date: Date for which to perform the analysis.
            use_llm: Whether to use LLM for analysis.

        Returns:
            Dictionary mapping tickers to MarketRegime objects.
        """
        results = {}

        for ticker in tickers:
            results[ticker] = self.analyze_asset(ticker, current_date, use_llm)

        return results

    def get_market_summary(self, analysis_results: Dict[str, MarketRegime]) -> str:
        """
        Generate a summary of the market analysis results.

        Args:
            analysis_results: Dictionary mapping tickers to MarketRegime objects.

        Returns:
            Summary string.
        """
        if not analysis_results:
            return "No market analysis results available."

        # Count regimes and sentiments
        regime_counts = {"bullish": 0, "bearish": 0, "neutral": 0, "volatile": 0}
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for ticker, analysis in analysis_results.items():
            regime_counts[analysis.regime] += 1
            sentiment_counts[analysis.sentiment_label] += 1

        # Determine overall market regime and sentiment
        total_assets = len(analysis_results)
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]

        # Generate summary
        summary = f"Market Analysis Summary ({total_assets} assets):\n"
        summary += f"Overall Market Regime: {dominant_regime.capitalize()} ({regime_counts[dominant_regime]}/{total_assets} assets)\n"
        summary += f"Overall Market Sentiment: {dominant_sentiment.capitalize()} ({sentiment_counts[dominant_sentiment]}/{total_assets} assets)\n\n"

        summary += "Asset Breakdown:\n"
        for ticker, analysis in analysis_results.items():
            summary += (
                f"- {ticker}: {analysis.regime.capitalize()} regime, "
                f"{analysis.sentiment_label.capitalize()} sentiment (score {analysis.sentiment:.2f}), "
                f"Confidence: {analysis.confidence:.2f}\n"
            )
            summary += (
                f"  Outlook ({analysis.outlook:.2f}): {analysis.short_term_outlook}\n"
            )

        return summary

    def analyze_assets_batch(
        self, tickers: List[str], current_date: date, use_llm: bool = True
    ) -> Dict[str, MarketRegime]:
        """Analyze multiple assets in a single API call to determine market regimes and sentiment."""
        if not use_llm or len(tickers) <= 1:
            # Fall back to standard approach for single assets or rule-based analysis
            return self.analyze_assets(tickers, current_date, use_llm)

        # Collect data for all tickers
        historical_data_dict = {}
        technical_indicators_dict = {}
        analyst_ratings_dict = {}

        earnings_dict = self.data_manager.load_earnings_data(current_date)
        for ticker in tickers:
            stock_data = self.data_manager.load_stock_data(
                ticker, end_date=current_date
            )
            technical_indicators = self.data_manager.get_technical_indicators(
                ticker, current_date
            )

            # Convert to dict format for prompt
            historical_data_dict[ticker] = [data.dict() for data in stock_data[-30:]]
            technical_indicators_dict[ticker] = technical_indicators

            # Load analyst ratings from prepared JSON files
            try:
                with open(f"data/{ticker}_analyst_actions.json", "r") as f:
                    analyst_data = json.load(f)

                # Get recent analyst actions (last 30 days)
                current_date_dt = pd.to_datetime(current_date)
                recent_dates = [
                    date
                    for date in analyst_data["data"].keys()
                    if (current_date_dt - pd.to_datetime(date)).days <= 30
                ]
                analyst_ratings_dict[ticker] = {
                    "dates": recent_dates,
                    "actions": {
                        date: analyst_data["data"][date] for date in recent_dates
                    },
                }
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                analyst_ratings_dict[ticker] = {"dates": [], "actions": {}}
                print(f"Warning: Could not load analyst ratings for {ticker}: {e}")

        # Load extra data: VIX and Fear-Greed
        vix_df = self.data_manager.load_vix_data()  # from data/vix.csv
        fear_greed_df = (
            self.data_manager.load_fear_greed_data()
        )  # from data/fear-greed-2011-2023.csv
        cpi_df = self.data_manager.load_cpi_timeline()  # <-- new loader
        # Build combined prompt for all assets
        prompt = """
        Analyze the market conditions for the following assets:
        """

        for ticker in tickers:
            prompt += f"\n\nASSET: {ticker}\nRecent Price Action:\n"

            # Add last 5 days of price data
            for day in historical_data_dict[ticker][-5:]:
                prompt += f"- {day['date']}: Open ${day['open']:.2f}, Close ${day['close']:.2f}, High ${day['high']:.2f}, Low ${day['low']:.2f}, Volume {day['volume']}\n"

            info = earnings_dict.get(ticker, {}).get(current_date)
            if info:
                prompt += (
                    f"\nEarnings Metrics for {ticker} as of {current_date}:\n"
                    f"- Days to next earnings: {info['days_to_next']}\n"
                    f"- Forecast revenue growth: {info['rev_growth']:.2f}%\n"
                )
            prompt += "\nTechnical Indicators:\n"
            for indicator, value in technical_indicators_dict[ticker].items():
                prompt += f"- {indicator}: {value}\n"

            # Add analyst ratings section
            if analyst_ratings_dict[ticker]["dates"]:
                prompt += "\nRecent Analyst Actions:\n"
                # Show up to 10 most recent analyst actions
                for date in analyst_ratings_dict[ticker]["dates"][:10]:
                    for action in analyst_ratings_dict[ticker]["actions"][date]:
                        prompt += f"- {date}: {action['company']} {action['action'].upper()} from {action['from']} to {action['to']}\n"

        # Append VIX data
        prompt += "\n\nRecent VIX Data:\n"
        recent_vix = vix_df[vix_df["Date"] <= pd.to_datetime(current_date)].tail(5)
        for _, row in recent_vix.iterrows():
            prompt += f"- {row['Date'].strftime('%Y-%m-%d')}: Open ${row['Open']:.2f}, Close ${row['Close']:.2f}, High ${row['High']:.2f}, Low ${row['Low']:.2f}\n"

        # Append Fear-Greed data
        prompt += "\nRecent Fear-Greed Index Data:\n"
        recent_fg = fear_greed_df[
            fear_greed_df["Date"] <= pd.to_datetime(current_date)
        ].tail(5)
        for _, row in recent_fg.iterrows():
            prompt += f"- {row['Date'].strftime('%Y-%m-%d')}: Fear Greed {row['Fear Greed']}\n"

        # **New: CPI**
        prompt += "\nRecent CPI Inflation Data:\n"
        recent_cpi = cpi_df[cpi_df["date"] <= pd.to_datetime(current_date)].tail(5)
        for _, row in recent_cpi.iterrows():
            # format dates or “N/A” if missing
            last_rel = (
                row["last_release_date"].strftime("%Y-%m-%d")
                if pd.notna(row["last_release_date"])
                else "N/A"
            )
            next_rel = (
                row["next_release_date"].strftime("%Y-%m-%d")
                if pd.notna(row["next_release_date"])
                else "N/A"
            )
            days_next = (
                int(row["days_until_next_release"])
                if pd.notna(row["days_until_next_release"])
                else "N/A"
            )

            prompt += (
                f"- {row['date'].strftime('%Y-%m-%d')}: "
                f"Inflation {row['inflation']:.2f}%, "
                f"Forecast {row['forecast']:.2f}%, "
                f"State {row['actual_state']}, "
                f"Last Release {last_rel}, "
                f"Next Release {next_rel}, "
                f"Days Until Next {days_next}\n"
            )

        prompt += """
        For EACH asset listed above, determine:
        1. The current market regime (bullish, bearish, neutral, or volatile)
        2. Market sentiment (positive, negative, or neutral)
        3. Short-term outlook for this asset
        4. Your confidence level in this analysis (0.0 to 1.0)
        5. Consider analyst actions and their sentiment direction when evaluating outlook.

        Format your response as a JSON array of objects, with each object containing:
        - asset: the ticker symbol
        - regime: the market regime
        - sentiment: the market sentiment
        - short_term_outlook: your outlook text
        - confidence: your confidence level (between 0 and 1)
        - days_to_next_earnings
        - revenue_growth
        """

        # Get combined response from LLM
        response = get_llm_response(prompt, BatchMarketAnalysisResponse)

        # Process response
        results = {}
        if isinstance(response, BatchMarketAnalysisResponse):
            for analysis in response.analyses:
                results[analysis.asset] = analysis
        else:
            # Fall back to individual analysis if batch fails
            print("Batch analysis failed, falling back to individual analysis")
            return self.analyze_assets(tickers, current_date, use_llm)

        return results
