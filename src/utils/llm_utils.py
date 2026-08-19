"""
Utility functions for interacting with the Perplexity API for LLM functionality.
This module handles prompt building, API calls, and response parsing.
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

# Import models
from src.models.pydantic_models import LLMRequest, LLMResponse, MarketRegime
import config

from openai import OpenAI
from google.genai import types
from google import genai

from src.models.simplified_llm_response_models import HedgingStrategyResponse

logger = logging.getLogger(__name__)


GEMINI_API_KEYS_ENV = os.getenv("GEMINI_API_KEYS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


LLM_USAGE_LOG_FILENAME = "llm_usage.jsonl"


def _log_llm_usage(
    response: Optional[LLMResponse] = None,
    *,
    prompt: Optional[str] = None,
    response_text: Optional[str] = None,
    error: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    """Persist metadata about LLM usage to a JSONL log.

    The helper is defensive—missing metadata or filesystem issues are ignored to
    avoid impacting the main execution path.
    """

    try:
        output_dir = getattr(config, "DEFAULT_OUTPUT_DIR", None)
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, LLM_USAGE_LOG_FILENAME)

        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": None,
            "prompt": None,
            "response": None,
            "latency_seconds": None,
            "prompt_tokens": None,
            "candidates_tokens": None,
            "total_tokens": None,
            "error": None,
        }

        if response:
            log_entry.update(
                {
                    "model": response.model or log_entry["model"],
                    "prompt": response.prompt or log_entry["prompt"],
                    "response": response.content or log_entry["response"],
                    "latency_seconds": response.latency_seconds,
                    "prompt_tokens": response.prompt_tokens,
                    "candidates_tokens": response.candidates_tokens,
                    "total_tokens": response.total_tokens,
                    "error": response.error,
                }
            )

        if prompt is not None:
            log_entry["prompt"] = prompt
        if response_text is not None:
            log_entry["response"] = response_text
        if error is not None:
            log_entry["error"] = error

        if status is None:
            status = "error" if log_entry["error"] else "success"
        log_entry["status"] = status

        # Ensure string values default to an empty string instead of None to ease parsing later.
        if log_entry["prompt"] is None:
            log_entry["prompt"] = ""
        if log_entry["response"] is None:
            log_entry["response"] = ""

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        # Logging failures must never interrupt main execution flow.
        return

# Import the Gemini client from Google’s GenAI library

class GeminiAPI:
    """Client for interacting with the Gemini API."""

    def __init__(self, api_keys: Optional[List[str]] = None):
        if api_keys is None:
            keys_env = GEMINI_API_KEYS_ENV or GEMINI_API_KEY
            api_keys = (
                [key.strip() for key in keys_env.split(",") if key.strip()]
                if keys_env
                else []
            )

        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")

        self.api_key = api_keys[0]
        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the Gemini API with exponential backoff on rate limits."""

        max_attempts = 5
        backoff_base_seconds = 1

        for attempt in range(max_attempts):
            try:
                pydantic_model = request.response_format["pydantic_model"]
                content = request.prompt
                start_time = time.perf_counter()
                response = self.client.models.generate_content(
                    model=request.model,
                    contents=content,
                    config=types.GenerateContentConfig(
                        max_output_tokens=request.max_tokens,
                        temperature=request.temperature,
                        response_mime_type="application/json",
                        response_schema=pydantic_model,
                    )
                )
                latency_seconds = time.perf_counter() - start_time

                usage_metadata = getattr(response, "usage_metadata", None)
                prompt_tokens = None
                candidates_tokens = None
                total_tokens = None

                if usage_metadata:
                    if isinstance(usage_metadata, dict):
                        prompt_tokens = usage_metadata.get("prompt_token_count")
                        candidates_tokens = usage_metadata.get("candidates_token_count")
                        total_tokens = usage_metadata.get("total_token_count")
                    else:
                        prompt_tokens = getattr(usage_metadata, "prompt_token_count", None)
                        candidates_tokens = getattr(usage_metadata, "candidates_token_count", None)
                        total_tokens = getattr(usage_metadata, "total_token_count", None)

                content = response.text
                parsed_json = None

                if request.response_format:
                    try:
                        parsed_json = json.loads(content)
                    except json.JSONDecodeError:
                        return LLMResponse(
                            content=content,
                            parsed_json=None,
                            error="Failed to parse JSON response",
                            prompt_tokens=prompt_tokens,
                            candidates_tokens=candidates_tokens,
                            total_tokens=total_tokens,
                            latency_seconds=latency_seconds,
                            model=request.model,
                            prompt=request.prompt,
                        )
                return LLMResponse(
                    content=content,
                    parsed_json=parsed_json,
                    prompt_tokens=prompt_tokens,
                    candidates_tokens=candidates_tokens,
                    total_tokens=total_tokens,
                    latency_seconds=latency_seconds,
                    model=request.model,
                    prompt=request.prompt,
                )

            except Exception as e:
                error_message = str(e)

                if "429 RESOURCE_EXHAUSTED" in error_message:
                    if attempt == max_attempts - 1:
                        return LLMResponse(
                            content="",
                            parsed_json=None,
                            error=f"Rate limit exceeded after retries: {error_message}",
                            prompt_tokens=None,
                            candidates_tokens=None,
                            total_tokens=None,
                            latency_seconds=None,
                            model=request.model,
                            prompt=request.prompt,
                        )

                    sleep_time = backoff_base_seconds * (2**attempt)
                    logger.warning(
                        "Rate limit exceeded. Attempt %s/%s. Retrying in %.1f seconds.",
                        attempt + 1,
                        max_attempts,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue

                return LLMResponse(
                    content="",
                    parsed_json=None,
                    error=f"API request failed: {error_message}",
                    prompt_tokens=None,
                    candidates_tokens=None,
                    total_tokens=None,
                    latency_seconds=None,
                    model=request.model,
                    prompt=request.prompt,
                )

        return LLMResponse(
            content="",
            parsed_json=None,
            error="Failed to generate response after retries",
            prompt_tokens=None,
            candidates_tokens=None,
            total_tokens=None,
            latency_seconds=None,
            model=request.model,
            prompt=request.prompt,
        )


class OpenAIAPI:
    """Client for interacting with the OpenAI API with structured outputs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        if not request.response_format or "pydantic_model" not in request.response_format:
            return LLMResponse(
                content="",
                parsed_json=None,
                error="Missing pydantic model for structured response.",
                prompt_tokens=None,
                candidates_tokens=None,
                total_tokens=None,
                latency_seconds=None,
                model=request.model,
                prompt=request.prompt,
            )

        pydantic_model = request.response_format["pydantic_model"]

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            start_time = time.perf_counter()
            if '5' in request.model:
                completion = self.client.chat.completions.parse(
                    model=request.model,
                    messages=messages,
                    response_format=pydantic_model,
                )
            else:
                completion = self.client.chat.completions.parse(
                    model=request.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_format=pydantic_model,
                )
            latency_seconds = time.perf_counter() - start_time

            choice = completion.choices[0]
            message = choice.message

            refusal = getattr(message, "refusal", None)
            if refusal:
                return LLMResponse(
                    content="",
                    parsed_json=None,
                    error=str(refusal),
                    prompt_tokens=None,
                    candidates_tokens=None,
                    total_tokens=None,
                    latency_seconds=latency_seconds,
                    model=request.model,
                    prompt=request.prompt,
                )

            parsed_instance = getattr(message, "parsed", None)
            parsed_json: Optional[Dict[str, Any]] = None
            content = ""

            if parsed_instance is not None:
                if isinstance(parsed_instance, BaseModel):
                    parsed_json = parsed_instance.model_dump()
                elif hasattr(parsed_instance, "model_dump"):
                    parsed_json = parsed_instance.model_dump()
                else:
                    parsed_json = parsed_instance

                content = json.dumps(parsed_json, ensure_ascii=False)
            else:
                raw_content = getattr(message, "content", "")
                if isinstance(raw_content, list):
                    content = "".join(
                        part.text for part in raw_content if hasattr(part, "text")
                    )
                else:
                    content = str(raw_content or "")

                try:
                    parsed_json = json.loads(content) if content else None
                except json.JSONDecodeError:
                    parsed_json = None

            usage = getattr(completion, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None

            return LLMResponse(
                content=content,
                parsed_json=parsed_json,
                prompt_tokens=prompt_tokens,
                candidates_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency_seconds,
                model=request.model,
                prompt=request.prompt,
            )

        except Exception as exc:  # pragma: no cover - defensive fallback
            return LLMResponse(
                content="",
                parsed_json=None,
                error=str(exc),
                prompt_tokens=None,
                candidates_tokens=None,
                total_tokens=None,
                latency_seconds=None,
                model=request.model,
                prompt=request.prompt,
            )


def get_llm_response_json_genai(prompt: str, response_model: Type[BaseModel], temperature: float = 0.5) -> Union[BaseModel, LLMResponse]:
    """
    Get a structured (JSON) response from the Gemini API and parse it into a Pydantic model.

    Since Gemini does not provide beta parsing yet, we instruct the model via the prompt
    to output a valid JSON object. The response is then parsed into the provided model.
    """
    json_instruction = """
    IMPORTANT: Your response must be a valid JSON object that matches the following structure.
    Do not include any explanations, only return a valid JSON object.

    JSON Structure:
    """
    full_prompt = f"{prompt}\n\n{json_instruction}{response_model.schema()}"

    request = LLMRequest(
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=10000,
        model=getattr(config, "LLM_MODEL", "gemini-2.5-flash-lite"),
        response_format={"pydantic_model": response_model},
    )
    # Use the persistent GeminiAPI instance instead of creating a new one
    api = get_gemini_api_instance()
    response = api.generate_response(request)

    if response.error:
        _log_llm_usage(response=response, status="error")
        return response

    if not response.parsed_json:
        error_message = "Failed to parse JSON response"
        error_response = LLMResponse(
            content=response.content,
            parsed_json=None,
            error=error_message,
            prompt_tokens=response.prompt_tokens,
            candidates_tokens=response.candidates_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            model=response.model,
            prompt=response.prompt or full_prompt,
        )
        _log_llm_usage(response=error_response, status="error")
        return error_response

    try:
        parsed_model = response_model.parse_obj(response.parsed_json)
        _log_llm_usage(response=response, status="success")
        return parsed_model
    except ValidationError as e:
        error_message = f"JSON validation failed: {str(e)}"
        error_response = LLMResponse(
            content=response.content,
            parsed_json=response.parsed_json,
            error=error_message,
            prompt_tokens=response.prompt_tokens,
            candidates_tokens=response.candidates_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            model=response.model,
            prompt=response.prompt or full_prompt,
        )
        _log_llm_usage(response=error_response, status="error")
        return error_response


def get_llm_response_json_openai(
    prompt: str, response_model: Type[BaseModel], temperature: float = 0.5
) -> Union[BaseModel, LLMResponse]:
    request = LLMRequest(
        prompt=prompt,
        temperature=temperature,
        max_tokens=4000,
        model=getattr(config, "LLM_MODEL", "gpt-4o-mini"),
        response_format={"pydantic_model": response_model},
    )

    api = get_openai_api_instance()
    response = api.generate_response(request)

    if response.error:
        _log_llm_usage(response=response, status="error")
        return response

    if not response.parsed_json:
        error_response = LLMResponse(
            content=response.content,
            parsed_json=None,
            error="Failed to parse JSON response",
            prompt_tokens=response.prompt_tokens,
            candidates_tokens=response.candidates_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            model=response.model,
            prompt=response.prompt or prompt,
        )
        _log_llm_usage(response=error_response, status="error")
        return error_response

    try:
        parsed_model = response_model.parse_obj(response.parsed_json)
        _log_llm_usage(response=response, status="success")
        return parsed_model
    except ValidationError as exc:
        error_message = f"JSON validation failed: {str(exc)}"
        error_response = LLMResponse(
            content=response.content,
            parsed_json=response.parsed_json,
            error=error_message,
            prompt_tokens=response.prompt_tokens,
            candidates_tokens=response.candidates_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            model=response.model,
            prompt=response.prompt or prompt,
        )
        _log_llm_usage(response=error_response, status="error")
        return error_response


# Prompt templates for different LLM tasks


def _coerce_market_regime(analysis: Union[MarketRegime, Dict[str, Any]]) -> MarketRegime:
    """Ensure market analysis is represented as a :class:`MarketRegime`."""

    if isinstance(analysis, MarketRegime):
        return analysis

    if isinstance(analysis, dict):
        return MarketRegime.model_validate(analysis)

    raise TypeError("Unsupported market analysis type; expected MarketRegime or dict")

def build_market_analysis_prompt(ticker: str, historical_data: List[Dict], technical_indicators: Dict) -> str:
    """
    Build a prompt for market analysis of a specific asset.

    Args:
        ticker: The stock ticker symbol.
        historical_data: List of historical price data points.
        technical_indicators: Dictionary of technical indicators.

    Returns:
        Formatted prompt string.
    """
    prompt = f"""
    Analyze the market conditions for {ticker} based on the following data:

    Recent Price Action:
    """

    # Add last 5 days of price data
    for i, day in enumerate(historical_data[-5:]):
        prompt += f"- {day['date']}: Open ${day['open']:.2f}, Close ${day['close']:.2f}, High ${day['high']:.2f}, Low ${day['low']:.2f}, Volume {day['volume']}\n"

    prompt += "\nTechnical Indicators:\n"

    # Add technical indicators
    for indicator, value in technical_indicators.items():
        prompt += f"- {indicator}: {value}\n"

    prompt += """
    Based on this information, determine:
    1. The current market regime (bullish, bearish, neutral, or volatile)
    2. A sentiment score between -1 and 1 (negative implies bearish pressure)
    3. A short-term outlook score between -1 and 1 (negative implies defensive posture)
    4. Your confidence level in this analysis (0.0 to 1.0)
    5. A concise short-term outlook narrative supporting the score

    Provide supporting rationale for your analysis.
    """

    return prompt


def build_portfolio_allocation_prompt(
    available_capital: float,
    assets: List[str],
    market_analysis: Dict[str, Any],
    current_positions: Optional[Dict] = None
) -> str:
    """
    Build a prompt for portfolio capital allocation across multiple assets.

    Args:
        available_capital: Total capital available for allocation.
        assets: List of asset ticker symbols.
        market_analysis: Dictionary of market analysis results for each asset.
        current_positions: Optional dictionary of current positions.

    Returns:
        Formatted prompt string.
    """
    prompt = f"""
    You are a hedge fund portfolio manager with ${available_capital:,.2f} to allocate across the following assets:
    {', '.join(assets)}

    Here is the market analysis for each asset:
    """

    for ticker, analysis in market_analysis.items():

        regime = _coerce_market_regime(analysis)
        prompt += f"""
        {ticker}:
        - Market Regime: {regime.regime}
        - Sentiment Score: {regime.sentiment:.2f} ({regime.sentiment_label})
        - Outlook Score: {regime.outlook:.2f} ({regime.outlook_label})
        - Narrative Outlook: {regime.short_term_outlook}
        - Confidence: {regime.confidence:.2f}
        """

    if current_positions:
        prompt += "\nCurrent Portfolio Positions:\n"
        for ticker, position in current_positions.items():
            prompt += f"- {ticker}: ${position['market_value']:,.2f} ({position['shares']} shares at ${position['average_price']:.2f})\n"

    prompt += f"""
    Determine how to allocate the ${available_capital:,.2f} across these assets. Consider:
    1. Market regime and sentiment for each asset
    2. Diversification principles
    3. Risk management

    For each asset, specify:
    1. The dollar amount to allocate
    2. The percentage of the total portfolio
    3. Brief rationale for the allocation

    You may also choose to keep some capital in cash reserves if appropriate (not much). Allocate some capital to all assets.
    """

    return prompt


def build_hedging_strategy_prompt(
    ticker: str,
    stock_position: Dict,
    available_options: List[Dict],
    market_analysis: Dict,
    strategy_type: str = "optimal"
) -> str:
    """
    Build a prompt for determining hedging strategies for a specific asset.

    Args:
        ticker: The stock ticker symbol.
        stock_position: Dictionary with details of the stock position.
        available_options: List of available options contracts.
        market_analysis: Market analysis for the asset.
        strategy_type: Type of hedging strategy to focus on.

    Returns:
        Formatted prompt string.
    """
    regime = _coerce_market_regime(market_analysis)

    prompt = f"""
    You are a hedge fund options strategist. Design a hedging strategy for the following position:

    Asset: {ticker}
    Current Position: {stock_position['shares']} shares at ${stock_position['average_price']:.2f} (Market Value: ${stock_position['market_value']:,.2f})
    Current Price: ${stock_position['current_price']:.2f}

    Market Analysis:
    - Regime: {regime.regime}
    - Sentiment Score: {regime.sentiment:.2f} ({regime.sentiment_label})
    - Outlook Score: {regime.outlook:.2f} ({regime.outlook_label})
    - Short-term Outlook: {regime.short_term_outlook}

    Available Options:
    """

    # Add available options data
    for option in available_options[:10]:  # Limit to 10 options to keep prompt size reasonable
        prompt += f"""
        - {option['option_type'].upper()} {option['strike']:.2f} exp {option['quote_date']} ({option['dte']} DTE)
          Last: ${option['last']:.2f}, Bid: ${option['bid']:.2f}, Ask: ${option['ask']:.2f}
          IV: {option['implied_volatility']:.2f}, Delta: {option['delta']:.2f}
        """

    if strategy_type == "collar":
        prompt += """
        Design a collar strategy by selecting:
        1. A protective put (typically 5-10% OTM)
        2. A covered call (typically 5-10% OTM)

        The goal is to protect against downside while generating premium income.
        """
    elif strategy_type == "straddle":
        prompt += """
        Design a straddle strategy by selecting:
        1. At-the-money call options
        2. At-the-money put options

        The goal is to profit from significant price movement in either direction.
        """
    elif strategy_type == "delta-neutral":
        prompt += """
        Design a delta-neutral strategy that:
        1. Balances positive and negative delta exposures
        2. Minimizes directional risk
        3. Potentially profits from volatility or time decay
        """
    else:  # optimal
        prompt += """
        Based on the market analysis and available options, recommend the optimal hedging strategy.
        Consider strategies like:
        - Collar (protective put + covered call)
        - Straddle/Strangle (for volatility)
        - Delta-neutral approaches
        - Simple protective puts

        Choose the strategy that best aligns with the market outlook and provides appropriate risk management.
        """

    prompt += """
    For your recommended strategy, provide:
    1. Specific option contracts to use (strike, expiration, type)
    2. Number of contracts
    3. Estimated cost/premium
    4. Expected protection/payoff profile
    5. Rationale for this strategy
    """

    return prompt



# Add this to llm_utils.py
def build_combined_allocation_split_prompt(
    available_capital: float,
    assets: List[str],
    market_analysis: Dict[str, MarketRegime],
    current_positions: Optional[Dict] = None
) -> str:
    """Build a prompt for both capital allocation and stock/options split decisions."""
    prompt = f"""
    You are a hedge fund portfolio manager with ${available_capital:,.2f} to allocate across the following assets:
    {', '.join(assets)}

    Here is the market analysis for each asset:
    """

    for ticker, analysis in market_analysis.items():
        regime = _coerce_market_regime(analysis)
        prompt += f"""
        {ticker}:
        - Market Regime: {regime.regime}
        - Sentiment Score: {regime.sentiment:.2f} ({regime.sentiment_label})
        - Outlook Score: {regime.outlook:.2f} ({regime.outlook_label})
        - Short-term Outlook: {regime.short_term_outlook}
        - Confidence: {regime.confidence:.2f}
        """

    if current_positions:
        prompt += "\nCurrent Portfolio Positions:\n"
        for ticker, position in current_positions.items():
            prompt += f"- {ticker}: ${position['market_value']:,.2f} ({position['shares']} shares at ${position['average_price']:.2f})\n"
    Short_selling_note = ""
    if getattr(config, "ALLOW_SHORT_SELLING", False):
        Short_selling_note = "\nNote: Short selling is enabled in the configuration."
    else:
        Short_selling_note = "\nNote: Short selling is not allowed based on current configuration."
    allowed_stock_actions = (
        '"buy" or "sell"' if getattr(config, "ALLOW_SHORT_SELLING", False) else '"buy" only'
    )
    prompt += f"""
    TASK 1: Determine how to allocate the ${available_capital:,.2f} across these assets, considering:
    - Market regime and sentiment for each asset
    - Diversification principles
    - Risk management

    TASK 2: For EACH asset you've allocated capital to, determine:
    - What percentage should be allocated to stock vs. options?

    For the options allocation on each asset, further split it among three hedging strategies based on current market conditions:
        - "collar": Use when protecting gains or limiting downside risk while capping upside.
        - "straddle": Use when expecting significant price movement in either direction (e.g., before earnings).
        - "delta_neutral": Use when hedging directional risk or profiting from volatility changes.
    Ensure that the percentages for these three strategies sum to 1 (or 100%). Do not allocate too muhc into collar strategy.

    Format your response as a JSON object with:
    - allocations: array of objects with asset, amount, percentage, and action ({allowed_stock_actions})
    - asset_splits: array of objects, one per allocated asset, each containing:
      asset, total_allocation, stock_percentage, stocks_action ({allowed_stock_actions}),
      options_percentage, stock_amount, options_amount, rationale, and hedging_strategies
    - hedging_strategies must be an object with exactly these numeric fields:
      collar, straddle, delta_neutral
    - cash_reserve: amount to keep in cash; use 0-2% of available capital in normal conditions
    - rationale: overall strategy explanation
    {Short_selling_note}

    Important: Keep cash very small in normal conditions. Target 0-2% cash and do not exceed 2% cash unless there is an explicit emergency drawdown or extreme market stress. Try not to allocate more than 10% for options unless extereme market conditions. If you believe stocks is in a rally you may not allocate defensive options.
    """

    return prompt

# At the module level, define persistent API client instances
_gemini_api_instance: Optional[GeminiAPI] = None
_openai_api_instance: Optional[OpenAIAPI] = None


def get_gemini_api_instance() -> GeminiAPI:
    global _gemini_api_instance
    if _gemini_api_instance is None:
        _gemini_api_instance = GeminiAPI()
    return _gemini_api_instance


def get_openai_api_instance() -> OpenAIAPI:
    global _openai_api_instance
    if _openai_api_instance is None:
        _openai_api_instance = OpenAIAPI()
    return _openai_api_instance


def get_llm_response(prompt: str, response_model: Type[BaseModel], temperature: float = 0.5):
    provider = getattr(config, "LLM_PROVIDER", "gemini").lower()

    if provider in {"gemini", "google"}:
        return get_llm_response_json_genai(prompt, response_model, temperature)
    if provider in {"gpt", "openai"}:
        return get_llm_response_json_openai(prompt, response_model, temperature)

    raise ValueError(f"Unsupported LLM provider: {provider}")
