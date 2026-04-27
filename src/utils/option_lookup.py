"""
Option lookup utilities for the portfolio risk-management simulation system.
These functions help convert simplified LLM responses to full option positions
by looking up option details from available options data.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Union, Any

import config

from src.models.pydantic_models import OptionPosition, HedgingStrategy
from src.models.simplified_llm_response_models import (
    SimplifiedOptionContractResponse,
    HedgingStrategyResponse,
    CollarStrategyResponse,
    StraddleStrategyResponse,
    DeltaNeutralStrategyResponse
)


def find_option_by_id(option_id: str, available_options: List[Dict]) -> Optional[Dict]:
    """
    Find an option in the available options list by its option_id.
    
    Args:
        option_id: The option ID to search for
        available_options: List of available options
        
    Returns:
        The matching option dictionary or None if not found
    """
    for option in available_options:
        if str(option.get('option_id', '')) == option_id:
            return option
    option_id_parts = option_id.split('_')
    # change first part to option_type
    option_id_parts[0] = 'call' if option_id_parts[0] == '1' else 'put'
    option_id = '_'.join(option_id_parts)
    for option in available_options:
        if str(option.get('option_id', '')) == option_id:
            return option
    return None


def simplified_contract_to_option_position(
    contract: SimplifiedOptionContractResponse,
    available_options: List[Dict],
    strategy_type: str,
    ticker: str = '',
    available_capital: float = 0.0,
    current_date: date = None
) -> OptionPosition:
    """
    Convert a simplified option contract to a full option position
    by looking up details from available options.
    
    Args:
        contract: SimplifiedOptionContractResponse from LLM
        available_options: List of available options
        strategy_type: Type of hedging strategy
        ticker: Stock ticker symbol (used if asset is not specified)
        current_date: Current simulation date (used if option not found)
        
    Returns:
        OptionPosition object with complete details
    """
    # Find the option in available options
    option_data = find_option_by_id(contract.option_id, available_options)
    contract_size = getattr(config, "OPTION_CONTRACT_SIZE", 100)

    # If option not found, create a basic position with default values
    if option_data is None:
        # Parse option_id to extract basic information
        # Format is typically: "{option_type}_{strike}_{expiration_date}"
        # Example: "1_150.000000_2023-02-03" or "call_150_2023-02-03"
        parts = contract.option_id.split('_')
        
        # Try to extract strike price
        strike = 0.0
        strike = float(parts[1].replace('.000000', ''))
        
        # Default expiration date (30 days from current date)
        if current_date is None:
            current_date = date.today()
        expiration_date = current_date + timedelta(days=30)
        
        default_premium = getattr(config, "DEFAULT_OPTION_PREMIUM", 10.0)
        total_premium = default_premium * contract.contracts * contract_size
        paid_amount = total_premium if contract.action == "buy" else -total_premium

        # Create option position with default values
        return OptionPosition(
            asset=ticker or getattr(contract, 'asset', ''),
            option_type=contract.option_type,
            strike=strike,
            option_id=contract.option_id,
            expiration_date=expiration_date,
            contracts=contract.contracts,
            premium_per_contract=default_premium,
            total_premium=total_premium,
            paid_amount=paid_amount,
            action=contract.action,
            strategy=strategy_type
        )
    
    # Create option position from found option data
    option_type = "call" if option_data.get('option_type') == 1 else "put"
    
    # Calculate premium based on action (buy at ask, sell at bid)
    premium = option_data.get('ask', 0) if contract.action == "buy" else option_data.get('bid', 0)
    total_premium = premium * contract.contracts * contract_size
    paid_amount = total_premium

    if contract.action == "sell":
        paid_amount = -total_premium
    elif available_capital > 0 and premium > 0:
        max_leg_allocation = available_capital * getattr(
            config, "MAX_OPTION_LEG_CAPITAL_RATIO", 1.0
        )
        max_affordable_contracts = int(max_leg_allocation // (premium * contract_size))
        if max_affordable_contracts <= 0:
            contract.contracts = 0
            paid_amount = 0.0
            total_premium = 0.0
        elif contract.contracts > max_affordable_contracts:
            contract.contracts = max_affordable_contracts
            total_premium = premium * contract.contracts * contract_size
            paid_amount = total_premium
    # Calculate expiration date
    if 'expire_date' in option_data:
        expiration_date = option_data['expire_date']
    elif 'quote_date' in option_data and 'dte' in option_data:
        expiration_date = option_data['quote_date'] + timedelta(days=option_data['dte'])
    else:
        # Default to 30 days from current date if no expiration info available
        if current_date is None:
            current_date = date.today()
        expiration_date = current_date + timedelta(days=30)
    
    return OptionPosition(
        asset=ticker or getattr(contract, 'asset', ''),
        option_type=option_type,
        strike=option_data.get('strike', 0),
        option_id=contract.option_id,
        expiration_date=expiration_date,
        contracts=contract.contracts,
        premium_per_contract=premium,
        total_premium=total_premium,
        paid_amount=paid_amount,
        action=contract.action,
        strategy=strategy_type
    )


def simplified_contracts_to_option_positions(
    contracts: List[SimplifiedOptionContractResponse],
    available_options: List[Dict],
    strategy_type: str,
    ticker: str = '',
    available_capital: float = 0.0,
    current_date: date = None
) -> List[OptionPosition]:
    """
    Convert a list of simplified option contracts to full option positions.
    
    Args:
        contracts: List of SimplifiedOptionContractResponse objects from LLM
        available_options: List of available options
        strategy_type: Type of hedging strategy
        ticker: Stock ticker symbol (used if asset is not specified)
        current_date: Current simulation date
        
    Returns:
        List of OptionPosition objects with complete details
    """
    positions = []
    
    for contract in contracts:
        position = simplified_contract_to_option_position(
            contract=contract,
            available_options=available_options,
            strategy_type=strategy_type,
            ticker=ticker,
            available_capital=available_capital,
            current_date=current_date
        )
        position.market_value = position.paid_amount
        positions.append(position)
    
    return positions


def convert_hedging_strategy_response(
    response: Union[HedgingStrategyResponse, CollarStrategyResponse, StraddleStrategyResponse, DeltaNeutralStrategyResponse],
    available_options: List[Dict],
    ticker: str,
    available_capital: float = 0.0,
    current_date: date = None
) -> HedgingStrategy:
    """
    Convert a hedging strategy response from LLM to a full HedgingStrategy object.
    
    Args:
        response: LLM response object
        available_options: List of available options
        ticker: Stock ticker symbol
        current_date: Current simulation date
        
    Returns:
        HedgingStrategy object with complete option positions
    """
    option_positions = []
    strategy_type = getattr(response, 'strategy_type', 'hedging')
    estimated_cost = 0.0
    rationale = ""
    
    # Handle different response types
    if isinstance(response, HedgingStrategyResponse):
        option_positions = simplified_contracts_to_option_positions(
            contracts=response.option_contracts,
            available_options=available_options,
            strategy_type=response.strategy_type,
            ticker=ticker,
            available_capital=available_capital,
            current_date=current_date
        )
        estimated_cost = response.estimated_cost
        rationale = "Hedging strategy based on market analysis."
        
    
    # Calculate actual cost based on option positions
    actual_cost = sum(abs(pos.paid_amount) for pos in option_positions)

    
    # Apply scaling if actual cost exceeds available capital
    if actual_cost > available_capital and available_capital > 0 and option_positions:
        # Calculate scaling factor
        scaling_factor = available_capital / actual_cost
        
        # Scale each position's contracts and paid_amount
        for pos in option_positions:
            # Save original values
            original_contracts = pos.contracts
            original_paid = pos.paid_amount
            
            # Scale down contracts (minimum 1 contract)
            new_contracts = max(1, int(pos.contracts * scaling_factor))
            
            # Adjust paid amount proportionally
            pos.paid_amount = (original_paid / original_contracts) * new_contracts
            pos.contracts = new_contracts
        
        # Recalculate actual cost after scaling
        actual_cost = sum(
            pos.paid_amount if pos.action == "buy" else -pos.paid_amount 
            for pos in option_positions
        )
        
        # Add scaling information to rationale
        rationale += f" Strategy scaled to {scaling_factor:.2%} of original size to fit capital constraints."
    
    
    return HedgingStrategy(
        asset=ticker,
        strategy_type=strategy_type,
        rationale=rationale,
        option_positions=option_positions,
        estimated_cost=actual_cost,  # Use actual calculated cost
        estimated_protection=0.8,  # Default value
        confidence=0.7  # Default value
    )
