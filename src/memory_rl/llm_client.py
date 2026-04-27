"""
LLM Client implementation for memory-based RL integration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Type

from pydantic import BaseModel

# Import LLM utilities
from src.utils.llm_utils import get_llm_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with LLM APIs for memory-based RL.
    This implementation uses the existing llm_utils functions.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name to use.
        """
        self.model = model
    
    def generate_text(self, prompt: str, response_model: Type[BaseModel]) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Prompt text.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated text.
        """
        logger.info(f"Generating text with prompt of length {len(prompt)}")
        
        try:
            # Use the existing llm_utils function
            response = get_llm_response(prompt, response_model)
            
            # If response is a string, return it directly
            if isinstance(response, response_model):
                return response
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating text: {str(e)}"
