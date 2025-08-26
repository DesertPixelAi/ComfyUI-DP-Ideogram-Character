"""
API client utilities for Ideogram API interaction
"""

import requests
import time
import logging
import json
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class IdeogramAPIClient:
    """Client for interacting with Ideogram API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.ideogram.ai"
        self.session = requests.Session()
        self.session.headers.update({
            'Api-Key': api_key,
            'User-Agent': 'ComfyUI-IdeogramCharacter/1.0'
        })
    
    def check_quota(self) -> Optional[Dict[str, Any]]:
        """Check API quota/credits remaining"""
        try:
            # Note: This endpoint might not exist, adjust based on actual API
            response = self.session.get(f"{self.base_url}/v1/account/quota", timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.info("Quota endpoint not available")
            else:
                logger.warning(f"Quota check returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to check quota: {e}")
        return None
    
    def get_generation_status(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a generation request"""
        try:
            response = self.session.get(
                f"{self.base_url}/v1/generations/{generation_id}",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Status check returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to get generation status: {e}")
        return None
    
    def validate_api_key(self) -> Tuple[bool, str]:
        """Validate API key by making a test request"""
        try:
            # Try to check quota or make a minimal API call
            quota = self.check_quota()
            if quota:
                return True, "API key is valid"
            
            # Alternative: try a simple API endpoint
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return True, "API key validation inconclusive"
        except Exception as e:
            logger.warning(f"API key validation error: {e}")
            return True, "Could not validate API key (will try anyway)"

def parse_api_error(response: requests.Response) -> str:
    """Parse error message from API response"""
    try:
        error_data = response.json()
        if 'message' in error_data:
            return error_data['message']
        elif 'error' in error_data:
            if isinstance(error_data['error'], dict):
                return error_data['error'].get('message', str(error_data['error']))
            else:
                return str(error_data['error'])
        else:
            return json.dumps(error_data)
    except:
        # If JSON parsing fails, return text
        return response.text[:500] if response.text else f"HTTP {response.status_code}"

def calculate_cost(image_count: int, render_speed: str, use_character: bool = True) -> float:
    """Calculate estimated cost for generation"""
    # Prices in USD per image
    base_prices = {
        "Turbo": 0.03,
        "Default": 0.06,
        "Quality": 0.09
    }
    
    # Character reference pricing (usually higher)
    character_prices = {
        "Turbo": 0.04,
        "Default": 0.07,
        "Quality": 0.10
    }
    
    prices = character_prices if use_character else base_prices
    price_per_image = prices.get(render_speed, 0.07)
    
    return price_per_image * image_count

def format_generation_info(generation_id: str, image_count: int, render_speed: str, 
                         seed: str, dimensions: str) -> str:
    """Format generation information for display"""
    cost = calculate_cost(image_count, render_speed, use_character=True)
    
    info = f"""
=== Generation Complete ===
ID: {generation_id}
Images: {image_count}
Speed: {render_speed}
Seed: {seed}
Dimensions: {dimensions}
Estimated Cost: ${cost:.3f}
=========================
"""
    return info.strip()