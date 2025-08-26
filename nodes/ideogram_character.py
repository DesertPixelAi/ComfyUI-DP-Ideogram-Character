"""
Ideogram Character Node for ComfyUI
Generates consistent character images using Ideogram API v3
"""

import torch
import numpy as np
from PIL import Image
import io
import json
import time
import random
import requests
from typing import Optional, Tuple, List, Dict, Any
import logging
import os
import sys

# Import utilities with robust path handling
try:
    # First try relative import
    from ..utils.image_utils import ensure_rgb, resize_to_limit, calculate_aspect_ratio
    from ..utils.api_client import IdeogramAPIClient
except (ImportError, ValueError):
    try:
        # Try absolute import
        from utils.image_utils import ensure_rgb, resize_to_limit, calculate_aspect_ratio
        from utils.api_client import IdeogramAPIClient
    except ImportError:
        # Fallback - add parent directory to path and import
        import sys
        import os
        
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory (the main package directory)
        parent_dir = os.path.dirname(current_dir)
        
        # Add to path if not already there
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            from utils.image_utils import ensure_rgb, resize_to_limit, calculate_aspect_ratio
            from utils.api_client import IdeogramAPIClient
        except ImportError as e:
            # Final fallback - direct file import
            import importlib.util
            
            # Import image_utils
            utils_dir = os.path.join(parent_dir, 'utils')
            image_utils_path = os.path.join(utils_dir, 'image_utils.py')
            api_client_path = os.path.join(utils_dir, 'api_client.py')
            
            # Load image_utils
            spec = importlib.util.spec_from_file_location("image_utils", image_utils_path)
            image_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(image_utils)
            
            # Load api_client
            spec = importlib.util.spec_from_file_location("api_client", api_client_path)
            api_client = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_client)
            
            # Extract needed functions
            ensure_rgb = image_utils.ensure_rgb
            resize_to_limit = image_utils.resize_to_limit
            calculate_aspect_ratio = image_utils.calculate_aspect_ratio
            IdeogramAPIClient = api_client.IdeogramAPIClient

# Set up logging
logger = logging.getLogger(__name__)

class DP_IdeogramCharacter:
    """
    Generate consistent character images using Ideogram API v3 with character reference
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "display": "password"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe what you want to see"
                }),
                "character_image": ("IMAGE",),
                "aspect_ratio": ([
                    "1:1 (1024x1024)", "16:9 (1456x832)", "9:16 (832x1456)", 
                    "4:3 (1152x896)", "3:4 (896x1152)", "10:16 (896x1408)", 
                    "16:10 (1344x832)", "3:2 (1216x832)", "2:3 (832x1216)",
                    "3:1 (1536x512)", "1:3 (512x1536)", "1:2 (832x1216)",
                    "2:1 (1408x704)", "4:5 (896x1152)", "5:4 (1152x896)"
                ], {
                    "default": "1:1 (1024x1024)"
                }),
                "image_count": ([1, 2, 3, 4], {
                    "default": 1
                }),
                "render_speed": (["Turbo", "Default", "Quality"], {
                    "default": "Default"
                }),
                "style_type": (["Auto", "General", "Realistic", "Design", "Fiction"], {
                    "default": "Auto",
                    "tooltip": "Auto: Let AI choose style, General: Balanced approach, Realistic: Photorealistic style, Design: Artistic/illustration style, Fiction: Fantasy/sci-fi style"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
                "magic_prompt": (["AUTO", "ON", "OFF"], {
                    "default": "AUTO"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    
    def __init__(self):
        self.api_url = "https://api.ideogram.ai/v1/ideogram-v3/generate"
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor (B,H,W,C) or (H,W,C) to PIL Image"""
        # Handle batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Take first image from batch
        elif len(tensor.shape) != 3:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")
        
        # Ensure tensor is on CPU and convert to numpy
        np_image = tensor.cpu().numpy()
        
        # Clip values to [0, 1] range and convert to uint8
        np_image = np.clip(np_image, 0, 1)
        np_image = (np_image * 255).astype(np.uint8)
        
        # Create PIL image
        if np_image.shape[2] == 1:
            # Grayscale
            return Image.fromarray(np_image.squeeze(), mode='L')
        elif np_image.shape[2] == 3:
            # RGB
            return Image.fromarray(np_image, mode='RGB')
        elif np_image.shape[2] == 4:
            # RGBA
            return Image.fromarray(np_image, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {np_image.shape[2]}")
    
    def pil_to_bytes(self, pil_image: Image.Image, format: str = 'PNG', quality: int = 95) -> bytes:
        """Convert PIL Image to bytes with size optimization"""
        # Ensure RGB mode for better compatibility
        pil_image = ensure_rgb(pil_image)
        
        # Resize if too large
        pil_image = resize_to_limit(pil_image, max_size_mb=9.5)  # Leave some margin
        
        buffer = io.BytesIO()
        
        # Try to save with specified format
        if format.upper() == 'PNG':
            pil_image.save(buffer, format='PNG', optimize=True)
        else:
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        # Check if size is still too large
        if buffer.tell() > 10 * 1024 * 1024:
            # Try JPEG with lower quality
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=80, optimize=True)
        
        buffer.seek(0)
        return buffer.getvalue()
    

    
    def download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL with retry logic and security validation"""
        # Security: Validate URL domain to prevent SSRF attacks
        if not url.startswith(('https://ideogram.ai/', 'https://api.ideogram.ai/')):
            logger.error(f"Security: Rejected download from untrusted domain: {url}")
            return None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=30, allow_redirects=False)
                response.raise_for_status()
                
                # Security: Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith(('image/jpeg', 'image/png', 'image/webp')):
                    logger.error(f"Security: Invalid content type: {content_type}")
                    return None
                
                # Security: Check content length
                content_length = len(response.content)
                if content_length > 20 * 1024 * 1024:  # 20MB limit
                    logger.error(f"Security: Image too large: {content_length} bytes")
                    return None
                
                return Image.open(io.BytesIO(response.content))
            except Exception as e:
                logger.warning(f"Failed to download image (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return None
    
    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI tensor format (B,H,W,C)"""
        # Ensure RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to torch tensor and add batch dimension
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        return tensor
    
    def build_request_data(self, **kwargs) -> Tuple[Dict[str, str], List[Tuple[str, Tuple[str, bytes, str]]]]:
        """Build multipart form data for API request"""
        fields = {}
        files = []
        
        # Basic required fields
        fields['prompt'] = kwargs['prompt']
        fields['image_count'] = str(kwargs['image_count'])
        
        # Map style type to API format
        style_map = {
            "Auto": "AUTO",
            "General": "GENERAL", 
            "Realistic": "REALISTIC",
            "Design": "DESIGN",
            "Fiction": "FICTION"
        }
        fields['style_type'] = style_map.get(kwargs['style_type'], 'AUTO')
        
        # Handle aspect ratio (custom option removed due to API compatibility issues)
        # Extract just the ratio part from "1:1 (1024x1024)" format
        aspect_ratio = kwargs['aspect_ratio'].split(' ')[0]  # Get "1:1" from "1:1 (1024x1024)"
        # Convert colon format to x format (1:1 -> 1x1)
        fields['aspect_ratio'] = aspect_ratio.replace(":", "x")
        print(f"[Ideogram Character] Using preset aspect ratio: {fields['aspect_ratio']}")
        
        # Map render speed to match Ideogram API v3 specification
        # Based on Edit API docs: TURBO, DEFAULT, QUALITY are correct values
        speed_map = {
            "Turbo": "TURBO", 
            "Default": "DEFAULT",
            "Quality": "QUALITY"
        }
        fields['rendering_speed'] = speed_map.get(kwargs['render_speed'], "DEFAULT")
        
        # Magic prompt - only add if not AUTO (API default)
        if kwargs['magic_prompt'] != 'AUTO':
            fields['magic_prompt'] = kwargs['magic_prompt']
        
        # Seed (if not random)
        if kwargs.get('seed', -1) != -1:
            fields['seed'] = str(kwargs['seed'])
        else:
            # Generate random seed
            fields['seed'] = str(random.randint(0, 2147483647))
        
        # Store seed for info
        self.used_seed = fields['seed']
        
        # Add character reference image
        char_image_bytes = kwargs['character_image_bytes']
        files.append(('character_reference_images', ('character.png', char_image_bytes, 'image/png')))
        
        return fields, files
    
    def make_api_request(self, api_key: str, fields: Dict, files: List) -> Dict:
        """Make API request with retry logic"""
        headers = {
            'Api-Key': api_key
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Making API request (attempt {attempt + 1})")
                
                # Prepare files for multipart upload
                files_prepared = []
                for field_name, (filename, data, content_type) in files:
                    files_prepared.append((field_name, (filename, io.BytesIO(data), content_type)))
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=fields,
                    files=files_prepared,
                    timeout=120
                )
                
                # Log response status
                logger.info(f"Response status: {response.status_code}")
                
                # Log response content for debugging (first 1000 chars for more detail)
                if response.status_code != 200:
                    logger.error(f"API Error Response: {response.text[:1000]}")
                    print(f"[Ideogram Character] API Error {response.status_code}: {response.text[:1000]}")
                    print(f"[Ideogram Character] Request fields that caused error: {fields}")
                else:
                    print(f"[Ideogram Character] ✓ API request successful")
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                # Check for other errors
                if response.status_code != 200:
                    error_msg = f"API returned status {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f": {error_data.get('message', 'Unknown error')}"
                    except:
                        error_msg += f": {response.text[:200]}"
                    
                    if response.status_code == 401:
                        raise ValueError("Invalid API key. Please check your Ideogram API key.")
                    elif response.status_code == 400:
                        raise ValueError(f"Bad request: {error_msg}")
                    else:
                        raise Exception(error_msg)
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"API request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        raise Exception(f"API request failed after {self.max_retries} attempts: {last_error}")
    
    def process_api_response(self, response: Dict) -> Tuple[List[torch.Tensor], str]:
        """Process API response and download images"""
        images = []
        
        # Extract image URLs from response
        # The response structure might be { "data": [...] } or { "images": [...] }
        data = response.get('data', response.get('images', []))
        
        if not data:
            logger.error(f"API Response structure: {json.dumps(response, indent=2)[:500]}")
            raise ValueError("No images returned from API. Check your API key and quota.")
        
        info_parts = []
        info_parts.append(f"Seed used: {getattr(self, 'used_seed', 'unknown')}")
        info_parts.append(f"Images generated: {len(data)}")
        
        for idx, item in enumerate(data):
            # Handle different response structures
            if isinstance(item, str):
                url = item
            elif isinstance(item, dict):
                url = item.get('url', item.get('image_url', item.get('link')))
            else:
                logger.warning(f"Unexpected item type: {type(item)}")
                continue
            
            if not url:
                logger.warning(f"No URL found for image {idx + 1}")
                continue
            
            logger.info(f"Downloading image {idx + 1}/{len(data)}")
            pil_image = self.download_image(url)
            
            if pil_image:
                tensor = self.pil_to_tensor(pil_image)
                images.append(tensor)
                info_parts.append(f"Image {idx + 1}: {pil_image.size[0]}x{pil_image.size[1]}")
            else:
                logger.error(f"Failed to download image {idx + 1}")
        
        if not images:
            raise ValueError("Failed to download any images from API response")
        
        # Stack images into batch
        if len(images) == 1:
            image_batch = images[0]
        else:
            image_batch = torch.cat(images, dim=0)
        
        info = "\n".join(info_parts)
        
        return image_batch, info
    
    def validate_inputs(self, api_key: str, prompt: str, **kwargs) -> None:
        """Validate required inputs"""
        if not api_key or api_key.strip() == "":
            raise ValueError("API key is required. Please enter your Ideogram API key.")
        
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt is required. Please describe what you want to see.")
        
        # Security: Basic prompt sanitization
        if len(prompt) > 2000:  # Reasonable limit for prompts
            raise ValueError("Prompt too long. Maximum 2000 characters allowed.")
        
        # Check for potentially malicious content
        suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', 'file://', 'ftp://']
        prompt_lower = prompt.lower()
        for pattern in suspicious_patterns:
            if pattern in prompt_lower:
                raise ValueError("Prompt contains potentially unsafe content.")
        
        # Validate image count
        image_count = kwargs.get('image_count', 1)
        if not isinstance(image_count, int) or image_count < 1 or image_count > 4:
            raise ValueError(f"Image count must be between 1 and 4, got: {image_count}")
        
        # Validate render speed
        render_speed = kwargs.get('render_speed', 'Default')
        if render_speed not in ['Turbo', 'Default', 'Quality']:
            raise ValueError(f"Invalid render speed: {render_speed}")
        
        # Validate magic prompt
        magic_prompt = kwargs.get('magic_prompt', 'AUTO')
        if magic_prompt not in ['AUTO', 'ON', 'OFF']:
            raise ValueError(f"Invalid magic prompt setting: {magic_prompt}")
        
        # Validate style type
        style_type = kwargs.get('style_type', 'Auto')
        if style_type not in ['Auto', 'General', 'Realistic', 'Design', 'Fiction']:
            raise ValueError(f"Invalid style type: {style_type}")
        

    
    def generate(self, api_key: str, prompt: str, character_image: torch.Tensor, 
                 aspect_ratio: str, image_count: int, render_speed: str, style_type: str,
                 seed: int = -1, magic_prompt: str = "AUTO", **kwargs) -> Tuple[torch.Tensor, str]:
        """Main generation function"""
        try:
            # Validate inputs
            self.validate_inputs(api_key, prompt, image_count=image_count, 
                               render_speed=render_speed, magic_prompt=magic_prompt,
                               style_type=style_type)
            
            logger.info("Starting Ideogram character generation")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Settings: {aspect_ratio}, {image_count} images, {render_speed} speed")
            logger.info(f"Magic prompt: {magic_prompt}, Seed: {seed}")
            
            # Debug console output for user
            print(f"[Ideogram Character] Starting generation:")
            print(f"[Ideogram Character] - Render speed: {render_speed}")
            print(f"[Ideogram Character] - Image count: {image_count}")
            print(f"[Ideogram Character] - Magic prompt: {magic_prompt}")
            print(f"[Ideogram Character] - Style type: {style_type}")
            print(f"[Ideogram Character] - Aspect ratio: {aspect_ratio}")
            print(f"[Ideogram Character] - Seed: {seed}")
            
            # Convert character image to bytes
            char_pil = self.tensor_to_pil(character_image)
            char_bytes = self.pil_to_bytes(char_pil)
            logger.info(f"Character image size: {len(char_bytes)} bytes")
            
            # Build request data
            fields, files = self.build_request_data(
                prompt=prompt,
                image_count=image_count,
                aspect_ratio=aspect_ratio,
                render_speed=render_speed,
                style_type=style_type,
                magic_prompt=magic_prompt,
                seed=seed,
                character_image_bytes=char_bytes
            )
            
            # Make API request
            logger.info("Sending request to Ideogram API")
            logger.info(f"Request fields: {fields}")
            logger.info(f"Files being sent: {[f[0] for f in files]}")
            print(f"[Ideogram Character] API URL: {self.api_url}")
            print(f"[Ideogram Character] Request fields: {fields}")
            print(f"[Ideogram Character] Files: {[f[0] for f in files]}")
            print(f"[Ideogram Character] Character image bytes: {len(char_bytes)} bytes")
            
            response = self.make_api_request(api_key, fields, files)
            
            # Process response
            logger.info("Processing API response")
            images, info = self.process_api_response(response)
            
            logger.info(f"Successfully generated {images.shape[0] if len(images.shape) == 4 else 1} images")
            return (images, info)
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            
            # Log detailed error information
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Print to console for ComfyUI debugging
            print(f"[Ideogram Character] ERROR: {error_msg}")
            print(f"[Ideogram Character] Full traceback:")
            traceback.print_exc()
            
            # Return a placeholder error image
            error_image = torch.zeros((1, 512, 512, 3))
            # Add red tint to indicate error
            error_image[:, :, :, 0] = 0.5
            
            return (error_image, error_msg)