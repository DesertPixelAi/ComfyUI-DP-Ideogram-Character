"""
Image utility functions for ComfyUI Ideogram Character Node
"""

import torch
import numpy as np
from PIL import Image
import io
from typing import Optional, Tuple

def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is in RGB mode"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
            return background
        else:
            return image.convert('RGB')
    return image

def resize_to_limit(image: Image.Image, max_size_mb: float = 10) -> Image.Image:
    """Resize image if it exceeds size limit"""
    # Estimate current size
    temp_buffer = io.BytesIO()
    image.save(temp_buffer, format='PNG')
    current_size_mb = temp_buffer.tell() / (1024 * 1024)
    
    if current_size_mb > max_size_mb:
        # Calculate resize factor
        resize_factor = np.sqrt(max_size_mb / current_size_mb) * 0.9  # 90% to be safe
        new_width = int(image.width * resize_factor)
        new_height = int(image.height * resize_factor)
        
        # Ensure minimum dimensions
        new_width = max(new_width, 256)
        new_height = max(new_height, 256)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def calculate_aspect_ratio(width: int, height: int) -> str:
    """Calculate closest standard aspect ratio from dimensions"""
    ratio = width / height
    
    aspect_ratios = {
        "1:1": 1.0,
        "4:3": 4/3,
        "3:4": 3/4,
        "16:9": 16/9,
        "9:16": 9/16,
        "3:2": 3/2,
        "2:3": 2/3,
        "5:4": 5/4,
        "4:5": 4/5,
        "16:10": 16/10,
        "10:16": 10/16,
        "3:1": 3.0,
        "1:3": 1/3,
        "2:1": 2.0,
        "1:2": 0.5
    }
    
    closest_ratio = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
    return closest_ratio[0]

def validate_image_dimensions(width: int, height: int) -> Tuple[bool, str]:
    """Validate image dimensions for Ideogram API"""
    min_dim = 256
    max_dim = 2048
    max_pixels = 4194304  # 4 megapixels
    
    if width < min_dim or height < min_dim:
        return False, f"Image dimensions too small. Minimum dimension is {min_dim}px"
    
    if width > max_dim or height > max_dim:
        return False, f"Image dimensions too large. Maximum dimension is {max_dim}px"
    
    if width * height > max_pixels:
        return False, f"Image resolution too high. Maximum is {max_pixels} pixels"
    
    return True, "Valid dimensions"

def create_error_image(message: str, width: int = 512, height: int = 512) -> Image.Image:
    """Create an error placeholder image with message"""
    # Create red-tinted image
    img = Image.new('RGB', (width, height), color=(128, 0, 0))
    
    # Add text if possible (requires PIL with text support)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to use a basic font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw error message
        text_lines = message.split('\n')
        y_offset = height // 2 - (len(text_lines) * 25)
        
        for line in text_lines[:10]:  # Limit lines to prevent overflow
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x_position = (width - text_width) // 2
            draw.text((x_position, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 25
    except:
        # If text drawing fails, just return the red image
        pass
    
    return img