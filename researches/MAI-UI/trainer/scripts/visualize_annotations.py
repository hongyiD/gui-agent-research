#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize action coordinates and thinking process on training images.

This script reads sft_train.jsonl files and annotates images with:
- Red dots at coordinate locations
- Red text showing the thinking process (in Chinese)

Usage:
    python visualize_annotations.py --input sft_train.jsonl
    python visualize_annotations.py --input sft_train.jsonl --output_dir annotated/ --point_size 15
    python visualize_annotations.py --input sft_train.jsonl --index 2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not available. Please install: pip install Pillow")


# =============================================================================
# Font Loading
# =============================================================================

def load_chinese_font(font_size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    """Load Chinese font if available.
    
    Args:
        font_size: Font size in pixels.
        
    Returns:
        Font object or None if not available.
    """
    if not HAS_PIL:
        return None
    
    # Try common Chinese font paths
    font_paths = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # Windows (if running on Windows)
        "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
        "C:/Windows/Fonts/simsun.ttc",  # SimSun
    ]
    
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                # For TTC files, we need to specify font index (usually 0)
                if font_path.endswith('.ttc'):
                    return ImageFont.truetype(font_path, font_size, index=0)
                else:
                    return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
    
    # Fallback to default font (may not support Chinese)
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_thinking_and_coordinates(assistant_content: str) -> tuple[str | None, list[dict]]:
    """Parse thinking process and coordinates from assistant content.
    
    Args:
        assistant_content: Text content from assistant message.
        
    Returns:
        Tuple of (thinking_text, coordinates_list).
        coordinates_list contains dicts with keys: 'coords', 'type' (coordinate/start_coordinate/end_coordinate)
    """
    thinking_text = None
    coordinates = []
    
    # Extract thinking
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", assistant_content, re.DOTALL)
    if thinking_match:
        thinking_text = thinking_match.group(1).strip()
    
    # Extract tool_call
    tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_content, re.DOTALL)
    if tool_call_match:
        try:
            tool_call_json = json.loads(tool_call_match.group(1).strip())
            arguments = tool_call_json.get("arguments", {})
            
            # Extract coordinates
            coord_fields = ["coordinate", "start_coordinate", "end_coordinate"]
            for field in coord_fields:
                if field in arguments:
                    coords = arguments[field]
                    if isinstance(coords, list) and len(coords) >= 2:
                        coordinates.append({
                            "coords": [int(coords[0]), int(coords[1])],
                            "type": field
                        })
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse tool_call JSON: {e}")
    
    return thinking_text, coordinates


def parse_last_step_from_history(user_content: str) -> tuple[str | None, list[dict]]:
    """Parse the last step's thinking and coordinates from Previous actions in user content.
    
    Args:
        user_content: Text content from user message (may contain Previous actions).
        
    Returns:
        Tuple of (thinking_text, coordinates_list) for the last step.
        coordinates_list contains dicts with keys: 'coords', 'type', 'step_num'.
    """
    thinking_text = None
    coordinates = []
    
    # Look for "Previous actions:" section
    if "Previous actions:" not in user_content:
        return None, []
    
    # Extract all Step sections
    step_pattern = r"Step (\d+):\s*<thinking>\s*(.*?)\s*</thinking>\s*<tool_call>\s*({.*?})\s*</tool_call>"
    step_matches = list(re.finditer(step_pattern, user_content, re.DOTALL))
    
    if not step_matches:
        return None, []
    
    # Get the last step (most recent)
    last_match = step_matches[-1]
    step_num = int(last_match.group(1))
    thinking_text = last_match.group(2).strip()
    tool_call_str = last_match.group(3)
    
    try:
        tool_call_json = json.loads(tool_call_str)
        arguments = tool_call_json.get("arguments", {})
        
        # Extract coordinates
        coord_fields = ["coordinate", "start_coordinate", "end_coordinate"]
        for field in coord_fields:
            if field in arguments:
                coords = arguments[field]
                if isinstance(coords, list) and len(coords) >= 2:
                    coordinates.append({
                        "coords": [int(coords[0]), int(coords[1])],
                        "type": field,
                        "step_num": step_num
                    })
    except json.JSONDecodeError:
        pass
    
    return thinking_text, coordinates


def extract_image_path(messages: list[dict], image_dir: Path) -> Path | None:
    """Extract image path from user message content.
    
    Args:
        messages: List of message dictionaries.
        image_dir: Base directory for images.
        
    Returns:
        Path to image file, or None if not found.
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        image_url = item.get("image_url", "")
                        if isinstance(image_url, str):
                            # Try to resolve image path
                            image_path = image_dir / image_url
                            if image_path.exists():
                                return image_path
                            # Try just the filename
                            image_path = image_dir / Path(image_url).name
                            if image_path.exists():
                                return image_path
    return None


# =============================================================================
# Drawing Functions
# =============================================================================

def wrap_text(text: str, max_width: int, font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None) -> list[str]:
    """Wrap text to fit within max_width.
    
    Args:
        text: Text to wrap.
        max_width: Maximum width in pixels.
        font: Font object for measuring text.
        
    Returns:
        List of text lines.
    """
    if not font:
        # Simple character-based wrapping if no font
        # For Chinese characters, estimate ~font_size pixels per character
        chars_per_line = max(1, max_width // 20)  # Rough estimate
        lines = []
        for i in range(0, len(text), chars_per_line):
            lines.append(text[i:i+chars_per_line])
        return lines if lines else [text]
    
    lines = []
    current_line = ""
    
    # Split by words (for English) and characters (for Chinese)
    # For mixed content, we'll process character by character for better control
    i = 0
    while i < len(text):
        char = text[i]
        
        # Try adding this character
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line = test_line
            i += 1
        else:
            # Current line is full, start new line
            if current_line:
                lines.append(current_line)
                current_line = ""
            else:
                # Single character is too wide, add it anyway
                current_line = char
                i += 1
    
    if current_line:
        lines.append(current_line)
    
    return lines if lines else [text]


def draw_annotations(
    image: Image.Image,
    thinking_text: str | None,
    coordinates: list[dict],
    point_size: int = 12,
    font_size: int = 20,
) -> Image.Image:
    """Draw annotations on image.
    
    Args:
        image: PIL Image to annotate.
        thinking_text: Thinking process text to display.
        coordinates: List of coordinate dicts with 'coords' and 'type'.
        point_size: Size of the red dot (radius).
        font_size: Font size for text.
        
    Returns:
        Annotated image.
    """
    if not HAS_PIL:
        return image
    
    # Create a copy to draw on
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Load font
    font = load_chinese_font(font_size)
    
    # Color: Red
    red_color = (255, 0, 0)
    white_color = (255, 255, 255)
    
    img_width, img_height = annotated.size
    
    # Store coordinate label position for thinking text placement
    coord_label_bottom_y = None
    coord_label_x = None
    
    # Draw coordinates if available
    if coordinates:
        for coord_info in coordinates:
            x, y = coord_info["coords"]
            coord_type = coord_info.get("type", "coordinate")
            
            # Draw a mouse pointer cursor pointing at the coordinate
            # Mouse pointer shape: arrow pointing to (x, y)
            pointer_size = max(point_size * 4, 40)  # Make it larger and more visible (at least 40px)
            
            # Calculate pointer points (arrow pointing to (x, y))
            # The tip of the pointer is at the coordinate
            tip_x, tip_y = int(x), int(y)
            
            # Arrow shape: triangle pointing to the coordinate
            # Points forming a right-angled triangle pointing down-right
            base_x = tip_x - pointer_size
            base_y = tip_y - pointer_size // 3
            mid_x = tip_x - pointer_size // 2
            mid_y = tip_y - pointer_size
            
            # Draw filled triangle (mouse pointer arrow)
            pointer_points = [(tip_x, tip_y), (base_x, base_y), (mid_x, mid_y)]
            
            # Draw a thick white outline first (for better visibility)
            draw.polygon(
                pointer_points,
                fill=None,
                outline=white_color,
                width=6
            )
            
            # Draw filled triangle
            draw.polygon(
                pointer_points,
                fill=red_color,
                outline=None
            )
            
            # Draw a red outline on top
            draw.polygon(
                pointer_points,
                fill=None,
                outline=red_color,
                width=3
            )
            
            # Also draw a red circle at the tip for extra visibility
            draw.ellipse(
                [tip_x - 5, tip_y - 5, tip_x + 5, tip_y + 5],
                fill=red_color,
                outline=white_color,
                width=2
            )
            
            # Draw coordinate text label next to the pointer
            # Don't show step_num for the last step (it's implied)
            coord_text = f"[{x}, {y}]"
            # Use a slightly smaller font for coordinates
            coord_font_size = max(font_size - 4, 14)
            coord_font = load_chinese_font(coord_font_size) if font else None
            
            if coord_font:
                coord_bbox = coord_font.getbbox(coord_text)
                coord_text_width = coord_bbox[2] - coord_bbox[0]
                coord_text_height = coord_bbox[3] - coord_bbox[1]
            else:
                coord_text_width = len(coord_text) * 8
                coord_text_height = coord_font_size
            
            # Position coordinate text to the right of the pointer
            coord_text_x = tip_x + pointer_size // 2 + 8
            coord_text_y = tip_y - coord_text_height // 2
            
            # Ensure text doesn't go beyond image bounds
            if coord_text_x + coord_text_width + 10 > img_width:
                coord_text_x = tip_x - coord_text_width - pointer_size // 2 - 8
            
            # Store position for thinking text placement
            coord_label_x = coord_text_x
            coord_label_bottom_y = coord_text_y + coord_text_height + 8  # Bottom of coordinate label + padding
            
            # Create background for coordinate text (more opaque for better visibility)
            coord_bg_img = Image.new('RGBA', (int(coord_text_width + 12), int(coord_text_height + 8)), (255, 255, 255, 240))
            coord_bg_draw = ImageDraw.Draw(coord_bg_img)
            
            # Draw border around coordinate text
            coord_bg_draw.rectangle(
                [0, 0, coord_text_width + 11, coord_text_height + 7],
                fill=None,
                outline=red_color,
                width=2
            )
            
            if coord_font:
                coord_bg_draw.text((6, 4), coord_text, fill=red_color, font=coord_font)
            else:
                coord_bg_draw.text((6, 4), coord_text, fill=red_color)
            
            # Paste coordinate text onto main image
            if 0 <= coord_text_x <= img_width - coord_text_width - 12 and 0 <= coord_text_y <= img_height - coord_text_height - 8:
                annotated.paste(coord_bg_img, (int(coord_text_x), int(coord_text_y)), coord_bg_img)
    
    # Draw thinking text
    if thinking_text:
        if coordinates and coord_label_bottom_y is not None and coord_label_x is not None:
            # Position thinking text below the coordinate label
            text_x = coord_label_x
            text_y = coord_label_bottom_y + 5  # Below coordinate label with spacing
            
            # Check if text would go beyond image bounds
            max_text_width = img_width - text_x - 20
            
            if max_text_width < 100 or text_y + font_size + 20 > img_height:  # Not enough space, adjust position
                x, y = coordinates[0]["coords"]
                pointer_size = max(point_size * 4, 40)
                text_x = x - pointer_size - 10
                text_y = y + pointer_size + 30  # Below pointer, with more spacing
                max_text_width = min(img_width - text_x - 10, img_width - 10)
        else:
            # No coordinates: display text in center-top area
            text_x = 20
            text_y = 20
            max_text_width = img_width - 40
        
        # Wrap text
        text_lines = wrap_text(thinking_text, max_text_width, font)
        
        # Draw text with background for better visibility
        line_height = font_size + 4 if font else 20
        for i, line in enumerate(text_lines):
            line_y = text_y + i * line_height
            
            # Draw text background (semi-transparent white)
            if font:
                bbox = font.getbbox(line)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(line) * 10  # Rough estimate
                text_height = font_size
            
            # Create a temporary image for the text with background
            text_img = Image.new('RGBA', (int(text_width + 10), int(text_height + 6)), (255, 255, 255, 200))
            text_draw = ImageDraw.Draw(text_img)
            
            if font:
                text_draw.text((5, 3), line, fill=red_color, font=font)
            else:
                text_draw.text((5, 3), line, fill=red_color)
            
            # Paste text image onto main image
            if text_x + text_width + 10 <= img_width and line_y + text_height + 6 <= img_height:
                annotated.paste(text_img, (int(text_x), int(line_y)), text_img)
    
    return annotated


# =============================================================================
# Main Processing
# =============================================================================

def process_training_file(
    input_file: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    point_size: int = 12,
    font_size: int = 20,
    index: int | None = None,
) -> dict[str, int]:
    """Process training file and create annotated images.
    
    Args:
        input_file: Path to sft_train.jsonl file.
        output_dir: Directory to save annotated images.
        image_dir: Directory containing original images (default: input_file.parent).
        point_size: Size of red dot.
        font_size: Font size for text.
        index: Specific index to process (0-based, None = process all).
        
    Returns:
        Statistics dictionary.
    """
    if not HAS_PIL:
        print("Error: PIL/Pillow is required. Please install: pip install Pillow")
        return {"processed": 0, "skipped": 0, "errors": 0}
    
    if image_dir is None:
        image_dir = input_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
    }
    
    # Read and process each line
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            # If index is specified, only process that line (0-based -> 1-based conversion)
            if index is not None:
                if line_num - 1 != index:
                    continue
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                # Extract image path
                image_path = extract_image_path(messages, Path(image_dir))
                if not image_path:
                    print(f"Warning: Line {line_num}: Image not found")
                    stats["skipped"] += 1
                    continue
                
                # Extract assistant content
                assistant_content = None
                for msg in messages:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if item.get("type") == "text":
                                    assistant_content = item.get("text", "")
                                    break
                        break
                
                if not assistant_content:
                    print(f"Warning: Line {line_num}: No assistant content found")
                    stats["skipped"] += 1
                    continue
                
                # Extract user content (may contain Previous actions with coordinates)
                user_content_text = None
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if item.get("type") == "text":
                                    user_content_text = item.get("text", "")
                                    break
                        break
                
                # Parse thinking and coordinates from assistant content (current step)
                current_thinking, current_coordinates = parse_thinking_and_coordinates(assistant_content)
                
                # Parse the last step from Previous actions in user content
                last_step_thinking = None
                last_step_coordinates = []
                if user_content_text:
                    last_step_thinking, last_step_coordinates = parse_last_step_from_history(user_content_text)
                
                # Use the last step's thinking and coordinates if available, otherwise use current step's
                if last_step_coordinates:
                    # Use last step's thinking and coordinates
                    thinking_text = last_step_thinking
                    coordinates = last_step_coordinates
                else:
                    # Use current step's thinking and coordinates
                    thinking_text = current_thinking
                    coordinates = current_coordinates
                
                # Skip only if neither thinking text nor coordinates are available
                if not thinking_text and not coordinates:
                    print(f"Warning: Line {line_num}: No thinking text or coordinates found")
                    stats["skipped"] += 1
                    continue
                
                # Inform user if no coordinates (some actions like 'type' don't have coordinates)
                if not coordinates:
                    print(f"Info: Line {line_num}: No coordinates found (action may not require coordinates, displaying thinking text only)")
                
                # Load image
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"Error: Line {line_num}: Failed to load image {image_path}: {e}")
                    stats["errors"] += 1
                    continue
                
                # Draw annotations
                annotated_image = draw_annotations(
                    image,
                    thinking_text,
                    coordinates,
                    point_size=point_size,
                    font_size=font_size,
                )
                
                # Save annotated image
                output_filename = f"{image_path.stem}_annotated.png"  # Always use PNG
                output_path = output_dir / output_filename
                
                # Ensure output directory exists and is writable
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Save as PNG
                    annotated_image.save(output_path, 'PNG')
                    
                    print(f"Processed: {image_path.name} -> {output_filename}")
                    stats["processed"] += 1
                except Exception as e:
                    print(f"Error: Line {line_num}: Failed to save image {output_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    stats["errors"] += 1
                    continue
                
                print(f"Processed: {image_path.name} -> {output_filename}")
                stats["processed"] += 1
                
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num}: Failed to parse JSON: {e}")
                stats["errors"] += 1
            except Exception as e:
                print(f"Error: Line {line_num}: Unexpected error: {e}")
                stats["errors"] += 1
    
    return stats


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize action coordinates and thinking process on training images"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to sft_train.jsonl file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory for annotated images (default: input_dir/annotated_images/)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing original images (default: input file directory)"
    )
    parser.add_argument(
        "--point_size",
        type=int,
        default=12,
        help="Size of red dot (radius in pixels, default: 12)"
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=20,
        help="Font size for text (default: 20)"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Process only the specified index (0-based). If not specified, process all entries."
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_file.parent / "annotated_images"
    
    # Determine image directory
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    if image_dir:
        print(f"Image directory: {image_dir}")
    print(f"Point size: {args.point_size}")
    print(f"Font size: {args.font_size}")
    if args.index is not None:
        print(f"Processing index: {args.index} (0-based)")
    print()
    
    # Process file
    stats = process_training_file(
        input_file=input_file,
        output_dir=output_dir,
        image_dir=image_dir,
        point_size=args.point_size,
        font_size=args.font_size,
        index=args.index,
    )
    
    print()
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
