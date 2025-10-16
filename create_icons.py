"""
Simple script to create extension icons
Run this to generate basic icons for the browser extension
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple icon with the 🎭 emoji or DF text"""
    # Create image with gradient background
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw gradient background (purple)
    for y in range(size):
        color_value = int(102 + (118 - 102) * (y / size))
        draw.rectangle([(0, y), (size, y+1)], fill=(color_value, 126, 234))
    
    # Draw circle background
    margin = size // 8
    draw.ellipse(
        [(margin, margin), (size - margin, size - margin)],
        fill=(255, 255, 255, 200)
    )
    
    # Try to add text
    try:
        # Use default font
        font_size = size // 2
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw "DF" text
    text = "DF"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((size - text_width) // 2, (size - text_height) // 2 - size // 10)
    
    # Draw text with shadow
    shadow_offset = max(1, size // 32)
    draw.text((position[0] + shadow_offset, position[1] + shadow_offset), text, 
              fill=(100, 100, 100), font=font)
    draw.text(position, text, fill=(102, 126, 234), font=font)
    
    # Save
    img.save(output_path, 'PNG')
    print(f"✓ Created {output_path}")

def main():
    """Create all required icon sizes"""
    icons_dir = os.path.join(os.path.dirname(__file__), 'extension', 'icons')
    os.makedirs(icons_dir, exist_ok=True)
    
    print("Creating extension icons...")
    print("=" * 50)
    
    sizes = [16, 48, 128]
    for size in sizes:
        output_path = os.path.join(icons_dir, f'icon{size}.png')
        create_icon(size, output_path)
    
    print("=" * 50)
    print("✓ All icons created successfully!")
    print(f"Icons saved to: {icons_dir}")

if __name__ == '__main__':
    main()
