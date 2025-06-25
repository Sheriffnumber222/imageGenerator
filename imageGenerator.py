import os
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter
from rembg import remove, new_session

# === Configuration ===
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
input_folder = os.path.join(desktop, "Image Generator")
image_extensions = ('.jpg', '.jpeg', '.png')

# === Get Latest Image ===
def get_latest_image(folder):
    images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    if not images:
        return None
    images.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    return images[0]

# === Get Dominant Edge Color ===
def get_dominant_edge_color(img, edge=20):
    np_img = np.array(img)
    top = np_img[:edge, :, :3]
    bottom = np_img[-edge:, :, :3]
    left = np_img[:, :edge, :3]
    right = np_img[:, -edge:, :3]
    top_avg = top.mean(axis=0).mean(axis=0)
    bottom_avg = bottom.mean(axis=0).mean(axis=0)
    left_avg = left.mean(axis=0).mean(axis=0)
    right_avg = right.mean(axis=0).mean(axis=0)
    avg_color = (top_avg + bottom_avg + left_avg + right_avg) / 4
    return tuple(map(int, avg_color))

# === Choose Black or White Text Based on Background Brightness ===
def pick_text_color(rgb):
    brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)

# === Create Gradient Background ===
def create_gradient(size, color_top, color_bottom):
    gradient = Image.new('RGB', size, color=0)
    draw = ImageDraw.Draw(gradient)

    for y in range(size[1]):
        ratio = y / size[1]
        r = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
        g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
        b = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))

    return gradient

# === Resize Large Images ===
def resize_if_needed(img, max_width=1024):
    if img.width > max_width:
        new_height = int((max_width / img.width) * img.height)
        return img.resize((max_width, new_height), Image.LANCZOS)
    return img

# === Add Drop Shadow ===
def add_shadow(image, offset=(5,5), background_color=0, shadow_color=0, border=20, iterations=10):
    total_width = image.width + abs(offset[0]) + 2 * border
    total_height = image.height + abs(offset[1]) + 2 * border

    shadow_image = Image.new('RGBA', (total_width, total_height), background_color)
    shadow = Image.new('RGBA', image.size, color=shadow_color)
    alpha = image.split()[3]
    shadow.putalpha(alpha)

    shadow_position = (border + max(offset[0], 0), border + max(offset[1], 0))
    shadow_image.paste(shadow, shadow_position, shadow)
    for _ in range(iterations):
        shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(2))

    image_position = (border - min(offset[0], 0), border - min(offset[1], 0))
    shadow_image.paste(image, image_position, image)
    return shadow_image

# === Compose Final Image with Subject and Text ===
def compose_image(cutout, bg_color_top, bg_color_bottom, text_lines, text_position="below"):
    width, height = cutout.size

    alpha = np.array(cutout.split()[-1])
    rows_with_data = np.where(np.max(alpha, axis=1) > 0)[0]
    subject_top = rows_with_data[0] if len(rows_with_data) > 0 else 0
    subject_bottom = rows_with_data[-1] if len(rows_with_data) > 0 else height

    font_path = "arialbd.ttf"
    longest_line = max(text_lines, key=lambda l: len(l)) if text_lines else ""
    font_size = 60
    while font_size > 10:
        font = ImageFont.truetype(font_path, font_size)
        if font.getlength(longest_line) <= 0.95 * width:
            break
        font_size -= 1

    text_height_total = sum([font.getbbox(line)[3] - font.getbbox(line)[1] + 10 for line in text_lines]) + 30

    if text_position == "above":
        overlap = 45
        text_y_start = 10
        cutout_y = text_height_total - overlap
        total_height = subject_bottom + text_height_total - overlap + 30
    else:
        cutout_y = 0
        text_y_start = subject_bottom + 30
        total_height = subject_bottom + text_height_total + 30

    gradient = create_gradient((width, total_height), bg_color_top, bg_color_bottom)
    result = Image.new("RGBA", (width, total_height))
    result.paste(gradient, (0, 0))

    cutout_with_shadow = add_shadow(cutout)
    result.paste(cutout_with_shadow, (0, cutout_y), cutout_with_shadow)

    draw = ImageDraw.Draw(result)
    text_color = pick_text_color(bg_color_bottom if text_position == "below" else bg_color_top)
    y = text_y_start
    for line in text_lines:
        text_width = font.getlength(line)
        x = (width - text_width) // 2
        draw.text((x, y), line, font=font, fill=text_color)
        y += font.getbbox(line)[3] - font.getbbox(line)[1] + 10

    return result.convert("RGB")  # <-- flatten before save

# === Background Removal + Gradient ===
def remove_background_and_add_gradient(input_path, output_path, text_input, text_position):
    with open(input_path, 'rb') as f:
        input_data = f.read()

    original_img = Image.open(input_path).convert("RGBA")
    resized_img = resize_if_needed(original_img)

    session = new_session("isnet-general-use")
    output_data = remove(input_data, session=session)

    cutout = Image.open(io.BytesIO(output_data)).convert("RGBA")
    cutout = resize_if_needed(cutout)

    text_lines = [line.strip() for line in text_input.split("//")] if text_input else []

    gradient_options = {
        "1": ("#2c3e50", "#bdc3c7"),
        "2": ("#0f2027", "#2c5364"),
        "3": ("#1a1a1a", "#272727"),
        "4": ("#4d217a", "#9b59b6"),
        "5": ("#1b1f3b", "#4d217a"),
        "6": ("#7f4fa3", "#4d217a")
    }

    while True:
        print("\nChoose gradient style:")
        print("1 = Soft Steel (#2c3e50 → #bdc3c7)")
        print("2 = Midnight Fade (#0f2027 → #2c5364)")
        print("3 = Subtle Charcoal (#1a1a1a → #272727)")
        print("4 = Brand to Soft Purple (#4d217a → #9b59b6)")
        print("5 = Navy to Brand (#1b1f3b → #4d217a)")
        print("6 = Brand Tint to Brand (#7f4fa3 → #4d217a)")
        print("custom = Enter your own two hex colors")
        print("Leave blank to auto-detect from image background")

        gradient_choice = input("Your choice (1–6, custom, or blank): ").strip()

        if gradient_choice.lower() == "custom":
            top_hex = input("Top gradient color (e.g. #ff0000): ").strip()
            bottom_hex = input("Bottom gradient color (e.g. #000000): ").strip()
            try:
                bg_color_top = ImageColor.getrgb(top_hex)
                bg_color_bottom = ImageColor.getrgb(bottom_hex)
            except ValueError:
                print("Invalid color. Reverting to auto background.")
                bg_color_top = get_dominant_edge_color(resized_img)
                bg_color_bottom = tuple(max(0, c - 40) for c in bg_color_top)
        elif gradient_choice in gradient_options:
            top_hex, bottom_hex = gradient_options[gradient_choice]
            bg_color_top = ImageColor.getrgb(top_hex)
            bg_color_bottom = ImageColor.getrgb(bottom_hex)
        else:
            bg_color_top = get_dominant_edge_color(resized_img)
            bg_color_bottom = tuple(max(0, c - 40) for c in bg_color_top)

        result = compose_image(cutout, bg_color_top, bg_color_bottom, text_lines, text_position)
        result.show()
        confirm = input("Use this gradient? (y/n): ").strip().lower()
        if confirm == 'y':
            result.save(output_path.replace(".png", ".jpg"), format="JPEG", quality=95)
            print(f"✅ Saved flattened version for Shopify: {output_path.replace('.png', '.jpg')}")
            break

# === Main Runner ===
def main():
    latest = get_latest_image(input_folder)
    if not latest:
        print("No image found in folder.")
        return

    input_path = os.path.join(input_folder, latest)
    base = os.path.splitext(latest)[0]

    user_text = input("Enter text (use // for line breaks): ").strip()
    text_position = input("Where should the text appear? (above/below): ").strip().lower()
    if text_position not in ["above", "below"]:
        print("Invalid input. Defaulting to 'below'.")
        text_position = "below"

    finished_folder = os.path.join(input_folder, "Finished")
    os.makedirs(finished_folder, exist_ok=True)

    output_name = f"{base} Transparent Background.png"
    output_path = os.path.join(finished_folder, output_name)

    remove_background_and_add_gradient(input_path, output_path, user_text, text_position)

if __name__ == "__main__":
    main()