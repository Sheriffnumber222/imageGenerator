import os
import io
from PIL import Image, ImageOps
from rembg import remove, new_session

# === Configuration ===
base_folder = r"C:\Users\LocalAdmin\Desktop\automation\imageGeneratorAssets"
input_folder = base_folder
finished_folder = os.path.join(base_folder, "Finished")
os.makedirs(finished_folder, exist_ok=True)
image_extensions = ('.jpg', '.jpeg', '.png', '.webp')

# Background and canvas padding
BACKGROUND_HEX = "#cb845f"   # Change this to whatever you want (e.g., "#1C1C1C")
PADDING = 56                  # Space around the cut-out on the new background

# ReMBG session (create once)
SESSION = new_session("isnet-general-use")

def hex_to_rgb(h: str):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch*2 for ch in h)
    if len(h) != 6:
        raise ValueError("BACKGROUND_HEX must be a 3- or 6-digit hex color like '#2A2A2A' or '#1C1'")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def get_latest_image(folder):
    images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    if not images:
        return None
    images.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    return images[0]

def remove_background_and_change_bg(input_path, output_path, bg_hex=BACKGROUND_HEX, padding=PADDING):
    # 1) Load & respect EXIF orientation
    with Image.open(input_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")

        # 2) Remove background
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        cutout_bytes = remove(buf.getvalue(), session=SESSION)

    subject = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")

    # 3) Create solid background canvas
    bg_rgb = hex_to_rgb(bg_hex)
    sw, sh = subject.size
    canvas = Image.new("RGBA", (sw + padding * 2, sh + padding * 2), bg_rgb + (255,))

    # 4) Center subject on background (no shadows, no glows)
    x = (canvas.width - sw) // 2
    y = (canvas.height - sh) // 2
    canvas.alpha_composite(subject, (x, y))

    # 5) Save
    canvas.convert("RGB").save(output_path, format="PNG")
    print(f"✅ Saved: {output_path}")

def main():
    latest = get_latest_image(input_folder)
    if not latest:
        print("No image found in folder.")
        return

    src = os.path.join(input_folder, latest)
    base, _ = os.path.splitext(latest)
    dst = os.path.join(finished_folder, f"{base} - SolidBG.png")
    remove_background_and_change_bg(src, dst)

if __name__ == "__main__":
    main()