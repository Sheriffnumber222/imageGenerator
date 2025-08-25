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

def get_latest_image(folder):
    images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    if not images:
        return None
    images.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    return images[0]

def remove_background(input_path, output_path):
    # Read original and respect EXIF orientation
    with Image.open(input_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        input_bytes = buf.getvalue()

    # Use rembg with ISNet (good general model)
    session = new_session("isnet-general-use")
    cutout_bytes = remove(input_bytes, session=session)

    # Save exactly as returned (transparent PNG)
    cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
    cutout.save(output_path, format="PNG")
    print(f"✅ Saved: {output_path}")

def main():
    latest = get_latest_image(input_folder)
    if not latest:
        print("No image found in folder.")
        return

    src = os.path.join(input_folder, latest)
    base, _ = os.path.splitext(latest)
    dst = os.path.join(finished_folder, f"{base} Transparent.png")
    remove_background(src, dst)

if __name__ == "__main__":
    main()