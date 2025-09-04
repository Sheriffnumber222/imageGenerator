import os
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter
from rembg import remove, new_session

# === Configuration ===
base_folder = r"C:\Users\LocalAdmin\Desktop\automation\imageGeneratorAssets"
input_folder = base_folder
finished_folder = os.path.join(base_folder, "Finished")
os.makedirs(finished_folder, exist_ok=True)
image_extensions = ('.jpg', '.jpeg', '.png')

# ---- Layout controls ----
SIDE_PADDING_PCT   = 0.00   # 0.00 = no side padding; 0.02 = 2% per side
TOP_PADDING_PCT    = 0.08
BOTTOM_PADDING_PCT = 0.08

# Text sizing targets (ratios of available text width)
TEXT_WIDTH_MIN_RATIO = 0.60    # aim for longest line ≥ 60% of available width
TEXT_WIDTH_MAX_RATIO = 0.90    # and ≤ 90% of available width

# [NEW] Orphan mitigation (gentle boost when a block’s longest line is very short)
ORPHAN_MIN_RATIO     = 0.40    # if longest line ≤ 40%, we try to nudge it up a bit
ORPHAN_TARGET_RATIO  = 0.50    # aim to reach ~50% if possible (without breaking other rules)
ORPHAN_UPSCALE_STEP  = 1.04    # per-step multiplier when nudging font size up
ORPHAN_UPSCALE_STEPS = 6       # at most ~25% total boost (1.04^6 ≈ 1.27), but other guards still apply

# Vertical rhythm
LINE_SPACING_PCT   = 0.024     # line spacing vs canvas height
GAP_PCT            = 0.02      # base gap between text blocks and product
MIN_GAP_PX         = 8         # never go below this when auto-shrinking gaps

# [NEW] Quiet zone (buffer around the product when adjacent to text)
QUIET_ZONE_FRAC_OF_PRODUCT = 0.065  # 6.5% of product height as min buffer (per side with text)

# [CHANGE] Optical bias is now expressed as explicit product-center anchors
# Keep this for single-sided layout bias (still useful)
OPTICAL_BIAS_PCT   = -0.03

# Product size protection
PRODUCT_MIN_FRAC_OF_SAFE = 0.35   # product should occupy at least this fraction of safe vertical space
PREFER_FULL_WIDTH_WHEN_POSSIBLE = True

# Optional: make top lines slightly larger than bottom lines (base multipliers)
TOP_FONT_MULT    = 1.00
BOTTOM_FONT_MULT = 1.00

# ---- Hierarchy enforcement (when both top and bottom exist) ----
BOTTOM_HIERARCHY_SCALE   = 0.92  # [NEW] bottom font ≈ 90–92% of top
BOTTOM_MAX_SAFE_FRAC     = 0.25  # [NEW] bottom block ≤ 25% of safe band

# ---- Shadow controls ----
SHADOW_OFFSET     = (6, 6)
SHADOW_BORDER     = 16
SHADOW_BLUR       = 2
SHADOW_ITERATIONS = 8
SHADOW_ALPHA      = 160  # 0..255

# ---- Text legibility (optional stroke/outline) ----
TEXT_STROKE_WIDTH = 0  # set to 1–2 to enable a safety outline around text

# ---- Centering behavior ----
CENTER_TEXT_BLOCKS = True  # center text blocks within their own sections

# ---- Single‑sided anchors (0.5 = exact center) ----
SINGLE_SIDE_ANCHOR_TOP    = 0.62  # only top text → product a bit lower
SINGLE_SIDE_ANCHOR_BOTTOM = 0.38  # only bottom text → product a bit higher

# ---- [NEW] Both‑sides product anchor (optical center) ----
PRODUCT_CENTER_RATIO_BOTH = 0.45  # product visual center at ~57% of safe band

# Platform presets
PLATFORM_SPECS = {
    "linkedin":  {"size": (1200, 1200), "desc": "LinkedIn square 1:1 (1200x1200)"},
    "facebook":  {"size": (1200, 1500), "desc": "Facebook feed 4:5 (1200x1500)"},
    "instagram": {"size": (1080, 1350), "desc": "Instagram feed 4:5 (1080x1350)"},
    "email":     {"size": (1200,  600), "desc": "Email hero 2:1 (1200x600)"}
}

# Cache rembg session once (performance)
RMBG_SESSION = new_session("isnet-general-use")

# === Utility ===
def get_latest_image(folder):
    images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    if not images:
        return None
    images.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    return images[0]

def get_dominant_edge_color(img, edge=20):
    np_img = np.array(img)
    top = np_img[:edge, :, :3]
    bottom = np_img[-edge:, :, :3]
    left = np_img[:, :edge, :3]
    right = np_img[:, -edge:, :3]
    avg = (top.mean((0,0)) + bottom.mean((0,0)) + left.mean((0,0)) + right.mean((0,0))) / 4
    return tuple(map(int, avg))

def pick_text_color(rgb):
    brightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)

def inverse_color(rgb):
    return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])

def create_gradient(size, color_top, color_bottom):
    w, h = size
    gradient = Image.new('RGB', size, color=0)
    draw = ImageDraw.Draw(gradient)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(color_top[0]*(1-t) + color_bottom[0]*t)
        g = int(color_top[1]*(1-t) + color_bottom[1]*t)
        b = int(color_top[2]*(1-t) + color_bottom[2]*t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return gradient

def resize_if_needed(img, max_width=2048):
    if img.width > max_width:
        nh = int((max_width / img.width) * img.height)
        return img.resize((max_width, nh), Image.LANCZOS)
    return img

def add_shadow(image,
               offset=SHADOW_OFFSET,
               background_color=(0, 0, 0, 0),
               shadow_color=None,
               border=SHADOW_BORDER,
               iterations=SHADOW_ITERATIONS,
               blur=SHADOW_BLUR):
    if shadow_color is None:
        shadow_color = (0, 0, 0, SHADOW_ALPHA)
    total_w = image.width + abs(offset[0]) + 2*border
    total_h = image.height + abs(offset[1]) + 2*border
    shadow_image = Image.new('RGBA', (total_w, total_h), background_color)
    shadow = Image.new('RGBA', image.size, color=shadow_color)
    alpha = image.split()[3]
    shadow.putalpha(alpha)
    shadow_pos = (border + max(offset[0], 0), border + max(offset[1], 0))
    shadow_image.paste(shadow, shadow_pos, shadow)
    for _ in range(iterations):
        shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(blur))
    image_pos = (border - min(offset[0], 0), border - min(offset[1], 0))
    shadow_image.paste(image, image_pos, image)
    return shadow_image

def load_font(size):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", size)
        except Exception:
            return ImageFont.load_default()

def compute_block_height(lines, font, line_spacing):
    if not lines:
        return 0
    total = 0
    for line in lines:
        bbox = font.getbbox(line if line else "Ag")
        total += (bbox[3] - bbox[1]) + line_spacing
    return total - line_spacing

# === Smart Compose ===
def compose_image(
    cutout,
    bg_color_top,
    bg_color_bottom,
    top_lines,
    bottom_lines,
    canvas_size=None,  # require explicit size
    side_padding_pct=SIDE_PADDING_PCT,
    top_padding_pct=TOP_PADDING_PCT,
    bottom_padding_pct=BOTTOM_PADDING_PCT,
    prefer_full_width=PREFER_FULL_WIDTH_WHEN_POSSIBLE,
    center_text_blocks=CENTER_TEXT_BLOCKS,
    single_side_anchor_top=SINGLE_SIDE_ANCHOR_TOP,
    single_side_anchor_bottom=SINGLE_SIDE_ANCHOR_BOTTOM,
):
    if canvas_size is None or len(canvas_size) != 2:
        raise ValueError("compose_image requires an explicit canvas_size=(width, height)")

    cw, ch = canvas_size
    side_pad   = int(cw * side_padding_pct)
    top_pad    = int(ch * top_padding_pct)
    bottom_pad = int(ch * bottom_padding_pct)
    base_line_spacing = int(ch * LINE_SPACING_PCT)
    base_gap = max(int(ch * GAP_PCT), MIN_GAP_PX)

    gradient = create_gradient((cw, ch), bg_color_top, bg_color_bottom)
    result = Image.new("RGBA", (cw, ch))
    result.paste(gradient, (0, 0))

    safe_h = ch - top_pad - bottom_pad
    max_text_width = cw - 2*side_pad

    # Shadow overhead (constant for a given cutout size/params)
    _probe = Image.new('RGBA', (1, 1), (255, 255, 255, 255))
    _probe_s = add_shadow(_probe)
    shadow_extra_w = _probe_s.width - 1
    shadow_extra_h = _probe_s.height - 1

    has_top = bool(top_lines)
    has_bottom = bool(bottom_lines)
    both_blocks = has_top and has_bottom

    # --- Search for the best base font size (fs controls overall scale) ---
    start_fs = max(int(cw * 0.11), 24)
    best = None  # (score, fs, eff_bottom_fs)

    def longest_ratio(lines, font_obj):
        if not lines:
            return 0.0
        widths = [font_obj.getlength(ln if ln else " ") for ln in lines]
        longest = max(widths) if widths else 0
        return 0 if max_text_width <= 0 else (longest / max_text_width)

    for fs in range(start_fs, 11, -2):
        # Top font at fs
        top_fs = int(fs * TOP_FONT_MULT)
        font_top = load_font(top_fs)

        # Bottom hierarchy: smaller when both blocks exist
        bottom_rel = BOTTOM_HIERARCHY_SCALE if both_blocks else 1.0
        bottom_fs = max(10, int(fs * BOTTOM_FONT_MULT * bottom_rel))
        font_bottom = load_font(bottom_fs)

        line_spacing = base_line_spacing
        top_h = compute_block_height(top_lines, font_top, line_spacing)
        bottom_h = compute_block_height(bottom_lines, font_bottom, line_spacing)

        # [NEW] Enforce bottom height cap when both blocks exist
        if both_blocks and bottom_h > int(BOTTOM_MAX_SAFE_FRAC * safe_h):
            cap_px = int(BOTTOM_MAX_SAFE_FRAC * safe_h)
            while bottom_h > cap_px and bottom_fs > 10:
                bottom_fs = max(10, int(bottom_fs * 0.96))
                font_bottom = load_font(bottom_fs)
                bottom_h = compute_block_height(bottom_lines, font_bottom, line_spacing)

        # Width ratios after bottom cap adjustment
        top_ratio = longest_ratio(top_lines, font_top)
        bottom_ratio = longest_ratio(bottom_lines, font_bottom)

        # Hard reject if anything exceeds max width
        if top_ratio > TEXT_WIDTH_MAX_RATIO + 0.02 or bottom_ratio > TEXT_WIDTH_MAX_RATIO + 0.02:
            continue

        # Base gaps if the block exists
        gap_top = base_gap if has_top else 0
        gap_bottom = base_gap if has_bottom else 0

        target_w = cw - 2*side_pad
        w_limit_inner = max(1, target_w - shadow_extra_w)

        # First-pass available height (without quiet zones)
        avail_h_inner = safe_h - (top_h + gap_top + bottom_h + gap_bottom) - shadow_extra_h

        # Candidate scales without quiet zones
        scale_by_w = w_limit_inner / cutout.width
        scale_by_h = avail_h_inner / cutout.height if cutout.height > 0 else 0
        if scale_by_w <= 0 and scale_by_h <= 0:
            continue

        # Width-first allowed only if product remains prominent enough
        def can_fit_full_width(g_top, g_bottom):
            if not (prefer_full_width and scale_by_w > 0):
                return False
            product_total_h = cutout.height * scale_by_w + shadow_extra_h
            if product_total_h > (safe_h - (top_h + g_top + bottom_h + g_bottom)):
                return False
            frac = product_total_h / max(1, safe_h)
            return frac >= PRODUCT_MIN_FRAC_OF_SAFE

        # Initial scale choice (without quiet zones)
        if can_fit_full_width(gap_top, gap_bottom):
            scale = scale_by_w
        else:
            cands = []
            if scale_by_w > 0: cands.append(scale_by_w)
            if scale_by_h > 0: cands.append(scale_by_h)
            if not cands:
                continue
            scale = min(cands)

        if not np.isfinite(scale) or scale <= 0:
            continue

        # [NEW] Quiet zones depend on product height → refine gaps and scale
        product_total_h = cutout.height * scale + shadow_extra_h
        quiet_top = int(QUIET_ZONE_FRAC_OF_PRODUCT * product_total_h) if has_top else 0
        quiet_bottom = int(QUIET_ZONE_FRAC_OF_PRODUCT * product_total_h) if has_bottom else 0

        gap_top_eff = max(gap_top, quiet_top)
        gap_bottom_eff = max(gap_bottom, quiet_bottom)

        avail_h_inner2 = safe_h - (top_h + gap_top_eff + bottom_h + gap_bottom_eff) - shadow_extra_h
        scale_by_h2 = avail_h_inner2 / cutout.height if cutout.height > 0 else 0

        if can_fit_full_width(gap_top_eff, gap_bottom_eff):
            scale = scale_by_w
        else:
            cands = []
            if scale_by_w > 0: cands.append(scale_by_w)
            if scale_by_h2 > 0: cands.append(scale_by_h2)
            if not cands:
                continue
            scale = min(cands)

        if not np.isfinite(scale) or scale <= 0:
            continue

        # Final product metrics for scoring
        product_total_h = cutout.height * scale + shadow_extra_h
        product_frac = product_total_h / max(1, safe_h)
        prominence_penalty = max(0.0, PRODUCT_MIN_FRAC_OF_SAFE - product_frac)

        def band_score(ratio):
            if ratio == 0:
                return 1.0
            if ratio < TEXT_WIDTH_MIN_RATIO:
                return max(0.0, 1.0 - (TEXT_WIDTH_MIN_RATIO - ratio) * 2.0)
            if ratio > TEXT_WIDTH_MAX_RATIO:
                return max(0.0, 1.0 - (ratio - TEXT_WIDTH_MAX_RATIO) * 4.0)
            return 1.0

        width_score = min(band_score(top_ratio), band_score(bottom_ratio))
        fs_norm = (fs - 12) / (start_fs - 12 + 1e-6)

        # Heavier penalty if width << target (protects against tiny orphan lines)
        small_width_penalty = 0.0
        if (has_top and top_ratio <= ORPHAN_MIN_RATIO) or (has_bottom and bottom_ratio <= ORPHAN_MIN_RATIO):
            small_width_penalty = 0.1  # mild; we'll also do a micro-upscale later

        score = (scale * 1.0) + (fs_norm * 0.25) + (width_score * 0.15) - (prominence_penalty * 1.0) - small_width_penalty

        if (best is None) or (score > best[0]):
            best = (score, fs, bottom_fs)

    # Fallback if no size worked at all
    if best is None:
        fs = 12
        bottom_fs = max(10, int(fs * BOTTOM_FONT_MULT * (BOTTOM_HIERARCHY_SCALE if both_blocks else 1.0)))
    else:
        _, fs, bottom_fs = best

    # --- Final sizing + safety passes ---
    # Base fonts
    top_fs = int(fs * TOP_FONT_MULT)
    font_top = load_font(top_fs)

    font_bottom = load_font(bottom_fs)

    # Final line spacing and block heights
    line_spacing = base_line_spacing
    top_h = compute_block_height(top_lines, font_top, line_spacing)
    bottom_h = compute_block_height(bottom_lines, font_bottom, line_spacing)

    # Enforce bottom cap again at the very end (more accurate with final fs)
    if both_blocks and bottom_h > int(BOTTOM_MAX_SAFE_FRAC * safe_h):
        cap_px = int(BOTTOM_MAX_SAFE_FRAC * safe_h)
        while bottom_h > cap_px and bottom_fs > 10:
            bottom_fs = max(10, int(bottom_fs * 0.96))
            font_bottom = load_font(bottom_fs)
            bottom_h = compute_block_height(bottom_lines, font_bottom, line_spacing)

    # Width safety: shrink to avoid overflow
    def block_too_wide(lines, font_obj):
        return any(font_obj.getlength(ln or " ") > max_text_width for ln in lines)

    while top_fs > 12 and block_too_wide(top_lines, font_top):
        top_fs -= 1
        font_top = load_font(top_fs)
        top_h = compute_block_height(top_lines, font_top, line_spacing)

    while bottom_fs > 12 and block_too_wide(bottom_lines, font_bottom):
        bottom_fs -= 1
        font_bottom = load_font(bottom_fs)
        bottom_h = compute_block_height(bottom_lines, font_bottom, line_spacing)

    # [NEW] Orphan mitigation: gentle upscaling per block (without breaking constraints)
    def longest_ratio_with(font_obj, lines):
        if not lines:
            return 0.0
        widths = [font_obj.getlength(ln if ln else " ") for ln in lines]
        longest = max(widths) if widths else 0
        return longest / max_text_width if max_text_width > 0 else 0

    # We will re-evaluate scale after this pass; only slight changes allowed.
    if has_top:
        tries = 0
        while tries < ORPHAN_UPSCALE_STEPS:
            tr = longest_ratio_with(font_top, top_lines)
            if tr > ORPHAN_MIN_RATIO and tr >= ORPHAN_TARGET_RATIO:
                break
            # Try a tiny increase but never allow overflow beyond TEXT_WIDTH_MAX_RATIO
            cand_fs = int(round(top_fs * ORPHAN_UPSCALE_STEP))
            cand_font = load_font(cand_fs)
            if longest_ratio_with(cand_font, top_lines) > (TEXT_WIDTH_MAX_RATIO + 0.01):
                break
            top_fs = cand_fs
            font_top = cand_font
            top_h = compute_block_height(top_lines, font_top, line_spacing)
            tries += 1

    if has_bottom:
        tries = 0
        while tries < ORPHAN_UPSCALE_STEPS:
            br = longest_ratio_with(font_bottom, bottom_lines)
            if br > ORPHAN_MIN_RATIO and br >= ORPHAN_TARGET_RATIO:
                break
            cand_fs = int(round(bottom_fs * ORPHAN_UPSCALE_STEP))
            cand_font = load_font(cand_fs)
            # Keep bottom smaller than top when both blocks exist
            if both_blocks and cand_fs > int(top_fs * BOTTOM_HIERARCHY_SCALE):
                break
            if longest_ratio_with(cand_font, bottom_lines) > (TEXT_WIDTH_MAX_RATIO + 0.01):
                break
            bottom_fs = cand_fs
            font_bottom = cand_font
            bottom_h = compute_block_height(bottom_lines, font_bottom, line_spacing)
            # Re-enforce 25% safe cap
            if both_blocks and bottom_h > int(BOTTOM_MAX_SAFE_FRAC * safe_h):
                break
            tries += 1

    # Final gaps (base + quiet zones later)
    gap_top = base_gap if has_top else 0
    gap_bottom = base_gap if has_bottom else 0

    target_w = cw - 2*side_pad
    w_limit_inner = max(1, target_w - shadow_extra_w)

    # First-pass scale
    avail_h_inner = safe_h - (top_h + gap_top + bottom_h + gap_bottom) - shadow_extra_h
    scale_by_w = w_limit_inner / cutout.width
    scale_by_h = avail_h_inner / cutout.height if cutout.height > 0 else 0

    def can_fit_full_width(g_top, g_bottom, scale_w):
        if not (prefer_full_width and scale_w > 0):
            return False
        prod_h = cutout.height * scale_w + shadow_extra_h
        if prod_h > (safe_h - (top_h + g_top + bottom_h + g_bottom)):
            return False
        frac = prod_h / max(1, safe_h)
        return frac >= PRODUCT_MIN_FRAC_OF_SAFE

    if can_fit_full_width(gap_top, gap_bottom, scale_by_w):
        scale = scale_by_w
    else:
        cands = []
        if scale_by_w > 0: cands.append(scale_by_w)
        if scale_by_h > 0: cands.append(scale_by_h)
        scale = min(cands) if cands else 0.5

    if not np.isfinite(scale) or scale <= 0:
        scale = 0.5

    # Add quiet zones (refine gaps and scale once more)
    product_total_h = cutout.height * scale + shadow_extra_h
    quiet_top = int(QUIET_ZONE_FRAC_OF_PRODUCT * product_total_h) if has_top else 0
    quiet_bottom = int(QUIET_ZONE_FRAC_OF_PRODUCT * product_total_h) if has_bottom else 0
    gap_top_eff = max(gap_top, quiet_top)
    gap_bottom_eff = max(gap_bottom, quiet_bottom)

    avail_h_inner2 = safe_h - (top_h + gap_top_eff + bottom_h + gap_bottom_eff) - shadow_extra_h
    scale_by_h2 = avail_h_inner2 / cutout.height if cutout.height > 0 else 0

    if can_fit_full_width(gap_top_eff, gap_bottom_eff, scale_by_w):
        scale = scale_by_w
    else:
        cands = []
        if scale_by_w > 0: cands.append(scale_by_w)
        if scale_by_h2 > 0: cands.append(scale_by_h2)
        scale = min(cands) if cands else scale

    if not np.isfinite(scale) or scale <= 0:
        scale = 0.5

    # Resize + shadow
    new_w = max(1, int(cutout.width * scale))
    new_h = max(1, int(cutout.height * scale))
    cutout_resized = cutout.resize((new_w, new_h), Image.LANCZOS)
    cutout_with_shadow = add_shadow(cutout_resized)

    # --- Product position (anchors) ---
    product_h_total = cutout_with_shadow.height
    safe_top_y = top_pad
    safe_bottom_y = ch - bottom_pad

    if both_blocks:
        # [NEW] Optical vertical bias via anchor
        centerY = safe_top_y + int(safe_h * PRODUCT_CENTER_RATIO_BOTH)
        product_y = int(centerY - product_h_total / 2)
        # Ensure room for text + quiet gaps
        min_y = safe_top_y + top_h + gap_top_eff
        max_y = safe_bottom_y - (bottom_h + gap_bottom_eff) - product_h_total
        product_y = max(min_y, min(product_y, max_y))
    elif has_top and not has_bottom:
        centerY = safe_top_y + int(safe_h * float(single_side_anchor_top))
        product_y = int(centerY - product_h_total / 2)
        min_y = safe_top_y + top_h + gap_top_eff
        max_y = safe_bottom_y - product_h_total
        product_y = max(min_y, min(product_y, max_y))
    elif has_bottom and not has_top:
        centerY = safe_top_y + int(safe_h * float(single_side_anchor_bottom))
        product_y = int(centerY - product_h_total / 2)
        min_y = safe_top_y
        max_y = safe_bottom_y - (bottom_h + gap_bottom_eff) - product_h_total
        product_y = max(min_y, min(product_y, max_y))
    else:
        # No text → simple center
        centerY = safe_top_y + int(safe_h * 0.5)
        product_y = int(centerY - product_h_total / 2)

    product_x = (cw - cutout_with_shadow.width) // 2

    draw = ImageDraw.Draw(result)

    # ---- DRAW: Vertically center text blocks in their sections (respect quiet gaps) ----
    if center_text_blocks:
        # TOP SECTION
        if has_top:
            top_sec_top = safe_top_y
            top_sec_bottom = product_y - gap_top_eff
            top_sec_avail = max(0, top_sec_bottom - top_sec_top)
            ty = top_sec_top + max(0, (top_sec_avail - top_h) // 2)

            text_color_top = pick_text_color(bg_color_top)
            stroke_col_top = inverse_color(text_color_top)
            for line in top_lines:
                t = line if line else " "
                tx = int((cw - font_top.getlength(t)) // 2)
                draw.text((tx, ty), t, font=font_top, fill=text_color_top,
                          stroke_width=TEXT_STROKE_WIDTH, stroke_fill=stroke_col_top)
                bb = font_top.getbbox(t)
                ty += (bb[3] - bb[1]) + line_spacing

        # Product
        result.paste(cutout_with_shadow, (product_x, product_y), cutout_with_shadow)

        # BOTTOM SECTION
        if has_bottom:
            bottom_start = product_y + product_h_total + gap_bottom_eff
            bottom_sec_bottom = safe_bottom_y
            bottom_sec_avail = max(0, bottom_sec_bottom - bottom_start)
            by = bottom_start + max(0, (bottom_sec_avail - bottom_h) // 2)

            text_color_bottom = pick_text_color(bg_color_bottom)
            stroke_col_bottom = inverse_color(text_color_bottom)
            for line in bottom_lines:
                t = line if line else " "
                tx = int((cw - font_bottom.getlength(t)) // 2)
                draw.text((tx, by), t, font=font_bottom, fill=text_color_bottom,
                          stroke_width=TEXT_STROKE_WIDTH, stroke_fill=stroke_col_bottom)
                bb = font_bottom.getbbox(t)
                by += (bb[3] - bb[1]) + line_spacing
    else:
        # Fallback drawing path (legacy)
        y = safe_top_y
        if has_top:
            text_color_top = pick_text_color(bg_color_top)
            stroke_col_top = inverse_color(text_color_top)
            ty = y
            for line in top_lines:
                t = line if line else " "
                tx = int((cw - font_top.getlength(t)) // 2)
                draw.text((tx, ty), t, font=font_top, fill=text_color_top,
                          stroke_width=TEXT_STROKE_WIDTH, stroke_fill=stroke_col_top)
                bb = font_top.getbbox(t)
                ty += (bb[3] - bb[1]) + line_spacing
            y = ty + gap_top_eff

        result.paste(cutout_with_shadow, (product_x, product_y), cutout_with_shadow)
        y = product_y + product_h_total + gap_bottom_eff

        if has_bottom:
            text_color_bottom = pick_text_color(bg_color_bottom)
            stroke_col_bottom = inverse_color(text_color_bottom)
            for line in bottom_lines:
                t = line if line else " "
                tx = int((cw - font_bottom.getlength(t)) // 2)
                draw.text((tx, y), t, font=font_bottom, fill=text_color_bottom,
                          stroke_width=TEXT_STROKE_WIDTH, stroke_fill=stroke_col_bottom)
                bb = font_bottom.getbbox(t)
                y += (bb[3] - bb[1]) + line_spacing

    return result.convert("RGB")

# === Parse text (/// switches top→bottom; // line breaks) ===
def parse_text_blocks(text_input):
    if not text_input:
        return [], []
    parts = text_input.strip().split("///", 1)
    top_raw = parts[0] if len(parts) > 0 else ""
    bottom_raw = parts[1] if len(parts) > 1 else ""
    top_lines = [s.strip() for s in top_raw.split("//")] if top_raw != "" else []
    bottom_lines = [s.strip() for s in bottom_raw.split("//")] if bottom_raw != "" else []
    return top_lines, bottom_lines

# === Background Removal + Gradient ===
def remove_background_and_add_gradient(input_path, output_path, text_input, canvas_size,
                                       side_padding_pct=SIDE_PADDING_PCT,
                                       top_padding_pct=TOP_PADDING_PCT,
                                       bottom_padding_pct=BOTTOM_PADDING_PCT):
    with open(input_path, 'rb') as f:
        input_data = f.read()

    original_img = Image.open(input_path).convert("RGBA")
    resized_img = resize_if_needed(original_img)

    output_data = remove(input_data, session=RMBG_SESSION)

    cutout = Image.open(io.BytesIO(output_data)).convert("RGBA")
    cutout = resize_if_needed(cutout)

    top_lines, bottom_lines = parse_text_blocks(text_input)

    gradient_options = {
        "1": ("#2c3e50", "#bdc3c7"),
        "2": ("#0f2027", "#2c5364"),
        "3": ("#1a1a1a", "#272727"),
        "4": ("#E7B95F", "#FFF4D6"),
        "5": ("#E7B95F", "#4D217A"),
        "6": ("#E7B95F", "#1A1A1A")
    }

    while True:
        print("\nChoose gradient style:")
        print("1 = Soft Steel (#2C3E50 → #BDC3C7)")
        print("2 = Midnight Fade (#0F2027 → #2C5364)")
        print("3 = Subtle Charcoal (#1A1A1A → #272727)")
        print("4 = Brand to Cream (#E7B95F → #FFF4D6)")
        print("5 = Brand to Brand Purple (#E7B95F → #4D217A)")
        print("6 = Brand Gold to Charcoal (#E7B95F → #1A1A1A)")
        print("custom = Enter your own two hex colors")
        print("Leave blank to auto-detect from image background")

        choice = input("Your choice (1–6, custom, or blank): ").strip()

        if choice.lower() == "custom":
            top_hex = input("Top gradient color (e.g. #ff0000): ").strip()
            bottom_hex = input("Bottom gradient color (e.g. #000000): ").strip()
            try:
                bg_top = ImageColor.getrgb(top_hex)
                bg_bot = ImageColor.getrgb(bottom_hex)
            except ValueError:
                print("Invalid color. Reverting to auto background.")
                bg_top = get_dominant_edge_color(resized_img)
                bg_bot = tuple(max(0, c - 40) for c in bg_top)
        elif choice in gradient_options:
            th, bh = gradient_options[choice]
            bg_top = ImageColor.getrgb(th)
            bg_bot = ImageColor.getrgb(bh)
        else:
            bg_top = get_dominant_edge_color(resized_img)
            bg_bot = tuple(max(0, c - 40) for c in bg_top)

        # Compose & preview
        result = compose_image(
            cutout, bg_top, bg_bot, top_lines, bottom_lines,
            canvas_size=canvas_size,
            side_padding_pct=side_padding_pct,
            top_padding_pct=top_padding_pct,
            bottom_padding_pct=bottom_padding_pct,
            prefer_full_width=PREFER_FULL_WIDTH_WHEN_POSSIBLE,
            center_text_blocks=CENTER_TEXT_BLOCKS,
            single_side_anchor_top=SINGLE_SIDE_ANCHOR_TOP,
            single_side_anchor_bottom=SINGLE_SIDE_ANCHOR_BOTTOM,
        )
        result.show()
        confirm = input("Use this gradient? (y/n): ").strip().lower()
        if confirm == 'y':
            result.save(output_path, format="JPEG", quality=95)
            print(f"✅ Saved: {output_path}")
            break

# === Main ===
def main():
    latest = get_latest_image(input_folder)
    if not latest:
        print("No image found in folder.")
        return
    input_path = os.path.join(input_folder, latest)
    base = os.path.splitext(latest)[0]

    print("\nSelect platform output:")
    for key, spec in PLATFORM_SPECS.items():
        print(f"- {key}: {spec['desc']}")
    platform_choice = input("Enter platform (facebook / instagram / email): ").strip().lower()
    if platform_choice not in PLATFORM_SPECS:
        print("Invalid platform. Defaulting to 'instagram'.")
        platform_choice = "instagram"
    canvas_size = PLATFORM_SPECS[platform_choice]["size"]

    user_text = input("Enter text (use // for line breaks; use /// to move text below): ").strip()

    w, h = canvas_size
    output_name = f"{base}_{platform_choice}_{w}x{h}.jpg"
    output_path = os.path.join(finished_folder, output_name)

    remove_background_and_add_gradient(
        input_path, output_path, user_text, canvas_size,
        side_padding_pct=SIDE_PADDING_PCT,
        top_padding_pct=TOP_PADDING_PCT,
        bottom_padding_pct=BOTTOM_PADDING_PCT
    )

if __name__ == "__main__":
    main()