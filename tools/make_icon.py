"""
Generate the 2D FOMO Meter icon and desktop assets with Pillow.

The mark is intentionally simple: radar rings, a cyan sweep, an amber contact,
and a small speech bubble. It reads at 16 px and at 1024 px.

Run:
    python tools/make_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

SIZE = 1024
BG = (7, 13, 20, 255)
BG_SOFT = (12, 23, 32, 255)
CYAN = (70, 217, 207, 255)
CYAN_DIM = (70, 217, 207, 80)
AMBER = (245, 184, 75, 255)
WHITE = (233, 241, 247, 255)
EDGE = (43, 73, 88, 255)


def _ring(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, color, width: int) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def build_master() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Rounded square body. Opaque base, so translucent icon details are
    # composited instead of punching holes through the background.
    draw.rounded_rectangle((36, 36, SIZE - 36, SIZE - 36), radius=230, fill=BG)
    gradient = Image.new("RGBA", image.size, (0, 0, 0, 0))
    gradient_draw = ImageDraw.Draw(gradient)
    for i in range(180):
        inset = 50 + i
        alpha = int(8 * (1 - i / 180))
        gradient_draw.rounded_rectangle(
            (inset, inset, SIZE - inset, SIZE - inset),
            radius=max(30, 220 - i),
            outline=(30, 70, 82, alpha),
            width=1,
        )
    image = Image.alpha_composite(image, gradient).convert("RGBA")

    # Radar rings around the centre, drawn on a translucent overlay.
    center = (500, 490)
    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    _ring(glow_draw, center, 292, (70, 217, 207, 54), 18)
    _ring(glow_draw, center, 196, (70, 217, 207, 40), 12)
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    image = Image.alpha_composite(image, glow).convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    _ring(od, center, 292, (70, 217, 207, 95), 4)
    _ring(od, center, 196, (70, 217, 207, 150), 4)
    _ring(od, center, 104, (70, 217, 207, 205), 4)

    # Crosshair.
    cx, cy = center
    od.line((cx - 300, cy, cx + 300, cy), fill=(70, 217, 207, 48), width=2)
    od.line((cx, cy - 300, cx, cy + 300), fill=(70, 217, 207, 48), width=2)

    # Radar sweep wedge and leading edge.
    sweep_bbox = (cx - 292, cy - 292, cx + 292, cy + 292)
    od.pieslice(sweep_bbox, start=-72, end=-24, fill=(70, 217, 207, 38))
    edge_angle = math.radians(-24)
    ex = cx + math.cos(edge_angle) * 292
    ey = cy + math.sin(edge_angle) * 292
    od.line((cx, cy, ex, ey), fill=(70, 217, 207, 235), width=7)

    # Contact blips.
    od.ellipse((cx + 146, cy - 116, cx + 190, cy - 72), fill=AMBER)
    od.ellipse((cx + 155, cy - 107, cx + 181, cy - 81), fill=(255, 223, 156, 255))
    od.ellipse((cx - 104, cy + 132, cx - 74, cy + 162), fill=(70, 217, 207, 235))

    # Small speech bubble — the "sentiment" part of the mark.
    bubble = (606, 600, 838, 776)
    od.rounded_rectangle(bubble, radius=52, fill=BG_SOFT, outline=(70, 217, 207, 230), width=7)
    od.polygon([(654, 770), (642, 842), (730, 774)], fill=BG_SOFT)
    od.line((656, 774, 720, 774), fill=(70, 217, 207, 230), width=7)
    od.line((660, 680, 780, 680), fill=(233, 241, 247, 235), width=9)
    od.line((660, 718, 742, 718), fill=(70, 217, 207, 235), width=9)
    od.ellipse((688, 744, 702, 758), fill=AMBER)
    od.ellipse((716, 744, 730, 758), fill=(70, 217, 207, 235))
    od.ellipse((744, 744, 758, 758), fill=(126, 158, 181, 200))

    image = Image.alpha_composite(image, overlay).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((36, 36, SIZE - 36, SIZE - 36), radius=230, outline=EDGE, width=4)
    return image


def main() -> None:
    master = build_master()
    master.save(ASSETS / "fomo_meter_1024.png")
    master.resize((512, 512), Image.Resampling.LANCZOS).save(ASSETS / "fomo_meter_512.png")
    master.resize((256, 256), Image.Resampling.LANCZOS).save(ASSETS / "fomo_meter_256.png")
    master.resize((64, 64), Image.Resampling.LANCZOS).save(ASSETS / "fomo_meter_64.png")

    ico_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    master.resize((256, 256), Image.Resampling.LANCZOS).save(
        ASSETS / "fomo_meter.ico",
        format="ICO",
        sizes=ico_sizes,
    )
    print(f"Created {ASSETS / 'fomo_meter.ico'}")
    print(f"Created {ASSETS / 'fomo_meter_1024.png'}")


if __name__ == "__main__":
    main()
