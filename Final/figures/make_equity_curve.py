"""Generates Fig. 9: cumulative test-period return, Hard RA-RL vs Soft RA-RL,
same seed (seed 1), same training data — the seed on which Hard RA-RL
collapsed (-13.9% final return) while Soft RA-RL stayed positive (+17.7%).

Source: cropped from the original seed_1_metrics_over_time.png plots
(results/seed_1_metrics_over_time.png, results/moe_run_3590623/seed_1_metrics_over_time.png),
which were rendered server-side during the Run B (job 3590623) evaluation.
Only the top "Cumulative Return" panel is used; the original title (which used
the pre-rename "Hard Routing"/"Soft MoE" naming and had an unreadable date-tick
band) is cropped out and replaced with clean RA-RL-consistent labels.
"""
from PIL import Image, ImageDraw, ImageFont

CROP_BOX = (0, 75, 2084, 640)  # excludes original title + illegible date-tick band

hard = Image.open("../results/seed_1_metrics_over_time.png").crop(CROP_BOX)
soft = Image.open("../results/moe_run_3590623/seed_1_metrics_over_time.png").crop(CROP_BOX)

w, h = hard.size
label_h = 50
gap = 12
canvas = Image.new("RGB", (w, label_h * 2 + h * 2 + gap), "white")
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
except Exception:
    font = ImageFont.load_default()

draw.text((10, 10), "Hard RA-RL — Seed 1 (Return -13.9%, Sharpe -0.10)", fill="#1a1a1a", font=font)
canvas.paste(hard, (0, label_h))
draw.text((10, label_h + h + gap + 10), "Soft RA-RL — Seed 1 (Return +17.7%, Sharpe +0.45)",
          fill="#1a1a1a", font=font)
canvas.paste(soft, (0, label_h * 2 + h + gap))

canvas.save("equity_curve_seed1.png", dpi=(300, 300))
print("Saved equity_curve_seed1.png")
