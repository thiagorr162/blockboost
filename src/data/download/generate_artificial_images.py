import json
import pathlib

import seaborn as sns

for noise_scale in [8, 32, 64]:
    for balanced in [True, False]:
        if balanced:
            patterns = [
                {
                    "start": (0, 0),
                    "end": (72, 178),
                    "n_colors": 12,
                    "colors": [[int(255 * v) for v in c] for c in sns.color_palette("Paired", 12)],
                    "noise_scale": noise_scale,
                },
                {
                    "start": (72, 0),
                    "end": (144, 178),
                    "n_colors": 12,
                    "colors": [[int(255 * v) for v in c] for c in sns.color_palette("flare", 12)],
                    "noise_scale": noise_scale,
                },
                {
                    "start": (144, 0),
                    "end": (218, 178),
                    "n_colors": 12,
                    "colors": [[int(255 * v) for v in c] for c in sns.color_palette("mako", 12)],
                    "noise_scale": noise_scale,
                },
            ]
            json.dump(
                patterns,
                open(pathlib.Path("research/data/raw/artificial_images") / f"balanced-{noise_scale}.json", "w"),
                indent=2,
            )
        else:
            patterns = [
                {
                    "start": (30, 30),
                    "end": (40, 40),
                    "n_colors": 6,
                    "colors": [[int(255 * v) for v in c] for c in sns.color_palette("Paired", 6)],
                    "noise_scale": noise_scale,
                },
                {
                    "start": (40, 40),
                    "end": (100, 90),
                    "n_colors": 12,
                    "colors": [[int(255 * v) for v in c] for c in sns.color_palette("flare", 12)],
                    "noise_scale": noise_scale,
                },
                {
                    "start": (100, 90),
                    "end": (218, 178),
                    "n_colors": 24,
                    "colors": [[int(255 * v) for v in c] for c in sns.color_palette("mako", 24)],
                    "noise_scale": noise_scale,
                },
            ]
            json.dump(
                patterns,
                open(pathlib.Path("research/data/raw/artificial_images") / f"unbalanced-{noise_scale}.json", "w"),
                indent=2,
            )
