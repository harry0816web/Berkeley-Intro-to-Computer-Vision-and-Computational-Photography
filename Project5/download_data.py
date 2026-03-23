from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np


LEGO_URL = "https://cal-cs180.github.io/fa26/hw/proj4/assets/lego_200x200.npz"
CAT_URL = "https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg"
PERSONAL_IMAGE_URL = (
    "https://raw.githubusercontent.com/harry0816web/"
    "Berkeley-Intro-to-Computer-Vision-and-Computational-Photography/"
    "main/Assignment2/person.png"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already present: {destination}")
        return
    print(f"Downloading {url}")
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)
    print(f"Saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-lego", action="store_true")
    parser.add_argument("--skip-cat", action="store_true")
    args = parser.parse_args()

    if not args.skip_lego:
        lego_path = args.data_dir / "lego_200x200.npz"
        download(LEGO_URL, lego_path)
        with np.load(lego_path) as data:
            required = {"images_train", "c2ws_train", "images_val", "c2ws_val", "c2ws_test", "focal"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Downloaded dataset is missing keys: {sorted(missing)}")
            print("Lego dataset shapes:", {key: data[key].shape for key in required})

    if not args.skip_cat:
        download(CAT_URL, args.data_dir / "part1" / "official_cat.jpg")
        download(PERSONAL_IMAGE_URL, args.data_dir / "part1" / "personal_photo.png")


if __name__ == "__main__":
    main()
