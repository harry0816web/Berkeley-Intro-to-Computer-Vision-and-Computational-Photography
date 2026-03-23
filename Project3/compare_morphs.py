"""Compare the original pair morph with a morph toward the IMM mean face."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from compute_imm_mean import BORDER_POINTS, normalized_to_pixels, parse_asf
from morph import morph, read_image


ROOT = Path(__file__).resolve().parent
FRACTIONS = np.linspace(0, 1, 46)
SELECTED_INDICES = [0, 11, 23, 34, 45]


def gif_durations(frame_count: int, fps: float = 30.0) -> list[int]:
    boundaries = np.round(np.arange(frame_count + 1) * (1000 / fps) / 10).astype(int) * 10
    return np.maximum(10, np.diff(boundaries)).tolist()


def save_gif(frames: list[np.ndarray], output_path: Path) -> None:
    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=gif_durations(len(pil_frames)),
        loop=0,
        disposal=2,
    )


def render_dataset_mean_morph() -> list[np.ndarray]:
    dataset_dir = ROOT / "IMM-Face"
    image = read_image(dataset_dir / "01-1m.jpg")
    mean_image = read_image(ROOT / "results/imm_mean/imm_mean_face.jpg")
    geometry = np.load(ROOT / "results/imm_mean/imm_mean_geometry.npz")
    triangles = geometry["triangles"]
    mean_points = geometry["mean_points"]
    height, width = image.shape[:2]
    source_normalized = np.concatenate(
        [parse_asf(dataset_dir / "01-1m.asf"), BORDER_POINTS], axis=0
    )
    source_points = normalized_to_pixels(source_normalized, height, width)

    frames: list[np.ndarray] = []
    for fraction in FRACTIONS:
        frame, _, _ = morph(
            image,
            mean_image,
            source_points,
            mean_points,
            triangles,
            warp_fraction=float(fraction),
            dissolve_fraction=float(fraction),
        )
        frames.append(np.round(frame * 255).astype(np.uint8))
    return frames


def render_original_morph() -> list[np.ndarray]:
    frame_dir = ROOT / "results/morph_frames"
    return [
        np.asarray(Image.open(frame_dir / f"frame_{index:02d}.png").convert("RGB"))
        for index in range(46)
    ]


def save_contact_sheet(
    original_frames: list[np.ndarray],
    dataset_frames: list[np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17, 9))
    titles = [f"t={FRACTIONS[index]:.2f}" for index in SELECTED_INDICES]
    for column, (index, title) in enumerate(zip(SELECTED_INDICES, titles)):
        axes[0, column].imshow(original_frames[index])
        axes[0, column].set_title(title)
        axes[1, column].imshow(dataset_frames[index])
        axes[1, column].set_title(title)
    axes[0, 0].set_ylabel("Eleven → George", fontsize=12)
    axes[1, 0].set_ylabel("IMM sample → IMM mean", fontsize=12)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Morph comparison: individual target vs. population mean", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_side_by_side_gif(
    original_frames: list[np.ndarray],
    dataset_frames: list[np.ndarray],
    output_path: Path,
) -> None:
    combined: list[np.ndarray] = []
    canvas_width = max(original_frames[0].shape[1], dataset_frames[0].shape[1])
    canvas_height = original_frames[0].shape[0] + dataset_frames[0].shape[0]
    for original, dataset in zip(original_frames, dataset_frames):
        canvas = Image.new("RGB", (canvas_width, canvas_height), (30, 30, 30))
        top = Image.fromarray(original)
        bottom = Image.fromarray(dataset)
        canvas.paste(top, ((canvas_width - top.width) // 2, 0))
        canvas.paste(bottom, ((canvas_width - bottom.width) // 2, top.height))
        combined.append(np.asarray(canvas))
    save_gif(combined, output_path)


def main() -> None:
    original_frames = render_original_morph()
    dataset_frames = render_dataset_mean_morph()
    output_dir = ROOT / "results/morph_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_contact_sheet(
        original_frames,
        dataset_frames,
        output_dir / "individual_vs_population_mean.png",
    )
    save_gif(dataset_frames, output_dir / "imm_sample_to_mean.gif")
    save_side_by_side_gif(
        original_frames,
        dataset_frames,
        output_dir / "eleven_to_george_vs_sample_to_mean.gif",
    )
    print(f"Saved comparison results to {output_dir}")


if __name__ == "__main__":
    main()
