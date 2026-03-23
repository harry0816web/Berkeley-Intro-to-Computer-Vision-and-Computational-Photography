from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_psnr(path: Path, section: str = "history") -> float:
    return float(load_json(path)[section][-1]["psnr"])


def main() -> None:
    root = Path("outputs")
    cat_metrics = root / "part1" / "cat" / "metrics.json"
    personal_metrics = root / "part1" / "personal" / "metrics.json"
    nerf_metrics = root / "part2" / "metrics.json"
    required = [cat_metrics, personal_metrics, nerf_metrics]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run all training cells before generating the report. Missing: {missing}")

    cat_psnr = final_psnr(cat_metrics)
    personal_psnr = final_psnr(personal_metrics)
    nerf = load_json(nerf_metrics)
    validation_psnr = float(nerf["validation"][-1]["psnr"])
    arguments = nerf["arguments"]
    report = f"""# CS180 Project 5: Neural Radiance Fields

## Part 1: Fit a Neural Field to a 2D Image

I represented each image as a continuous function from normalized pixel coordinates to RGB. The input coordinates were expanded with sinusoidal positional encoding, including the original coordinates and frequencies from 2^0 through 2^9. The baseline MLP uses three 256-channel hidden layers with ReLU activations and a final sigmoid layer. I optimized mean squared error with Adam at learning rate 1e-2, using 10,000 randomly sampled pixels per iteration.

The official cat image reached **{cat_psnr:.2f} dB PSNR**. The image from my collection reached **{personal_psnr:.2f} dB PSNR**.

![Cat training progression](part1/cat/training_progression.png)

![Cat PSNR curve](part1/cat/psnr_curve.png)

![Personal image training progression](part1/personal/training_progression.png)

![Personal image PSNR curve](part1/personal/psnr_curve.png)

### Hyperparameter tuning

I varied two hyperparameters in a small factorial experiment: the highest positional-encoding frequency (L=10 versus L=4) and network depth (three versus two hidden layers). The comparison keeps the optimizer, batch size, random seed, and image fixed.

![Hyperparameter comparison](part1/cat/hyperparameter_sweep/comparison.png)

## Part 2: Fit a Neural Radiance Field from Multi-view Images

### Ray construction and sampling

For every sampled pixel center `(u+0.5, v+0.5)`, I inverted the camera intrinsic matrix to obtain a camera-space point at unit depth, transformed it into world space with the camera-to-world matrix, and normalized the vector from the camera center to that point. During training, I sampled points in {arguments['samples']} stratified intervals between near={arguments['near']} and far={arguments['far']}.

![Cameras, rays, and samples](part2/rays_and_samples.png)

### NeRF architecture and volume rendering

The NeRF uses an eight-layer, 256-channel position MLP. Position encoding uses L=10, direction encoding uses L=4, and the encoded position is concatenated back into the network at layer four. Density is constrained nonnegative with ReLU; RGB is conditioned on direction and constrained to [0,1] with sigmoid. I used the discrete volume-rendering equation with alpha compositing and exclusive accumulated transmittance.

Training used Adam with learning rate {arguments['learning_rate']}, an effective batch of {arguments['batch_size']} rays, and ray microbatches of {arguments['ray_microbatch']} to control GPU memory. Validation PSNR over six held-out views reached **{validation_psnr:.2f} dB**.

![Training and validation PSNR](part2/psnr_curves.png)

The `validation/` directory contains predicted images across training iterations. The final spherical novel-view rendering is available as [MP4](part2/lego_novel_views.mp4) and [GIF](part2/lego_novel_views.gif).
"""
    output = root / "REPORT.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
