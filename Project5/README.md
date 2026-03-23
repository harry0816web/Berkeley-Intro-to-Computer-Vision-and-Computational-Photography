# CS180 Project 5: Neural Fields and NeRF

This directory contains a complete implementation of both required parts:

- Part 1: a positional-encoded 2D neural field, random pixel sampling, PSNR tracking, two-image experiments, and a two-factor hyperparameter sweep.
- Part 2: camera transforms, pixel-to-ray conversion, stratified ray sampling, a view-conditioned NeRF, differentiable volume rendering, validation rendering, checkpoints, and a spherical novel-view video.

## Recommended workflow

Use VS Code to edit and a hosted Colab GPU to train. Open `project5_colab.ipynb`, select **Kernel > Colab > Auto Connect**, then run cells in order. The notebook downloads the official data, runs tests, creates all required visualizations, performs both Part 1 experiments, trains Part 2, and generates `outputs/REPORT.md`.

The official Google extension is `google.colab`; it requires a recent VS Code release and also installs the Jupyter dependency.

## Local Mac setup

```bash
cd Project5
UV_CACHE_DIR=/private/tmp/cs180-project5-uv uv sync --extra dev --python 3.12
uv run pytest
uv run python download_data.py
```

PyTorch automatically selects CUDA, Apple MPS, or CPU. For an M1 Pro with 16 GB unified memory, use a smaller local NeRF command:

```bash
uv run python train_nerf.py --batch-size 2048 --ray-microbatch 256 --samples 32 --skip-video
```

## Full Colab commands

```bash
python download_data.py
pytest -q
python visualize_rays.py
python train_2d.py --image data/part1/official_cat.jpg --output-dir outputs/part1/cat --sweep
python train_2d.py --image data/part1/personal_photo.png --output-dir outputs/part1/personal
python train_nerf.py
python generate_report.py
```

`train_nerf.py` saves `checkpoint_latest.pt` periodically. The target iteration must be larger than the checkpoint step when resuming. For example, continue a 1000-step run to 3000 steps with:

```bash
python train_nerf.py --iterations 3000 --resume outputs/part2/checkpoint_latest.pt
```

## Expected outputs

```text
outputs/
  part1/cat/
    psnr_curve.png
    training_progression.png
    hyperparameter_sweep/comparison.png
  part1/personal/
    psnr_curve.png
    training_progression.png
  part2/
    rays_and_samples.png
    psnr_curves.png
    validation/step_*.png
    lego_novel_views.mp4
    lego_novel_views.gif
    checkpoint_latest.pt
  REPORT.md
```
