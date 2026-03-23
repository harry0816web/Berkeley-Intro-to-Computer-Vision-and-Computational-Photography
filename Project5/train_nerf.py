from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import trange

from nerf_project.models import NeRF
from nerf_project.rays import RaysData, get_rays_for_camera, make_intrinsics
from nerf_project.rendering import render_rays
from nerf_project.utils import choose_device, mse_to_psnr, seed_everything, write_json


def load_dataset(path: Path, device: torch.device) -> dict[str, torch.Tensor | float]:
    raw = np.load(path)
    return {
        "images_train": torch.from_numpy(raw["images_train"].astype(np.float32) / 255.0).to(device),
        "c2ws_train": torch.from_numpy(raw["c2ws_train"].astype(np.float32)).to(device),
        "images_val": torch.from_numpy(raw["images_val"].astype(np.float32) / 255.0).to(device),
        "c2ws_val": torch.from_numpy(raw["c2ws_val"].astype(np.float32)).to(device),
        "c2ws_test": torch.from_numpy(raw["c2ws_test"].astype(np.float32)).to(device),
        "focal": float(raw["focal"]),
    }


@torch.inference_mode()
def render_image(
    model: NeRF,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    height: int,
    width: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    origins, directions = get_rays_for_camera(intrinsics, c2w, height, width)
    rendered_chunks: list[torch.Tensor] = []
    for start in range(0, origins.shape[0], args.render_ray_chunk):
        rendered_chunks.append(
            render_rays(
                model,
                origins[start : start + args.render_ray_chunk],
                directions[start : start + args.render_ray_chunk],
                near=args.near,
                far=args.far,
                num_samples=args.samples,
                perturb=False,
                point_chunk_size=args.point_chunk,
            )
        )
    return torch.cat(rendered_chunks).reshape(height, width, 3)


def save_validation_grid(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    output: Path,
    step: int,
) -> None:
    count = len(predictions)
    fig, axes = plt.subplots(2, count, figsize=(3 * count, 6), squeeze=False)
    for index in range(count):
        axes[0, index].imshow(np.clip(targets[index], 0.0, 1.0))
        axes[0, index].set_title(f"Target {index}")
        axes[1, index].imshow(np.clip(predictions[index], 0.0, 1.0))
        axes[1, index].set_title(f"Predicted {index}")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    fig.suptitle(f"Validation views at iteration {step}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def evaluate(
    model: NeRF,
    images: torch.Tensor,
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
    args: argparse.Namespace,
    step: int,
) -> tuple[float, list[np.ndarray]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    mses: list[float] = []
    count = min(args.val_images, images.shape[0])
    height, width = images.shape[1:3]
    for index in range(count):
        prediction = render_image(model, intrinsics, c2ws[index], height, width, args)
        mse = torch.mean((prediction - images[index]) ** 2).item()
        mses.append(mse)
        predictions.append(prediction.cpu().numpy())
        targets.append(images[index].cpu().numpy())
    mean_psnr = mse_to_psnr(float(np.mean(mses)))
    save_validation_grid(
        predictions,
        targets,
        args.output_dir / "validation" / f"step_{step:06d}.png",
        step,
    )
    model.train()
    return mean_psnr, predictions


def save_curves(train_history: list[dict[str, float]], val_history: list[dict[str, float]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [entry["iteration"] for entry in train_history],
        [entry["psnr"] for entry in train_history],
    )
    axes[0].set_title("Training PSNR")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].plot(
        [entry["iteration"] for entry in val_history],
        [entry["psnr"] for entry in val_history],
        marker="o",
    )
    axes[1].set_title("Validation PSNR (6 images)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("PSNR (dB)")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


@torch.inference_mode()
def render_video(
    model: NeRF,
    c2ws_test: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    args: argparse.Namespace,
) -> None:
    frames: list[np.ndarray] = []
    progress = trange(c2ws_test.shape[0], desc="Novel-view video")
    for index in progress:
        frame = render_image(model, intrinsics, c2ws_test[index], height, width, args)
        frames.append((frame.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.mimsave(args.output_dir / "lego_novel_views.mp4", frames, fps=24, quality=8)
    imageio.mimsave(args.output_dir / "lego_novel_views.gif", frames, fps=12, loop=0)


def save_checkpoint(
    path: Path,
    model: NeRF,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_history: list[dict[str, float]],
    val_history: list[dict[str, float]],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_history": train_history,
            "val_history": val_history,
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.layers,
                "position_frequencies": args.position_frequencies,
                "direction_frequencies": args.direction_frequencies,
                "skip_layer": args.skip_layer,
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    dataset = load_dataset(args.data, device)
    images_train = dataset["images_train"]
    c2ws_train = dataset["c2ws_train"]
    images_val = dataset["images_val"]
    c2ws_val = dataset["c2ws_val"]
    c2ws_test = dataset["c2ws_test"]
    assert isinstance(images_train, torch.Tensor)
    assert isinstance(c2ws_train, torch.Tensor)
    assert isinstance(images_val, torch.Tensor)
    assert isinstance(c2ws_val, torch.Tensor)
    assert isinstance(c2ws_test, torch.Tensor)

    height, width = images_train.shape[1:3]
    intrinsics = make_intrinsics(height, width, float(dataset["focal"]), device=device)
    ray_dataset = RaysData(images_train, intrinsics, c2ws_train)
    model = NeRF(
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        position_frequencies=args.position_frequencies,
        direction_frequencies=args.direction_frequencies,
        skip_layer=args.skip_layer,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 1
    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        train_history = checkpoint.get("train_history", [])
        val_history = checkpoint.get("val_history", [])
        print(f"Resumed at iteration {start_step}")

    if start_step == 1 and not args.skip_initial_eval:
        initial_psnr, _ = evaluate(model, images_val, c2ws_val, intrinsics, args, 0)
        val_history.append({"iteration": 0, "psnr": initial_psnr})
        print(f"Initial validation PSNR: {initial_psnr:.2f} dB")

    progress = trange(start_step, args.iterations + 1, desc="NeRF training")
    for step in progress:
        origins, directions, target_colors = ray_dataset.sample_rays(args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for start in range(0, args.batch_size, args.ray_microbatch):
            end = min(start + args.ray_microbatch, args.batch_size)
            fraction = (end - start) / args.batch_size
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predicted_colors = render_rays(
                    model,
                    origins[start:end],
                    directions[start:end],
                    near=args.near,
                    far=args.far,
                    num_samples=args.samples,
                    perturb=True,
                    point_chunk_size=args.point_chunk,
                )
                micro_loss = loss_function(predicted_colors, target_colors[start:end])
                weighted_loss = micro_loss * fraction
            scaler.scale(weighted_loss).backward()
            total_loss += float(micro_loss.detach()) * fraction
        scaler.step(optimizer)
        scaler.update()

        train_psnr = mse_to_psnr(total_loss)
        train_history.append({"iteration": step, "mse": total_loss, "psnr": train_psnr})
        progress.set_postfix(train_psnr=f"{train_psnr:.2f}")

        if step % args.eval_every == 0 or step == args.iterations:
            val_psnr, _ = evaluate(model, images_val, c2ws_val, intrinsics, args, step)
            val_history.append({"iteration": step, "psnr": val_psnr})
            progress.set_postfix(train_psnr=f"{train_psnr:.2f}", val_psnr=f"{val_psnr:.2f}")
            save_curves(train_history, val_history, args.output_dir / "psnr_curves.png")
            write_json(
                args.output_dir / "metrics.json",
                {
                    "train": train_history,
                    "validation": val_history,
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
            )

        if step % args.checkpoint_every == 0 or step == args.iterations:
            save_checkpoint(
                args.output_dir / "checkpoint_latest.pt",
                model,
                optimizer,
                step,
                train_history,
                val_history,
                args,
            )

    if not args.skip_video:
        render_video(model, c2ws_test, intrinsics, height, width, args)
    print(f"Final validation PSNR: {val_history[-1]['psnr']:.2f} dB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NeRF on the CS180 Lego dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/lego_200x200.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part2"))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--ray-microbatch", type=int, default=1024)
    parser.add_argument("--samples", type=int, choices=[32, 64], default=64)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--position-frequencies", type=int, default=10)
    parser.add_argument("--direction-frequencies", type=int, default=4)
    parser.add_argument("--skip-layer", type=int, default=4)
    parser.add_argument("--point-chunk", type=int, default=65536)
    parser.add_argument("--render-ray-chunk", type=int, default=2048)
    parser.add_argument("--val-images", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
