from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import trange

from nerf_project.models import NeuralField2D
from nerf_project.utils import choose_device, mse_to_psnr, seed_everything, write_json


@dataclass(frozen=True)
class FieldConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    num_frequencies: int = 10
    learning_rate: float = 1e-2


def load_image(path: Path, max_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)


def make_coordinates(height: int, width: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([x / width, y / height], dim=-1).reshape(-1, 2)


@torch.inference_mode()
def render_full(
    model: nn.Module,
    coordinates: torch.Tensor,
    height: int,
    width: int,
    chunk_size: int = 65536,
) -> torch.Tensor:
    predictions = []
    for start in range(0, coordinates.shape[0], chunk_size):
        predictions.append(model(coordinates[start : start + chunk_size]))
    return torch.cat(predictions).reshape(height, width, 3)


def save_psnr_curve(history: list[dict[str, float]], output: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot([entry["iteration"] for entry in history], [entry["psnr"] for entry in history])
    plt.xlabel("Iteration")
    plt.ylabel("Training PSNR (dB)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_progression(snapshots: list[tuple[int, np.ndarray]], target: np.ndarray, output: Path) -> None:
    images = [("Target", target)] + [(f"Iter {step}", image) for step, image in snapshots]
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis, (label, image) in zip(axes.flat, images):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(label)
        axis.axis("off")
    for axis in axes.flat[len(images) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def train_field(
    image: torch.Tensor,
    config: FieldConfig,
    iterations: int,
    batch_size: int,
    eval_every: int,
    device: torch.device,
    seed: int,
    capture_snapshots: bool = True,
) -> tuple[NeuralField2D, list[dict[str, float]], list[tuple[int, np.ndarray]]]:
    seed_everything(seed)
    height, width = image.shape[:2]
    target = image.to(device)
    flat_target = target.reshape(-1, 3)
    coordinates = make_coordinates(height, width, device)
    model = NeuralField2D(
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        num_frequencies=config.num_frequencies,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    history: list[dict[str, float]] = []
    snapshots: list[tuple[int, np.ndarray]] = []
    snapshot_steps = {0, iterations // 10, iterations // 4, iterations // 2, iterations}

    def evaluate(step: int) -> None:
        prediction = render_full(model, coordinates, height, width)
        mse = torch.mean((prediction - target) ** 2).item()
        history.append({"iteration": step, "mse": mse, "psnr": mse_to_psnr(mse)})
        if capture_snapshots and step in snapshot_steps:
            snapshots.append((step, prediction.detach().cpu().numpy()))

    evaluate(0)
    progress = trange(1, iterations + 1, desc="2D field")
    for step in progress:
        indices = torch.randint(flat_target.shape[0], (batch_size,), device=device)
        predicted = model(coordinates[indices])
        loss = loss_function(predicted, flat_target[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == iterations:
            evaluate(step)
            progress.set_postfix(psnr=f"{history[-1]['psnr']:.2f}")
    return model, history, snapshots


def run_single(args: argparse.Namespace, image: torch.Tensor, device: torch.device) -> None:
    config = FieldConfig(args.hidden_dim, args.layers, args.frequencies, args.learning_rate)
    model, history, snapshots = train_field(
        image,
        config,
        args.iterations,
        args.batch_size,
        args.eval_every,
        device,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(config)}, args.output_dir / "model.pt")
    write_json(args.output_dir / "metrics.json", {"config": asdict(config), "history": history})
    save_psnr_curve(history, args.output_dir / "psnr_curve.png", "2D Neural Field Training")
    save_progression(snapshots, image.numpy(), args.output_dir / "training_progression.png")
    print(f"Final PSNR: {history[-1]['psnr']:.2f} dB")


def run_sweep(args: argparse.Namespace, image: torch.Tensor, device: torch.device) -> None:
    baseline = FieldConfig(args.hidden_dim, args.layers, args.frequencies, args.learning_rate)
    configurations = {
        "baseline_L10_layers3": baseline,
        "lower_frequency_L4": FieldConfig(baseline.hidden_dim, baseline.num_hidden_layers, 4, baseline.learning_rate),
        "shallower_layers2": FieldConfig(baseline.hidden_dim, 2, baseline.num_frequencies, baseline.learning_rate),
        "L4_layers2": FieldConfig(baseline.hidden_dim, 2, 4, baseline.learning_rate),
    }
    sweep_dir = args.output_dir / "hyperparameter_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    all_histories: dict[str, list[dict[str, float]]] = {}
    rows: list[dict[str, float | int | str]] = []
    for name, config in configurations.items():
        print(f"\nRunning sweep configuration: {name}")
        _, history, _ = train_field(
            image,
            config,
            args.sweep_iterations,
            args.batch_size,
            args.eval_every,
            device,
            args.seed,
            capture_snapshots=False,
        )
        all_histories[name] = history
        rows.append({"name": name, **asdict(config), "final_psnr": history[-1]["psnr"]})

    plt.figure(figsize=(8, 5))
    for name, history in all_histories.items():
        plt.plot(
            [entry["iteration"] for entry in history],
            [entry["psnr"] for entry in history],
            label=name,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Training PSNR (dB)")
    plt.title("Hyperparameter Tuning")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(sweep_dir / "comparison.png", dpi=180)
    plt.close()
    with (sweep_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(sweep_dir / "histories.json", all_histories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a neural field to one 2D image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part1/image"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--frequencies", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device: {device}")
    image = load_image(args.image, args.max_size)
    run_single(args, image, device)
    if args.sweep:
        run_sweep(args, image, device)


if __name__ == "__main__":
    main()
