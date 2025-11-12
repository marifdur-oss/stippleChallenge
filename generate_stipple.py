#!/usr/bin/env python3
"""
Generate a blue-noise stipple rendering and progressive GIF for a source image.
Based on the modified void-and-cluster workflow from the Stipple Challenge guide.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt


@dataclass
class StippleConfig:
    percentage: float = 0.085
    sigma: float = 1.0
    content_bias: float = 0.95
    noise_scale_factor: float = 0.08
    extreme_downweight: float = 0.55
    extreme_threshold_low: float = 0.25
    extreme_threshold_high: float = 0.85
    extreme_sigma: float = 0.12
    mid_tone_boost: float = 0.5
    mid_tone_center: float = 0.68
    mid_tone_sigma: float = 0.18
    resize_max: int = 512
    rng_seed: int = 20241111
    frame_increment: int = 120
    gif_duration: float = 0.5


def load_grayscale_image(path: pathlib.Path, max_size: int | None) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")

    if max_size is not None and max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        return arr
    return arr[..., 0]


def compute_importance(
    gray_img: np.ndarray,
    extreme_downweight: float,
    extreme_threshold_low: float,
    extreme_threshold_high: float,
    extreme_sigma: float,
    mid_tone_boost: float,
    mid_tone_center: float,
    mid_tone_sigma: float,
) -> np.ndarray:
    I = np.clip(gray_img, 0.0, 1.0)
    I_inverted = 1.0 - I

    dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * extreme_sigma**2))
    dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
    if dark_mask.max() > 0:
        dark_mask /= dark_mask.max()

    light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * extreme_sigma**2))
    light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
    if light_mask.max() > 0:
        light_mask /= light_mask.max()

    extreme_mask = np.maximum(dark_mask, light_mask)
    importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)

    mid_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * mid_tone_sigma**2))
    if mid_gaussian.max() > 0:
        mid_gaussian /= mid_gaussian.max()
    importance *= 1.0 + mid_tone_boost * mid_gaussian

    m, M = importance.min(), importance.max()
    if M > m:
        importance = (importance - m) / (M - m)
    return importance


def toroidal_gaussian_kernel(h: int, w: int, sigma: float) -> np.ndarray:
    y = np.arange(h)
    x = np.arange(w)
    dy = np.minimum(y, h - y)[:, None]
    dx = np.minimum(x, w - x)[None, :]
    kernel = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel


def void_and_cluster(
    image: np.ndarray,
    importance: np.ndarray,
    config: StippleConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    kernel = toroidal_gaussian_kernel(h, w, config.sigma)
    energy = -importance * config.content_bias
    stipple = np.ones_like(image)
    samples: list[tuple[int, int, float]] = []

    def energy_splat(y: int, x: int) -> np.ndarray:
        return np.roll(np.roll(kernel, shift=y, axis=0), shift=x, axis=1)

    num_points = max(1, int(image.size * config.percentage))
    cy, cx = h // 2, w // 2
    r = max(1, min(20, h // 10, w // 10))
    ys = slice(max(0, cy - r), min(h, cy + r))
    xs = slice(max(0, cx - r), min(w, cx + r))
    region = energy[ys, xs]
    flat = np.argmin(region)
    y0 = flat // region.shape[1] + (cy - r)
    x0 = flat % region.shape[1] + (cx - r)

    energy += energy_splat(y0, x0)
    energy[y0, x0] = np.inf
    samples.append((y0, x0, image[y0, x0]))
    stipple[y0, x0] = 0.0

    rng = np.random.default_rng(config.rng_seed)

    for i in range(1, num_points):
        exploration = 1.0 - (i / num_points) * 0.5
        noise = rng.normal(
            0.0,
            config.noise_scale_factor * config.content_bias * exploration,
            size=energy.shape,
        )
        energy_with_noise = energy + noise
        pos_flat = np.argmin(energy_with_noise)
        y = pos_flat // w
        x = pos_flat % w
        energy += energy_splat(y, x)
        energy[y, x] = np.inf
        samples.append((y, x, image[y, x]))
        stipple[y, x] = 0.0

    return stipple, np.asarray(samples)


def render_comparison(
    original: np.ndarray,
    importance: np.ndarray,
    stipple: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, data, title in zip(
        axes,
        (original, importance, stipple),
        ("Original", "Importance Map", "Blue Noise Stipple"),
    ):
        ax.imshow(data, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_progressive_gif(
    samples: np.ndarray,
    image_shape: Sequence[int],
    output_path: pathlib.Path,
    frame_increment: int,
    duration: float,
) -> None:
    h, w = image_shape
    canvas = np.ones((h, w), dtype=np.float32)
    frames: list[np.ndarray] = []

    for idx, (y, x, _) in enumerate(samples, start=1):
        canvas[int(y), int(x)] = 0.0
        if idx == 1 or idx % frame_increment == 0 or idx == len(samples):
            frames.append(canvas.copy())

    frames_uint8 = [np.uint8(frame * 255) for frame in frames]
    imageio.mimsave(output_path, frames_uint8, duration=duration)


def save_stipple_image(stipple: np.ndarray, output_path: pathlib.Path) -> None:
    img = Image.fromarray(np.uint8(np.clip(stipple, 0.0, 1.0) * 255.0), mode="L")
    img.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=pathlib.Path,
        required=True,
        help="Path to the source image.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory to store generated assets.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=StippleConfig.percentage,
        help="Fraction of pixels that become stipple points.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=StippleConfig.frame_increment,
        help="Point increment between GIF frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StippleConfig(percentage=args.percentage, frame_increment=args.frame_step)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading image: {args.image}")
    image = load_grayscale_image(args.image, config.resize_max)
    print(f"Working resolution: {image.shape[0]} x {image.shape[1]}")

    importance = compute_importance(
        image,
        config.extreme_downweight,
        config.extreme_threshold_low,
        config.extreme_threshold_high,
        config.extreme_sigma,
        config.mid_tone_boost,
        config.mid_tone_center,
        config.mid_tone_sigma,
    )

    stipple, samples = void_and_cluster(image, importance, config)
    print(f"Generated {len(samples)} stipple samples.")

    stipple_path = output_dir / f"{args.image.stem}_stipple.png"
    comparison_path = output_dir / f"{args.image.stem}_comparison.png"
    gif_path = output_dir / f"{args.image.stem}_progressive.gif"
    samples_path = output_dir / f"{args.image.stem}_samples.npy"

    save_stipple_image(stipple, stipple_path)
    render_comparison(image, importance, stipple, comparison_path)
    save_progressive_gif(
        samples,
        image.shape,
        gif_path,
        config.frame_increment,
        config.gif_duration,
    )
    np.save(samples_path, samples)

    print(f"Saved stipple image: {stipple_path}")
    print(f"Saved comparison figure: {comparison_path}")
    print(f"Saved progressive GIF: {gif_path}")
    print(f"Stored samples array: {samples_path}")


if __name__ == "__main__":
    main()

