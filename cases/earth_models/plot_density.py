#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one or more radial Earth density profiles."
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="CSV density profile files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output image path.",
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )

    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help=(
            "Curve label. Repeat once per input file. "
            "If omitted, labels are derived from file names."
        ),
    )

    return parser.parse_args()


def default_label(path: Path) -> str:
    label = path.stem.replace("_", " ")

    words = []

    for word in label.split():
        if word.lower() == "prem":
            words.append("PREM")
        else:
            words.append(word.capitalize())

    return " ".join(words)


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Density profile not found: {path}")

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
    )

    required = {
        "radius_km",
        "density_g_cm3",
    }

    if data.dtype.names is None:
        raise ValueError(f"CSV has no named columns: {path}")

    if not required.issubset(data.dtype.names):
        raise ValueError(
            f"CSV must contain columns 'radius_km' and 'density_g_cm3': {path}"
        )

    radius = np.asarray(
        data["radius_km"],
        dtype=float,
    )

    density = np.asarray(
        data["density_g_cm3"],
        dtype=float,
    )

    if radius.ndim != 1 or density.ndim != 1:
        raise ValueError(f"Invalid density profile shape: {path}")

    if radius.size != density.size:
        raise ValueError(f"Radius and density lengths differ: {path}")

    if radius.size < 2:
        raise ValueError(f"Density profile is too short: {path}")

    return radius, density


def main() -> None:
    args = parse_arguments()

    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError(
            "--label must either be omitted or supplied once per input file"
        )

    labels = (
        args.labels
        if args.labels is not None
        else [default_label(path) for path in args.inputs]
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 5.2),
        layout="constrained",
    )

    for path, label in zip(args.inputs, labels):
        radius, density = load_profile(path)

        ax.plot(
            radius,
            density,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("Radius [km]")
    ax.set_ylabel(r"Density [g cm$^{-3}$]")

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    if args.title:
        ax.set_title(args.title)

    ax.grid(
        True,
        linewidth=0.7,
        alpha=0.22,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(args.inputs) > 1:
        ax.legend(
            frameon=False,
            ncol=2,
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=240,
    )

    plt.close(fig)

    print(f"[plot] {args.output}")


if __name__ == "__main__":
    main()
