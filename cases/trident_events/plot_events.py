#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANNOTATE_HEATMAP = False
ANNOTATE_COSZENITH_HISTOGRAM = False
ANNOTATE_ENERGY_HISTOGRAM = False

HEATMAP_CMAP = "viridis"
HISTOGRAM_COLOR = "#22b8cf"


def load_events(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    required = {
        "coszenith_bin",
        "proxy_energy_bin",
        "coszenith_low",
        "coszenith_center",
        "coszenith_high",
        "proxy_energy_low_gev",
        "proxy_energy_center_gev",
        "proxy_energy_high_gev",
        "events",
    }

    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"Unexpected event CSV format: {path}")

    z_index = np.asarray(data["coszenith_bin"], dtype=int)
    e_index = np.asarray(data["proxy_energy_bin"], dtype=int)

    n_z = int(z_index.max()) + 1
    n_e = int(e_index.max()) + 1

    expected_z = np.repeat(np.arange(n_z), n_e)
    expected_e = np.tile(np.arange(n_e), n_z)

    if not np.array_equal(z_index, expected_z) or not np.array_equal(
        e_index, expected_e
    ):
        raise ValueError("Event CSV rows are not in [coszenith, proxy_energy] order")

    events = np.asarray(data["events"], dtype=float).reshape(n_z, n_e)

    coszenith_center = np.asarray(data["coszenith_center"], dtype=float).reshape(
        n_z, n_e
    )[:, 0]
    coszenith_low = np.asarray(data["coszenith_low"], dtype=float).reshape(n_z, n_e)[
        :, 0
    ]
    coszenith_high = np.asarray(data["coszenith_high"], dtype=float).reshape(n_z, n_e)[
        :, 0
    ]

    energy_center = np.asarray(data["proxy_energy_center_gev"], dtype=float).reshape(
        n_z, n_e
    )[0, :]
    energy_low = np.asarray(data["proxy_energy_low_gev"], dtype=float).reshape(
        n_z, n_e
    )[0, :]
    energy_high = np.asarray(data["proxy_energy_high_gev"], dtype=float).reshape(
        n_z, n_e
    )[0, :]

    coszenith_edges = np.concatenate([coszenith_low, [coszenith_high[-1]]])
    energy_edges = np.concatenate([energy_low, [energy_high[-1]]])

    return events, coszenith_center, coszenith_edges, energy_center, energy_edges


def event_label(value: float) -> str:
    if value >= 1000.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def prepare_axis(ax):
    ax.grid(True, linewidth=0.7, alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_heatmap(
    events, coszenith_center, coszenith_edges, energy_center, energy_edges, output: Path
):
    fig, ax = plt.subplots(figsize=(10.4, 7.0), layout="constrained")

    mesh = ax.pcolormesh(
        coszenith_edges,
        energy_edges,
        events.T,
        shading="flat",
        cmap=HEATMAP_CMAP,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Cosine zenith angle")
    ax.set_ylabel("Proxy energy [GeV]")
    ax.set_title("TRIDENT expected event distribution")

    ax.set_xlim(coszenith_edges[0], coszenith_edges[-1])
    ax.set_ylim(energy_edges[0], energy_edges[-1])

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Expected events")

    if ANNOTATE_HEATMAP:
        for z, cz in enumerate(coszenith_center):
            for e, energy in enumerate(energy_center):
                ax.text(
                    cz,
                    energy,
                    event_label(events[z, e]),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                )

    fig.savefig(output, dpi=240)
    plt.close(fig)

    print(f"[plot] {output}")


def plot_coszenith_histogram(events, coszenith_center, coszenith_edges, output: Path):
    values = events.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), layout="constrained")

    ax.stairs(
        values,
        coszenith_edges,
        color=HISTOGRAM_COLOR,
        linewidth=1.8,
        fill=True,
        alpha=0.24,
    )
    ax.stairs(
        values,
        coszenith_edges,
        color=HISTOGRAM_COLOR,
        linewidth=1.8,
    )

    ax.set_xlabel("Cosine zenith angle")
    ax.set_ylabel("Expected events")
    ax.set_title("Events summed over proxy energy")
    ax.set_xlim(coszenith_edges[0], coszenith_edges[-1])
    ax.set_ylim(bottom=0.0)

    prepare_axis(ax)

    if ANNOTATE_COSZENITH_HISTOGRAM:
        for x, y in zip(coszenith_center, values):
            ax.annotate(
                event_label(y),
                xy=(x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.savefig(output, dpi=240)
    plt.close(fig)

    print(f"[plot] {output}")


def plot_energy_histogram(events, energy_center, energy_edges, output: Path):
    values = events.sum(axis=0)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), layout="constrained")

    ax.stairs(
        values,
        energy_edges,
        color=HISTOGRAM_COLOR,
        linewidth=1.8,
        fill=True,
        alpha=0.24,
    )
    ax.stairs(
        values,
        energy_edges,
        color=HISTOGRAM_COLOR,
        linewidth=1.8,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Proxy energy [GeV]")
    ax.set_ylabel("Expected events")
    ax.set_title("Events summed over cosine zenith")
    ax.set_xlim(energy_edges[0], energy_edges[-1])
    ax.set_ylim(bottom=0.0)

    prepare_axis(ax)

    if ANNOTATE_ENERGY_HISTOGRAM:
        for x, y in zip(energy_center, values):
            ax.annotate(
                event_label(y),
                xy=(x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.savefig(output, dpi=240)
    plt.close(fig)

    print(f"[plot] {output}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_events.py <events.csv> <output-directory>")

    csv_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not csv_file.is_file():
        raise FileNotFoundError(f"Event CSV not found: {csv_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    events, coszenith_center, coszenith_edges, energy_center, energy_edges = (
        load_events(csv_file)
    )

    plot_heatmap(
        events,
        coszenith_center,
        coszenith_edges,
        energy_center,
        energy_edges,
        output_dir / "event_heatmap.png",
    )

    plot_coszenith_histogram(
        events,
        coszenith_center,
        coszenith_edges,
        output_dir / "events_vs_coszenith.png",
    )

    plot_energy_histogram(
        events,
        energy_center,
        energy_edges,
        output_dir / "events_vs_proxy_energy.png",
    )


if __name__ == "__main__":
    main()
