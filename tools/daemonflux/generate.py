#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from importlib.metadata import version as package_version
from pathlib import Path

import h5py
import numpy as np

from daemonflux import Flux as DaemonFlux

# ============================================================
# Project layout
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DAEMONFLUX_VERSION = "0.8.2"

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "daemonflux"
    / f"daemonflux_{DAEMONFLUX_VERSION}.h5"
)


# ============================================================
# HDF5 schema
# ============================================================

SCHEMA_NAME = "neutrino_tomography_daemonflux"
SCHEMA_VERSION = 1


# ============================================================
# Master grid
# ============================================================

# DaemonFlux itself supports a much wider energy range.
#
# For the tomography project we keep a master table wider than
# the current analysis range so that the C++ side can interpolate
# onto different analysis grids without regenerating DaemonFlux.
#
# 100 GeV -- 1 PeV
# 401 points -> Delta log10(E/GeV) = 0.01 exactly.
ENERGY_MIN_GEV = 1.0e2
ENERGY_MAX_GEV = 1.0e6
N_ENERGY = 401


# Project detector zenith convention:
#
#   coszenith = -1 : vertically up-going
#   coszenith =  0 : horizontal
#
# DaemonFlux 0.8.2 uses atmospheric/down-going zenith angles in
# the range 0 -- 90 degrees for the spline tables we use.
#
# Therefore:
#
#   theta_DF = acos(-coszenith)
#
# gives
#
#   detector coszenith = -1  -> theta_DF = 0 deg
#   detector coszenith =  0  -> theta_DF = 90 deg
#
# The master grid is uniform in detector coszenith.
#
# 2001 points -> Delta coszenith = 5e-4 exactly.
COSZENITH_MIN = -1.0
COSZENITH_MAX = 0.0
N_COSZENITH = 2001


# These are the locations already used by the project.
LOCATIONS = (
    "generic",
    "IceCube",
    "Kamioka",
)


# We explicitly request the calibrated DaemonFlux baseline.
#
# This matches the default behavior of:
#
#     DaemonFlux(location=...)
#
# used by the old embedded-Python implementation.
USE_CALIBRATION = True


# ============================================================
# Grid construction
# ============================================================


def make_grid() -> tuple[np.ndarray, np.ndarray]:
    energy_gev = np.logspace(
        math.log10(ENERGY_MIN_GEV),
        math.log10(ENERGY_MAX_GEV),
        N_ENERGY,
        dtype=np.float64,
    )

    coszenith = np.linspace(
        COSZENITH_MIN,
        COSZENITH_MAX,
        N_COSZENITH,
        dtype=np.float64,
    )

    return coszenith, energy_gev


# ============================================================
# Zenith convention conversion
# ============================================================


def detector_coszenith_to_daemonflux_zenith(
    coszenith: np.ndarray,
) -> np.ndarray:
    coszenith = np.asarray(
        coszenith,
        dtype=np.float64,
    )

    if coszenith.ndim != 1:
        raise ValueError("coszenith must be a one-dimensional array")

    if not np.all(np.isfinite(coszenith)):
        raise ValueError("coszenith contains non-finite values")

    if np.any(coszenith < -1.0) or np.any(coszenith > 0.0):
        raise ValueError("up-going detector coszenith must be in [-1, 0]")

    # Guard against tiny floating-point excursions.
    argument = np.clip(
        -coszenith,
        0.0,
        1.0,
    )

    zenith_deg = np.rad2deg(np.arccos(argument))

    # DaemonFlux requires a sorted ascending angle array.
    if np.any(np.diff(zenith_deg) < 0.0):
        raise RuntimeError(
            "converted DaemonFlux zenith angles are not sorted in ascending order"
        )

    return zenith_deg


# ============================================================
# DaemonFlux output conversion
# ============================================================


def normalize_daemonflux_shape(
    raw: np.ndarray,
    n_coszenith: int,
    n_energy: int,
    quantity: str,
) -> np.ndarray:
    """
    Return flux in project storage order:

        [coszenith, energy]

    DaemonFlux 0.8.2 returns a vectorized multi-angle query in
    [energy, zenith] order. The alternate shape is accepted here
    as a defensive compatibility check.
    """

    array = np.asarray(
        raw,
        dtype=np.float64,
    )

    if array.shape == (n_energy, n_coszenith):
        return np.ascontiguousarray(array.T)

    if array.shape == (n_coszenith, n_energy):
        return np.ascontiguousarray(array)

    raise RuntimeError(
        f"DaemonFlux returned unexpected shape for {quantity}: "
        f"{array.shape}; expected "
        f"({n_energy}, {n_coszenith}) or "
        f"({n_coszenith}, {n_energy})"
    )


# ============================================================
# Flux evaluation
# ============================================================


def evaluate_location(
    location: str,
    coszenith: np.ndarray,
    energy_gev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"[daemonflux] location = {location}")

    model = DaemonFlux(
        location=location,
        use_calibration=USE_CALIBRATION,
    )

    daemonflux_zenith_deg = detector_coszenith_to_daemonflux_zenith(coszenith)

    print(f"             detector coszenith: [{coszenith[0]:.6f}, {coszenith[-1]:.6f}]")

    print(
        "             DaemonFlux zenith:   "
        f"[{daemonflux_zenith_deg[0]:.6f}, "
        f"{daemonflux_zenith_deg[-1]:.6f}] deg"
    )

    # --------------------------------------------------------
    # DaemonFlux supports vectorized zenith queries.
    #
    # This is much cheaper than calling model.flux() once for
    # every master-grid angle.
    # --------------------------------------------------------

    raw_numu = model.flux(
        energy_gev,
        daemonflux_zenith_deg,
        "total_numu",
    )

    raw_antinumu = model.flux(
        energy_gev,
        daemonflux_zenith_deg,
        "total_antinumu",
    )

    numu = normalize_daemonflux_shape(
        raw_numu,
        n_coszenith=coszenith.size,
        n_energy=energy_gev.size,
        quantity="total_numu",
    )

    antinumu = normalize_daemonflux_shape(
        raw_antinumu,
        n_coszenith=coszenith.size,
        n_energy=energy_gev.size,
        quantity="total_antinumu",
    )

    # --------------------------------------------------------
    # DaemonFlux returns E^3 * Phi.
    #
    # Store the ordinary differential flux:
    #
    #     Phi = dPhi/dE
    #
    # with units:
    #
    #     GeV^-1 cm^-2 s^-1 sr^-1
    #
    # The energy axis is in GeV, so division by energy_gev**3
    # exactly removes the DaemonFlux E^3 weighting.
    # --------------------------------------------------------

    e3 = energy_gev[np.newaxis, :] ** 3

    numu = numu / e3
    antinumu = antinumu / e3

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    for quantity, array in (
        ("numu", numu),
        ("antinumu", antinumu),
    ):
        expected_shape = (
            coszenith.size,
            energy_gev.size,
        )

        if array.shape != expected_shape:
            raise RuntimeError(
                f"{location}/{quantity}: unexpected shape "
                f"{array.shape}; expected {expected_shape}"
            )

        if not np.all(np.isfinite(array)):
            bad = np.argwhere(~np.isfinite(array))

            raise RuntimeError(
                f"{location}/{quantity}: non-finite values "
                f"found; first bad index = "
                f"{tuple(bad[0])}"
            )

        if np.any(array < 0.0):
            bad = np.argwhere(array < 0.0)

            raise RuntimeError(
                f"{location}/{quantity}: negative values "
                f"found; first bad index = "
                f"{tuple(bad[0])}"
            )

    print(f"             numu=[{numu.min():.6e}, {numu.max():.6e}]")

    print(f"             antinumu=[{antinumu.min():.6e}, {antinumu.max():.6e}]")

    return numu, antinumu


# ============================================================
# HDF5 helpers
# ============================================================


def write_flux_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    daemonflux_quantity: str,
) -> None:
    dataset = group.create_dataset(
        name,
        data=data,
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=(
            min(64, data.shape[0]),
            min(256, data.shape[1]),
        ),
    )

    dataset.attrs["daemonflux_quantity"] = daemonflux_quantity

    dataset.attrs["quantity"] = "ordinary differential atmospheric neutrino flux"

    dataset.attrs["units"] = "GeV^-1 cm^-2 s^-1 sr^-1"

    dataset.attrs["axis_order"] = "coszenith, energy_GeV"


# ============================================================
# HDF5 generation
# ============================================================


def generate(output: Path) -> None:
    installed_version = package_version("daemonflux")

    if installed_version != DAEMONFLUX_VERSION:
        raise RuntimeError(
            "Unexpected DaemonFlux version: "
            f"{installed_version}; "
            f"expected {DAEMONFLUX_VERSION}"
        )

    output = output.resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coszenith, energy_gev = make_grid()

    daemonflux_zenith_deg = detector_coszenith_to_daemonflux_zenith(coszenith)

    print(f"[daemonflux] version = {installed_version}")

    print(f"[grid] coszenith={coszenith.size}, energy={energy_gev.size}")

    print(f"[grid] coszenith range=[{coszenith[0]}, {coszenith[-1]}]")

    print(
        "[grid] "
        f"DaemonFlux zenith range="
        f"[{daemonflux_zenith_deg[0]}, "
        f"{daemonflux_zenith_deg[-1]}] deg"
    )

    print(f"[grid] energy range=[{energy_gev[0]}, {energy_gev[-1]}] GeV")

    print(f"[grid] Delta coszenith={coszenith[1] - coszenith[0]:.8g}")

    print(
        "[grid] "
        "Delta log10(E/GeV)="
        f"{np.log10(energy_gev[1]) - np.log10(energy_gev[0]):.8g}"
    )

    # --------------------------------------------------------
    # Atomic-ish generation:
    #
    # Never leave a partially generated file using the final
    # filename.
    # --------------------------------------------------------

    temporary_output = output.with_name(output.name + ".tmp")

    if temporary_output.exists():
        temporary_output.unlink()

    try:
        with h5py.File(
            temporary_output,
            "w",
        ) as file:
            # =================================================
            # Global metadata
            # =================================================

            file.attrs["schema_name"] = SCHEMA_NAME

            file.attrs["schema_version"] = SCHEMA_VERSION

            file.attrs["generator"] = "tools/daemonflux/generate.py"

            file.attrs["daemonflux_version"] = installed_version

            file.attrs["daemonflux_use_calibration"] = USE_CALIBRATION

            file.attrs["daemonflux_numu_quantity"] = "total_numu"

            file.attrs["daemonflux_antinumu_quantity"] = "total_antinumu"

            file.attrs["includes_prompt_flux"] = True

            file.attrs["stored_flux_units"] = "GeV^-1 cm^-2 s^-1 sr^-1"

            file.attrs["daemonflux_raw_units"] = "GeV^2 cm^-2 s^-1 sr^-1"

            file.attrs["daemonflux_raw_scaling"] = "E_GeV^3 * differential_flux"

            file.attrs["stored_scaling"] = (
                "ordinary differential flux; DaemonFlux output divided by E_GeV^3"
            )

            file.attrs["detector_coszenith_convention"] = (
                "-1 = vertically up-going; 0 = horizontal"
            )

            file.attrs["daemonflux_zenith_mapping"] = (
                "theta_DF_deg = acos(-detector_coszenith) * 180/pi"
            )

            file.attrs["daemonflux_zenith_range_deg"] = "0 to 90"

            # =================================================
            # Axes
            # =================================================

            axes = file.create_group("axes")

            ds_coszenith = axes.create_dataset(
                "coszenith",
                data=coszenith,
                dtype=np.float64,
            )

            ds_coszenith.attrs["quantity"] = "detector zenith cosine"

            ds_coszenith.attrs["units"] = "dimensionless"

            ds_coszenith.attrs["convention"] = (
                "-1 = vertically up-going; 0 = horizontal"
            )

            ds_energy = axes.create_dataset(
                "energy_GeV",
                data=energy_gev,
                dtype=np.float64,
            )

            ds_energy.attrs["quantity"] = "neutrino energy"

            ds_energy.attrs["units"] = "GeV"

            ds_energy.attrs["spacing"] = "uniform in log10(E/GeV)"

            # =================================================
            # Flux data
            # =================================================

            flux_root = file.create_group("flux")

            for location in LOCATIONS:
                numu, antinumu = evaluate_location(
                    location=location,
                    coszenith=coszenith,
                    energy_gev=energy_gev,
                )

                location_group = flux_root.create_group(location)

                location_group.attrs["daemonflux_location"] = location

                calibrated_group = location_group.create_group("calibrated")

                calibrated_group.attrs["use_calibration"] = True

                write_flux_dataset(
                    calibrated_group,
                    "numu",
                    numu,
                    "total_numu",
                )

                write_flux_dataset(
                    calibrated_group,
                    "antinumu",
                    antinumu,
                    "total_antinumu",
                )

            file.flush()

        # Only expose the final filename after the complete
        # HDF5 file has been written successfully.
        temporary_output.replace(output)

    except BaseException:
        if temporary_output.exists():
            temporary_output.unlink()

        raise

    size_mib = output.stat().st_size / (1024.0 * 1024.0)

    print()
    print(f"[saved] {output}")

    print(f"[size]  {size_mib:.2f} MiB")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate the project-local DaemonFlux master flux table.")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(f"output HDF5 path (default: {DEFAULT_OUTPUT})"),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=("overwrite an existing generated table"),
    )

    args = parser.parse_args()

    output = args.output.resolve()

    if output.exists():
        if not args.force:
            print(f"[skip] DaemonFlux table already exists: {output}")
            return

        print(f"[remove] existing DaemonFlux table: {output}")

        output.unlink()

    generate(output)


if __name__ == "__main__":
    main()
