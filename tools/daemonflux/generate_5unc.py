#!/usr/bin/env python3
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from generate import (
    DAEMONFLUX_VERSION,
    PROJECT_ROOT,
    detector_coszenith_to_daemonflux_zenith,
    make_grid,
    normalize_daemonflux_shape,
    write_flux_dataset,
)

from daemonflux import Flux as DaemonFlux

PARAMETERS = ("K+_2P", "K-_2P", "pi+_20T", "pi+_2P", "pi-_20T")
QUANTITIES = ("numu", "antinumu")

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/generated/daemonflux"
    / f"daemonflux_{DAEMONFLUX_VERSION}_5unc.h5"
)


def conditional_covariance(model):
    names = tuple(str(name) for name in model.params.known_parameters)
    errors = np.asarray(model.params.errors, dtype=np.float64)
    cov = np.asarray(model.params.cov, dtype=np.float64)

    # Parameter shifts passed to daemonflux are measured in units
    # of each parameter's marginal standard deviation.
    correlation = cov / np.outer(errors, errors)
    correlation = 0.5 * (correlation + correlation.T)

    selected = [names.index(name) for name in PARAMETERS]

    # Conditional covariance with every unselected parameter fixed.
    precision = np.linalg.inv(correlation)[np.ix_(selected, selected)]
    precision = 0.5 * (precision + precision.T)

    conditional = np.linalg.inv(precision)
    conditional = 0.5 * (conditional + conditional.T)

    return conditional, np.linalg.cholesky(conditional)


def evaluate(model, energy, angles, quantity, params):
    raw = model.flux(energy, angles, quantity, params=params)

    array = normalize_daemonflux_shape(
        raw,
        angles.size,
        energy.size,
        quantity,
    )

    # Restore increasing detector coszenith and remove E^3 weighting.
    return np.ascontiguousarray(array[::-1, :] / energy[np.newaxis, :] ** 3)


def generate(output):
    if version("daemonflux") != DAEMONFLUX_VERSION:
        raise RuntimeError(f"Expected daemonflux {DAEMONFLUX_VERSION}")

    coszenith, energy = make_grid()

    # daemonflux requires ascending zenith angles.
    angles = np.ascontiguousarray(
        detector_coszenith_to_daemonflux_zenith(coszenith)[::-1]
    )

    model = DaemonFlux(
        location="IceCube",
        use_calibration=True,
        debug=0,
    )

    conditional, cholesky = conditional_covariance(model)

    baseline = {
        quantity: evaluate(model, energy, angles, quantity, {})
        for quantity in QUANTITIES
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")

    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["schema_name"] = "neutrino_tomography_daemonflux_5unc"
            file.attrs["schema_version"] = 1
            file.attrs["daemonflux_version"] = DAEMONFLUX_VERSION
            file.attrs["location"] = "IceCube"
            file.attrs["use_calibration"] = True
            file.attrs["generator"] = "tools/daemonflux/generate_5unc.py"

            axes = file.create_group("axes")
            axes.create_dataset("coszenith", data=coszenith)
            axes.create_dataset("energy_GeV", data=energy)

            # Keep the existing central-flux dataset layout so the
            # C++ load_daemonflux() function can read this file.
            central = file.create_group("flux/IceCube/calibrated")

            for quantity in QUANTITIES:
                write_flux_dataset(
                    central,
                    quantity,
                    baseline[quantity],
                    quantity,
                )

            nuisance = file.create_group("nuisance")

            nuisance.attrs["mapping"] = "s = L @ z; Phi = Phi0 + sum(s[a] * T[a])"
            nuisance.attrs["shift_units"] = "marginal standard deviations"

            nuisance.create_dataset(
                "parameter_names",
                data=np.asarray(PARAMETERS, dtype="S16"),
            )
            nuisance.create_dataset(
                "conditional_covariance",
                data=conditional,
            )
            nuisance.create_dataset(
                "cholesky",
                data=cholesky,
            )

            templates = nuisance.create_group("templates")

            for name in PARAMETERS:
                print(f"[template] {name}", flush=True)
                group = templates.create_group(name)

                for quantity in QUANTITIES:
                    shifted = evaluate(
                        model,
                        energy,
                        angles,
                        quantity,
                        {name: 1.0},
                    )

                    # These are signed changes, not physical fluxes.
                    template = shifted - baseline[quantity]

                    write_flux_dataset(
                        group,
                        quantity,
                        template,
                        quantity,
                    )

                    group[quantity].attrs["quantity"] = (
                        "signed differential flux change for a +1 sigma shift"
                    )

        temporary.replace(output)

    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"[saved] {output}")


def main():
    parser = argparse.ArgumentParser(description="Export five daemonflux nuisances.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    output = args.output.resolve()

    if output.exists() and not args.force:
        print(f"[skip] {output}")
        return

    generate(output)


if __name__ == "__main__":
    main()
