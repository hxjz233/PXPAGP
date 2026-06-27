"""Reproduce Fig. 11 from Phys. Rev. X 10, 041017 (2020).

The paper studies the regularized adiabatic gauge potential (AGP) norm for the
XXZ chain perturbed by a next-nearest-neighbor Ising interaction,

    H = H_XXZ + J2 * sum_i sigma^z_i sigma^z_{i+2},

and measures the AGP with respect to the XXZ anisotropy Delta.

This script performs exact diagonalization in the zero-magnetization sector
with QuSpin, evaluates the regularized AGP norm,

    ||A_Delta||^2 = (1/D) sum_{n != m} |<n|dH/dDelta|m>|^2
                    * w_nm^2 / (w_nm^2 + mu^2)^2,

with mu = L / D as used in the paper, and plots ||A_Delta||^2 / L versus
system size for several J2 values.

The default system sizes are intentionally modest so the script stays practical
with dense diagonalization. Increase --l-max if you want to push it harder.
The script also overlays simple exponential tail fits for the nonzero J2 curves,
which makes the semilog plot closer to the figure in the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

FIG_DIR = Path(__file__).resolve().parent.parent / "fig"


@dataclass(frozen=True)
class ModelData:
    """Dense operators needed for one system size."""

    l: int
    dimension: int
    h_base: np.ndarray
    h_nnn: np.ndarray
    dh_ddelta: np.ndarray


def build_model(l: int, delta: float) -> ModelData:
    """Construct dense Hamiltonian blocks for a fixed chain length."""

    if l % 2 != 0:
        raise ValueError("This reproduction uses the zero-magnetization sector, so L must be even.")

    basis = spin_basis_1d(l, Nup=l // 2, pauli=True)

    nn_xx = [[1.0, i, i + 1] for i in range(l - 1)]
    nn_yy = [[1.0, i, i + 1] for i in range(l - 1)]
    nn_zz = [[delta, i, i + 1] for i in range(l - 1)]
    nnn_zz = [[1.0, i, i + 2] for i in range(l - 2)]

    h_base = hamiltonian(
        [["xx", nn_xx], ["yy", nn_yy], ["zz", nn_zz]],
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
    )
    h_nnn = hamiltonian(
        [["zz", nnn_zz]],
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
    )
    dh_ddelta = hamiltonian(
        [["zz", [[1.0, i, i + 1] for i in range(l - 1)]]],
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
    )

    return ModelData(
        l=l,
        dimension=basis.Ns,
        h_base=np.asarray(h_base.todense(), dtype=np.float64),
        h_nnn=np.asarray(h_nnn.todense(), dtype=np.float64),
        dh_ddelta=np.asarray(dh_ddelta.todense(), dtype=np.float64),
    )


def regularized_agp_norm(h: np.ndarray, dh_ddelta: np.ndarray, mu: float) -> float:
    """Return the regularized AGP norm for a dense Hamiltonian.

    The implementation follows Eq. (5) in the paper.
    """

    evals, evecs = np.linalg.eigh(h)
    dh_eig = evecs.conj().T @ dh_ddelta @ evecs

    omega = evals[:, None] - evals[None, :]
    weight = (omega * omega) / (omega * omega + mu * mu) ** 2
    np.fill_diagonal(weight, 0.0)

    norm = np.sum(np.abs(dh_eig) ** 2 * weight) / h.shape[0]
    return float(np.real_if_close(norm))


def compute_curve(
    l_values: Iterable[int],
    delta: float,
    j2_values: Iterable[float],
) -> dict[float, list[tuple[int, float]]]:
    """Compute ||A_Delta||^2 / L for each J2 and chain length."""

    cached_models: dict[int, ModelData] = {}
    results: dict[float, list[tuple[int, float]]] = {j2: [] for j2 in j2_values}

    for l in l_values:
        model = cached_models.setdefault(l, build_model(l, delta))
        mu = l / model.dimension

        for j2 in j2_values:
            h = model.h_base + j2 * model.h_nnn
            norm = regularized_agp_norm(h, model.dh_ddelta, mu)
            results[j2].append((l, norm / l))

    return results


def plot_results(results: dict[float, list[tuple[int, float]]], output_path: Path) -> None:
    """Create a semilog plot similar to Fig. 11."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for j2, points in results.items():
        lengths = np.array([p[0] for p in points], dtype=float)
        values = np.array([p[1] for p in points], dtype=float)
        label = r"$J_2 = {:.3g}$".format(j2)
        ax.semilogy(lengths, values, marker="o", linewidth=1.8, label=label)

        if j2 > 0 and len(points) >= 3:
            tail_lengths = lengths[-3:]
            tail_log_values = np.log(values[-3:])
            slope, intercept = np.polyfit(tail_lengths, tail_log_values, 1)
            fit_values = np.exp(intercept + slope * lengths)
            ax.semilogy(lengths, fit_values, linestyle="--", linewidth=1.1, alpha=0.7)

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\|A_\Delta\|^2 / L$")
    ax.set_title("XXZ chain with NNN interaction: AGP norm scaling")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, default=1.1, help="XXZ anisotropy Delta.")
    parser.add_argument(
        "--l-min",
        type=int,
        default=6,
        help="Smallest even system size to include.",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=12,
        help="Largest even system size to include (dense ED gets expensive quickly).",
    )
    parser.add_argument(
        "--j2",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        help="NNN interaction strengths to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIG_DIR / "fig11_agp_scaling.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    l_values = list(range(args.l_min, args.l_max + 1, 2))
    if not l_values:
        raise ValueError("No system sizes selected.")

    results = compute_curve(l_values, args.delta, args.j2)
    plot_results(results, args.output)

    print(f"Saved {args.output}")
    for j2, points in results.items():
        last_l, last_value = points[-1]
        print(f"J2={j2:g}: L={last_l}, ||A_Delta||^2/L={last_value:.6g}")


if __name__ == "__main__":
    main()