"""Compute and plot AGP norm for the PXP model (PXPZ with hxz=0).

The AGP is computed with respect to the hxz coupling parameter, which couples
remote Z and X operators. The script uses the PXP_infra infrastructure for
diagonalization and scales to multiple system sizes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from quspin.operators import hamiltonian

# Add PXP_infra to path
sys.path.insert(0, str(Path(__file__).parent / "PXP_infra"))

from diag import GetBasis, GetHam, Diag
from sym_kin import op_dict


def regularized_agp_norm_pxp(
    h_base: np.ndarray,
    dh_dhxz: np.ndarray,
    mu: float,
) -> float:
    """Compute regularized AGP norm for hxz coupling.

    The AGP is computed as the norm of the response to changing hxz:
        ||A_hxz||^2 = (1/D) sum_{n != m} |<n|dH/dhxz|m>|^2 * w_nm^2 / (w_nm^2 + mu^2)^2

    Parameters:
        h_base: The Hamiltonian matrix (at hxz=0 or the base point)
        dh_dhxz: The operator representing dH/dhxz
        mu: Regularization cutoff parameter
    """

    # Diagonalize the base Hamiltonian
    evals, evecs = np.linalg.eigh(h_base)
    dh_eig = evecs.conj().T @ dh_dhxz @ evecs

    # Compute the weighted sum (regularized AGP formula)
    omega = evals[:, None] - evals[None, :]
    weight = (omega * omega) / (omega * omega + mu * mu) ** 2
    np.fill_diagonal(weight, 0.0)

    norm = np.sum(np.abs(dh_eig) ** 2 * weight) / h_base.shape[0]
    return float(np.real_if_close(norm))


def compute_pxp_agp(
    l_values: Iterable[int],
    hxz: float = 0.0,
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
) -> list[tuple[int, float, int]]:
    """Compute AGP norm for PXP model at multiple system sizes.

    The AGP is computed with respect to the hxz parameter. Since the ZX and XZ
    terms in the Hamiltonian are proportional to hxz, dH/dhxz is simply the sum
    of those two operators with unit coefficient.

    Returns list of (L, ||A_hxz||^2 / L, basis_dim) tuples.
    """

    results = []

    for l in l_values:
        gen_dict = dict(N=l, hxz=hxz, sym=symmetry, model="PXPZ", bound=boundary)
        basis = GetBasis(gen_dict)
        basis_dim = basis.Ns

        # Construct base Hamiltonian at hxz value
        h_base = GetHam(gen_dict, basis)
        h_base_dense = np.asarray(h_base.toarray(), dtype=np.float64)

        # Construct dH/dhxz operator (the ZX and XZ terms with coefficient 1.0)
        isPBC = 1 if boundary == "PBC" else 0
        if isPBC:
            zx_list = [[1.0, i, (i + 2) % l] for i in range(l)]
            xz_list = [[1.0, (i - 2) % l, i] for i in range(l)]
        else:
            zx_list = [[1.0, i - 2, i] for i in range(2, l)]
            xz_list = [[1.0, i, i + 2] for i in range(l - 2)]

        dh_dhxz = hamiltonian(
            [["zx", zx_list], ["xz", xz_list]],
            [],
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_pcon=False,
            check_herm=False,
        )
        dh_dhxz_dense = np.asarray(dh_dhxz.toarray(), dtype=np.float64)

        # Compute cutoff as in the paper: mu = L / D
        mu = l / basis_dim

        # Compute regularized AGP norm
        norm_sq = regularized_agp_norm_pxp(h_base_dense, dh_dhxz_dense, mu)
        norm_sq_per_l = norm_sq / l

        results.append((l, norm_sq_per_l, basis_dim))
        print(f"L={l:2d}, D={basis_dim:6d}, ||A_hxz||^2/L={norm_sq_per_l:.6e}")

    return results


def plot_pxp_agp(results: list[tuple[int, float, int]], output_path: Path) -> None:
    """Plot AGP scaling for PXP model."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    lengths = np.array([r[0] for r in results], dtype=float)
    values = np.array([r[1] for r in results], dtype=float)

    ax.semilogy(lengths, values, marker="o", linewidth=2.0, markersize=8, label="PXP AGP norm")

    # Add an exponential fit if we have enough points
    if len(results) >= 3:
        tail_lengths = lengths[-3:]
        tail_log_values = np.log(values[-3:])
        slope, intercept = np.polyfit(tail_lengths, tail_log_values, 1)
        fit_values = np.exp(intercept + slope * lengths)
        ax.semilogy(lengths, fit_values, linestyle="--", linewidth=1.2, alpha=0.7, label="Exponential fit")
        print(f"Exponential slope (last 3 points): {slope:.4f}")

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\|A_{hxz}\|^2 / L$")
    ax.set_title("PXP model: Regularized AGP norm scaling")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--l-min",
        type=int,
        default=10,
        help="Smallest system size to include.",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=18,
        help="Largest system size to include.",
    )
    parser.add_argument(
        "--hxz",
        type=float,
        default=0.0,
        help="Base hxz coupling for AGP computation (usually 0 at integrable point).",
    )
    parser.add_argument(
        "--boundary",
        choices=["OBC", "PBC"],
        default="OBC",
        help="Boundary conditions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pxp_agp_scaling.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    l_values = list(range(args.l_min, args.l_max + 1, 2))
    if not l_values:
        raise ValueError("No system sizes selected.")

    print(f"Computing PXP AGP norm for L = {l_values}")
    results = compute_pxp_agp(
        l_values,
        hxz=args.hxz,
        symmetry=(False, False),
        boundary=args.boundary,
    )

    plot_pxp_agp(results, args.output)


if __name__ == "__main__":
    main()
