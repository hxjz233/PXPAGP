"""Compute and plot AGP norm for the PXPZ model.

The script uses the PXP_infra infrastructure for exact diagonalization of the
PXPZ Hamiltonian and computes the regularized AGP norm with respect to hxz.
It can plot either AGP versus hxz for several fixed system sizes L, or the
previous system-size scaling curve at fixed hxz.

size sweep: python pxp_agp_scaling.py --mode size --l-values 10 12 --hxz-fixed 0.0
hxz series: python pxp_agp_scaling.py --mode hxz --l-values 10 12 --hxz-min 0.0 --hxz-max 0.1 --hxz-count 3
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

from diag import GetBasis, GetHam


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


def compute_pxp_agp_series(
    l_values: Iterable[int],
    hxz_values: Iterable[float],
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
) -> dict[int, list[tuple[float, float]]]:
    """Compute AGP norm versus hxz for several fixed system sizes.

    The AGP is computed with respect to the hxz parameter. Since the ZX and XZ
    terms in the Hamiltonian are proportional to hxz, dH/dhxz is simply the sum
    of those two operators with unit coefficient.

    Returns a mapping L -> list[(hxz, ||A_hxz||^2 / L)].
    """

    results: dict[int, list[tuple[float, float]]] = {l: [] for l in l_values}

    for l in l_values:
        gen_dict = dict(N=l, hxz=0.0, sym=symmetry, model="PXPZ", bound=boundary)
        basis = GetBasis(gen_dict)
        basis_dim = basis.Ns

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

        for hxz in hxz_values:
            gen_dict["hxz"] = hxz
            h_base = GetHam(gen_dict, basis)
            h_base_dense = np.asarray(h_base.toarray(), dtype=np.float64)

            # Compute regularized AGP norm at this hxz value.
            norm_sq = regularized_agp_norm_pxp(h_base_dense, dh_dhxz_dense, mu)
            norm_sq_per_l = norm_sq / l

            results[l].append((hxz, norm_sq_per_l))
            print(f"L={l:2d}, hxz={hxz: .5f}, D={basis_dim:6d}, ||A_hxz||^2/L={norm_sq_per_l:.6e}")

    return results


def compute_pxp_agp_size_series(
    l_values: Iterable[int],
    hxz: float = 0.0,
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
) -> list[tuple[int, float, int]]:
    """Compute AGP norm versus system size at a fixed hxz value.

    Returns list of (L, ||A_hxz||^2 / L, basis_dim) tuples.
    """

    results: list[tuple[int, float, int]] = []

    for l in l_values:
        gen_dict = dict(N=l, hxz=hxz, sym=symmetry, model="PXPZ", bound=boundary)
        basis = GetBasis(gen_dict)
        basis_dim = basis.Ns

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

        h_base = GetHam(gen_dict, basis)
        h_base_dense = np.asarray(h_base.toarray(), dtype=np.float64)

        mu = l / basis_dim
        norm_sq = regularized_agp_norm_pxp(h_base_dense, dh_dhxz_dense, mu)
        norm_sq_per_l = norm_sq / l

        results.append((l, norm_sq_per_l, basis_dim))
        print(f"L={l:2d}, hxz={hxz: .5f}, D={basis_dim:6d}, ||A_hxz||^2/L={norm_sq_per_l:.6e}")

    return results


def plot_pxp_agp_series(results: dict[int, list[tuple[float, float]]], output_path: Path) -> None:
    """Plot AGP versus hxz for several fixed system sizes."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for l, points in results.items():
        hxz_values = np.array([p[0] for p in points], dtype=float)
        agp_values = np.array([p[1] for p in points], dtype=float)
        ax.semilogy(hxz_values, agp_values, marker="o", linewidth=2.0, markersize=6, label=fr"$L={l}$")

    ax.set_xlabel(r"Coupling $h_{xz}$")
    ax.set_ylabel(r"$\|A_{hxz}\|^2 / L$")
    ax.set_title("PXPZ model: regularized AGP versus $h_{xz}$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def plot_pxp_agp_size_series(results: list[tuple[int, float, int]], output_path: Path) -> None:
    """Plot AGP versus system size at fixed hxz."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    lengths = np.array([r[0] for r in results], dtype=float)
    values = np.array([r[1] for r in results], dtype=float)

    ax.semilogy(lengths, values, marker="o", linewidth=2.0, markersize=8, label=r"$\|A_{h_{xz}}\|^2/L$")

    if len(results) >= 3:
        tail_lengths = lengths[-3:]
        tail_log_values = np.log(values[-3:])
        slope, intercept = np.polyfit(tail_lengths, tail_log_values, 1)
        fit_values = np.exp(intercept + slope * lengths)
        ax.semilogy(lengths, fit_values, linestyle="--", linewidth=1.2, alpha=0.7, label="Exponential fit")
        print(f"Exponential slope (last 3 points): {slope:.4f}")

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\|A_{h_{xz}}\|^2 / L$")
    ax.set_title("PXPZ model: regularized AGP scaling with system size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["hxz", "size"],
        default="hxz",
        help="Plot AGP versus hxz at fixed L values or AGP versus L at fixed hxz.",
    )
    parser.add_argument(
        "--l-values",
        type=int,
        nargs="+",
        default=[10, 12, 14, 16],
        help="System sizes to use.",
    )
    parser.add_argument(
        "--hxz-fixed",
        type=float,
        default=0.0,
        help="Fixed hxz value for size-scaling mode.",
    )
    parser.add_argument(
        "--hxz-min",
        type=float,
        default=0.0,
        help="Minimum hxz value in the sweep.",
    )
    parser.add_argument(
        "--hxz-max",
        type=float,
        default=0.25,
        help="Maximum hxz value in the sweep.",
    )
    parser.add_argument(
        "--hxz-count",
        type=int,
        default=9,
        help="Number of hxz points in the sweep.",
    )
    parser.add_argument(
        "--l-min",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
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

    if args.l_min is not None or args.l_max is not None:
        if args.l_min is None or args.l_max is None:
            raise ValueError("Specify both --l-min and --l-max, or use --l-values.")
        l_values = list(range(args.l_min, args.l_max + 1, 2))
    else:
        l_values = list(args.l_values)
    if not l_values:
        raise ValueError("No system sizes selected.")

    if args.mode == "hxz":
        hxz_values = np.linspace(args.hxz_min, args.hxz_max, args.hxz_count)
        if len(hxz_values) == 0:
            raise ValueError("No hxz values selected.")

        print(f"Computing PXP AGP norm for L = {l_values}")
        print(f"Sweeping hxz over {hxz_values[0]:.5f} to {hxz_values[-1]:.5f} in {len(hxz_values)} steps")
        results = compute_pxp_agp_series(
            l_values,
            symmetry=(False, False),
            boundary=args.boundary,
            hxz_values=hxz_values,
        )
        plot_pxp_agp_series(results, args.output)
    else:
        print(f"Computing PXP AGP norm versus system size for hxz={args.hxz_fixed:.5f} and L = {l_values}")
        results = compute_pxp_agp_size_series(
            l_values,
            hxz=args.hxz_fixed,
            symmetry=(False, False),
            boundary=args.boundary,
        )
        plot_pxp_agp_size_series(results, args.output)


if __name__ == "__main__":
    main()
