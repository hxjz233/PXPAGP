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
import os
import json
import hashlib


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


def typical_susceptibility_pxp(
    h_base: np.ndarray,
    dh_dhxz: np.ndarray
) -> float:
    """Compute the typical susceptibility over eigenstates.

    chi_n = sum_{m != n} |<n|dH/dhxz|m>|^2 / (E_n - E_m)^2
    chi_typ = exp(average(log(chi_n)))
    """

    evals, evecs = np.linalg.eigh(h_base)
    dh_eig = evecs.conj().T @ dh_dhxz @ evecs

    omega = evals[:, None] - evals[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_matrix = np.abs(dh_eig) ** 2 / (omega * omega)
    np.fill_diagonal(chi_matrix, 0.0)

    chi_n = np.sum(chi_matrix, axis=1)
    chi_n = np.asarray(chi_n, dtype=float)
    chi_n = chi_n[np.isfinite(chi_n) & (chi_n > 0.0)]
    if chi_n.size == 0:
        return float("nan")

    return float(np.exp(np.mean(np.log(chi_n))))


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


def compute_pxp_chi_typ_series(
    l_values: Iterable[int],
    hxz_values: Iterable[float],
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
) -> dict[int, list[tuple[float, float]]]:
    """Compute typical susceptibility versus hxz for several fixed system sizes."""

    results: dict[int, list[tuple[float, float]]] = {l: [] for l in l_values}

    for l in l_values:
        gen_dict = dict(N=l, hxz=0.0, sym=symmetry, model="PXPZ", bound=boundary)
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

        for hxz in hxz_values:
            gen_dict["hxz"] = hxz
            h_base = GetHam(gen_dict, basis)
            h_base_dense = np.asarray(h_base.toarray(), dtype=np.float64)

            chi_typ = typical_susceptibility_pxp(h_base_dense, dh_dhxz_dense)
            chi_typ_per_l = chi_typ / l
            results[l].append((hxz, chi_typ_per_l))
            print(f"L={l:2d}, hxz={hxz: .5f}, D={basis_dim:6d}, chi_typ/L={chi_typ_per_l:.6e}")

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


def _make_cache_path(mode: str, params: dict) -> Path:
    cache_dir = Path("res")
    cache_dir.mkdir(exist_ok=True)
    # create deterministic json and hash it for filename
    j = json.dumps(params, sort_keys=True)
    h = hashlib.sha1(j.encode()).hexdigest()[:16]
    return cache_dir / f"pxp_agp_{mode}_{h}.npz"


def save_hxz_results(results: dict[int, list[tuple[float, float]]], cache_path: Path) -> None:
    # convert to arrays per L
    npz_dict = {}
    for l, points in results.items():
        hxz = np.array([p[0] for p in points], dtype=float)
        vals = np.array([p[1] for p in points], dtype=float)
        npz_dict[f"L_{l}_hxz"] = hxz
        npz_dict[f"L_{l}_vals"] = vals
    np.savez_compressed(cache_path, **npz_dict)


def load_hxz_results(cache_path: Path) -> dict[int, list[tuple[float, float]]]:
    arr = np.load(cache_path)
    results = {}
    # keys are L_{l}_hxz and L_{l}_vals
    keys = list(arr.keys())
    ls = sorted(set(k.split("_")[1] for k in keys))
    for l in ls:
        hxz = arr[f"L_{l}_hxz"]
        vals = arr[f"L_{l}_vals"]
        results[int(l)] = list(zip(hxz.tolist(), vals.tolist()))
    return results


def save_chi_typ_results(results: dict[int, list[tuple[float, float]]], cache_path: Path) -> None:
    npz_dict = {}
    for l, points in results.items():
        hxz = np.array([p[0] for p in points], dtype=float)
        vals = np.array([p[1] for p in points], dtype=float)
        npz_dict[f"L_{l}_hxz"] = hxz
        npz_dict[f"L_{l}_vals"] = vals
    np.savez_compressed(cache_path, **npz_dict)


def load_chi_typ_results(cache_path: Path) -> dict[int, list[tuple[float, float]]]:
    arr = np.load(cache_path)
    results = {}
    keys = list(arr.keys())
    ls = sorted(set(k.split("_")[1] for k in keys))
    for l in ls:
        hxz = arr[f"L_{l}_hxz"]
        vals = arr[f"L_{l}_vals"]
        results[int(l)] = list(zip(hxz.tolist(), vals.tolist()))
    return results


def save_size_results(results: list[tuple[int, float, int]], cache_path: Path) -> None:
    L = np.array([r[0] for r in results], dtype=int)
    vals = np.array([r[1] for r in results], dtype=float)
    dims = np.array([r[2] for r in results], dtype=int)
    np.savez_compressed(cache_path, L=L, vals=vals, dims=dims)


def load_size_results(cache_path: Path) -> list[tuple[int, float, int]]:
    arr = np.load(cache_path)
    L = arr["L"]
    vals = arr["vals"]
    dims = arr["dims"]
    return [(int(L[i]), float(vals[i]), int(dims[i])) for i in range(len(L))]


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


def plot_pxp_chi_typ_series(results: dict[int, list[tuple[float, float]]], output_path: Path) -> None:
    """Plot typical susceptibility versus hxz for several fixed system sizes."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for l, points in results.items():
        hxz_values = np.array([p[0] for p in points], dtype=float)
        chi_values = np.array([p[1] for p in points], dtype=float)
        ax.semilogy(hxz_values, chi_values, marker="o", linewidth=2.0, markersize=6, label=fr"$L={l}$")

    ax.set_xlabel(r"Coupling $h_{xz}$")
    ax.set_ylabel(r"$\chi_{\mathrm{typ}}$")
    ax.set_title("PXPZ model: typical susceptibility versus $h_{xz}$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def plot_pxp_agp_normalized_log_series(results: dict[int, list[tuple[float, float]]], output_path: Path) -> None:
    """Plot slope and intercept of log(AGP/L) vs L at each hxz.
    
    For each hxz value, fit log(AGP/L) = slope * L + intercept across all system sizes,
    then plot slope and intercept as functions of hxz.
    """
    
    # Reorganize results by hxz point
    hxz_to_data = {}
    for l, points in results.items():
        for hxz, agp_val in points:
            if hxz not in hxz_to_data:
                hxz_to_data[hxz] = []
            hxz_to_data[hxz].append((l, agp_val))
    
    slopes = []
    intercepts = []
    hxz_values = sorted(hxz_to_data.keys())
    
    # For each hxz, fit log(AGP/L) vs L
    for hxz in hxz_values:
        l_vals = np.array([x[0] for x in hxz_to_data[hxz]], dtype=float)
        agp_vals = np.array([x[1] for x in hxz_to_data[hxz]], dtype=float)
        log_agp_vals = np.log(agp_vals)
        
        # Fit: log(AGP/L) = slope * L + intercept
        slope, intercept = np.polyfit(l_vals, log_agp_vals, 1)
        slopes.append(slope)
        intercepts.append(intercept)
    
    hxz_array = np.array(hxz_values, dtype=float)
    slopes = np.array(slopes, dtype=float)
    intercepts = np.array(intercepts, dtype=float)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 4.8), constrained_layout=True)
    
    # Plot slope vs hxz
    ax1.plot(hxz_array, slopes, marker="o", linewidth=2.0, markersize=6, color="C0")
    ax1.axhline(np.log((np.sqrt(5)+1)/2), color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_xlabel(r"Coupling $h_{xz}$")
    ax1.set_ylabel(r"Slope of $\log(\|A_{hxz}\|^2 / L)$ vs $L$")
    ax1.set_title("Slope of log(AGP/L) vs system size")
    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    
    # Plot intercept vs hxz
    ax2.plot(hxz_array, intercepts, marker="s", linewidth=2.0, markersize=6, color="C1")
    ax2.set_xlabel(r"Coupling $h_{xz}$")
    ax2.set_ylabel(r"Intercept of $\log(\|A_{hxz}\|^2 / L)$ vs $L$")
    ax2.set_title("Intercept of log(AGP/L) vs system size")
    ax2.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")
    
    # Print summary
    print("\nSlope and intercept summary:")
    for i, hxz in enumerate(hxz_array):
        print(f"hxz={hxz:.5f}: slope={slopes[i]:.6e}, intercept={intercepts[i]:.6f}")


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
        "--force",
        action="store_true",
        help="Bypass cached results and recompute.",
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
        params = dict(l_values=list(l_values), hxz_min=float(hxz_values[0]), hxz_max=float(hxz_values[-1]), hxz_count=int(len(hxz_values)), boundary=args.boundary)
        cache_path = _make_cache_path("hxz", params)
        if (not args.force) and cache_path.exists():
            print(f"Loading cached hxz results from {cache_path}")
            results = load_hxz_results(cache_path)
        else:
            results = compute_pxp_agp_series(
                l_values,
                symmetry=(False, False),
                boundary=args.boundary,
                hxz_values=hxz_values,
            )
            save_hxz_results(results, cache_path)

        plot_pxp_agp_series(results, args.output)
        
        # Also produce log(AGP/L) vs hxz plot
        log_output = args.output.parent / f"{args.output.stem}_log{args.output.suffix}"
        plot_pxp_agp_normalized_log_series(results, log_output)

        chi_typ_params = dict(
            l_values=list(l_values),
            hxz_min=float(hxz_values[0]),
            hxz_max=float(hxz_values[-1]),
            hxz_count=int(len(hxz_values)),
            boundary=args.boundary,
        )
        chi_typ_cache_path = _make_cache_path("chi_typ", chi_typ_params)
        if (not args.force) and chi_typ_cache_path.exists():
            print(f"Loading cached chi_typ results from {chi_typ_cache_path}")
            chi_typ_results = load_chi_typ_results(chi_typ_cache_path)
        else:
            chi_typ_results = compute_pxp_chi_typ_series(
                l_values,
                symmetry=(False, False),
                boundary=args.boundary,
                hxz_values=hxz_values,
            )
            save_chi_typ_results(chi_typ_results, chi_typ_cache_path)

        chi_typ_output = args.output.parent / f"{args.output.stem}_chi_typ{args.output.suffix}"
        plot_pxp_chi_typ_series(chi_typ_results, chi_typ_output)
    else:
        print(f"Computing PXP AGP norm versus system size for hxz={args.hxz_fixed:.5f} and L = {l_values}")
        params = dict(l_values=list(l_values), hxz=float(args.hxz_fixed), boundary=args.boundary)
        cache_path = _make_cache_path("size", params)
        if (not args.force) and cache_path.exists():
            print(f"Loading cached size results from {cache_path}")
            results = load_size_results(cache_path)
        else:
            results = compute_pxp_agp_size_series(
                l_values,
                hxz=args.hxz_fixed,
                symmetry=(False, False),
                boundary=args.boundary,
            )
            save_size_results(results, cache_path)

        plot_pxp_agp_size_series(results, args.output)


if __name__ == "__main__":
    main()
