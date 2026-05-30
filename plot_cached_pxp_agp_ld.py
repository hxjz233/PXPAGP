"""Temporary helper to replot cached PXP AGP data as ||A||^2/(L D).

The cached hxz results produced by pxp_agp_scaling.py store ||A||^2 / L.
This script reads one such cache, reconstructs the Hilbert-space dimension D
for each L, and plots ||A||^2 / (L D) instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "PXP_infra"))

from diag import GetBasis


def load_cached_hxz(cache_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    arr = np.load(cache_path)
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ls = sorted({int(key.split("_")[1]) for key in arr.keys() if key.endswith("_hxz")})
    for l in ls:
        hxz = np.asarray(arr[f"L_{l}_hxz"], dtype=float)
        vals = np.asarray(arr[f"L_{l}_vals"], dtype=float)
        results[l] = (hxz, vals)
    return results


def hilbert_space_dimension(l: int, symmetry: tuple[bool, bool], boundary: str) -> int:
    basis = GetBasis(dict(N=l, hxz=0.0, sym=symmetry, model="PXPZ", bound=boundary))
    return int(basis.Ns)


def plot_ld_scaled(
    results: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    output: Path,
    boundary: str,
    symmetry: tuple[bool, bool],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for l, (hxz, vals_over_l) in sorted(results.items()):
        d = hilbert_space_dimension(l, symmetry, boundary)
        vals_over_ld = vals_over_l / float(d)
        ax.semilogy(hxz, vals_over_ld, marker="o", linewidth=2.0, markersize=5, label=fr"$L={l}$")

    ax.set_xlabel(r"Coupling $h_{xz}$")
    ax.set_ylabel(r"$\|A_{h_{xz}}\|^2 / (L D)$")
    # ax.set_title(r"Cached PXPZ AGP rescaled by Hilbert-space dimension $D$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache",
        nargs="?",
        type=Path,
        default=Path("res"),
        help="Path to a cached hxz .npz file or directory containing pxp_agp_hxz_*.npz files. Default: res/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path for single-cache mode. If omitted when processing multiple caches, files are written to --out-dir.",
    )
    parser.add_argument(
        "--boundary",
        choices=["OBC", "PBC"],
        default="OBC",
        help="Boundary condition used when the cache was generated.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("res"),
        help="Output directory when processing multiple caches. Default: res/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symmetry = (False, False)
    cache_path = Path(args.cache)
    if cache_path.is_dir():
        pattern = cache_path / "pxp_agp_hxz_*.npz"
        files = sorted(cache_path.glob("pxp_agp_hxz_*.npz"))
        if not files:
            raise FileNotFoundError(f"No pxp_agp_hxz_*.npz files found in {cache_path}")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            results = load_cached_hxz(f)
            out_path = out_dir / f"{f.stem}_ld.png"
            print(f"Processing cache {f} -> {out_path}")
            plot_ld_scaled(results, output=out_path, boundary=args.boundary, symmetry=symmetry)
    else:
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file {cache_path} not found")
        results = load_cached_hxz(cache_path)
        out = args.output or (cache_path.parent / f"{cache_path.stem}_ld.png")
        plot_ld_scaled(results, output=out, boundary=args.boundary, symmetry=symmetry)


if __name__ == "__main__":
    main()