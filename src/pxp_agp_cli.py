from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from pxp_agp_common import (
    FIG_DIR,
    cache_params_with_inv_sector,
    fig_output_path,
    get_perturbation_spec,
    hxz_shard_cache_path,
    make_cache_path,
    shard_hxz_values,
)
from pxp_agp_plots import (
    plot_pxp_agp_normalized_log_series,
    plot_pxp_agp_series,
    plot_pxp_agp_size_series,
    plot_pxp_chi_typ_series,
    plot_pxp_spacing_series,
    plot_pxp_spectral_series,
)
from pxp_agp_series import (
    compute_pxp_agp_series,
    compute_pxp_agp_size_series,
    compute_pxp_chi_typ_series,
    compute_pxp_spacing_series,
    compute_pxp_spectral_series,
    merge_hxz_results,
    load_chi_typ_results,
    load_hxz_results,
    load_size_results,
    load_spectral_results,
    load_spacing_results,
    save_chi_typ_results,
    save_hxz_results,
    save_size_results,
    save_spectral_results,
    save_spacing_results,
)

CPUNUM = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["hxz", "z", "zz", "ss", "size", "spectral", "spacing", "typical"],
        default="hxz",
        help="Plot AGP versus a perturbation coupling, AGP versus L, standalone spectral function, or mean level-spacing ratio.",
    )
    parser.add_argument("--l-values", type=int, nargs="+", default=[10, 12, 14, 16], help="System sizes to use.")
    parser.add_argument("--hxz-fixed", type=float, default=0.0, help="Fixed hxz value for size-scaling mode.")
    parser.add_argument("--hxz-min", type=float, default=0.0, help="Minimum hxz value in the sweep.")
    parser.add_argument("--hxz-max", type=float, default=0.25, help="Maximum hxz value in the sweep.")
    parser.add_argument("--hxz-count", type=int, default=9, help="Number of hxz points in the sweep.")
    parser.add_argument("--z-min", type=float, default=0.0, help="Minimum z-coupling value in the sweep.")
    parser.add_argument("--z-max", type=float, default=0.25, help="Maximum z-coupling value in the sweep.")
    parser.add_argument("--z-count", type=int, default=9, help="Number of z-coupling points in the sweep.")
    parser.add_argument("--zz-min", type=float, default=0.0, help="Minimum zz-coupling value in the sweep.")
    parser.add_argument("--zz-max", type=float, default=0.25, help="Maximum zz-coupling value in the sweep.")
    parser.add_argument("--zz-count", type=int, default=9, help="Number of zz-coupling points in the sweep.")
    parser.add_argument("--ss-min", type=float, default=0.0, help="Minimum ss-coupling value in the sweep.")
    parser.add_argument("--ss-max", type=float, default=0.25, help="Maximum ss-coupling value in the sweep.")
    parser.add_argument("--ss-count", type=int, default=9, help="Number of ss-coupling points in the sweep.")
    parser.add_argument(
        "--hxz-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used to split the hxz sweep across jobs.",
    )
    parser.add_argument(
        "--hxz-shard-count",
        type=int,
        default=1,
        help="Total number of hxz shards; values are assigned round-robin.",
    )
    parser.add_argument(
        "--collect-hxz-shards",
        action="store_true",
        help="Load all shard caches for the current hxz sweep and rebuild the full result.",
    )
    parser.add_argument("--l-min", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--l-max", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--boundary", choices=["OBC", "PBC"], default="OBC", help="Boundary conditions.")
    parser.add_argument(
        "--inv-sector",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional inversion-parity sector; pass 0 or 1 to use that symmetry sector.",
    )
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu", help="Linear-algebra backend.")
    parser.add_argument("--force", action="store_true", help="Bypass cached results and recompute.")
    parser.add_argument("--output", type=Path, default=FIG_DIR / "pxp_agp_scaling.png", help="Output image path.")
    parser.add_argument(
        "--spectral-output",
        type=Path,
        default=FIG_DIR / "pxp_spectral_weight.png",
        help="Output image path for the spectral-weight plot.",
    )
    parser.add_argument(
        "--spacing-output",
        type=Path,
        default=FIG_DIR / "pxp_spacing_ratio.png",
        help="Output image path for the level-spacing-ratio plot.",
    )
    parser.add_argument(
        "--chi-typ-output",
        type=Path,
        default=FIG_DIR / "pxp_chi_typ.png",
        help="Output image path for the typical-susceptibility plot.",
    )
    parser.add_argument(
        "--z-output",
        type=Path,
        default=FIG_DIR / "pxp_z_scaling.png",
        help="Output image path for the Z-perturbation plot.",
    )
    parser.add_argument(
        "--zz-output",
        type=Path,
        default=FIG_DIR / "pxp_zz_scaling.png",
        help="Output image path for the ZZ-perturbation plot.",
    )
    parser.add_argument(
        "--ss-output",
        type=Path,
        default=FIG_DIR / "pxp_ss_scaling.png",
        help="Output image path for the SS-perturbation plot.",
    )
    parser.add_argument(
        "--spectral-bins",
        type=int,
        default=200,
        help="Number of log-spaced frequency bins for the spectral-weight plot.",
    )
    parser.add_argument("--spectral-hxz", type=float, default=0.0, help="Fixed hxz value used for the spectral plot.")
    return parser.parse_args()


def _selected_l_values(args: argparse.Namespace) -> list[int]:
    if args.l_min is not None or args.l_max is not None:
        if args.l_min is None or args.l_max is None:
            raise ValueError("Specify both --l-min and --l-max, or use --l-values.")
        l_values = list(range(args.l_min, args.l_max + 1, 2))
    else:
        l_values = list(args.l_values)
    if not l_values:
        raise ValueError("No system sizes selected.")
    return l_values


def main() -> None:
    args = parse_args()
    print(f"Using {CPUNUM} CPU thread(s) for linear algebra")

    args.output = fig_output_path(args.output)
    args.spectral_output = fig_output_path(args.spectral_output)
    args.spacing_output = fig_output_path(args.spacing_output)
    args.chi_typ_output = fig_output_path(args.chi_typ_output)
    args.z_output = fig_output_path(args.z_output)
    args.zz_output = fig_output_path(args.zz_output)
    args.ss_output = fig_output_path(args.ss_output)

    symmetry = (False, args.inv_sector) if args.inv_sector is not None else (False, False)
    l_values = _selected_l_values(args)
    pxpz_spec = get_perturbation_spec("pxpz")
    z_spec = get_perturbation_spec("z")
    zz_spec = get_perturbation_spec("zz")
    ss_spec = get_perturbation_spec("ss")

    if args.mode == "hxz":
        all_hxz_values = np.linspace(args.hxz_min, args.hxz_max, args.hxz_count)
        if len(all_hxz_values) == 0:
            raise ValueError("No hxz values selected.")

        print(f"Sweeping hxz over {all_hxz_values[0]:.5f} to {all_hxz_values[-1]:.5f} in {len(all_hxz_values)} steps")
        params = dict(
            l_values=list(l_values),
            hxz_min=float(all_hxz_values[0]),
            hxz_max=float(all_hxz_values[-1]),
            hxz_count=int(len(all_hxz_values)),
            boundary=args.boundary,
        )
        params = cache_params_with_inv_sector(params, args.inv_sector)
        shard_count = int(args.hxz_shard_count)
        shard_index = int(args.hxz_shard_index)
        if shard_count > 1:
            print(f"Using hxz shard {shard_index + 1}/{shard_count}")

        if args.collect_hxz_shards:
            shard_paths = [hxz_shard_cache_path("hxz", params, i, shard_count) for i in range(shard_count)]
            print(f"Collecting {len(shard_paths)} hxz shard caches")
            results = merge_hxz_results(shard_paths)
            cache_path = make_cache_path("hxz", params)
            save_hxz_results(results, cache_path)
            print(f"Saved merged hxz cache to {cache_path}")
            plot_pxp_agp_series(results, args.output)
            log_output = args.output.parent / f"{args.output.stem}_fit{args.output.suffix}"
            plot_pxp_agp_normalized_log_series(
                results,
                log_output,
                perturbation_label=pxpz_spec.display_name,
                coupling_label=pxpz_spec.coupling_label,
            )
        else:
            hxz_values = shard_hxz_values(all_hxz_values, shard_index, shard_count)
            if len(hxz_values) == 0:
                raise ValueError("This hxz shard has no assigned points.")

            cache_path = hxz_shard_cache_path("hxz", params, shard_index, shard_count)
            if (not args.force) and cache_path.exists():
                print(f"Loading cached hxz results from {cache_path}")
                results = load_hxz_results(cache_path)
            else:
                results = compute_pxp_agp_series(
                    l_values,
                    symmetry=symmetry,
                    boundary=args.boundary,
                    coupling_values=hxz_values,
                    backend=args.backend,
                )
                save_hxz_results(results, cache_path)
                print(f"Saved shard cache to {cache_path}")

            if shard_count == 1:
                plot_pxp_agp_series(results, args.output)
                log_output = args.output.parent / f"{args.output.stem}_fit{args.output.suffix}"
                plot_pxp_agp_normalized_log_series(
                    results,
                    log_output,
                    perturbation_label=pxpz_spec.display_name,
                    coupling_label=pxpz_spec.coupling_label,
                )
            else:
                print("Shard mode skips plotting; run with --collect-hxz-shards to build the combined figure.")
    elif args.mode == "z":
        z_values = np.linspace(args.z_min, args.z_max, args.z_count)
        if len(z_values) == 0:
            raise ValueError("No z values selected.")

        print(f"Sweeping z over {z_values[0]:.5f} to {z_values[-1]:.5f} in {len(z_values)} steps")
        params = dict(
            l_values=list(l_values),
            coupling_min=float(z_values[0]),
            coupling_max=float(z_values[-1]),
            coupling_count=int(len(z_values)),
            boundary=args.boundary,
            perturbation=z_spec.cache_tag,
        )
        params = cache_params_with_inv_sector(params, args.inv_sector)
        cache_path = make_cache_path(z_spec.cache_tag, params)
        if (not args.force) and cache_path.exists():
            print(f"Loading cached z results from {cache_path}")
            results = load_hxz_results(cache_path)
        else:
            results = compute_pxp_agp_series(
                l_values,
                coupling_values=z_values,
                symmetry=symmetry,
                boundary=args.boundary,
                backend=args.backend,
                perturbation_kind=z_spec.kind,
            )
            save_hxz_results(results, cache_path)
            print(f"Saved z cache to {cache_path}")

        plot_pxp_agp_series(
            results,
            args.z_output,
            perturbation_label=z_spec.display_name,
            coupling_label=z_spec.coupling_label,
        )
        log_output = args.z_output.parent / f"{args.z_output.stem}_fit{args.z_output.suffix}"
        plot_pxp_agp_normalized_log_series(
            results,
            log_output,
            perturbation_label=z_spec.display_name,
            coupling_label=z_spec.coupling_label,
        )
    elif args.mode == "zz":
        zz_values = np.linspace(args.zz_min, args.zz_max, args.zz_count)
        if len(zz_values) == 0:
            raise ValueError("No zz values selected.")

        print(f"Sweeping zz over {zz_values[0]:.5f} to {zz_values[-1]:.5f} in {len(zz_values)} steps")
        params = dict(
            l_values=list(l_values),
            coupling_min=float(zz_values[0]),
            coupling_max=float(zz_values[-1]),
            coupling_count=int(len(zz_values)),
            boundary=args.boundary,
            perturbation=zz_spec.cache_tag,
        )
        params = cache_params_with_inv_sector(params, args.inv_sector)
        cache_path = make_cache_path(zz_spec.cache_tag, params)
        if (not args.force) and cache_path.exists():
            print(f"Loading cached zz results from {cache_path}")
            results = load_hxz_results(cache_path)
        else:
            results = compute_pxp_agp_series(
                l_values,
                coupling_values=zz_values,
                symmetry=symmetry,
                boundary=args.boundary,
                backend=args.backend,
                perturbation_kind=zz_spec.kind,
            )
            save_hxz_results(results, cache_path)
            print(f"Saved zz cache to {cache_path}")

        plot_pxp_agp_series(
            results,
            args.zz_output,
            perturbation_label=zz_spec.display_name,
            coupling_label=zz_spec.coupling_label,
        )
        log_output = args.zz_output.parent / f"{args.zz_output.stem}_fit{args.zz_output.suffix}"
        plot_pxp_agp_normalized_log_series(
            results,
            log_output,
            perturbation_label=zz_spec.display_name,
            coupling_label=zz_spec.coupling_label,
        )
    elif args.mode == "ss":
        ss_values = np.linspace(args.ss_min, args.ss_max, args.ss_count)
        if len(ss_values) == 0:
            raise ValueError("No ss values selected.")

        print(f"Sweeping ss over {ss_values[0]:.5f} to {ss_values[-1]:.5f} in {len(ss_values)} steps")
        params = dict(
            l_values=list(l_values),
            coupling_min=float(ss_values[0]),
            coupling_max=float(ss_values[-1]),
            coupling_count=int(len(ss_values)),
            boundary=args.boundary,
            perturbation=ss_spec.cache_tag,
        )
        params = cache_params_with_inv_sector(params, args.inv_sector)
        cache_path = make_cache_path(ss_spec.cache_tag, params)
        if (not args.force) and cache_path.exists():
            print(f"Loading cached ss results from {cache_path}")
            results = load_hxz_results(cache_path)
        else:
            results = compute_pxp_agp_series(
                l_values,
                coupling_values=ss_values,
                symmetry=symmetry,
                boundary=args.boundary,
                backend=args.backend,
                perturbation_kind=ss_spec.kind,
            )
            save_hxz_results(results, cache_path)
            print(f"Saved ss cache to {cache_path}")

        plot_pxp_agp_series(
            results,
            args.ss_output,
            perturbation_label=ss_spec.display_name,
            coupling_label=ss_spec.coupling_label,
        )
        log_output = args.ss_output.parent / f"{args.ss_output.stem}_fit{args.ss_output.suffix}"
        plot_pxp_agp_normalized_log_series(
            results,
            log_output,
            perturbation_label=ss_spec.display_name,
            coupling_label=ss_spec.coupling_label,
        )
    elif args.mode == "size":
        print(f"Computing PXP AGP norm versus system size for hxz={args.hxz_fixed:.5f} and L = {l_values}")
        params = cache_params_with_inv_sector(
            dict(l_values=list(l_values), hxz=float(args.hxz_fixed), boundary=args.boundary),
            args.inv_sector,
        )
        cache_path = make_cache_path("size", params)
        if (not args.force) and cache_path.exists():
            print(f"Loading cached size results from {cache_path}")
            results = load_size_results(cache_path)
        else:
            results = compute_pxp_agp_size_series(
                l_values,
                coupling=args.hxz_fixed,
                symmetry=symmetry,
                boundary=args.boundary,
                backend=args.backend,
            )
            save_size_results(results, cache_path)

        plot_pxp_agp_size_series(results, args.output)
    elif args.mode == "spectral":
        spectral_hxz = float(args.spectral_hxz)
        spectral_bins = int(args.spectral_bins)

        print(f"Computing PXP spectral weight for hxz={spectral_hxz:.5f} and L = {l_values}")
        spectral_params = dict(l_values=list(l_values), hxz=spectral_hxz, bins=spectral_bins, boundary=args.boundary)
        spectral_params = cache_params_with_inv_sector(spectral_params, args.inv_sector)
        spectral_cache_path = make_cache_path("spectral_hxz", spectral_params)
        if (not args.force) and spectral_cache_path.exists():
            print(f"Loading cached spectral results from {spectral_cache_path}")
            spectral_results = load_spectral_results(spectral_cache_path)
        else:
            spectral_results = compute_pxp_spectral_series(
                l_values,
                coupling=spectral_hxz,
                bins=spectral_bins,
                symmetry=symmetry,
                boundary=args.boundary,
                backend=args.backend,
            )
            save_spectral_results(spectral_results, spectral_cache_path)

        plot_pxp_spectral_series(spectral_results, args.spectral_output)
    elif args.mode == "spacing":
        hxz_values = np.linspace(args.hxz_min, args.hxz_max, args.hxz_count)
        if len(hxz_values) == 0:
            raise ValueError("No hxz values selected.")

        print(f"Computing mean level spacing ratio for L = {l_values}")
        print(f"Sweeping hxz over {hxz_values[0]:.5f} to {hxz_values[-1]:.5f} in {len(hxz_values)} steps")
        spacing_params = dict(
            l_values=list(l_values),
            hxz_min=float(hxz_values[0]),
            hxz_max=float(hxz_values[-1]),
            hxz_count=int(len(hxz_values)),
            boundary=args.boundary,
            symmetry=[False, 0],
        )
        spacing_cache_path = make_cache_path("spacing", spacing_params)
        if (not args.force) and spacing_cache_path.exists():
            print(f"Loading cached spacing results from {spacing_cache_path}")
            spacing_results = load_spacing_results(spacing_cache_path)
        else:
            spacing_results = compute_pxp_spacing_series(
                l_values,
                coupling_values=hxz_values,
                symmetry=(False, 0),
                boundary=args.boundary,
                backend=args.backend,
            )
            save_spacing_results(spacing_results, spacing_cache_path)

        plot_pxp_spacing_series(spacing_results, args.spacing_output)
    elif args.mode == "typical":
        hxz_values = np.linspace(args.hxz_min, args.hxz_max, args.hxz_count)
        if len(hxz_values) == 0:
            raise ValueError("No hxz values selected.")

        print(f"Computing typical susceptibility for L = {l_values}")
        print(f"Sweeping hxz over {hxz_values[0]:.5f} to {hxz_values[-1]:.5f} in {len(hxz_values)} steps")
        chi_params = dict(
            l_values=list(l_values),
            hxz_min=float(hxz_values[0]),
            hxz_max=float(hxz_values[-1]),
            hxz_count=int(len(hxz_values)),
            boundary=args.boundary,
        )
        chi_params = cache_params_with_inv_sector(chi_params, args.inv_sector)
        chi_cache_path = make_cache_path("chi_typ", chi_params)
        if (not args.force) and chi_cache_path.exists():
            print(f"Loading cached chi_typ results from {chi_cache_path}")
            chi_results = load_chi_typ_results(chi_cache_path)
        else:
            chi_results = compute_pxp_chi_typ_series(
                l_values,
                coupling_values=hxz_values,
                symmetry=symmetry,
                boundary=args.boundary,
                backend=args.backend,
            )
            save_chi_typ_results(chi_results, chi_cache_path)

        plot_pxp_chi_typ_series(chi_results, args.chi_typ_output)
