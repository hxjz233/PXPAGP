from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from quspin.operators import hamiltonian

from pxp_agp_common import (
    build_model_hamiltonian_dense,
    get_array_module,
    get_perturbation_spec,
    load_grouped_xy_series,
    load_size_series,
    load_spectral_series,
    prepare_perturbation_context,
    save_grouped_xy_series,
    save_size_series,
    save_spectral_series,
    to_numpy,
)


def eigh_backend(matrix: np.ndarray, backend: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    xp = get_array_module(backend)
    if xp is np:
        evals, evecs = np.linalg.eigh(matrix)
    else:
        evals, evecs = xp.linalg.eigh(xp.asarray(matrix))
    return to_numpy(evals, backend), to_numpy(evecs, backend)


def eigvalsh_backend(matrix: np.ndarray, backend: str = "cpu") -> np.ndarray:
    xp = get_array_module(backend)
    if xp is np:
        evals = np.linalg.eigvalsh(matrix)
    else:
        evals = xp.linalg.eigvalsh(xp.asarray(matrix))
    return to_numpy(evals, backend)


def regularized_agp_norm_pxp(
    h_base: np.ndarray,
    dh_dhxz: np.ndarray,
    mu: float,
    backend: str = "cpu",
) -> float:
    xp = get_array_module(backend)
    h_base_xp = xp.asarray(h_base) if backend == "gpu" else h_base
    dh_dhxz_xp = xp.asarray(dh_dhxz) if backend == "gpu" else dh_dhxz

    evals, evecs = eigh_backend(h_base_xp, backend=backend)
    if backend == "gpu":
        evals = xp.asarray(evals)
        evecs = xp.asarray(evecs)
        dh_dhxz_xp = xp.asarray(dh_dhxz_xp)
    dh_eig = evecs.conj().T @ dh_dhxz_xp @ evecs

    omega = evals[:, None] - evals[None, :]
    weight = (omega * omega) / (omega * omega + mu * mu) ** 2
    xp.fill_diagonal(weight, 0.0)

    norm = xp.sum(xp.abs(dh_eig) ** 2 * weight) / h_base.shape[0]
    if backend == "gpu":
        norm = xp.asnumpy(norm)
    return float(np.real_if_close(norm))


def typical_susceptibility_pxp(
    h_base: np.ndarray,
    dh_dhxz: np.ndarray,
    backend: str = "cpu",
) -> float:
    xp = get_array_module(backend)
    h_base_xp = xp.asarray(h_base) if backend == "gpu" else h_base
    dh_dhxz_xp = xp.asarray(dh_dhxz) if backend == "gpu" else dh_dhxz

    evals, evecs = eigh_backend(h_base_xp, backend=backend)
    if backend == "gpu":
        evecs = xp.asarray(evecs)
        dh_dhxz_xp = xp.asarray(dh_dhxz_xp)
    dh_eig = evecs.conj().T @ dh_dhxz_xp @ evecs

    omega = evals[:, None] - evals[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_matrix = xp.abs(dh_eig) ** 2 / (omega * omega)
    xp.fill_diagonal(chi_matrix, 0.0)

    chi_n = xp.sum(chi_matrix, axis=1)
    chi_n = to_numpy(chi_n, backend).astype(float, copy=False)
    chi_n = chi_n[np.isfinite(chi_n) & (chi_n > 0.0)]
    if chi_n.size == 0:
        return float("nan")

    return float(np.exp(np.mean(np.log(chi_n))))


def spectral_weight_from_matrix(
    h: np.ndarray,
    dh: np.ndarray,
    mu: float,
    omega_grid: np.ndarray,
    backend: str = "cpu",
) -> np.ndarray:
    xp = get_array_module(backend)
    h_xp = xp.asarray(h) if backend == "gpu" else h
    dh_xp = xp.asarray(dh) if backend == "gpu" else dh
    omega_grid_xp = xp.asarray(omega_grid) if backend == "gpu" else omega_grid

    evals, evecs = eigh_backend(h_xp, backend=backend)
    if backend == "gpu":
        evals = xp.asarray(evals)
        evecs = xp.asarray(evecs)
        dh_xp = xp.asarray(dh_xp)
        omega_grid_xp = xp.asarray(omega_grid_xp)

    dh_eig = evecs.conj().T @ dh_xp @ evecs

    omegas = (evals[:, None] - evals[None, :]).reshape(-1)
    m2 = xp.abs(dh_eig) ** 2
    xp.fill_diagonal(m2, 0.0)
    m2_flat = m2.reshape(-1)

    D = h.shape[0]
    spectral = xp.zeros_like(omega_grid_xp, dtype=float)
    for idx, omega in enumerate(omega_grid_xp):
        x = omega - omegas
        lorentz = (1.0 / np.pi) * (mu / (x * x + mu * mu))
        spectral[idx] = (1.0 / D) * xp.sum(m2_flat * lorentz)

    return to_numpy(spectral, backend).astype(float, copy=False)


def mean_level_spacing_ratio(evals: np.ndarray) -> float:
    evals = np.asarray(evals, dtype=float)
    if evals.size < 3:
        return float("nan")

    evals = np.sort(evals)
    spacings = np.diff(evals)
    if spacings.size < 2:
        return float("nan")

    s1 = spacings[:-1]
    s2 = spacings[1:]
    denom = np.maximum(s1, s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.minimum(s1, s2) / denom
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan")
    return float(np.mean(ratios))


def mean_level_spacing_ratio_middle_third(evals: np.ndarray) -> float:
    evals = np.asarray(evals, dtype=float)
    if evals.size < 3:
        return float("nan")

    evals = np.sort(evals)
    start = evals.size // 3
    stop = (2 * evals.size) // 3
    middle = evals[start:stop]
    return mean_level_spacing_ratio(middle)


def _run_perturbation_series(
    l_values: Iterable[int],
    coupling_values: Iterable[float],
    *,
    symmetry: tuple,
    boundary: str,
    backend: str,
    perturbation_kind: str,
    value_fn: Callable[[np.ndarray, np.ndarray, float, int, int, str], float],
    series_name: str,
    include_base: bool = True,
) -> dict[int, list[tuple[float, float]]]:
    results: dict[int, list[tuple[float, float]]] = {l: [] for l in l_values}
    total_start = time.perf_counter()
    spec = get_perturbation_spec(perturbation_kind)

    for l in l_values:
        l_start = time.perf_counter()
        context = prepare_perturbation_context(l, symmetry, boundary, perturbation_kind)
        print(
            f"Computing {series_name} for L={l}, basis dimension D={context.basis_dim}, "
            f"perturbation={spec.display_name}"
        )

        for coupling in coupling_values:
            coupling_start = time.perf_counter()
            h_base_dense = build_model_hamiltonian_dense(
                context.basis,
                l,
                boundary,
                perturbation_kind,
                coupling,
                symmetry,
                include_base=include_base,
            )
            # print(f"** step start: L={l:2d}, {spec.coupling_name}={coupling: .5f}")
            # print(h_base_dense)
            value = value_fn(h_base_dense, context.dh_dense, context.mu, context.basis_dim, l, backend)
            results[l].append((coupling, value))
            print(
                f"** step time: L={l:2d}, {spec.coupling_name}={coupling: .5f}, "
                f"elapsed={time.perf_counter() - coupling_start:.2f}s"
            )

        print(f"## L={l:2d} {series_name} sweep finished in {time.perf_counter() - l_start:.2f}s")

    print(f"## {series_name} perturbation-series computation finished in {time.perf_counter() - total_start:.2f}s")
    return results


def compute_pxp_agp_series(
    l_values: Iterable[int],
    coupling_values: Iterable[float],
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
    backend: str = "cpu",
    perturbation_kind: str = "pxpz",
    include_base: bool = True,
) -> dict[int, list[tuple[float, float]]]:
    def _evaluate_agp(
        h_base_dense: np.ndarray,
        dh_dense: np.ndarray,
        mu: float,
        basis_dim: int,
        l: int,
        backend_name: str,
    ) -> float:
        norm_sq = regularized_agp_norm_pxp(h_base_dense, dh_dense, mu, backend=backend_name)
        return norm_sq / (l * basis_dim)

    return _run_perturbation_series(
        l_values,
        coupling_values,
        symmetry=symmetry,
        boundary=boundary,
        backend=backend,
        perturbation_kind=perturbation_kind,
        value_fn=_evaluate_agp,
        series_name="AGP",
        include_base=include_base,
    )


def compute_pxp_chi_typ_series(
    l_values: Iterable[int],
    coupling_values: Iterable[float],
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
    backend: str = "cpu",
    perturbation_kind: str = "pxpz",
    include_base: bool = True,
) -> dict[int, list[tuple[float, float]]]:
    def _evaluate_chi_typ(
        h_base_dense: np.ndarray,
        dh_dense: np.ndarray,
        mu: float,
        basis_dim: int,
        l: int,
        backend_name: str,
    ) -> float:
        chi_typ = typical_susceptibility_pxp(h_base_dense, dh_dense, backend=backend_name)
        return chi_typ / (l * basis_dim)

    return _run_perturbation_series(
        l_values,
        coupling_values,
        symmetry=symmetry,
        boundary=boundary,
        backend=backend,
        perturbation_kind=perturbation_kind,
        value_fn=_evaluate_chi_typ,
        series_name="chi_typ",
        include_base=include_base,
    )


def compute_pxp_spacing_series(
    l_values: Iterable[int],
    coupling_values: Iterable[float],
    symmetry: tuple = (False, 0),
    boundary: str = "OBC",
    backend: str = "cpu",
    perturbation_kind: str = "pxpz",
    include_base: bool = True,
) -> dict[int, list[tuple[float, float]]]:
    def _evaluate_spacing(
        h_base_dense: np.ndarray,
        dh_dense: np.ndarray,
        mu: float,
        basis_dim: int,
        l: int,
        backend_name: str,
    ) -> float:
        evals = eigvalsh_backend(h_base_dense, backend=backend_name)
        return mean_level_spacing_ratio_middle_third(evals)

    return _run_perturbation_series(
        l_values,
        coupling_values,
        symmetry=symmetry,
        boundary=boundary,
        backend=backend,
        perturbation_kind=perturbation_kind,
        value_fn=_evaluate_spacing,
        series_name="spacing",
        include_base=include_base,
    )


def compute_pxp_spectral_series(
    l_values: Iterable[int],
    coupling: float = 0.0,
    bins: int = 200,
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
    backend: str = "cpu",
    perturbation_kind: str = "pxpz",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    total_start = time.perf_counter()

    for l in l_values:
        l_start = time.perf_counter()
        context = prepare_perturbation_context(l, symmetry, boundary, perturbation_kind)
        h_base_dense = build_model_hamiltonian_dense(context.basis, l, boundary, perturbation_kind, coupling, symmetry)

        mu = context.mu
        evals = eigvalsh_backend(h_base_dense, backend=backend)
        max_omega = float(np.max(evals) - np.min(evals))
        omega_min = max(mu * 1e-1, 1e-12)
        omega_grid = np.logspace(np.log10(omega_min), np.log10(max_omega + 1e-12), bins)

        spectral = spectral_weight_from_matrix(h_base_dense, context.dh_dense, mu, omega_grid, backend=backend)
        spectral /= l
        results[int(l)] = (omega_grid, spectral)
        print(f"Computed spectral L={l}, {context.perturbation.coupling_name}={coupling:.5f}, D={context.basis_dim:6d}")
        print(f"Timing: L={l:2d} spectral computation finished in {time.perf_counter() - l_start:.2f}s")

    print(f"Timing: spectral computation finished in {time.perf_counter() - total_start:.2f}s")
    return results


def compute_pxp_agp_size_series(
    l_values: Iterable[int],
    coupling: float = 0.0,
    symmetry: tuple = (False, False),
    boundary: str = "OBC",
    backend: str = "cpu",
    perturbation_kind: str = "pxpz",
) -> list[tuple[int, float, int]]:
    results: list[tuple[int, float, int]] = []
    total_start = time.perf_counter()

    for l in l_values:
        l_start = time.perf_counter()
        context = prepare_perturbation_context(l, symmetry, boundary, perturbation_kind)
        h_base_dense = build_model_hamiltonian_dense(context.basis, l, boundary, perturbation_kind, coupling, symmetry)

        norm_sq = regularized_agp_norm_pxp(h_base_dense, context.dh_dense, context.mu, backend=backend)
        norm_sq_per_ld = norm_sq / (l * context.basis_dim)

        results.append((l, norm_sq_per_ld, context.basis_dim))
        print(
            f"L={l:2d}, {context.perturbation.coupling_name}={coupling: .5f}, D={context.basis_dim:6d}, "
            f"||A||^2/(L D)={norm_sq_per_ld:.6e}"
        )
        print(f"Timing: L={l:2d} size-scaling point finished in {time.perf_counter() - l_start:.2f}s")

    print(f"Timing: AGP size-series computation finished in {time.perf_counter() - total_start:.2f}s")
    return results


def save_spacing_results(results: dict[int, list[tuple[float, float]]], cache_path: Path) -> None:
    save_grouped_xy_series(results, cache_path)


def load_spacing_results(cache_path: Path) -> dict[int, list[tuple[float, float]]]:
    return load_grouped_xy_series(cache_path)


def save_spectral_results(results: Dict[int, Tuple[np.ndarray, np.ndarray]], cache_path: Path) -> None:
    save_spectral_series(results, cache_path)


def load_spectral_results(cache_path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    return load_spectral_series(cache_path)


def save_hxz_results(results: dict[int, list[tuple[float, float]]], cache_path: Path) -> None:
    save_grouped_xy_series(results, cache_path)


def load_hxz_results(cache_path: Path) -> dict[int, list[tuple[float, float]]]:
    return load_grouped_xy_series(cache_path)


def save_chi_typ_results(results: dict[int, list[tuple[float, float]]], cache_path: Path) -> None:
    save_grouped_xy_series(results, cache_path)


def load_chi_typ_results(cache_path: Path) -> dict[int, list[tuple[float, float]]]:
    return load_grouped_xy_series(cache_path)


def save_size_results(results: list[tuple[int, float, int]], cache_path: Path) -> None:
    save_size_series(results, cache_path)


def load_size_results(cache_path: Path) -> list[tuple[int, float, int]]:
    return load_size_series(cache_path)


def merge_hxz_results(shard_paths: Iterable[Path]) -> dict[int, list[tuple[float, float]]]:
    merged: dict[int, dict[float, float]] = {}
    for cache_path in shard_paths:
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing shard cache: {cache_path}")
        shard_results = load_hxz_results(cache_path)
        for l, points in shard_results.items():
            bucket = merged.setdefault(l, {})
            for hxz_value, value in points:
                bucket[float(hxz_value)] = float(value)

    combined: dict[int, list[tuple[float, float]]] = {}
    for l, hxz_map in merged.items():
        combined[l] = sorted(hxz_map.items(), key=lambda item: item[0])
    return combined


def _cache_params_with_inv_sector(params: dict, inv_sector: int | None) -> dict:
    if inv_sector is None:
        return params
    merged = dict(params)
    merged["inv_sector"] = int(inv_sector)
    return merged
