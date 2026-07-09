from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from quspin.operators import hamiltonian

try:
    import cupy as cp  # pyright: ignore[reportMissingImports]
except ImportError:
    cp = None

sys.path.insert(0, str(Path(__file__).parent / "PXP_infra"))

from diag import GetBasis  # pyright: ignore[reportMissingImports]

ROOT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT_DIR / "fig"
RES_DIR = ROOT_DIR / "res"


@dataclass(frozen=True)
class PerturbationSpec:
    kind: str
    cache_tag: str
    coupling_name: str
    coupling_label: str
    display_name: str


@dataclass(frozen=True)
class PerturbationSweepContext:
    l: int
    perturbation: PerturbationSpec
    basis: object
    basis_dim: int
    dh_dense: np.ndarray
    mu: float


def fig_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "fig":
        return ROOT_DIR / path
    return FIG_DIR / path


def get_array_module(backend: str):
    if backend == "gpu":
        if cp is None:
            raise RuntimeError("CuPy is not available in this environment, but --backend gpu was requested.")
        return cp
    return np


def to_numpy(array, backend: str):
    if backend == "gpu":
        return cp.asnumpy(array)
    return np.asarray(array)


def get_perturbation_spec(kind: str) -> PerturbationSpec:
    normalized = kind.lower().strip()
    if normalized in {"hxz", "xz", "pxpz"}:
        return PerturbationSpec(
            kind="pxpz",
            cache_tag="pxpz",
            coupling_name="hxz",
            coupling_label=r"h_{xz}",
            display_name="PXPZ",
        )
    if normalized in {"z", "hz"}:
        return PerturbationSpec(
            kind="z",
            cache_tag="z",
            coupling_name="hz",
            coupling_label=r"h_z",
            display_name="PXPZ with Z perturbation",
        )
    if normalized in {"zz", "hzz"}:
        return PerturbationSpec(
            kind="zz",
            cache_tag="zz",
            coupling_name="hzz",
            coupling_label=r"h_{zz}",
            display_name="PXPZ with ZZ perturbation",
        )
    raise ValueError(f"Unsupported perturbation kind: {kind}")


def make_gen_dict(l: int, hxz: float, symmetry: tuple, boundary: str) -> dict:
    return dict(N=l, hxz=hxz, sym=symmetry, model="PXPZ", bound=boundary)


def build_offset_term_lists(
    l: int,
    boundary: str,
    left_op: str,
    right_op: str,
    offset: int = 2,
) -> list[tuple[str, list[list[float]]]]:
    is_pbc = boundary == "PBC"
    if is_pbc:
        left_list = [[1.0, i, (i + offset) % l] for i in range(l)]
        right_list = [[1.0, (i - offset) % l, i] for i in range(l)]
    else:
        left_list = [[1.0, i - offset, i] for i in range(offset, l)]
        right_list = [[1.0, i, i + offset] for i in range(l - offset)]
    return [(left_op, left_list), (right_op, right_list)]


def build_single_offset_term_list(l: int, boundary: str, op_name: str, offset: int = 2) -> list[tuple[str, list[list[float]]]]:
    is_pbc = boundary == "PBC"
    if is_pbc:
        term_list = [[1.0, i, (i + offset) % l] for i in range(l)]
    else:
        term_list = [[1.0, i, i + offset] for i in range(l - offset)]
    return [(op_name, term_list)]


def _scale_term_lists(term_specs: list[tuple[str, list[list[float]]]], scale: float) -> list[tuple[str, list[list[float]]]]:
    if scale == 1.0:
        return term_specs
    scaled: list[tuple[str, list[list[float]]]] = []
    for op_name, term_list in term_specs:
        scaled.append((op_name, [[scale * float(term[0]), *term[1:]] for term in term_list]))
    return scaled


def build_base_term_lists(l: int) -> list[tuple[str, list[list[float]]]]:
    return [("x", [[1.0, i] for i in range(l)])]


def build_perturbation_term_lists(
    l: int,
    boundary: str,
    perturbation_kind: str,
    coupling: float = 1.0,
) -> list[tuple[str, list[list[float]]]]:
    spec = get_perturbation_spec(perturbation_kind)
    if spec.kind == "pxpz":
        term_specs = build_offset_term_lists(l, boundary, "zx", "xz", offset=2)
        return _scale_term_lists(term_specs, coupling)
    if spec.kind == "z":
        return [("z", [[coupling, i] for i in range(l)])]
    if spec.kind == "zz":
        return _scale_term_lists(build_single_offset_term_list(l, boundary, "zz", offset=2), coupling)
    raise ValueError(f"Unsupported perturbation kind: {perturbation_kind}")


def _requires_complex_dtype(boundary: str, symmetry: tuple) -> bool:
    return boundary == "PBC" and bool(symmetry and symmetry[0])


def build_model_term_lists(
    l: int,
    boundary: str,
    perturbation_kind: str,
    coupling: float,
) -> list[tuple[str, list[list[float]]]]:
    static = build_base_term_lists(l)
    static.extend(build_perturbation_term_lists(l, boundary, perturbation_kind, coupling=coupling))
    return static


def build_perturbation_operator_dense(basis: object, l: int, boundary: str, perturbation_kind: str) -> np.ndarray:
    term_specs = build_perturbation_term_lists(l, boundary, perturbation_kind, coupling=1.0)
    dtype = np.float64
    dh_dhxz = hamiltonian(
        [[op_name, term_list] for op_name, term_list in term_specs],
        [],
        basis=basis,
        dtype=dtype,
        check_symm=False,
        check_pcon=False,
        check_herm=False,
    )
    return np.asarray(dh_dhxz.toarray(), dtype=np.float64)


def build_model_hamiltonian_dense(
    basis: object,
    l: int,
    boundary: str,
    perturbation_kind: str,
    coupling: float,
    symmetry: tuple,
) -> np.ndarray:
    static = build_model_term_lists(l, boundary, perturbation_kind, coupling)
    dtype = np.complex128 if _requires_complex_dtype(boundary, symmetry) else np.float64
    hamiltonian_dense = hamiltonian(
        [[op_name, term_list] for op_name, term_list in static],
        [],
        basis=basis,
        dtype=dtype,
        check_symm=False,
        check_pcon=False,
        check_herm=False,
    )
    return np.asarray(hamiltonian_dense.toarray(), dtype=np.float64)


def build_hxz_operator_dense(basis: object, l: int, boundary: str) -> np.ndarray:
    return build_perturbation_operator_dense(basis, l, boundary, "pxpz")


def build_z_operator_dense(basis: object, l: int, boundary: str) -> np.ndarray:
    return build_perturbation_operator_dense(basis, l, boundary, "z")


def build_zz_operator_dense(basis: object, l: int, boundary: str) -> np.ndarray:
    return build_perturbation_operator_dense(basis, l, boundary, "zz")


def prepare_perturbation_context(
    l: int,
    symmetry: tuple,
    boundary: str,
    perturbation_kind: str = "pxpz",
) -> PerturbationSweepContext:
    spec = get_perturbation_spec(perturbation_kind)
    gen_dict = make_gen_dict(l, 0.0, symmetry, boundary)
    basis = GetBasis(gen_dict)
    basis_dim = basis.Ns
    dh_dense = build_perturbation_operator_dense(basis, l, boundary, spec.kind)
    mu = l / basis_dim
    return PerturbationSweepContext(
        l=l,
        perturbation=spec,
        basis=basis,
        basis_dim=basis_dim,
        dh_dense=dh_dense,
        mu=mu,
    )


def prepare_hxz_context(l: int, symmetry: tuple, boundary: str) -> PerturbationSweepContext:
    return prepare_perturbation_context(l, symmetry, boundary, "pxpz")


def run_parameter_grid(
    x_values: Sequence[float],
    y_values: Sequence[float],
    evaluator: Callable[[float, float], float],
) -> np.ndarray:
    grid = np.empty((len(y_values), len(x_values)), dtype=float)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            grid[y_index, x_index] = evaluator(x_value, y_value)
    return grid


def save_grouped_xy_series(results: Mapping[int, Sequence[tuple[float, float]]], cache_path: Path) -> None:
    npz_dict = {}
    for l, points in results.items():
        x_values = np.array([point[0] for point in points], dtype=float)
        y_values = np.array([point[1] for point in points], dtype=float)
        npz_dict[f"L_{l}_coupling"] = x_values
        npz_dict[f"L_{l}_vals"] = y_values
    np.savez_compressed(cache_path, **npz_dict)


def load_grouped_xy_series(cache_path: Path) -> dict[int, list[tuple[float, float]]]:
    arr = np.load(cache_path)
    results: dict[int, list[tuple[float, float]]] = {}
    keys = list(arr.keys())
    ls = sorted(set(k.split("_")[1] for k in keys))
    for l in ls:
        coupling_key = f"L_{l}_coupling"
        if coupling_key in arr:
            x_values = arr[coupling_key]
        else:
            x_values = arr[f"L_{l}_hxz"]
        vals = arr[f"L_{l}_vals"]
        results[int(l)] = list(zip(x_values.tolist(), vals.tolist()))
    return results


def save_size_series(results: Sequence[tuple[int, float, int]], cache_path: Path) -> None:
    lengths = np.array([row[0] for row in results], dtype=int)
    values = np.array([row[1] for row in results], dtype=float)
    dims = np.array([row[2] for row in results], dtype=int)
    np.savez_compressed(cache_path, L=lengths, vals=values, dims=dims)


def load_size_series(cache_path: Path) -> list[tuple[int, float, int]]:
    arr = np.load(cache_path)
    lengths = arr["L"]
    values = arr["vals"]
    dims = arr["dims"]
    return [(int(lengths[i]), float(values[i]), int(dims[i])) for i in range(len(lengths))]


def save_spectral_series(results: Mapping[int, tuple[np.ndarray, np.ndarray]], cache_path: Path) -> None:
    npz = {}
    for l, (omega, spectral) in results.items():
        npz[f"L_{l}_omega"] = omega
        npz[f"L_{l}_S"] = spectral
    np.savez_compressed(cache_path, **npz)


def load_spectral_series(cache_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    arr = np.load(cache_path)
    keys = list(arr.keys())
    ls = sorted(set(k.split("_")[1] for k in keys))
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for l in ls:
        omega = arr[f"L_{l}_omega"]
        spectral = arr[f"L_{l}_S"]
        results[int(l)] = (omega, spectral)
    return results


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def make_cache_path(mode: str, params: dict) -> Path:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(params, sort_keys=True)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:16]
    return RES_DIR / f"pxp_agp_{mode}_{digest}.npz"


def cache_params_with_inv_sector(params: dict, inv_sector: int | None) -> dict:
    if inv_sector is None:
        return params
    merged = dict(params)
    merged["inv_sector"] = int(inv_sector)
    return merged


def shard_hxz_values(hxz_values: np.ndarray, shard_index: int, shard_count: int) -> np.ndarray:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if shard_count == 1:
        return hxz_values
    return hxz_values[shard_index::shard_count]


def hxz_shard_cache_path(mode: str, params: dict, shard_index: int, shard_count: int) -> Path:
    shard_params = dict(params)
    shard_params["hxz_shard_index"] = int(shard_index)
    shard_params["hxz_shard_count"] = int(shard_count)
    return make_cache_path(mode, shard_params)
