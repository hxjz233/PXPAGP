from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pxp_agp_common import save_figure


def plot_pxp_spacing_series(results: dict[int, list[tuple[float, float]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for l, points in results.items():
        hxz_values = np.array([p[0] for p in points], dtype=float)
        r_values = np.array([p[1] for p in points], dtype=float)
        ax.plot(hxz_values, r_values, marker="o", linewidth=2.0, markersize=6, label=fr"$L={l}$")

    ax.set_xlabel(r"Coupling $h_{xz}$")
    ax.set_ylabel(r"Mean level spacing ratio $\langle r \rangle$")
    ax.axhline(0.386, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Poisson")
    ax.axhline(0.530, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="GOE")
    ax.set_title("PXPZ model: mean level spacing ratio versus $h_{xz}$")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    save_figure(fig, output_path)


def plot_pxp_spectral_series(results: dict[int, tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for l, (omega, spectral) in results.items():
        ax.loglog(omega, spectral, marker="", linewidth=1.5, label=fr"$L={l}$")

    ax.set_xlabel(r"Frequency $\omega$")
    ax.set_ylabel(r"$|f_{h_{xz}}(\omega)|^2$")
    ax.set_title(r"PXPZ: spectral weight $|f_{h_{xz}}(\omega)|^2$ (fixed $h_{xz}$)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    save_figure(fig, output_path)


def plot_pxp_agp_series(
    results: dict[int, list[tuple[float, float]]],
    output_path: Path,
    *,
    perturbation_label: str = "PXPZ",
    coupling_label: str = r"h_{xz}",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for l, points in results.items():
        hxz_values = np.array([p[0] for p in points], dtype=float)
        agp_values = np.array([p[1] for p in points], dtype=float)
        ax.semilogy(hxz_values, agp_values, marker="o", linewidth=2.0, markersize=6, label=fr"$L={l}$")

    ax.set_xlabel(fr"Coupling ${coupling_label}$")
    ax.set_ylabel(r"$\|A\|^2 / (L D)$")
    ax.set_title(fr"{perturbation_label} model: regularized AGP/(L D) versus ${coupling_label}$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    save_figure(fig, output_path)


def plot_pxp_chi_typ_series(results: dict[int, list[tuple[float, float]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for l, points in results.items():
        hxz_values = np.array([p[0] for p in points], dtype=float)
        chi_values = np.array([p[1] for p in points], dtype=float)
        ax.semilogy(hxz_values, chi_values, marker="o", linewidth=2.0, markersize=6, label=fr"$L={l}$")

    ax.set_xlabel(r"Coupling $h_{xz}$")
    ax.set_ylabel(r"$\chi_{\mathrm{typ}} / (L D)$")
    ax.set_title("PXPZ model: typical susceptibility/(L D) versus $h_{xz}$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    save_figure(fig, output_path)


def plot_pxp_agp_normalized_log_series(
    results: dict[int, list[tuple[float, float]]],
    output_path: Path,
    *,
    perturbation_label: str = "PXPZ",
    coupling_label: str = r"h_{xz}",
) -> None:
    coupling_to_data: dict[float, list[tuple[int, float]]] = {}
    for l, points in results.items():
        for coupling, agp_val in points:
            coupling_to_data.setdefault(coupling, []).append((l, agp_val))

    slopes = []
    intercepts = []
    coupling_values = sorted(coupling_to_data.keys())

    for coupling in coupling_values:
        l_vals = np.array([x[0] for x in coupling_to_data[coupling]], dtype=float)
        agp_vals = np.array([x[1] for x in coupling_to_data[coupling]], dtype=float)
        log_agp_vals = np.log(agp_vals)
        slope, intercept = np.polyfit(l_vals, log_agp_vals, 1)
        slopes.append(slope)
        intercepts.append(intercept)

    coupling_array = np.array(coupling_values, dtype=float)
    slopes = np.array(slopes, dtype=float)
    intercepts = np.array(intercepts, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 4.8), constrained_layout=True)

    ax1.plot(coupling_array, slopes, marker="o", linewidth=2.0, markersize=6, color="C0")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.axhline(np.log((np.sqrt(5) + 1) / 2), color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_xlabel(fr"Coupling ${coupling_label}$")
    ax1.set_ylabel(r"Slope of $\log(\|A\|^2 / (L D))$ vs $L$")
    ax1.set_title(fr"{perturbation_label}: slope of log(AGP/(L D)) vs system size")
    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    ax2.plot(coupling_array, intercepts, marker="s", linewidth=2.0, markersize=6, color="C1")
    ax2.set_xlabel(fr"Coupling ${coupling_label}$")
    ax2.set_ylabel(r"Intercept of $\log(\|A\|^2 / (L D))$ vs $L$")
    ax2.set_title(fr"{perturbation_label}: intercept of log(AGP/(L D)) vs system size")
    ax2.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    save_figure(fig, output_path)

    print("\nSlope and intercept summary:")
    for i, coupling in enumerate(coupling_array):
        print(f"{coupling_label}={coupling:.5f}: slope={slopes[i]:.6e}, intercept={intercepts[i]:.6f}")


def plot_pxp_agp_size_series(results: list[tuple[int, float, int]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    lengths = np.array([r[0] for r in results], dtype=float)
    values = np.array([r[1] for r in results], dtype=float)

    ax.semilogy(lengths, values, marker="o", linewidth=2.0, markersize=8, label=r"$\|A_{h_{xz}}\|^2/(L D)$")

    if len(results) >= 3:
        tail_lengths = lengths[-3:]
        tail_log_values = np.log(values[-3:])
        slope, intercept = np.polyfit(tail_lengths, tail_log_values, 1)
        fit_values = np.exp(intercept + slope * lengths)
        ax.semilogy(lengths, fit_values, linestyle="--", linewidth=1.2, alpha=0.7, label="Exponential fit")
        print(f"Exponential slope (last 3 points): {slope:.4f}")

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\|A_{h_{xz}}\|^2 / (L D)$")
    ax.set_title("PXPZ model: regularized AGP/(L D) scaling with system size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    save_figure(fig, output_path)
