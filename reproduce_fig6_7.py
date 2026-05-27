"""Reproduce Figs. 6 and 7 from Phys. Rev. X 10, 041017 (2020).

This script computes the spectral weight

    |f_\lambda(\omega)|^2 = (1/D) \sum_{n\ne m} |<n|\partial_\lambda H|m>|^2
                                  * L_mu(\omega - (E_n-E_m))

where the delta functions are replaced by Lorentzians L_mu(x) = (1/pi) mu/(x^2+mu^2)
with width mu = L * 2^{-L} as in the paper. We average/plot the spectral
weight on a logarithmically spaced frequency grid.

By default the script computes the data for modest system sizes (L=12,14)
to keep the test quick. Increase sizes and bins for production runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def build_xxz_with_defect(l: int, delta: float, eps_d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense H, dH/dDelta, and V (defect operator) for given params.

    H = H_XXZ + eps_d * V, with V = sigma^z_{center}.
    dH/dDelta = sum_i sigma^z_i sigma^z_{i+1} (nearest-neighbor ZZ).
    """
    if l % 2 != 0:
        raise ValueError("Use even L for zero-magnetization sector")

    basis = spin_basis_1d(l, Nup=l // 2, pauli=True)

    nn_xx = [[1.0, i, i + 1] for i in range(l - 1)]
    nn_yy = [[1.0, i, i + 1] for i in range(l - 1)]
    nn_zz = [[delta, i, i + 1] for i in range(l - 1)]

    h_base = hamiltonian([["xx", nn_xx], ["yy", nn_yy], ["zz", nn_zz]], [], basis=basis, dtype=np.float64, check_herm=False)
    dh_ddelta = hamiltonian([["zz", [[1.0, i, i + 1] for i in range(l - 1)]]], [], basis=basis, dtype=np.float64, check_herm=False)

    center = (l + 1) // 2 - 1
    v_list = [[1.0, center]]
    Vop = hamiltonian([["z", v_list]], [], basis=basis, dtype=np.float64, check_herm=False)

    H = np.asarray(h_base.todense(), dtype=np.float64) + float(eps_d) * np.asarray(Vop.todense(), dtype=np.float64)
    return H, np.asarray(dh_ddelta.todense(), dtype=np.float64), np.asarray(Vop.todense(), dtype=np.float64)


def spectral_weight_from_matrix(h: np.ndarray, dh: np.ndarray, mu: float, omega_grid: np.ndarray) -> np.ndarray:
    """Compute |f_lambda(omega)|^2 on omega_grid using Lorentzian broadening mu.

    Returns array of same shape as omega_grid.
    """
    evals, evecs = np.linalg.eigh(h)
    dh_eig = evecs.conj().T @ dh @ evecs

    En = evals[:, None]
    Em = evals[None, :]
    omegas = (En - Em)  # matrix of E_n - E_m

    M2 = np.abs(dh_eig) ** 2
    np.fill_diagonal(M2, 0.0)

    D = h.shape[0]

    # compute spectral function S(omega) = (1/D) * sum_{n!=m} M2_nm * Lorentzian(omega - omega_nm)
    # Lorentzian normalized as (1/pi) * mu/(x^2+mu^2)
    omega_flat = omegas.reshape(-1)
    M2_flat = M2.reshape(-1)

    S = np.zeros_like(omega_grid, dtype=float)
    for i, w in enumerate(omega_grid):
        x = w - omega_flat
        L = (1.0 / np.pi) * (mu / (x * x + mu * mu))
        S[i] = (1.0 / D) * np.sum(M2_flat * L)

    return S


def compute_and_plot_fig6(l_values: Iterable[int], delta: float, epsd_vals: Iterable[float], bins: int, output: Path) -> None:
    """Compute and plot spectral weight for integrable perturbation (Fig. 6).

    Plot |f_lambda(omega)|^2 vs omega for lambda=Delta at different epsd (including 0 and small epsd).
    """
    omega_min_factor = 1e-6
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for epsd in epsd_vals:
        for L in l_values:
            H, dh_ddelta, V = build_xxz_with_defect(L, delta, epsd)
            mu = L * 2 ** (-L)

            # define log-spaced omega grid between mu*1e-2 and max energy diff
            evals = np.linalg.eigvalsh(H)
            max_omega = np.max(evals) - np.min(evals)
            omega_grid = np.logspace(np.log10(max(mu * 1e-3, 1e-12)), np.log10(max_omega + 1e-12), bins)

            S = spectral_weight_from_matrix(H, dh_ddelta, mu, omega_grid)

            label = fr"L={L}, epsd={epsd:.3g}" if epsd != 0.0 else fr"L={L}, epsd=0"
            ax.loglog(omega_grid, S, label=label)

    ax.set_xlabel(r"Frequency $\omega$")
    ax.set_ylabel(r"$|f_\lambda(\omega)|^2$")
    ax.set_title("Fig. 6 reproduction: spectral weight for integrable perturbation (\lambda=\Delta)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize="small", ncol=2)
    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


def compute_and_plot_fig7(l_values: Iterable[int], delta: float, output: Path, bins: int = 200) -> None:
    """Compute and plot spectral weights for nonintegrable perturbations (Fig. 7).

    Top: lambda = epsd at epsd=0 (integrable point). Bottom: lambda=Delta at epsd=0.5 (strongly nonintegrable).
    """
    # top panel: lambda = epsd, epsd=0
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.6), constrained_layout=True)

    # Top: lambda = epsd, at epsd=0 (integrable)
    epsd = 0.0
    ax = axes[0]
    for L in l_values:
        H, dh_ddelta, V = build_xxz_with_defect(L, delta, epsd)
        mu = L * 2 ** (-L)
        evals = np.linalg.eigvalsh(H)
        max_omega = np.max(evals) - np.min(evals)
        omega_grid = np.logspace(np.log10(max(mu * 1e-3, 1e-12)), np.log10(max_omega + 1e-12), bins)
        S = spectral_weight_from_matrix(H, V, mu, omega_grid)  # derivative wrt epsd is V
        ax.loglog(omega_grid, S, label=fr"L={L}")
    ax.set_title("Fig.7 top: spectral weight for lambda=epsd at epsd=0 (integrable)")
    ax.set_ylabel(r"$|f_\lambda(\omega)|^2$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize="small")

    # Bottom: lambda = Delta at epsd = 0.5 (strongly nonintegrable)
    epsd = 0.5
    ax = axes[1]
    for L in l_values:
        H, dh_ddelta, V = build_xxz_with_defect(L, delta, epsd)
        mu = L * 2 ** (-L)
        evals = np.linalg.eigvalsh(H)
        max_omega = np.max(evals) - np.min(evals)
        omega_grid = np.logspace(np.log10(max(mu * 1e-3, 1e-12)), np.log10(max_omega + 1e-12), bins)
        S = spectral_weight_from_matrix(H, dh_ddelta, mu, omega_grid)
        ax.loglog(omega_grid, S, label=fr"L={L}")
    ax.set_title("Fig.7 bottom: spectral weight for lambda=Delta at epsd=0.5 (nonintegrable)")
    ax.set_xlabel(r"Frequency $\omega$")
    ax.set_ylabel(r"$|f_\lambda(\omega)|^2$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize="small")

    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l-values", type=int, nargs="+", default=[12, 14], help="System sizes")
    p.add_argument("--delta", type=float, default=1.1, help="XXZ anisotropy")
    p.add_argument("--bins", type=int, default=200, help="Number of frequency bins (log-spaced)")
    p.add_argument("--output6", type=Path, default=Path("fig6_spectral.png"))
    p.add_argument("--output7", type=Path, default=Path("fig7_spectral.png"))
    return p.parse_args()


def main():
    args = parse_args()
    l_values = args.l_values
    delta = args.delta
    bins = args.bins

    # Fig. 6: lambda = Delta, epsd = 0 and 0.05
    compute_and_plot_fig6(l_values, delta, epsd_vals=[0.0, 0.05], bins=bins, output=args.output6)

    # Fig. 7: two-panel plot
    compute_and_plot_fig7(l_values, delta, output=args.output7, bins=bins)


if __name__ == "__main__":
    main()
