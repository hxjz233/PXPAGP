import time
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # uncomment this line if omp error occurs on OSX for python 3
)
os.environ["OMP_NUM_THREADS"] = "8"  # set number of OpenMP threads to run in parallel  # seems not affecting diagonalization?
os.environ["MKL_NUM_THREADS"] = "8"  # set number of MKL threads to run in parallel

import numpy as np  # generic math functions

from diag import GetBasis, GetHam, Diag
from visual import PlotLevelDiff, PlotS_E, PlotAdjacency
from paramdef import param_sets
from utils import LoadResults, SaveResults, GetMeta
from statobs import GetEmid, GetEDiff, degeneracy_check
from config import PROJECT_ROOT


for param in param_sets:

    gen_dict, io_dict = param["gen_dict"], param["io_dict"]
    
    if "bound" not in gen_dict:
        gen_dict["bound"] = "PBC"
    if "bound" == "OBC":
        gen_dict["sym"][0] = False
    if gen_dict["bound"] not in ["PBC", "OBC"]:
        raise ValueError("Invalid value for 'bound'. It must be either 'PBC' or 'OBC'.")
    
    N, hxz, t_param, p_param = gen_dict["N"], gen_dict["hxz"], gen_dict["sym"][0], gen_dict["sym"][1]
    [E, V, S, Emid] = [None] * 4
    basis = GetBasis(gen_dict)
    output_prefix = GetMeta(gen_dict)

    print(f"== running {output_prefix} ==")
    print(f"Hilbert space dimension: {basis.Ns}")
    # read E, V, S
    E, V, S = LoadResults(f"{PROJECT_ROOT}/res/{output_prefix}", io_dict)
    if "task_dict" in param and E is None:
        task_dict = param["task_dict"]
        calc_E = task_dict.get("calc_E", False)
        calc_V = task_dict.get("calc_V", False)
        if calc_E:
            start_time = time.time()

            result = Diag(gen_dict, basis, calc_V)
            if calc_V:
                E, V = result
            else:
                E = result

            end_time = time.time()
            degeneracy_check(E)
            print(f"Diagonalization time: {end_time - start_time:.2f}s")
    
    prec = param.get("proc_dict", {}).get("prec", 1e-6)
    mid_sample = param.get("proc_dict", {}).get("mid_sample", (100, False))
    middle_index, mid_sample, Emid = GetEmid(E, prec, mid_sample)

    SaveResults(f"{PROJECT_ROOT}/res/{output_prefix}", io_dict, E, V, None)

    if "task_dict" in param:
        task_dict = param["task_dict"]
        calc_S = task_dict.get("calc_S", False)
        if calc_S:  # S length determined by proc_dict
            start_time = time.time()
            indices = np.arange(middle_index, middle_index + mid_sample)
            if S is None:
                S = np.zeros(len(indices))
            else:
                # Pad S with zeros if necessary to match the length of indices
                if len(S) < len(indices):
                    S = np.pad(S, (0, len(indices) - len(S)), 'constant')
            
            for idx, i in enumerate(indices):
                if S[idx] == 0:  # Only calculate if S[idx] is not already set
                    S[idx] = basis.ent_entropy(V[:, i], density=True)["Sent_A"]
                    elapsed_time = time.time() - start_time
                    print(f"Progress: {idx + 1}/{len(indices)}, Elapsed time: {elapsed_time:.2f}s", end="\r")
            
            end_time = time.time()
            print(f"\nEntropy calculation time: {end_time - start_time:.2f}s")
            
    SaveResults(f"{PROJECT_ROOT}/res/{output_prefix}", io_dict, None, None, S)
    
    if "plot_dict" in param:
        plot_dict = param["plot_dict"]
        if "plotEdiff" in plot_dict and plot_dict["plotEdiff"]:
            fig, ax = PlotLevelDiff(gen_dict, *GetEDiff(Emid), plot_dict.get("plot_args", {}))
            fig.savefig(f'{PROJECT_ROOT}/pic/Ediff_{output_prefix}.png')  # To save the plot to a file
            print("Saved Ediff plot")
        if "plotS_E" in plot_dict and plot_dict["plotS_E"]:
            fig, ax = PlotS_E(gen_dict, Emid, S)
            fig.savefig(f'{PROJECT_ROOT}/pic/SE_{output_prefix}.png')  # To save the plot to a file
            print("Saved S-E plot")
        if "plotAdjacency" in plot_dict and plot_dict["plotAdjacency"]:
            fig, ax = PlotAdjacency(GetHam(gen_dict, basis).toarray())
            fig.savefig(f'{PROJECT_ROOT}/pic/Adjacency_{output_prefix}.png')  # To save the plot to a file
            print("Saved Adjacency plot")
