import numpy as np
import os
import glob
from utils import ParseFilename


# hxz_values = np.unique(np.concatenate((
#     np.arange(-1.0, 0.8, 0.1), 
#     np.arange(-0.25, 0.25, 0.01), 
#     np.arange(-0.68, -0.32, 0.02),
#     np.array([0.0])  # Ensure 0.0 is included explicitly
# )))
hxz_values = [0.2,0.24]
N_values = [14,16,18]

sym_values = [(False, False)]

param_sets = [
    dict(
        gen_dict=dict(N=N, hxz=hxz, sym=sym, model="PXPZ", bound="OBC"),
        proc_dict=dict(prec=1e-6, mid_sample=(False, 0.33)),

        io_dict=dict(read_E=False, read_V=False, read_S=False, 
                        write_E=True, write_V=True, write_S=True),
        task_dict=dict(calc_E=True, calc_V=True, calc_S=True),

        # io_dict=dict(read_E=True, read_V=True, read_S=True, 
        #                 write_E=False, write_V=False, write_S=False),
        # plot_dict=dict(plotS_E=True, plotEdiff=True, plotAdjacency=False, 
        #                plot_args=dict(bins_Ediff=40, bins_Shist=15)),

        # io_dict=dict(read_E=True, read_V=True, read_S=True, 
        #                 write_E=False, write_V=False, write_S=True),
        # task_dict=dict(calc_E=False, calc_V=False, calc_S=True)
    )
    for hxz in hxz_values
    for sym in sym_values
    for N in N_values
    ]

# def explore_folder_and_generate_dicts(folder_path):
#     dicts = []
#     seen_filenames = set()
#     for filepath in glob.glob(os.path.join(folder_path, "*.npy")):
#         filename = os.path.basename(filepath)
#         if filename[:-5] in seen_filenames:
#             continue
#         seen_filenames.add(filename[:-5])
#         gen_dict = ParseFilename(filename)
#         proc_dict = dict(prec=1e-6, mid_sample=(100, False))
#         io_dict = dict(read_E=True, read_V=True, read_S=False, write_E=False, write_V=False, write_S=True)
#         task_dict = dict(calc_E=False, calc_V=False, calc_S=True)
        
#         param_set = dict(
#             gen_dict=gen_dict,
#             proc_dict=proc_dict,
#             io_dict=io_dict,
#             task_dict=task_dict
#         )
#         dicts.append(param_set)
#     return dicts

# folder_path = "res"
# param_sets = explore_folder_and_generate_dicts(folder_path)