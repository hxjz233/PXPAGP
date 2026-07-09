from __future__ import annotations

import os

CPUNUM = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = str(CPUNUM)

from pxp_agp_cli import main


if __name__ == "__main__":
    main()
