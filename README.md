# AGP workspace

This repo is centered on `src/pxp_agp_scaling.py`. The `src/PXP_infra/` package is treated as a single existing block and is not modified by the workspace restructuring.

## Layout

- `src/` main entrypoint and supporting package code
- `scripts/` Windows batch launchers and reproduction scripts
- `configs/` reserved for optional config files
- `fig/` generated figures
- `res/` cached results and intermediate outputs
- `archive/` historical data and outputs
- `TODO` work notes

## Running

The batch files in `scripts/` activate the local `quspin` environment and call the `src/` entrypoint:

- `scripts/run_hxz.bat`
- `scripts/run_z.bat`
- `scripts/run_zz.bat`
- `scripts/run_ss.bat`
- `scripts/run_size.bat`
- `scripts/run_spectral.bat`
- `scripts/run_spacing.bat`
- `scripts/run_typical.bat`
- `scripts/reproduce_fig11.bat`
- `scripts/reproduce_fig6_7.bat`

Each launcher accepts extra CLI arguments after the defaults if you want to override sizes, output paths, or backend settings.
