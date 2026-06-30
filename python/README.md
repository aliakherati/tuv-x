# tuvx_photolysis — JAX port of the TUV-x actinic-flux & photolysis-rate calculation

A Python/JAX reimplementation of the TUV-x "Option A" pipeline: from
altitude / latitude / longitude / time-of-year and per-reaction cross sections and quantum yields,
solve the radiation field (delta-Eddington two-stream) and integrate the photolysis rate constants

```
J = Σ_λ F(λ, z) · σ(λ, T) · φ(λ)
```

for each reaction. Built to feed the `frank-model` gas-phase chemistry box model (`../../frank-model`)
and to be JAX-native (jit / vmap / autodiff) for its planned Diffrax phase.

The Fortran sources under `../src` are the **authoritative reference** for all physics (the C++ tree
in `../include` is only a placeholder).

## Status

Staged port (see `/Users/ali/.claude/plans/breezy-mapping-bear.md`). Stage 1 = scaffolding.

| Module | Stage | Fortran reference |
|---|---|---|
| `data.py` | 2 | `cross_section.F90`, `quantum_yield.F90`, `netcdf.F90`, `profiles/`, `grid.F90` |
| `grids.py` | 3 | `grid.F90`, `interpolate.F90` |
| `geometry.py` | 4 | `spherical_geometry.F90`, `profiles/earth_sun_distance.F90`, `profile_utils.F90` |
| `radiators.py` | 5 | `radiative_transfer/radiator.F90`, `radiators/` |
| `solver.py` | 6 | `radiative_transfer/solvers/delta_eddington.F90`, `linear_algebras/linpack.F90` |
| `photolysis.py`, `api.py` | 7 | `photolysis_rates.F90`, `core.F90`, `tuvx.F90` |

Out of scope for the first version: Lyman-α/Schumann-Runge band O2 parameterization
(`la_sr_bands.F90`), dose rates, heating rates.

## Install (venv + pip)

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # CPU-only JAX
pytest
```

## Validation

Numerical and visual consistency are checked against the Fortran reference output. Build TUV-x
(`cmake .. && make` in a `build/` dir), run `./tuv-x examples/tuv_5_4.json` with diagnostics on, and
compare the radiation field and `photolysis_rate_constants.nc` against this port (see
`validation/validate_plots.py`, Stage 8).
