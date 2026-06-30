# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Delta-Eddington two-stream radiative-transfer solver (JAX).

Stage 6. Reference: ``src/radiative_transfer/solvers/delta_eddington.F90`` and the tridiagonal
solve in ``src/linear_algebras/linpack.F90``. Delta-scaling -> two-stream coefficients
(Toon et al. 1989) -> eigen/exponential coefficients -> 2N x 2N tridiagonal system -> reconstruct
direct/diffuse-down/diffuse-up actinic flux. Vectorized over wavelength with ``vmap``; the
per-wavelength tridiagonal solve is a ``lax.scan`` Thomas algorithm. Divide-by-zero guards from the
Fortran (e.g. ``divisr`` floor) are kept explicit so gradients stay finite.
"""
