"""Matrix-defined SPD filter and response calibration; no dataset branches.

The upper bound is Gershgorin's row bound for D^-1 A. For right-triangle scalar
diffusion it is approximately 2, recovering the validated analytic interval.
For other SPD discretizations it adapts to A rather than a PDE name.
"""
from __future__ import annotations
import torch
from polynomial_joint import chebyshev
from coupled_coarse import factor_modes, mm


def spectral_upper(system, action):
    if hasattr(system, '_response_spectral_upper'):
        return system._response_spectral_upper
    matrix = action.matrix
    row = torch.repeat_interleave(torch.arange(matrix.shape[0],device=matrix.device),
                                  matrix.crow_indices()[1:]-matrix.crow_indices()[:-1])
    row_sum = torch.zeros_like(system.rhs).scatter_add_(0,row,matrix.values().abs())
    # Tiny safety margin covers floating summation; this is not an error target.
    upper = float((row_sum*system.levels[0].diagonal_inverse).amax())*(1+1e-6)
    if not upper > 0:
        raise ValueError('A positive diagonal and SPD reduced system are required.')
    system._response_spectral_upper = upper
    return upper


def fixed_filter(system, action, ae, initial, steps, cutoff=.001):
    upper = spectral_upper(system, action)
    return chebyshev(system,action,ae,initial,steps,lower=cutoff*upper/2,upper=upper)


def calibrate_dictionary(modes, context, system, ae, opts):
    upper = spectral_upper(system,context.compiled.action)
    inverse = system.levels[0].diagonal_inverse[:,None]
    # The factor 4/3 gives the old 2/3 Jacobi weight when the bound is 2.
    for _ in range(opts.harmonic_steps):
        modes = modes-(4/(3*upper))*inverse*mm(context.compiled.action.matrix,modes)
    return factor_modes(modes,context,ae)


def calibrated_update(prepared, system, ae, initial, context, sweeps=1):
    """Fixed small-system ridge debiasing; no additional fine-grid A action.

    H=G+eta I. The t-sweep coordinate is
      sum_{j=0}^{t-1} (eta H^-1)^j H^-1 Z^T r.
    On a Gram eigenvalue lambda, its projection gain is
      1-(eta/(lambda+eta))^t, always in [0,1].
    One sweep is exactly the existing regularized calibration.
    """
    if sweeps < 1:
        raise ValueError('Use a positive fixed calibration count.')
    modes, _, factor, _ = prepared
    residual = system.rhs-context.compiled.action.action(0, initial, ae)
    rhs = (modes.double().T@residual.double())[:, None]
    increment = torch.cholesky_solve(rhs, factor)
    coordinate = increment
    for _ in range(sweeps-1):
        increment = torch.cholesky_solve(1e-5*increment, factor)
        coordinate = coordinate+increment
    return initial+modes@coordinate[:, 0].to(initial.dtype)
