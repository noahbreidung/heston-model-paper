"""Spectral simulation of the stochastic heat equation on (0, 1).

Model:
    du(t) = A u(t) dt + dW(t),
    u(t, 0) = u(t, 1) = 0,
    A = d^2/dx^2.

The implementation uses the sine eigenbasis
    e_k(x) = sqrt(2) sin(k pi x),  lambda_k = (k pi)^2,
and the exact Ornstein-Uhlenbeck transition for the Fourier modes
    u_k^{n+1} = exp(-lambda_k dt) u_k^n + eta_k^n,
    eta_k^n ~ N(0, (1 - exp(-2 lambda_k dt)) / (2 lambda_k)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for an equidistant time-grid simulation."""

    n_modes: int = 96
    final_time: float = 0.25
    n_time_steps: int = 250
    seed: int | None = 1234

    @property
    def dt(self) -> float:
        return self.final_time / self.n_time_steps


def initial_condition(x: Array) -> Array:
    """Initial condition u_0(x) = 10*(-y^4 - y^3 + y^2 + y), y = 2*(x - 0.5)."""

    x = np.asarray(x)
    y = 2.0 * (x - 0.5)
    return 10.0 * (-(y**4) - y**3 + y**2 + y)


def eigenvalues(n_modes: int) -> Array:
    """Return lambda_k = (k pi)^2 for k=1,...,N."""

    k = np.arange(1, n_modes + 1, dtype=float)
    return (np.pi * k) ** 2


def sine_basis(x: Array, n_modes: int) -> Array:
    """Return matrix E with E[j, k-1] = e_k(x_j)."""

    x = np.asarray(x, dtype=float)
    k = np.arange(1, n_modes + 1, dtype=float)
    return np.sqrt(2.0) * np.sin(np.pi * np.outer(x, k))


def time_grid(final_time: float, n_time_steps: int) -> Array:
    """Equidistant grid t_n = n dt."""

    return np.linspace(0.0, final_time, n_time_steps + 1)


def initial_fourier_coefficients(
    n_modes: int,
    quadrature_order: int | None = None,
) -> Array:
    """Compute u_{0,k} = int_0^1 u_0(x) e_k(x) dx by Gauss-Legendre quadrature."""

    if quadrature_order is None:
        quadrature_order = max(512, 4 * n_modes)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    x = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    basis = sine_basis(x, n_modes)
    return (w * initial_condition(x)) @ basis


def ou_step_parameters(lambdas: Array, dt: float) -> tuple[Array, Array]:
    """Return exact OU decay factors and innovation standard deviations."""

    decay = np.exp(-lambdas * dt)
    variance = -np.expm1(-2.0 * lambdas * dt) / (2.0 * lambdas)
    return decay, np.sqrt(variance)


def simulate_fourier_modes(config: SimulationConfig) -> tuple[Array, Array]:
    """Simulate one path and store all Fourier coefficients on the time grid.

    Returns:
        times: shape (n_time_steps + 1,)
        coeffs: shape (n_time_steps + 1, n_modes)
    """

    rng = np.random.default_rng(config.seed)
    lambdas = eigenvalues(config.n_modes)
    decay, innovation_std = ou_step_parameters(lambdas, config.dt)
    times = time_grid(config.final_time, config.n_time_steps)

    coeffs = np.empty((config.n_time_steps + 1, config.n_modes), dtype=float)
    coeffs[0] = initial_fourier_coefficients(config.n_modes)

    for n in range(config.n_time_steps):
        noise = rng.normal(loc=0.0, scale=innovation_std, size=config.n_modes)
        coeffs[n + 1] = decay * coeffs[n] + noise

    return times, coeffs


def simulate_terminal_coefficients(
    n_paths: int,
    n_modes: int,
    final_time: float,
    n_time_steps: int,
    seed: int | None = None,
) -> Array:
    """Simulate terminal Fourier coefficients for several independent paths.

    The full path is not stored, which keeps convergence experiments cheap.
    The update still uses the exact OU transition on an equidistant grid.
    """

    rng = np.random.default_rng(seed)
    dt = final_time / n_time_steps
    lambdas = eigenvalues(n_modes)
    decay, innovation_std = ou_step_parameters(lambdas, dt)

    initial = initial_fourier_coefficients(n_modes)
    coeffs = np.broadcast_to(initial, (n_paths, n_modes)).copy()

    for _ in range(n_time_steps):
        noise = rng.normal(loc=0.0, scale=innovation_std, size=(n_paths, n_modes))
        coeffs = coeffs * decay + noise

    return coeffs


def deterministic_coefficients(
    n_modes: int,
    times: Array,
    u0_coeffs: Array | None = None,
) -> Array:
    """Fourier coefficients of the deterministic heat equation without noise."""

    times = np.asarray(times, dtype=float)
    lambdas = eigenvalues(n_modes)
    if u0_coeffs is None:
        u0_coeffs = initial_fourier_coefficients(n_modes)
    return np.exp(-np.outer(times, lambdas)) * u0_coeffs


def expected_coefficients(
    n_modes: int,
    times: Array,
    u0_coeffs: Array | None = None,
) -> Array:
    """E[u_k(t)] for the stochastic heat equation.

    The stochastic convolution has mean zero, so this equals the deterministic
    Fourier evolution exp(-lambda_k t) u_{0,k}.
    """

    return deterministic_coefficients(n_modes, times, u0_coeffs)


def reconstruct(coefficients: Array, x: Array) -> Array:
    """Reconstruct u^N from Fourier coefficients on spatial grid x.

    The last axis of coefficients is interpreted as the Fourier-mode axis.
    """

    coefficients = np.asarray(coefficients, dtype=float)
    basis = sine_basis(x, coefficients.shape[-1])
    return np.tensordot(coefficients, basis.T, axes=([-1], [0]))
