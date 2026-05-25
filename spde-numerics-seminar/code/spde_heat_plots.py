"""Create plots and an animation for the stochastic heat equation simulation.

Run from this folder with:
    python spde_heat_plots.py

The numerical method itself lives in spde_heat_simulation.py. This file only
creates visual output.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from spde_heat_simulation import (
    SimulationConfig,
    deterministic_coefficients,
    expected_coefficients,
    reconstruct,
    simulate_fourier_modes,
    simulate_terminal_coefficients,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "spde_outputs"


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap_single_path(times: np.ndarray, x: np.ndarray, values: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    image = ax.imshow(
        values,
        extent=[x[0], x[-1], times[0], times[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Stochastische Heat Equation: eine Realisierung")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("u(t, x, omega)")
    return save_figure(fig, OUTPUT_DIR / "01_heatmap_single_path.png")


def plot_multiple_realizations(
    x: np.ndarray,
    final_time: float,
    n_modes: int,
    n_time_steps: int,
    n_paths: int = 7,
    seed: int = 2026,
) -> Path:
    terminal_coeffs = simulate_terminal_coefficients(
        n_paths=n_paths,
        n_modes=n_modes,
        final_time=final_time,
        n_time_steps=n_time_steps,
        seed=seed,
    )
    terminal_values = reconstruct(terminal_coeffs, x)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for path_index, path_values in enumerate(terminal_values, start=1):
        ax.plot(x, path_values, lw=1.4, label=fr"$\omega_{path_index}$")

    ax.set_xlabel("x")
    ax.set_ylabel("u(t, x, omega)")
    ax.set_title(fr"Mehrere Realisierungen bei festem t = {final_time:g}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    return save_figure(fig, OUTPUT_DIR / "02_multiple_realizations_fixed_time.png")


def plot_expected_fourier_modes(n_modes: int, times: np.ndarray) -> Path:
    means = expected_coefficients(n_modes=n_modes, times=times)
    k = np.arange(1, n_modes + 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for time_value, mode_means in zip(times, means):
        ax.semilogy(k, np.maximum(np.abs(mode_means), 1e-16), marker="o", ms=3, lw=1.2, label=fr"$t={time_value:g}$")

    ax.set_xlabel("Fouriermode k")
    ax.set_ylabel(r"$|\mathbb{E}[u_k(t)]|$")
    ax.set_title("Abklingen der Erwartungswerte der Fouriermoden")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return save_figure(fig, OUTPUT_DIR / "03_expected_fourier_modes.png")


def plot_l2_error_convergence(
    final_time: float,
    n_time_steps: int,
    n_ref: int = 1024,
    n_paths: int = 300,
    seed: int = 31415,
) -> Path:
    n_values = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    terminal_ref_coeffs = simulate_terminal_coefficients(
        n_paths=n_paths,
        n_modes=n_ref,
        final_time=final_time,
        n_time_steps=n_time_steps,
        seed=seed,
    )

    mean_l2_errors = []
    for n_modes in n_values:
        tail = terminal_ref_coeffs[:, n_modes:]
        l2_errors = np.linalg.norm(tail, axis=1)
        mean_l2_errors.append(np.mean(l2_errors))
    mean_l2_errors = np.array(mean_l2_errors)

    fitted_slope, fitted_intercept = np.polyfit(np.log(n_values), np.log(mean_l2_errors), deg=1)
    reference_line = mean_l2_errors[0] * (n_values / n_values[0]) ** (-0.5)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.loglog(n_values, mean_l2_errors, "o-", lw=1.6, label=fr"Monte Carlo, Steigung {fitted_slope:.2f}")
    ax.loglog(n_values, reference_line, "--", lw=1.4, label=r"Referenz $N^{-1/2}$")

    ax.set_xlabel("Anzahl Moden N")
    ax.set_ylabel(r"$\mathbb{E}\|u^{N_{\mathrm{ref}}}(t)-u^N(t)\|_{L^2}$")
    ax.set_title(fr"L2-Fehler gegen Referenzloesung, t = {final_time:g}, $N_{{ref}}={n_ref}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return save_figure(fig, OUTPUT_DIR / "04_l2_error_convergence_loglog.png")


def create_solution_animation(
    times: np.ndarray,
    x: np.ndarray,
    stochastic_values: np.ndarray,
    n_modes: int,
) -> Path:
    deterministic_values = reconstruct(deterministic_coefficients(n_modes, times), x)
    frame_indices = np.unique(np.linspace(0, len(times) - 1, min(120, len(times))).astype(int))

    ymin = min(float(stochastic_values.min()), float(deterministic_values.min()))
    ymax = max(float(stochastic_values.max()), float(deterministic_values.max()))
    margin = 0.08 * max(ymax - ymin, 1.0)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    stochastic_line, = ax.plot([], [], lw=1.6, color="tab:blue", label="stochastisch")
    deterministic_line, = ax.plot([], [], lw=1.8, color="tab:orange", label="deterministisch")
    time_text = ax.text(0.02, 0.94, "", transform=ax.transAxes)

    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel("x")
    ax.set_ylabel("u(t, x)")
    ax.set_title("Zeitentwicklung: stochastisch vs. deterministisch")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    def init() -> tuple:
        stochastic_line.set_data([], [])
        deterministic_line.set_data([], [])
        time_text.set_text("")
        return stochastic_line, deterministic_line, time_text

    def update(frame_index: int) -> tuple:
        stochastic_line.set_data(x, stochastic_values[frame_index])
        deterministic_line.set_data(x, deterministic_values[frame_index])
        time_text.set_text(fr"$t={times[frame_index]:.3f}$")
        return stochastic_line, deterministic_line, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        blit=True,
        interval=50,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        path = OUTPUT_DIR / "05_solution_animation.mp4"
        writer = animation.FFMpegWriter(fps=20, bitrate=2200)
    else:
        path = OUTPUT_DIR / "05_solution_animation.gif"
        writer = animation.PillowWriter(fps=20)

    ani.save(path, writer=writer)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(n_modes=96, final_time=0.25, n_time_steps=250, seed=42)
    x = np.linspace(0.0, 1.0, 320)

    times, coeffs = simulate_fourier_modes(config)
    stochastic_values = reconstruct(coeffs, x)

    created_paths = [
        plot_heatmap_single_path(times, x, stochastic_values),
        plot_multiple_realizations(
            x=x,
            final_time=config.final_time,
            n_modes=config.n_modes,
            n_time_steps=config.n_time_steps,
        ),
        plot_expected_fourier_modes(
            n_modes=70,
            times=np.array([0.0, 0.005, 0.02, 0.08, config.final_time]),
        ),
        plot_l2_error_convergence(
            final_time=config.final_time,
            n_time_steps=config.n_time_steps,
        ),
        create_solution_animation(times, x, stochastic_values, config.n_modes),
    ]

    print("Created files:")
    for path in created_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
