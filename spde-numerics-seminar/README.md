# Numerical Approximation of a Stochastic Heat Equation

Readable web version of the seminar notes **Numerical Approximation of a Stochastic Heat Equation**.

**Read the seminar here:**  
https://noahbreidung.github.io/heston-model-paper/spde-numerics-seminar/

The page is designed as a MathJax-rendered reader for the LaTeX seminar notes. The original PDF, LaTeX source, simulation code, plot-generation code, and generated numerical figures are included in the repository.

## Overview

The seminar studies the one-dimensional stochastic heat equation with additive space-time white noise. It formulates the equation as an abstract SPDE on \(L^2(0,1)\), derives the mild solution and its spectral representation, and analyzes a spectral Galerkin approximation with exact Ornstein--Uhlenbeck time stepping.

## Topics

- deterministic heat equation and heat semigroup
- cylindrical Wiener processes and space-time white noise
- mild solutions and stochastic convolution
- spectral Galerkin approximation
- exact simulation of Ornstein--Uhlenbeck modes
- mean-square convergence with root mean-square rate \(N^{-1/2}\)

## Repository Layout

- `index.html`: static web reader with MathJax-rendered formulas
- `assets/spde-numerics-seminar.pdf`: seminar notes PDF
- `assets/spde_numerics_seminar.tex`: LaTeX source
- `code/spde_heat_simulation.py`: path generation and numerical routines
- `code/spde_heat_plots.py`: plotting and animation script
- `spde_outputs/`: generated figures and animation
- `index.html`: deployed through the parent GitHub Pages repository

## Status

This is a seminar and study project, not a production SPDE simulation library.
