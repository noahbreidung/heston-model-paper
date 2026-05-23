# The Heston Model

Readable web version of the seminar paper **The Heston Model: Theory, Numerical Simulation and Calibration**.

Live page: https://noahbreidung.github.io/heston-model-paper/

## Overview

The paper studies the Heston stochastic volatility model under the risk-neutral measure. It combines the mathematical structure of the CIR variance process with numerical simulation, Monte Carlo option pricing and calibration to SPY implied-volatility data.

The public page in this repository turns the PDF into a chapter-based web reader with a left-hand table of contents. The original PDF remains available as the source document.

## Topics

- Heston stochastic volatility model
- CIR variance process and Feller condition
- Full-Truncation Euler and Log-Euler simulation
- Monte Carlo pricing
- Calibration to implied-volatility smiles

## Repository Layout

- `index.html`: chapter-based static web reader generated from the PDF text
- `assets/heston-model-paper.pdf`: seminar paper PDF
- `assets/heston-paper-cover-01.png`: rendered cover preview
- `assets/pdf-images/`: figures extracted from the PDF
- `.github/workflows/pages.yml`: GitHub Pages deployment workflow

## Status

This is a seminar and study project, not a production pricing library. The repository is kept public mainly to make the written work and selected numerical results easy to access.
