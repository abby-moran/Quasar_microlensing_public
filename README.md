# Quasar Microlensing — Intensity Interferometry Fisher Analysis

Code for reproducing the figures and main results of

> **"Measuring Extragalactic Microlens Masses and Motions in Strongly Lensed Quasar Systems with Intensity Interferometry"**

## Overview

The code computes Fisher information matrices that forecast how well an intensity
interferometry (II) telescope array — combined with photometric magnification
monitoring — can constrain the parameters of a stellar microlens passing in front of
image A of the gravitationally lensed quasar **B1422+231** (source redshift z_s = 3.62,
lens redshift z_l = 0.34).

For each position on a 2D grid in the microlens plane the code estimates the
1-σ uncertainties on four parameters:

| Parameter | Symbol | Unit |
|-----------|--------|------|
| Quasar source radius | r_s | μas |
| Microlens mass | m_l | M☉ |
| Microlens position x | θ_x | μas |
| Microlens position y | θ_y | μas |

The Fisher matrix is constructed as a Gram matrix **F = Jᵀ J / σ²**, which is
guaranteed positive semi-definite. Gradients are estimated by central (magnification)
or forward (II, centroid) finite differences with step sizes of 1% r_s, 5% m_l,
and 0.5 pixels for positions.

## Repository structure

```
II_funcs.py            — core physics library (lensing, II correlation, Fisher)
run_II_fisher.py       — compute II Fisher matrix on a 2D lens-position grid
run_mag_fisher.py      — compute magnification Fisher matrix on the same grid
run_centroid.py        — compute image-centroid Jacobian on the same grid
analysis_grid.ipynb    — main analysis notebook: loads data, combines Fisher
                         matrices, plots 2D uncertainty maps
analysis_comparison.ipynb — same analysis with comparison to old Hessian results
Plots.ipynb            — additional figures
interp_grid01.ipynb    — interpolation grid notebook (M=0.1 M☉)
data/                  — pre-computed output files (see below)
figures/               — saved PDF figures
```

## Data files

Pre-computed data live in `data/`.  The naming convention is
`{prefix}_{observable}.txt` where the prefix encodes mass and grid extent:

| Prefix | Mass | Grid extent | Grid size |
|--------|------|-------------|-----------|
| `gram_m04_78tE` | 0.4 M☉ | 7.8 θ_E | 12×12 |
| `m04_78tE` | 0.4 M☉ | 7.8 θ_E | 12×12 |
| `m01_72tE` | 0.1 M☉ | 7.2 θ_E | 10×10 |

Suffixes: `_II.txt` (II Fisher + covariance), `_mag.txt` (magnification Fisher),
`_chi2_II.txt`, `_chi2_mag.txt` (chi-squared maps), `_cent.txt` (centroid Jacobian).

The `gram_*` files use the standardised finite-difference step sizes described above.
The `m04_78tE` / `m01_72tE` files are produced by the current run scripts and can be
regenerated (see below).

## Running the scripts

The scripts require JAX (with 64-bit precision) and the `quasar_smf` conda environment:

```bash
conda activate quasar_smf

# II Fisher matrix — M=0.4, 12×12 grid
python run_II_fisher.py --mass 0.4 --grid_extent 7.8 --grid_size 12 \
                        --output_prefix data/gram_m04_78tE

# Magnification Fisher matrix
python run_mag_fisher.py --mass 0.4 --grid_extent 7.8 --grid_size 12 \
                         --output_prefix data/gram_m04_78tE

# Image-centroid Jacobian
python run_centroid.py --mass 0.4 --grid_extent 7.8 --grid_size 12 \
                       --output_prefix data/gram_m04_78tE
```

The II script is the slowest (requires full uv-plane resolution at θ_max = 2×10⁻¹⁰ rad
on an 8000×8000 image grid) and benefits significantly from a GPU.  The magnification
and centroid scripts can run on CPU in reasonable time.

## Analysis notebooks

Open `analysis_grid.ipynb` and set `MASS`, `GRID_EXTENT`, `GRID_SIZE` in the config
cell, then **Run All**.  The notebook will call the production scripts automatically if
output files are missing, then produce:

- χ² map (II + magnification combined)
- Fractional source-radius uncertainty map σ[r_s] / r_s
- Fractional mass uncertainty map σ[m_l] / m_l
- Position uncertainty map √(σ_x² + σ_y²) in μas

All figures are saved to `figures/` as PDFs.

`analysis_comparison.ipynb` additionally loads an older Hessian-based dataset
(`gram_m04_78tE`) for a side-by-side comparison of covariance matrices and eigenvalue
distributions.

## Physical setup

The macro-lens convergence and shear for image A of B1422+231 are fixed at:

| Parameter | Value |
|-----------|-------|
| κ_bulk | 0.3848 |
| γ₁ | 0.2543 |
| γ₂ | 0.4099 |
| θ_500 (source radius) | ~1.5 μas |

The Einstein radius θ_E scales as √M / D_eff and is printed by the notebook at runtime.
