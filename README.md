# Quasar Microlensing — Intensity Interferometry Fisher Analysis

Code for reproducing the figures and main results of

> **"Measuring Extragalactic Microlens Masses and Motions in Strongly Lensed Quasar Systems with Intensity Interferometry"**

> **2026-07 review fixes** corrected flux-normalization prefactor (16π/3, was 4π) in
> `II_funcs.R_from_flux`; centroid-shift Δθ_c now taken from the stored image
> centroids instead of `J @ ones(4)`; centroid metrics use det^(1/4) and √λ_min;
> magnification/II Fisher matrices combined in consistent units; reflection sign
> fixes and shear-frame rotation for interpolated matrices; `interp_grid01` grid
> scale fixed (image_size 7000→8000).  Lens trajectories intentionally move at the
> *apparent* proper motion ν̃ = A⁻¹ν (image-fixed-frame convention, now documented
> in the `get_velocities` docstring and paper §3.2).  
**Data provenance:** everything in `data/` was
> generated *before* these fixes, with θ_500 = 2.2614e-13 rad (pre-16π/3 value) and
> images at λ = 806 nm; the notebooks pin these legacy values so the shipped grids
> remain self-consistent, while fresh runs of the scripts use the corrected
> normalization and the paper's λ = 570 nm.

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
Plots.ipynb            — legacy/setup figures (lensing cartoon, derivative maps,
                         comparison, sampling); see caveat below — currently
                         missing its input .npy files, not runnable as-is
interp_grid01.ipynb    — interpolation grid notebook (M=0.1 M☉)
interp_grid02.ipynb    — interpolation grid notebook (M=0.2 M☉)
interp_grid04.ipynb    — interpolation grid notebook (M=0.4 M☉; also makes the
                         trajectory figures from avgs_04_8yr_cent.npz)
interp_grid08.ipynb    — interpolation grid notebook (M=0.8 M☉)
avgs_0{1,2,4,8}_8yr_cent.npz — per-mass Monte-Carlo-averaged 8-year lens-trajectory
                         statistics (see below)
data/                  — pre-computed Fisher/covariance grids + macro-lens fit (see below)
figures/               — saved PDF figures, one per `savefig` call in the notebooks
                         above; this is the *source* copy — figures are copied by
                         hand into the paper directory for compilation (see the
                         paper's own `CHANGES_review.md` for the copy history)
```

## Data files

### `data/` — Fisher/covariance grids

Pre-computed data live in `data/`.  The naming convention is
`{prefix}_{observable}.txt` where the prefix encodes mass and grid extent:

| Prefix | Mass | Grid extent | Grid size | Consumed by |
|--------|------|-------------|-----------|-------------|
| `gram_m01_10tE` | 0.1 M☉ | 10 θ_E | 20×20 | `interp_grid01.ipynb` |
| `gram_m02_10tE` | 0.2 M☉ | 10 θ_E | 20×20 | `interp_grid02.ipynb` |
| `gram_m04_10tE` | 0.4 M☉ | 10 θ_E | 20×20 | `interp_grid04.ipynb` |
| `gram_m08_10tE` | 0.8 M☉ | 10 θ_E | 20×20 | `interp_grid08.ipynb` |
| `gram_m04_78tE` | 0.4 M☉ | 7.8 θ_E | 12×12 | `analysis_comparison.ipynb` (old-Hessian comparison) |
| `m04_78tE` | 0.4 M☉ | 7.8 θ_E | 12×12 | `analysis_grid.ipynb` |
| `m01_72tE` | 0.1 M☉ | 7.2 θ_E | 10×10 | `analysis_grid.ipynb` |

Suffixes: `_II.txt` (II Fisher + covariance), `_mag.txt` (magnification Fisher),
`_chi2_II.txt`, `_chi2_mag.txt` (chi-squared maps), `_cent.txt` (centroid Jacobian).

The `gram_m0{1,2,4,8}_10tE_*` grids are the ones that feed the manuscript figures
(see mapping table below). The `m04_78tE` / `m01_72tE` / `gram_m04_78tE` files are
lower-resolution, older-vintage grids used only by `analysis_grid.ipynb` /
`analysis_comparison.ipynb` for diagnostic Fisher-vs-Hessian comparisons — **none of
their outputs appear in the paper.** They are produced by the current run scripts
and can be regenerated (see below).

`data/macro_lens_results.npz` — the macro-lens (B1422+231 image A) inverse-Jacobian
fit (`inv_jacobian_fit`), used by every `interp_grid0X.ipynb` and `analysis_grid.ipynb`
to convert rest-frame lens velocities into apparent lensed proper motion
(`get_velocities`, §3.2 of the paper). It is **not produced by any script in this
repo** — it comes from an external macro-lens model fit and is treated as a fixed
input.

`data/pre_regen_backup/` — a manual snapshot of the `gram_m01_10tE_*` and
`gram_m04_10tE_*` files taken 2026-07 immediately before regenerating `data/` with
the corrected flux-normalization prefactor (16π/3 fix, see top of this file). Not
read by any notebook or script; kept only as a rollback reference.

### Root-level `.npz` — trajectory Monte Carlo

`avgs_0{1,2,4,8}_8yr_cent.npz` hold per-mass Monte-Carlo-averaged results for a
lens transiting the source over an 8-year baseline, binned by impact parameter
(`impact_params`). Each `interp_grid0X.ipynb` loads its own file to pick a
representative trajectory index for the corner plots; **only `avgs_04_8yr_cent.npz`
feeds paper figures** — the 5 trajectory-vs-impact-parameter panels of
`fig:traj_res` (chi/mass/pos/velocity/centroid-acceleration uncertainty vs. |ξ|).
Cell 57 of `interp_grid04.ipynb` re-runs the underlying 2000×38 Monte Carlo and
*appends* to this file — do not run it during a routine figure regeneration or the
npz will grow and break downstream array shapes (see the paper's review memory for
the safe regeneration recipe).

## Figure → manuscript mapping

Every PDF in `figures/` corresponds to a `\label{fig:...}` in the paper's `main.tex`.
Generating notebook and underlying `data/` grid are listed below (mass suffix `0X`
∈ {01, 02, 04, 08} runs through the same notebook/data pattern).

| `fig:` label in paper | PDF(s) in `figures/` | Notebook | Underlying data |
|---|---|---|---|
| `basic_lensing` | `basic_lensing.pdf` | `Plots.ipynb` | — (setup cartoon; see caveat) |
| `derivs` | `derivs.pdf` | `Plots.ipynb` | `dC_d{mass,r_lensed,x,y}.npy` (**missing**, see caveat) |
| `sampling` | `sampling.pdf` | `Plots.ipynb` | — |
| `corner_ex` | `corner04_ell.pdf` | `interp_grid04.ipynb` | `gram_m04_10tE_*`, `macro_lens_results.npz` |
| `chi2_2` | `chi04.pdf` | `interp_grid04.ipynb` | `gram_m04_10tE_chi2_{II,mag}.txt` |
| `rad_04` | `rad_04.pdf` | `interp_grid04.ipynb` | `gram_m04_10tE_*` |
| `lens_grid` | `mass_04.pdf`, `pos_04.pdf`, `thetac1_04.pdf`, `thetac2_04.pdf` | `interp_grid04.ipynb` | `gram_m04_10tE_*` |
| `thetac_res` | `thetac1res_04.pdf`, `thetac2res_04.pdf` | `interp_grid04.ipynb` | `gram_m04_10tE_*`, `avgs_04_8yr_cent.npz` (macro centroid) |
| `traj_res` | `chi_traj_04.pdf`, `mass_traj_04.pdf`, `pos_traj_04.pdf`, `vel_traj_04.pdf`, `thetac_acc_04.pdf` | `interp_grid04.ipynb` | `avgs_04_8yr_cent.npz` |
| `lens_comp` | `lensing_comparison.pdf` | `Plots.ipynb` | — (see caveat) |
| `other_mass_grid` | `mass_0X.pdf`, `pos_0X.pdf` (X=1,2,4,8) | `interp_grid0X.ipynb` | `gram_m0X_10tE_*` |
| `other_centroids` | `thetac1_0X.pdf`, `thetac2_0X.pdf` (X=1,2,4,8) | `interp_grid0X.ipynb` | `gram_m0X_10tE_*` |
| `other_centres` | `thetac1res_0X.pdf`, `thetac2res_0X.pdf` (X=1,2,4,8) | `interp_grid0X.ipynb` | `gram_m0X_10tE_*` |
| `m1_corners` | `corner01_res.pdf`, `corner01_ell.pdf`, `corner01_unres.pdf` | `interp_grid01.ipynb` | `gram_m01_10tE_*`, `macro_lens_results.npz`, `avgs_01_8yr_cent.npz` |
| `m8_corners` | `corner_res.pdf`, `corner_unres.pdf` (paper's naming — copied from `corner08_res.pdf` / `corner08_unres.pdf`) | `interp_grid08.ipynb` | `gram_m08_10tE_*`, `macro_lens_results.npz`, `avgs_08_8yr_cent.npz` |

**Caveat on `Plots.ipynb`:** it produces `basic_lensing.pdf`, `derivs.pdf`,
`lensing_comparison.pdf`, and `sampling.pdf`, but (a) it `savefig`s to `../Figures/`,
not this repo's `figures/` — a path left over from an older directory layout — and
(b) the `derivs.pdf` cell loads `dC_dmass.npy`, `dC_dr_lensed.npy`, `dC_dx.npy`,
`dC_dy.npy`, none of which are shipped in this repo. These four figures are
illustrative/setup panels, not outputs of the Fisher pipeline above; the checked-in
PDFs in `figures/` are the last good copies, and the notebook is not currently
runnable end-to-end without the missing derivative arrays.

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
