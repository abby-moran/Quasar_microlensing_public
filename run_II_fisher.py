"""
Compute the II (intensity interferometry) Fisher matrix on a 2D grid of lens positions.

For each grid point the script:
  1. Places a microlens of the given mass at that position.
  2. Generates the microlensed sparse image and samples the correlation function.
  3. Calls fisher_three_pt to get the 4x4 Fisher matrix for (r_s, m_l, x, y).
  4. Appends the position, per-parameter uncertainties, full covariance, and chi2
     to the output text file.

Usage:
    python run_II_fisher.py --mass 0.1 --grid_extent 7.2 --grid_size 10 \
                            --output_prefix m01_72tE

For M=0.4 Msun (the M04 case from the old M04/m04_78tE_II.py):
    python run_II_fisher.py --mass 0.4 --grid_extent 7.8 --grid_size 12 \
                            --output_prefix m04_78tE

NOTE on theta_max: the II Fisher calculation needs the full 2e-10 rad image to resolve
structure in uv-space.  The magnification script (run_mag_fisher.py) intentionally uses
a smaller theta_max=1.5e-10 rad because it does not need uv resolution — keep them separate.
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import numpy as np
import jax.numpy as jnp
from II_funcs import (
    G, c, D_s, D_l, D_ls, rad_to_arcsec, kappa_bulk, gamma1, gamma2,
    theta_500, error_C,
    alpha_many_stars, beta, sparse_quasar_image, quasar_int_sparse,
    calcC_sparse, pad_image, get_rot_angle, place_star_radial,
    chi2, parameter_uncertainties,
    model_C, fisher_gram,
)


def build_grid(mass, grid_extent, grid_size, image_size, theta_max, phi_0):
    """
    Return an (N²,2) array of pixel-space positions for the grid sweep.

    The grid covers [0, grid_extent * theta_E] along two axes aligned with the
    lensed quasar plane, spanning the second quadrant (x: 0→+, y: 0→−).
    """
    theta_step = 2 * theta_max / image_size

    max_y = (np.array(place_star_radial(
        jnp.asarray(mass), grid_extent, angle=-np.pi / 2,
        image_size=image_size, phi_0=phi_0, theta_max=theta_max,
    )) / image_size - 0.5) * theta_step * image_size

    max_x = (np.array(place_star_radial(
        jnp.asarray(mass), -grid_extent, angle=0,
        image_size=image_size, phi_0=phi_0, theta_max=theta_max,
    )) / image_size - 0.5) * theta_step * image_size

    ex = max_x[0]
    ey = max_y[0]

    x_vals = np.linspace(0, 1, grid_size)
    y_vals = -np.linspace(0, 1, grid_size)
    pos0, pos1 = np.meshgrid(x_vals, y_vals, indexing="xy")
    positions_plot = (pos0[..., None] * ex + pos1[..., None] * ey).reshape(-1, 2)
    positions = positions_plot / theta_step + (image_size / 2)
    return positions


def run(mass, grid_extent, grid_size, output_prefix):
    # ── image parameters ────────────────────────────────────────────────────
    theta_max  = 2e-10          # rad; full image needed to resolve uv structure
    image_size = 8000
    theta_step = 2 * theta_max / image_size
    beta_min   = 0.9e-11        # sparse-image cutoff radius (rad)
    pad_frac   = 0

    g_params = (kappa_bulk, gamma1, gamma2)
    rad      = theta_500
    rad_mas  = rad * rad_to_arcsec   # source radius in microarcseconds

    vec_theta = np.linspace(-theta_max, theta_max, image_size)
    arr_theta = np.asarray(np.meshgrid(vec_theta, vec_theta, indexing="xy"))

    sigma_C = error_C(
        mag=15.76, wavelength=570e-7, R=2e4,
        sigma_t=3e-12, A=10 * (500**2) * np.pi, t_obs=8 * 3600,
    )

    # ── baseline macrolensed image (zero-mass star, just for geometry) ──────
    arr_rands_0 = jnp.array([[(300 - 239), (500 - 68)]])
    star_params_0 = (jnp.asarray([0]), D_s, D_l, D_ls, theta_max, arr_rands_0)
    _, beta_0 = beta(g_params, star_params_0, arr_theta)
    image_0, _, _ = sparse_quasar_image(beta_0, rad, theta_max, beta_min, quasar_int_sparse)
    upts, vpts, corr_0 = calcC_sparse(image_0, theta_step, ellipse=True)
    elps_pts = jnp.asarray((vpts, upts))   # (v, u) order matches calcC_sparse indexing
    phi_0 = -get_rot_angle(beta_0)

    # ── position grid ────────────────────────────────────────────────────────
    positions = build_grid(mass, grid_extent, grid_size, image_size, theta_max, phi_0)

    # ── output files ─────────────────────────────────────────────────────────
    out_II   = f"{output_prefix}_II.txt"
    out_chi2 = f"{output_prefix}_chi2_II.txt"

    # ── per-position loop ─────────────────────────────────────────────────────
    # Step sizes for finite differences.
    # theta_step is the image pixel size; half a pixel is the natural step for
    # a discretised image — above the float64 noise floor, below caustic scale.
    delta_default = [
        rad_mas * 1e-2,                    # source radius — 1% of r_s (μas)
        mass * 5e-2,                       # lens mass — 5% of mass (M_sun)
        0.5 * theta_step * rad_to_arcsec,  # x position — 0.5 pixel (μas)
        0.5 * theta_step * rad_to_arcsec,  # y position — 0.5 pixel (μas)
    ]

    for pos in positions:
        arr_rands = jnp.array([[pos[0] * theta_step, pos[1] * theta_step]])
        star_params = (jnp.asarray([mass]), D_s, D_l, D_ls, theta_max, arr_rands)
        _, beta_ = beta(g_params, star_params, arr_theta)
        image_new, _, _ = sparse_quasar_image(beta_, rad, theta_max, beta_min, quasar_int_sparse)

        total_I_0      = image_new.data.sum()
        _, _, corr_lensed = calcC_sparse(image_new, theta_step, box_size=100, ellipse=False)
        corr_lensed    = corr_lensed[vpts, upts]
        err_c_full     = jnp.full_like(corr_lensed, sigma_C / jnp.sqrt(10))

        input_test = [total_I_0, 100, arr_theta, sparse_quasar_image,
                      theta_max, D_s, D_l, D_ls, g_params]
        params_initial = [
            rad_mas,
            [jnp.asarray([mass])],
            arr_rands.T[0] * rad_to_arcsec,
            arr_rands.T[1] * rad_to_arcsec,
        ]

        delta = list(delta_default)
        F_II = fisher_gram(
            params_initial, input_test, 1, err_c_full,
            delta, elps_pts, pad_frac=0,
        )
        cov_matrix, uncertainties = parameter_uncertainties(
            F_II, theta_step, numstars=1, printout=False,
        )

        chi2_val = chi2(corr_lensed, corr_0, err_c_full[0])

        with open(out_II, "a") as f:
            f.write(
                f"{pos}\t{uncertainties[0]:.6e}\t"
                f"{uncertainties[1]:.6e}\t{uncertainties[2]:.6e}\t"
                f"{uncertainties[3]:.6e}\t{cov_matrix}\n"
            )
        with open(out_chi2, "a") as f:
            f.write(f"{pos}\t{chi2_val}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute II Fisher matrix on a position grid.")
    parser.add_argument("--mass",          type=float, required=True,
                        help="Lens mass in solar masses (e.g. 0.1 or 0.4)")
    parser.add_argument("--grid_extent",   type=float, default=7.2,
                        help="Grid half-extent in units of theta_E for the given mass")
    parser.add_argument("--grid_size",     type=int,   default=10,
                        help="Number of grid points along each axis")
    parser.add_argument("--output_prefix", type=str,   default=None,
                        help="Prefix for output files.  Defaults to m{mass}_grid.")
    args = parser.parse_args()

    if args.output_prefix is None:
        mass_str   = str(args.mass).replace(".", "")
        extent_str = str(args.grid_extent).replace(".", "")
        args.output_prefix = f"m{mass_str}_{extent_str}tE"

    run(args.mass, args.grid_extent, args.grid_size, args.output_prefix)
