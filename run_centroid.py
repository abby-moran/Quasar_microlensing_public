"""
Compute the image-centroid Jacobian on a 2D grid of lens positions.

For each grid point the script:
  1. Places a microlens of the given mass at that position.
  2. Computes d(centroid)/d(params) via forward differences.
  3. Appends the position and Jacobian to the output text file.

Usage:
    python run_centroid.py --mass 0.1 --grid_extent 7.4 --grid_size 10 \
                           --output_prefix m01_74tE

For M=0.4 Msun:
    python run_centroid.py --mass 0.4 --grid_extent 7.8 --grid_size 12 \
                           --output_prefix m04_78tE
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import numpy as np
import jax.numpy as jnp
from II_funcs import (
    D_s, D_l, D_ls, rad_to_arcsec, kappa_bulk, gamma1, gamma2,
    theta_500, error_C,
    beta, sparse_quasar_image, quasar_int_sparse,
    get_rot_angle, place_star_radial,
)


def compute_centroid(params, inputs, num_stars, beta_min=0.9e-11):
    """
    Return the flux-weighted centroid (x, y) in radians.

    params: (r_s_mas, [m_l], [x_mas], [y_mas])
    inputs: (theta, theta_max, Ds, D_l, DLS, g_params)
    """
    source_rad, m_stars, xpos, ypos = params
    theta, theta_max, Ds, D_l, DLS, g_params = inputs
    xpos = jnp.array(xpos) / rad_to_arcsec
    ypos = jnp.array(ypos) / rad_to_arcsec

    star_params = (jnp.array(m_stars), Ds, D_l, DLS, theta_max, jnp.array((xpos, ypos)))
    _, beta_sparse = beta(g_params, star_params, theta)

    image, _, _ = sparse_quasar_image(
        beta_sparse, source_rad / rad_to_arcsec, theta_max, beta_min, quasar_int_sparse,
    )
    img = image.toarray()
    cent_x = np.sum(img * theta[0]) / np.sum(img)
    cent_y = np.sum(img * theta[1]) / np.sum(img)
    return np.array([cent_x, cent_y])


def compute_jacobian_fwd(params_init, model_func, delta, inputs, num_stars):
    """
    Forward-difference Jacobian of model_func (returns a 2-vector) w.r.t. params.
    Returns J of shape (4, 2).
    """
    npar = 4
    source_rad, m_stars, xpos, ypos = params_init
    flat = [source_rad] + list(m_stars) + list(xpos) + list(ypos)
    y0 = model_func(flat.copy(), inputs, num_stars)
    J = []
    for i in range(npar):
        perturbed = flat.copy()
        perturbed[i] += delta[i]
        yi = model_func(perturbed, inputs, num_stars)
        J.append(np.array([(yi - y0) / delta[i]]))
    return J, y0


def build_grid(mass, grid_extent, grid_size, image_size, theta_max, phi_0):
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
    # ── image parameters ──────────────────────────────────────────────────────
    theta_max  = 2e-10
    image_size = 8000
    theta_step = 2 * theta_max / image_size
    beta_min   = 0.9e-11

    g_params = (kappa_bulk, gamma1, gamma2)
    rad      = theta_500
    rad_mas  = rad * rad_to_arcsec

    vec_theta = np.linspace(-theta_max, theta_max, image_size)
    arr_theta = np.asarray(np.meshgrid(vec_theta, vec_theta, indexing="xy"))

    # ── baseline image (geometry only) ────────────────────────────────────────
    arr_rands_0 = jnp.array([[(300 - 239), (500 - 68)]])
    star_params_0 = (jnp.asarray([0]), D_s, D_l, D_ls, theta_max, arr_rands_0)
    _, beta_0 = beta(g_params, star_params_0, arr_theta)
    phi_0 = -get_rot_angle(beta_0)

    # ── position grid ─────────────────────────────────────────────────────────
    positions = build_grid(mass, grid_extent, grid_size, image_size, theta_max, phi_0)

    # ── output file — name derived from parameters to avoid hardcoding bugs ───
    out_cent = f"{output_prefix}_cent.txt"

    # ── per-position loop ─────────────────────────────────────────────────────
    for pos in positions:
        arr_rands = jnp.array([[pos[0] * theta_step, pos[1] * theta_step]])
        star_params = (jnp.asarray([mass]), D_s, D_l, D_ls, theta_max, arr_rands)
        _, beta_ = beta(g_params, star_params, arr_theta)
        image_new, _, _ = sparse_quasar_image(beta_, rad, theta_max, beta_min, quasar_int_sparse)
        total_I_0 = image_new.data.sum()

        params_initial = [
            rad_mas,
            [jnp.asarray([mass])],
            arr_rands.T[0] * rad_to_arcsec,
            arr_rands.T[1] * rad_to_arcsec,
        ]
        delta = [
            rad_mas * 1e-2,                    # source radius — 1% of r_s (μas)
            mass * 5e-2,                       # lens mass — 5% of mass (M_sun)
            0.5 * theta_step * rad_to_arcsec,  # x position — 0.5 pixel (μas)
            0.5 * theta_step * rad_to_arcsec,  # y position — 0.5 pixel (μas)
        ]
        input_test = [arr_theta, theta_max, D_s, D_l, D_ls, g_params]

        J_cent, cent0 = compute_jacobian_fwd(
            params_initial, compute_centroid, delta, input_test, 1,
        )

        with open(out_cent, "a") as f:
            f.write(f"{pos}\t{J_cent}\t{cent0}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute centroid Jacobian on a position grid.",
    )
    parser.add_argument("--mass",          type=float, required=True)
    parser.add_argument("--grid_extent",   type=float, default=7.4)
    parser.add_argument("--grid_size",     type=int,   default=10)
    parser.add_argument("--output_prefix", type=str,   default=None)
    args = parser.parse_args()

    if args.output_prefix is None:
        mass_str   = str(args.mass).replace(".", "")
        extent_str = str(args.grid_extent).replace(".", "")
        args.output_prefix = f"m{mass_str}_{extent_str}tE"

    run(args.mass, args.grid_extent, args.grid_size, args.output_prefix)
