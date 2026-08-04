
import re
import gc
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
import pandas as pd
from II_funcs import *
from matplotlib.patches import Ellipse

def apply_residual_cov_fast(cent_cov_traj, dtheta):
    I2 = np.eye(2)
    Cinv = np.linalg.inv(cent_cov_traj)

    prior = I2 / (dtheta[...,None,None]**2 + 1e-30)

    return np.linalg.inv(Cinv + prior)


def acceleration_sigma_from_cov_traj_fast(cent_cov_traj, times):
    n_imp, n_traj, n_t = times.shape
    t = times * 86400.0
    G = np.zeros((n_imp, n_traj, n_t, 2, 6))

    G[:,:,:,0,0] = 1
    G[:,:,:,0,2] = t
    G[:,:,:,0,4] = 0.5*t**2

    G[:,:,:,1,1] = 1
    G[:,:,:,1,3] = t
    G[:,:,:,1,5] = 0.5*t**2
    Cinv = np.linalg.inv(cent_cov_traj)
    GT = np.swapaxes(G, -1, -2)
    F = np.einsum('...ij,...jk,...kl->...il', GT, Cinv, G)
    F = np.sum(F, axis=2)
    cov = np.linalg.inv(F)
    return np.sqrt(np.diagonal(cov[:,:,4:6,4:6], axis1=-2, axis2=-1))
def acceleration_sigma_from_cov_traj_eig(cent_cov_traj, times):
    """
    Like acceleration_sigma_from_cov_traj_fast, but instead of returning the
    sqrt of the diagonal entries of the 2x2 acceleration covariance block
    (i.e. per-axis sigmas), this diagonalizes that 2x2 block and returns the
    sqrt of its eigenvalues, sorted ascending.

    Returns
    -------
    sigma_min, sigma_max : arrays of shape (n_imp, n_traj)
        sigma_min is sqrt(min eigenvalue) -- the tightest-constrained direction.
        sigma_max is sqrt(max eigenvalue) -- the loosest-constrained direction.
    """
    n_imp, n_traj, n_t = times.shape
    t = times * 86400.0
    G = np.zeros((n_imp, n_traj, n_t, 2, 6))

    G[:,:,:,0,0] = 1
    G[:,:,:,0,2] = t
    G[:,:,:,0,4] = 0.5*t**2

    G[:,:,:,1,1] = 1
    G[:,:,:,1,3] = t
    G[:,:,:,1,5] = 0.5*t**2

    Cinv = np.linalg.inv(cent_cov_traj)
    GT = np.swapaxes(G, -1, -2)
    F = np.einsum('...ij,...jk,...kl->...il', GT, Cinv, G)
    F = np.sum(F, axis=2)
    cov = np.linalg.inv(F)

    # 2x2 acceleration block: indices 4:6 correspond to (ax, ay)
    acc_block = cov[:, :, 4:6, 4:6]  # shape (n_imp, n_traj, 2, 2)

    # eigvalsh is for symmetric/Hermitian matrices and returns eigenvalues
    # in ascending order -- exactly what we want (min first, max second)
    eigvals = np.linalg.eigvalsh(acc_block)  # shape (n_imp, n_traj, 2)

    sigma_min = np.sqrt(eigvals[..., 0])
    sigma_max = np.sqrt(eigvals[..., 1])

    return sigma_min, sigma_max


def rotate_data_pos(xpts, ypts, theta):
    "xpts, ypts are relative to 0 0 in the right bottom corner of the image. theta in radians"
    xpts=np.array(xpts)
    ypts=np.array(ypts)
    xpts=xpts-0.5
    ypts=ypts-0.5
    rs=np.sqrt(xpts**2+ypts**2)
    x2=rs*np.cos(np.arctan2(xpts, ypts)+theta)
    y2=rs*np.sin(np.arctan2(xpts, ypts)+theta)
    return x2, y2

def load_file(file, skip_lines=0, matrix=True, force_shape=(4,4)):
    """
    Parse file with entries where each entry has a header (5 or 6 numeric tokens)
    followed by a matrix block between [[ and ]].
    Returns:
      out_xval: (N,2) array (second column nan if header had single x)
      out_rad, out_mass, out_x, out_y: 1D arrays length N
      out_mat: list of numpy arrays (each either reshaped to force_shape if possible, else 1D)
    """
    # token regex matches floats (with optional exp), ints, or 'nan'
    token_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+|nan', flags=re.I)
    with open(file, 'r') as fh:
        lines = fh.readlines()[skip_lines:]
    raw = "".join(lines)

    out_x1, out_x2 = [], []
    out_rad, out_mass, out_x, out_y = [], [], [], []
    out_mat = []

    # find all matrix blocks; iterate in order and take header as text since previous end
    prev_end = 0
    for m in re.finditer(r'\[\[.*?\]\]', raw, flags=re.S):
        header_text = raw[prev_end:m.start()].strip()
        matrix_text = m.group()
        prev_end = m.end()

        # extract numeric-ish tokens from header (includes 'nan')
        header_tokens = token_re.findall(header_text)
        if not header_tokens:
            raise ValueError(f"No numeric tokens found in header preceding matrix at pos {m.start()}")

        # convert tokens to floats, treating 'nan' correctly
        def tok_to_float(t):
            return np.nan if t.lower() == 'nan' else float(t)
        header_vals = [tok_to_float(t) for t in header_tokens]

        # map header tokens -> fields (6-token or 5-token)
        if len(header_vals) == 6:
            x1, x2, rad, mass, xpos, ypos = header_vals
        elif len(header_vals) == 5:
            x1, x2 = header_vals[0], np.nan
            rad, mass, xpos, ypos = header_vals[1], header_vals[2], header_vals[3], header_vals[4]
        else:
            raise ValueError(
                f"Unexpected number of header tokens ({len(header_vals)}) found: {header_tokens!r}\n"
                "Expected 5 or 6 tokens."
            )

        out_x1.append(x1)
        out_x2.append(x2)
        out_rad.append(rad)
        out_mass.append(mass)
        out_x.append(xpos)
        out_y.append(ypos)

        if matrix:
            mat_tokens = token_re.findall(matrix_text)
            if not mat_tokens:
                mat_arr = np.array([], dtype=float)
            else:
                mat_vals = [tok_to_float(t) for t in mat_tokens]
                mat_arr = np.array(mat_vals, dtype=float)
                # reshape if shape matches force_shape
                if force_shape is not None and mat_arr.size == np.prod(force_shape):
                    mat_arr = mat_arr.reshape(force_shape)
            out_mat.append(mat_arr)

    # if there was trailing text after last matrix containing another valid entry without matrix,
    # it won't be processed. That's intentional: function associates matrices with preceding headers.

    out_xval = np.column_stack([np.array(out_x1, dtype=float), np.array(out_x2, dtype=float)])
    return (out_xval,
            np.array(out_rad, dtype=float),
            np.array(out_mass, dtype=float),
            np.array(out_x, dtype=float),
            np.array(out_y, dtype=float),
            out_mat if matrix else None)
    
def load_mag_file(file, skip_lines=0, matrix=True, force_shape=(4,4)):
    """
    Parse file with entries where each entry has a header (5 or 6 numeric tokens)
    followed by a matrix block between [[ and ]].
    Returns:
      out_xval: (N,2) array (second column nan if header had single x)
      out_rad, out_mass, out_x, out_y: 1D arrays length N
      out_mat: list of numpy arrays (each either reshaped to force_shape if possible, else 1D)
    """
    # token regex matches floats (with optional exp), ints, or 'nan'
    token_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+|nan', flags=re.I)
    with open(file, 'r') as fh:
        lines = fh.readlines()[skip_lines:]
    raw = "".join(lines)

    out_x1, out_x2 = [], []
    out_mat = []

    # find all matrix blocks; iterate in order and take header as text since previous end
    prev_end = 0
    for m in re.finditer(r'\[\[.*?\]\]', raw, flags=re.S):
        header_text = raw[prev_end:m.start()].strip()
        matrix_text = m.group()
        prev_end = m.end()

        # extract numeric-ish tokens from header (includes 'nan')
        header_tokens = token_re.findall(header_text)
        if not header_tokens:
            raise ValueError(f"No numeric tokens found in header preceding matrix at pos {m.start()}")

        # convert tokens to floats, treating 'nan' correctly
        def tok_to_float(t):
            return np.nan if t.lower() == 'nan' else float(t)
        header_vals = [tok_to_float(t) for t in header_tokens]

        # map header tokens -> fields (6-token or 5-token)
        if len(header_vals) == 2:
            x1, x2 = header_vals
        if len(header_vals) == 1:
            x1 = header_vals
            x2=0
        #elif len(header_vals) == 5:
        #    x1, x2 = header_vals[0], np.nan
        #    rad, mass, xpos, ypos = header_vals[1], header_vals[2], header_vals[3], header_vals[4]
        #else:
        #    raise ValueError(
        #        f"Unexpected number of header tokens ({len(header_vals)}) found: {header_tokens!r}\n"
        #        "Expected 5 or 6 tokens."
        #    )

        out_x1.append(x1)
        out_x2.append(x2)

        if matrix:
            mat_tokens = token_re.findall(matrix_text)
            if not mat_tokens:
                mat_arr = np.array([], dtype=float)
            else:
                mat_vals = [tok_to_float(t) for t in mat_tokens]
                mat_arr = np.array(mat_vals, dtype=float)
                # reshape if shape matches force_shape
                if force_shape is not None and mat_arr.size == np.prod(force_shape):
                    mat_arr = mat_arr.reshape(force_shape)
            out_mat.append(mat_arr)

    # if there was trailing text after last matrix containing another valid entry without matrix,
    # it won't be processed. That's intentional: function associates matrices with preceding headers.
    if np.array(out_x2).any!=0:
        out_xval = np.column_stack([np.array(out_x1, dtype=float), np.array(out_x2, dtype=float)])
    else:
        out_xval = np.column_stack([np.array(out_x1, dtype=float)])
    return (out_xval,
            out_mat if matrix else None)

def load_cents(file):
    """
    Parse a *_cent.txt file.  Each line: lens position [pixels], centroid Jacobian J
    (4 blocks of (1,2)), and the microlensed-image centroid cent0 [radians].
    Returns (positions (N,2), jacobians (N,2,4) with J[a,i]=dcent_a/dp_i, cent0 (N,2)).
    NOTE: cent0 IS the microlensing centroid shift Delta theta_c of Eq. (23) of the
    paper -- the macro-only image is symmetric under theta -> -theta, so its centroid
    vanishes identically.  (Previously cent0 was ignored and Delta theta_c was
    approximated by J @ ones(4), which is not a physical quantity.)
    """
    import re as _re
    tok = _re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan', flags=_re.I)
    centroids, positions, cent0s = [], [], []
    with open(file, 'r') as fh:
        for line in fh:
            parts = line.split('\t')
            pos = [float(x) for x in tok.findall(parts[0])]
            jvals = [float(x) for x in tok.findall(parts[1])]
            c0 = [float(x) for x in tok.findall(parts[2])]
            assert len(jvals) == 8 and len(c0) == 2, (len(jvals), len(c0))
            positions.append(np.array(pos))
            centroids.append(np.array(jvals).reshape(4, 2).T)
            cent0s.append(np.array(c0))
    return np.array(positions), np.array(centroids), np.array(cent0s)

def get_combo_cov(fisher_mag, cov_II, theta_step_mag, theta_step_II, numstars):
    """
    Combine the magnification Fisher matrix and the II covariance.
    Both come from the run scripts in the SAME parameter units (uas, M_sun, uas, uas),
    so the Fisher matrices are summed directly: F_tot = F_mag + F_II.

    BUGFIX: the magnification Fisher was previously conjugated by
    diag(1,1,theta_step,theta_step) while the II term's identical rescaling was
    cancelled by an inverse rescaling of its covariance; the net effect suppressed
    the magnification term's position information and its r_s--position cross terms
    by factors ~4e-14, i.e. the two terms were combined in inconsistent units.
    The theta_step arguments are kept for signature compatibility but unused.
    """
    fisher_II = np.linalg.inv(np.asarray(cov_II).reshape(1 + 3*numstars, 1 + 3*numstars))
    fisher_tot = np.asarray(fisher_mag) + fisher_II
    cov_tot, uncertainties_tot = parameter_uncertainties(
        fisher_tot, theta_step=1.0, printout=False, numstars=numstars)
    return cov_tot, uncertainties_tot

def rescale_fisher_to_radians(F, theta_step, numstars):
    n_params = 1 + 3*numstars
    J = jnp.eye(n_params)

    # indices
    x_start = 1 + numstars
    y_start = 1 + 2*numstars

    # scale x and y
    J = J.at[x_start:x_start+numstars, x_start:x_start+numstars].set(theta_step)
    J = J.at[y_start:y_start+numstars, y_start:y_start+numstars].set(theta_step)

    # transform Fisher matrix
    return J.T @ F @ J

def load_chi2s(file, skip_lines=0):
    """
    get chi2s from file
    """
    # token regex matches floats (with optional exp), ints, or 'nan'
    token_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+|nan', flags=re.I)
    with open(file, 'r') as fh:
        lines = fh.readlines()[skip_lines:]
    raw = "".join(lines)
    out_chis = []
    for line in lines:
        header_tokens = token_re.findall(line)
        if not header_tokens:
            raise ValueError(f"No numeric tokens found in header preceding matrix at pos {m.start()}")
        if len(header_tokens) == 3:
            x1, x2, chi = header_tokens        
        out_chis.append(chi)
    return np.array(out_chis, dtype=float)

def get_velocities(v_rest, B_matrix, redshift, lens_dist):
    """
    Apparent angular proper motion of a lens star in the image-fixed frame.

    CONVENTION (paper Sec. 3.2): the full relative source--lens motion is attributed
    to the lens population, and we work in the frame where the quasar macroimage is
    stationary by construction.  Relative motion displaces the macroimage, amplified
    in the image plane by B = A^{-1}; boosting to the image-fixed frame transfers
    that amplification to the apparent lens motion:
        nu_tilde = B_matrix @ [ v_perp / ((1+z_l) D_l) ].
    Returns the apparent proper motion in rad per unit time (units of
    lens_dist / velocity).  Pass B_matrix = identity to obtain the unlensed rate.
    """
    mu_rest = v_rest / (1 + redshift) * (1 / lens_dist)
    mu_x = mu_rest[0] * B_matrix[0, 0] + mu_rest[1] * B_matrix[0, 1]
    mu_y = mu_rest[0] * B_matrix[1, 0] + mu_rest[1] * B_matrix[1, 1]
    return np.array((mu_x, mu_y))

def rescale_cov_to_radians(C, theta_step, numstars):
    n_params = 1 + 3*numstars
    J = jnp.eye(n_params)

    # indices
    x_start = 1 + numstars
    y_start = 1 + 2*numstars

    # scale x and y
    J = J.at[x_start:x_start+numstars, x_start:x_start+numstars].set(theta_step)
    J = J.at[y_start:y_start+numstars, y_start:y_start+numstars].set(theta_step)

    return J @ C @ J.T


def sim_rand_traj(bulk_vel, bulk_vel_err, vel_disp, vel_disp_err, theta_max, N, redshift, lens_dist, B_matrix, dist, 
                  dt_days=182.0, max_time_days= 5 * 365.0):
    """
    Simulate a single star trajectory in radians on an image grid
    Returns dict {'times_days', 'positions_rad' (radians Nx2), 'pixel_indices' (Nx2 ints), 'initial_position_rad', 'velocity_kms}

    Inputs:
        bulk_vel: km/s, asusmed to be entirely in the x direction (arb)
        bulk_vel_err: km/s
        vel_disp: km/s, velocity dispersion of the lensing galaxy
        theta_max: image smapens -theta_max to theta_max in x and y, radians
        N: number of pixels
        redshift: redshift of star
        lens_dist: in cm 
        B_matrix: inv_jacobian of the macrolensing stuff
        dt_days: cadence in days
        max_time_days: safety if we get a slow star     
    Returns:
        times_Deays: 1D array
        positions_rad: Nx2 array in rad
        pixel_indices: Nx2, ints, -1 for outside image
    The star passes at closest approach (impact parameter) = dist at t = 0.
    We integrate forward and backward in time with cadence dt_days.

    """

    km_to_cm = 1e5
    day_to_sec = 86400.0
    dt_sec = dt_days * day_to_sec
    max_time_sec = max_time_days * day_to_sec

    total_span = 2.0 * theta_max
    pixel_size = total_span / N

    # vel in km/s
    tot_vel=np.sqrt(bulk_vel**2+vel_disp**2)
    vel_angle=np.random.uniform(0, 2 * np.pi)
    
    v_x_kms =np.cos(vel_angle)*tot_vel #(np.random.normal(loc=bulk_vel, scale=bulk_vel_err)
               #+ np.random.normal(loc=0, scale=np.sqrt(vel_disp**2 + vel_disp_err**2)))
    v_y_kms = np.sin(vel_angle)*tot_vel#np.random.normal(loc=0, scale=np.sqrt(vel_disp**2 + vel_disp_err**2))

    # to angular
    v_rest = np.array([v_x_kms, v_y_kms]) * km_to_cm #to match lens_dist units
    v_ang = get_velocities(v_rest, B_matrix, redshift, lens_dist)
    v_x_ang, v_y_ang = v_ang

    v_norm = np.hypot(v_x_ang, v_y_ang)
    if v_norm == 0:
        raise ValueError("Zero velocity — cannot define trajectory direction.")

    # position at closest approach
    # direction perpendicular to velocity vector
    #phi = np.random.uniform(0, 2 * np.pi)  # pick random orientation of velocity
    # rotate velocity vector by angle phi around center to randomize orientation
    # rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    #v_ang_rot = rot @ np.array([v_x_ang, v_y_ang])
    #v_x_ang, v_y_ang = v_ang_rot

    # perpendicular unit vector
    n_perp = np.array([-v_y_ang, v_x_ang]) / np.hypot(v_x_ang, v_y_ang)
    theta0 = dist * n_perp  # position at closest approach
    theta_x0, theta_y0 = theta0
    times = np.arange(-max_time_sec, max_time_sec + dt_sec, dt_sec)

    theta_x = theta_x0 + v_x_ang * times
    theta_y = theta_y0 + v_y_ang * times
    positions = np.column_stack([theta_x, theta_y])
    pix_x = np.floor((theta_x + theta_max) / pixel_size).astype(int)
    pix_y = np.floor((theta_y + theta_max) / pixel_size).astype(int)
    
    inside = (pix_x >= 0) & (pix_x < N) & (pix_y >= 0) & (pix_y < N)
    
    """
    if not np.all(inside):
        first_out = np.argmax(~inside)   # first False
        # truncate arrays
        times = times[:first_out]
        theta_x = theta_x[:first_out]
        theta_y = theta_y[:first_out]
        pix_x = pix_x[:first_out]
        pix_y = pix_y[:first_out]
    """
    
    outside_mask = (pix_x < 0) | (pix_x >= N) | (pix_y < 0) | (pix_y >= N)
    pix_x[outside_mask] = -1
    pix_y[outside_mask] = -1
    pixel_indices = np.column_stack([pix_x, pix_y])

    times_days = times / day_to_sec

    return {"times_days": times_days, "positions_rad": positions, "pixel_indices": pixel_indices,
        "closest_approach_rad": (theta_x0, theta_y0), "velocity_kms": (v_x_kms, v_y_kms),
        "proper_motion": (v_x_ang, v_y_ang)} # rad/s
def one_quad_to_full(quad_arr, n):
    """
    Reflect one computed quadrant of SCALAR data across the two shear-frame axes.
    Grid convention: array axis 0 is x,
    axis 1 is y, and the computed quadrant has x <= 0, y >= 0.
    Only valid for reflection-invariant quantities (chi^2, per-parameter sigmas,
    |Delta theta_c|, det/eigenvalues of the centroid covariance, ...).
    """
    arr_quadrant = quad_arr.reshape(n, n)
    y_reflect = np.flipud(arr_quadrant)
    x_reflect = np.fliplr(arr_quadrant)
    xy_reflect = np.flipud(np.fliplr(arr_quadrant))
    right = np.concatenate((xy_reflect, y_reflect[:, 1:]), axis=1)
    left = np.concatenate((np.fliplr(arr_quadrant), np.fliplr(x_reflect)[:, 1:]), axis=1)
    return np.concatenate((right[:-1, :], left), axis=0)

def matrix_to_full(quad_arr, n):
    """
    Reflect one quadrant of 4x4 Fisher/covariance matrices (params r_s, m_l, x, y).
    SIGN FIX: under x -> -x the cross terms F[x, p] (p != x) are odd (likewise for
    y -> -y), so mirrored copies must be conjugated by diag(1,1,-1,1)/diag(1,1,1,-1).
    The previous version copied the matrices unmodified, giving wrong cross-term
    signs in three of the four quadrants (harmless for the scalar maps, but biasing
    the trajectory analyses, which interpolate F across quadrants).
    Exact only when the (x, y) parameters are expressed in the shear-aligned frame of
    the grid -- the loading cell now rotates all matrices accordingly.
    """
    arr = np.asarray(quad_arr).reshape(n, n, 4, 4)
    full = np.empty((2*n - 1, 2*n - 1, 4, 4))
    for a in range(4):
        for b in range(4):
            full[:, :, a, b] = one_quad_to_full(arr[:, :, a, b], n)
    m = 2*n - 1
    sgn = np.ones((m, m, 4))
    sgn[n:, :, 2] = -1.0      # x > 0 half: x parameter mirrored
    sgn[:, :n - 1, 3] = -1.0  # y < 0 half: y parameter mirrored
    return full * sgn[:, :, :, None] * sgn[:, :, None, :]

def Jac_to_full(quad_arr, n):
    """
    Reflect one quadrant of centroid Jacobians J[a, i] = d cent_a / d p_i
    (a in {x, y} centroid components, i in {r_s, m_l, x, y}), with the same sign
    logic as matrix_to_full; the centroid components flip together with their axis.
    """
    arr = np.asarray(quad_arr).reshape(n, n, 2, 4)
    full = np.empty((2*n - 1, 2*n - 1, 2, 4))
    for a in range(2):
        for i in range(4):
            full[:, :, a, i] = one_quad_to_full(arr[:, :, a, i], n)
    m = 2*n - 1
    sp = np.ones((m, m, 4)); sc = np.ones((m, m, 2))
    sp[n:, :, 2] = -1.0;   sc[n:, :, 0] = -1.0     # x > 0 half
    sp[:, :n - 1, 3] = -1.0; sc[:, :n - 1, 1] = -1.0  # y < 0 half
    return full * sc[:, :, :, None] * sp[:, :, None, :]

def shear_frame_matrix(angle_rot):
    """
    Orthogonal (improper) map used by rotate_data_pos: (x2, y2) = M @ (x, y) with
    M = [[-sin a, cos a], [cos a, sin a]] = M^T = M^-1.  Conjugating the position
    block of Fisher/covariance matrices (and the centroid Jacobian) by M expresses
    them in the shear-aligned frame of the grids and trajectories.
    """
    return np.array([[-np.sin(angle_rot), np.cos(angle_rot)],
                     [np.cos(angle_rot),  np.sin(angle_rot)]])
def get_velocities(v_rest, B_matrix, redshift, lens_dist):
    """
    Apparent angular proper motion of a lens star in the image-fixed frame.

    CONVENTION (paper Sec. 3.2): the full relative source--lens motion is attributed
    to the lens population, and we work in the frame where the quasar macroimage is
    stationary by construction.  Relative motion displaces the macroimage, amplified
    in the image plane by B = A^{-1}; boosting to the image-fixed frame transfers
    that amplification to the apparent lens motion:
        nu_tilde = B_matrix @ [ v_perp / ((1+z_l) D_l) ].
    Returns the apparent proper motion in rad per unit time (units of
    lens_dist / velocity).  Pass B_matrix = identity to obtain the unlensed rate.
    """
    mu_rest = v_rest / (1 + redshift) * (1 / lens_dist)
    mu_x = mu_rest[0] * B_matrix[0, 0] + mu_rest[1] * B_matrix[0, 1]
    mu_y = mu_rest[0] * B_matrix[1, 0] + mu_rest[1] * B_matrix[1, 1]
    return np.array((mu_x, mu_y))

def sim_rand_traj(bulk_vel, bulk_vel_err, vel_disp, vel_disp_err, theta_max, N, redshift, lens_dist, B_matrix, dist, 
                  dt_days=182.0, max_time_days= 5 * 365.0):
    """
    Simulate a single star trajectory in radians on an image grid
    Returns dict {'times_days', 'positions_rad' (radians Nx2), 'pixel_indices' (Nx2 ints), 'initial_position_rad', 'velocity_kms}

    Inputs:
        bulk_vel: km/s, asusmed to be entirely in the x direction (arb)
        bulk_vel_err: km/s
        vel_disp: km/s, velocity dispersion of the lensing galaxy
        theta_max: image smapens -theta_max to theta_max in x and y, radians
        N: number of pixels
        redshift: redshift of star
        lens_dist: in cm 
        B_matrix: inv_jacobian of the macrolensing stuff
        dt_days: cadence in days
        max_time_days: safety if we get a slow star     
    Returns:
        times_Deays: 1D array
        positions_rad: Nx2 array in rad
        pixel_indices: Nx2, ints, -1 for outside image
    The star passes at closest approach (impact parameter) = dist at t = 0.
    We integrate forward and backward in time with cadence dt_days.

    """

    km_to_cm = 1e5
    day_to_sec = 86400.0
    dt_sec = dt_days * day_to_sec
    max_time_sec = max_time_days * day_to_sec

    total_span = 2.0 * theta_max
    pixel_size = total_span / N

    # vel in km/s
    tot_vel=np.sqrt(bulk_vel**2+vel_disp**2)
    vel_angle=np.random.uniform(0, 2 * np.pi)
    
    v_x_kms =np.cos(vel_angle)*tot_vel #(np.random.normal(loc=bulk_vel, scale=bulk_vel_err)
               #+ np.random.normal(loc=0, scale=np.sqrt(vel_disp**2 + vel_disp_err**2)))
    v_y_kms = np.sin(vel_angle)*tot_vel#np.random.normal(loc=0, scale=np.sqrt(vel_disp**2 + vel_disp_err**2))

    # to angular
    v_rest = np.array([v_x_kms, v_y_kms]) * km_to_cm #to match lens_dist units
    v_ang = get_velocities(v_rest, B_matrix, redshift, lens_dist)
    v_x_ang, v_y_ang = v_ang

    v_norm = np.hypot(v_x_ang, v_y_ang)
    if v_norm == 0:
        raise ValueError("Zero velocity — cannot define trajectory direction.")

    # perpendicular unit vector
    n_perp = np.array([-v_y_ang, v_x_ang]) / np.hypot(v_x_ang, v_y_ang)
    theta0 = dist * n_perp  # position at closest approach
    theta_x0, theta_y0 = theta0
    times = np.arange(-max_time_sec, max_time_sec + dt_sec, dt_sec)

    theta_x = theta_x0 + v_x_ang * times
    theta_y = theta_y0 + v_y_ang * times
    positions = np.column_stack([theta_x, theta_y])
    pix_x = np.floor((theta_x + theta_max) / pixel_size).astype(int)
    pix_y = np.floor((theta_y + theta_max) / pixel_size).astype(int)
    
    inside = (pix_x >= 0) & (pix_x < N) & (pix_y >= 0) & (pix_y < N)
    
    
    outside_mask = (pix_x < 0) | (pix_x >= N) | (pix_y < 0) | (pix_y >= N)
    pix_x[outside_mask] = -1
    pix_y[outside_mask] = -1
    pixel_indices = np.column_stack([pix_x, pix_y])

    times_days = times / day_to_sec

    return {"times_days": times_days, "positions_rad": positions, "pixel_indices": pixel_indices,
        "closest_approach_rad": (theta_x0, theta_y0), "velocity_kms": (v_x_kms, v_y_kms),
        "proper_motion": (v_x_ang, v_y_ang)} # rad/s


def unlens_sigma_from_cov(cov_Theta, B_matrix, redshift, lens_dist_cm):
    """
    1-sigma uncertainties on the *physical* transverse velocity (vx, vy) in km/s
    from the 6x6 Theta covariance.  The fitted proper motion is the apparent,
    image-fixed-frame one (see get_velocities), so the conversion back to a physical
    velocity is v = (1 + z_l) D_l B^{-1} nu_tilde.
    """
    cov_mu = cov_Theta[4:6, 4:6]   # 2x2 apparent angular-velocity covariance

    B_inv = np.linalg.inv(B_matrix)
    J = (1 + redshift) * lens_dist_cm * B_inv   # Jacobian

    cov_v_cm = J @ cov_mu @ J.T                 # cm^2/s^2
    cov_v_kms = cov_v_cm / (1e5**2)             # (km/s)^2

    sigma_v_kms = np.sqrt(np.diag(cov_v_kms))
    return cov_v_cm, cov_v_kms, sigma_v_kms

def centroid_acceleration_uncertainty(centroid_cov_traj, times_days):
    """
    Estimate uncertainty on centroid acceleration by fitting a quadratic
    model c(t) = c0 + v*t + 0.5*a*t^2 to the centroid trajectory.

    Parameters
    ----------
    centroid_cov_traj : array, shape (N, 2, 2)
        Centroid position covariance at each timestep (radians^2)
    times_days : array, shape (N,)
        Observation times in days

    Returns
    -------
    cov_acc : array, shape (2, 2)
        Covariance of centroid acceleration [a_x, a_y] (rad/s^2)^2
    sigma_acc : array, shape (2,)
        1-sigma uncertainty on acceleration in each sky direction
    """
    times_s = times_days * 86400.0
    F = np.zeros((6, 6))

    for i, t in enumerate(times_s):
        # Design matrix: maps [cx0, cy0, vx, vy, ax, ay] -> [cx(t), cy(t)]
        A_i = np.array([[1, 0, t, 0, t**2/2, 0],
            [0, 1, 0, t, 0,t**2/2]])  # shape (2, 6)
        C_inv = np.linalg.inv(centroid_cov_traj[i])
        F += A_i.T @ C_inv @ A_i

    cov_full = np.linalg.inv(F)         # full (6,6) covariance
    cov_acc  = cov_full[4:6, 4:6]      # acceleration block only
    sigma_acc = np.sqrt(np.diag(cov_acc))
    return cov_acc, sigma_acc

def unlens_sigma_from_cov(cov_Theta, B_matrix, redshift, lens_dist_cm):
    """
    1-sigma uncertainties on the *physical* transverse velocity (vx, vy) in km/s
    from the 6x6 Theta covariance.  The fitted proper motion is the apparent,
    image-fixed-frame one (see get_velocities), so the conversion back to a physical
    velocity is v = (1 + z_l) D_l B^{-1} nu_tilde.
    """
    cov_mu = cov_Theta[4:6, 4:6]   # 2x2 apparent angular-velocity covariance

    B_inv = np.linalg.inv(B_matrix)
    J = (1 + redshift) * lens_dist_cm * B_inv   # Jacobian

    cov_v_cm = J @ cov_mu @ J.T                 # cm^2/s^2
    cov_v_kms = cov_v_cm / (1e5**2)             # (km/s)^2

    sigma_v_kms = np.sqrt(np.diag(cov_v_kms))
    return cov_v_cm, cov_v_kms, sigma_v_kms

def plot_star_trajectory(traj, theta_max, show_start_end=True, ax=None):
    """
    Plot a single star trajectory
    
    show_start_end : bool, optional
        Mark starting and ending positions with scatter points
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    theta_x, theta_y = traj["positions_rad"].T

    # Plot trajectory
    ls = '-'
    ax.plot(theta_x, theta_y, ls, color='C0', marker='.', linewidth=1.5)
    if show_start_end:
        ax.scatter(theta_x[0], theta_y[0], c='green', s=40, label='Start', zorder=3)
        ax.scatter(theta_x[-1], theta_y[-1], c='red', s=40, label='End', zorder=3)

    ax.set_xlim(-theta_max, theta_max)
    ax.set_ylim(-theta_max, theta_max)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.grid(True, linestyle='--', alpha=0.3)

    ax.set_xlabel(r'$\theta_x [\text{rad}]$', fontsize=15)
    ax.set_ylabel(r'$\theta_y [\text{rad}]$', fontsize=15)
    ax.set_aspect('equal', 'box')
    ax.legend(fontsize=14)
    return ax

def f_alpha_to_f_Theta(f_alpha, times):
    """
    Convert f_alpha matrices (time-dependent) into a symmetric 6x6 f_Theta matrix
    inputs:
        f_alpha, shape (len(times), 4, 4)
        times: with t=0 at closest approach
    Returns f_theta, of shape 6,6
    """
    f_Theta = np.zeros((6, 6))

    for time, f_a in zip(times, f_alpha):
        # Upper 4x4 block
        f_Theta[:4, :4] += f_a

        # 5th row
        f_Theta[4, :4] += time * f_a[2, :]
        f_Theta[:4, 4] = f_Theta[4, :4]  # symmetry
        f_Theta[4, 4] += (time ** 2) * f_a[2, 2]

        # 6th row 
        f_Theta[5, :4] += time * f_a[3, :]
        f_Theta[:4, 5] = f_Theta[5, :4]  
        f_Theta[5, 4] += (time ** 2) * f_a[2, 3]
        f_Theta[4, 5] = f_Theta[5, 4]    
        f_Theta[5, 5] += (time ** 2) * f_a[3, 3]

    return f_Theta

def plot_cov_ellipse(mean, cov, ax, n_sigma=1.0, **kwargs):
    """
    Plotting function to make perfect ellipses on our corner plots
    """
    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)

    # must be strictly positive
    if np.any(vals <= 0):
        return

    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    width = 2 * n_sigma * np.sqrt(vals[0])
    height = 2 * n_sigma * np.sqrt(vals[1])

    if not np.isfinite(width) or not np.isfinite(height):
        return

    ell = Ellipse(xy=mean,width=width,height=height,angle=theta,fill=False,**kwargs)
    ax.add_patch(ell)

def gaussian_1d(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2*np.pi) * sigma)

def centroid_acceleration_uncertainty(centroid_cov_traj, times_days):
    """
    Estimate uncertainty on centroid acceleration by fitting a quadratic
    model c(t) = c0 + v*t + 0.5*a*t^2 to the centroid trajectory.

    Parameters
    ----------
    centroid_cov_traj : array, shape (N, 2, 2)
        Centroid position covariance at each timestep (radians^2)
    times_days : array, shape (N,)
        Observation times in days

    Returns
    -------
    cov_acc : array, shape (2, 2)
        Covariance of centroid acceleration [a_x, a_y] (rad/s^2)^2
    sigma_acc : array, shape (2,)
        1-sigma uncertainty on acceleration in each sky direction
    """
    times_s = times_days * 86400.0
    F = np.zeros((6, 6))

    for i, t in enumerate(times_s):
        # Design matrix: maps [cx0, cy0, vx, vy, ax, ay] -> [cx(t), cy(t)]
        A_i = np.array([[1, 0, t, 0, t**2/2, 0],
            [0, 1, 0, t, 0,t**2/2]])  # shape (2, 6)
        C_inv = np.linalg.inv(centroid_cov_traj[i])
        F += A_i.T @ C_inv @ A_i

    cov_full = np.linalg.inv(F)         # full (6,6) covariance
    cov_acc  = cov_full[4:6, 4:6]      # acceleration block only
    sigma_acc = np.sqrt(np.diag(cov_acc))
    return cov_acc, sigma_acc

def unlens_sigma_from_cov(cov_Theta, B_matrix, redshift, lens_dist_cm):
    """
    1-sigma uncertainties on the *physical* transverse velocity (vx, vy) in km/s
    from the 6x6 Theta covariance.  The fitted proper motion is the apparent,
    image-fixed-frame one (see get_velocities), so the conversion back to a physical
    velocity is v = (1 + z_l) D_l B^{-1} nu_tilde.
    """
    cov_mu = cov_Theta[4:6, 4:6]   # 2x2 apparent angular-velocity covariance

    B_inv = np.linalg.inv(B_matrix)
    J = (1 + redshift) * lens_dist_cm * B_inv   # Jacobian

    cov_v_cm = J @ cov_mu @ J.T                 # cm^2/s^2
    cov_v_kms = cov_v_cm / (1e5**2)             # (km/s)^2

    sigma_v_kms = np.sqrt(np.diag(cov_v_kms))
    return cov_v_cm, cov_v_kms, sigma_v_kms