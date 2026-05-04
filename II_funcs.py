import numpy as np
import matplotlib
from astropy import units as u
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time, TimeDelta
from scipy.sparse import coo_matrix
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from scipy import special
from scipy.sparse import coo_matrix
from scipy.ndimage import rotate
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from scipy.stats import qmc

import math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.signal import savgol_filter

### constants ###
G=6.6743 * 10**(-8) #cm3 g-1 s-2
c=3e10 #cm/s
h=6.6261e-27 #cm2 g s-1
K=1.3807e-16 #cm2 g s-2 K-1
T_500 = 28776
ZP_791 = 1.23e-9               # erg cm^2 s^-1 A^-1
wavelength_791 = 7969e-8       # cm
mag_fit = 6.86

Dq = 4.6e27  
rad_to_arcsec=2.0626e+11

# define these explictly, from the data above 
kappa_star=0.05 
kappa_bulk=0.38479219
gamma1=0.25426193864
gamma2=0.4098991987
z_s=3.62
# D_q=4.6e+27 # KVT: Why is there a separate Dq and Ds?
hbar=1.0546e-27 #cm2 g s-1
z_l=0.34

D_l=cosmo.angular_diameter_distance(z_l).value*u.Mpc.to(u.cm)
D_s = cosmo.angular_diameter_distance(z_s).value*u.Mpc.to(u.cm)
D_ls=cosmo.angular_diameter_distance_z1z2(z_l, z_s).value*u.Mpc.to(u.cm)

## Defining quasar functions ###
def flux_from_mag(mag, ZP):    
    flux=ZP*10**(-.4*mag)*1e8 #cm, not angstrom
    return flux
def R_from_flux(flux, distance, z, wavelength):
    flux_den_const=4*np.pi*1.93216*(h**(-5/3))*c**(-2/3)*K**(8/3)
    T_500=28776
    theta_500=np.sqrt(flux/(flux_den_const*(T_500**(8/3))*wavelength**(-7/3)))
    R_500=theta_500*(distance*(1+z)**(4/3))
    return theta_500, R_500


flux_791 = flux_from_mag(15.88, ZP_791)
theta_500_791, R_500_791 = R_from_flux(flux_791, Dq, z_s,  wavelength_791) #gives theta_lensed, R lensed (not physically meaningful)
theta_500_791_UL = theta_500_791/np.sqrt(mag_fit) #unlensed angular size, use for normalization 
#R_500_791 /= mag_fit
theta_500 = theta_500_791_UL

theta_E_1 = np.array([np.sqrt(4 * G * (2e33) / c**2 * D_ls / (D_s * D_l))])
theta_E_4 = np.array([np.sqrt(4 * G * (0.4 * 2e33) / c**2 * D_ls / (D_s * D_l))])

def get_I(mag, deltav):
    """
    Go from magnitude to Intensity as defined in equation 2.48 of https://arxiv.org/pdf/2307.06989
    KVT: What is deltav? What are the units of the output?
    """
    zeropt= 2635e-23 #3.631e-20 #
    flux_den=(zeropt)*10**(-.4*mag) #erg/s/cm2/Hz
    #Ephot=(h*c)/wavelength #ergs, wavelength in cm
    #f_photon_v=flux_den/Ephot

    return flux_den*deltav
def chi2(data, unlensed, error):
    return jnp.sum((data-unlensed)**2/error**2)

def error_C(mag, wavelength, R, sigma_t=30e-12, t_obs=1e4, eta1=.5, eta2=.5, A=(200**2)*np.pi, mag_I=True):
    """
    Define error on our intensity interferometer according to equation 2.48 in https://arxiv.org/pdf/2307.06989
    Everything should be in cgs 
    """

    #R= kbar/sigma_k
    #first calculate the frequency bandidth sigma_f
    kbar=(2*np.pi)/wavelength #angular wavenumber
    sigma_f = (kbar*c)/(R*2*np.pi)

    #get I in units of g/s^3 (weird but cancels right)
    if mag_I:
        I1 = get_I(mag, sigma_f) 
    else:
        I1=mag
    #print(I1)
    sigma_c = np.sqrt(1/(np.sqrt(4*np.pi)*sigma_t*t_obs))*(hbar*c*kbar)/(I1*np.sqrt(eta1*eta2*(A**2)))
    return sigma_c

def calcC_sparse(sparse_image, theta_step, box_size=2e10, max_norm=True, ellipse=True):
    """
    Calculate the absolute square of the Fourier transform of a sparse image and extract a region of fixed size in uv-space.

    Parameters:
        sparse_image: scipy.sparse matrix
        theta_step: original step size of our image in radians before any padding
        box_size: desired size of box. **This is no longer being used in this version of the code**
        max_norm: bool, fixed normalization from an earlier version
        ellipse: full image or just some sampled points in the ellipse?
    Returns:
        2D array of the normalized power spectrum region or sampled ellipse points of that (if ellipse=True)
    """
    csr_image = sparse_image.tocsr()
    N_padded = csr_image.shape[0]  # after any padding
    N_orig = sparse_image.shape[0] if not hasattr(sparse_image, "pad_origin_shape") else sparse_image.pad_origin_shape

    dense_image = csr_image.toarray()

    # Correct theta_step if padded
    theta_step_eff = theta_step * N_orig / N_padded

    # FFT and power spectrum
    dense_image_jax = jnp.asarray(dense_image, dtype=jnp.complex128)
    
    fourier_transform = jnp.fft.fftshift(jnp.fft.fft2(dense_image_jax)) #jnp.fft.fftshift(jnp.fft.fft2(dense_image, workers=-1))
    correlation = jnp.abs(fourier_transform) ** 2
    #correlation=jnp.flipud(correlation)
    u_vals = 2 * jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(N_padded, d=theta_step_eff))
    v_vals = u_vals.copy()  # square FFT, so same axis
    if max_norm:
        tot = jnp.max(correlation)
    else:
        tot = jnp.nansum(correlation)

    correlation = correlation / tot

    if not ellipse:
        return u_vals, v_vals, correlation
        
    if ellipse:
        _, _, _, uv_ellipse=get_uv_elipse(correlation, u_vals, v_vals)
    
        u_indices = jnp.searchsorted(u_vals, uv_ellipse[:, 0]) - 1
        v_indices = jnp.searchsorted(v_vals, uv_ellipse[:, 1]) - 1
    
        # Clip to valid bounds
        u_indices = jnp.clip(u_indices, 0, len(u_vals) - 1)
        v_indices = jnp.clip(v_indices, 0, len(v_vals) - 1)
        
        # Gather values, swap u and v because of how we index (xy vs ij indexing causes this and I've checked many times to make sure we
        # should be doing this) 
        sampled_corr_vals = correlation[v_indices, u_indices] 
        return u_indices, v_indices, sampled_corr_vals


def get_uv_elipse(corr_uv, u_vals, v_vals, numpts=1000, limit=0.1): 
    """
    Fit an ellipse to corr_uv > limit (u,v centered).
    Return ellipse params (a, b, phi) and sample numpts points *inside* the ellipse.
    """
   # Build meshgrid and apply threshold
    U, V = jnp.meshgrid(u_vals, v_vals, indexing='ij')
    mask = corr_uv > limit
    u_points = U[mask]
    v_points = V[mask]

    if len(u_points) < 5:
        num_thresh_pts = len(u_points)
        print(f"Not enough points above threshold to fit ellipse. Only {num_thresh_pts} points")

    coords = jnp.vstack([u_points, v_points])

    # Covariance and eigen-decomposition
    cov = np.cov(coords)
    evals, evecs = jnp.linalg.eigh(cov)

    # Sort eigenvalues (major → minor)
    order = jnp.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    a = jnp.sqrt(evals[0])
    b = jnp.sqrt(evals[1])
    phi = jnp.arctan2(evecs[1, 0], evecs[0, 0])  # angle in radians

    # Determine grid resolution
    numpts_per_axis = int(jnp.sqrt(numpts))

    sampler = qmc.Halton(d=2, scramble=False)
    points = sampler.random(n=numpts)

    # Map unit square [0,1] x [0,1] to unit circle
    r = jnp.sqrt(points[:, 0])
    theta = 2 * jnp.pi * points[:, 1]
    u_inner = 1 * a * r * jnp.cos(theta)
    v_inner = 1 * b * r * jnp.sin(theta)
    
    # Rotate
    rot = jnp.array([[jnp.cos(phi), -jnp.sin(phi)],
                     [jnp.sin(phi),  jnp.cos(phi)]])
    ellipse_points = rot @ jnp.vstack([u_inner, v_inner])
    sampled_uv = ellipse_points.T

    return a, b, phi, sampled_uv

def pad_image(image, npad):
    """
    Pad an image with zeros to increase resolution when tranformed to UV space
    """
    if npad==0:
        return image
    if scipy.sparse.issparse(image):
        image_padded = np.pad(image.toarray(), npad, mode='constant', constant_values=0)
        return scipy.sparse.csr_matrix(image_padded)  # Returning as a sparse matrix
    else:
        return jnp.pad(image, npad, mode='constant', constant_values=0)

def alpha_g(kappa, shear1, shear2, theta):
    """
    Calculate the smooth (global) deflection field.
    Inputs:
        theta: shape 2, N, N for a grid OR shape 2 for a single point
        shears and kappa: constants
    Output:
        Array of (x, y) positions, shape 2, N, N or shape 2 for a point
    """
    theta = np.atleast_2d(theta)  # Convert to (2, N, N) if it's not already

    theta_x = theta[0]
    theta_y = theta[1]
    # Calculate deflections for each position
    alpha_x = (kappa + shear1) * theta_x + shear2 * theta_y  
    alpha_y = (kappa - shear1) * theta_y + shear2 * theta_x  

    return np.array([alpha_x, alpha_y])  # Output shape: (2, N, N) or (2,)
    
 
def alpha_pt(const, arr_theta):
    """
    In current code, we don't use this
    Deflection angle of a point mass at coordinates each point in arr_theta.
    Inputs:
        const: constant related to point mass
        arr_theta: shape 2, 2N-1, 2N-1 for a grid OR shape 2 for a single point
    Outputs:
        alpha: shape 2, 2N-1, 2N-1 or shape 2
    """
    norm_theta = jnp.linalg.norm(arr_theta, axis=0)
    mask = norm_theta > 0  # Avoid division by zero
    alpha = jnp.zeros_like(arr_theta)
    alpha=alpha.at[:, mask].set(const * arr_theta[:, mask] / norm_theta[mask]**2)
    return alpha


def alpha_many_stars(m_star, Ds, Dl, DLS, arr_theta, theta_max, arr_rand_idx):
    """
    Defelction from a single star, despite the name of the function.
    m_star should be in solar masses, and everything else should be in cgs units 
    """
    m_star=jnp.asarray(m_star)
    m_star=m_star*u.solMass.to(u.g)
    const = (4 * G * m_star / c**2) * (DLS / (Ds * Dl))
    ## bc of how we index (ie where we call 0) we need to shift the star by -theta_max so that a star at (0,0) is really at 
    # the center of the image, not the lower left corner
    arr_rand_idx=jnp.asarray(arr_rand_idx)
    arr_rand_idx-=theta_max 
    
    delta_theta_x=arr_theta[0]-arr_rand_idx.T[0]
    delta_theta_y=arr_theta[1]-arr_rand_idx.T[1]
    delta_theta = jnp.asarray((delta_theta_x, delta_theta_y))
    
    # Flatten arr_rand_idx to shape (2,)
    # Compute norm, avoid exact zero to prevent singularity issues
    norm_theta = jnp.linalg.norm(delta_theta, axis=0)
    eps = 1e-20  # tiny number to avoid division by zero
    safe_norm = jnp.where(norm_theta > 0, norm_theta, eps)

    alpha = const * delta_theta / safe_norm**2
    return alpha
    

def beta(gparams, star_params, theta):
    "subtracts defelction alpha off from theta, both of which are grids of shape 2,N,N"
    alpha_all= alpha_tot(gparams, star_params, theta)
    beta_pos = jnp.subtract(theta, alpha_all)
    return alpha_all, beta_pos

def alpha_tot(g_params, star_params, theta):
    """
    Calculate total alpha for both smooth deflection and point masses.
    Inputs:
        theta: grid or single point
    Outputs:
        Total deflection
    """
    kappa, shear1, shear2 = g_params
    m_stars, Ds, Dl, DLS, theta_max, arr_rand_idx= star_params
    alpha_gval = alpha_g(kappa, shear1, shear2, theta)
    alpha_stars = alpha_many_stars(m_stars, Ds, Dl,DLS, theta, theta_max, arr_rand_idx)
    return alpha_gval + alpha_stars
    
def get_rot_angle(coords):
    """
    Given a 2xN array of positions (e.g. beta positions),
    compute the orientation angle of the major axis.
    KVT: What does this function do exactly?
    """
    # Covariance and eigen-decomposition
    coords = coords.reshape(2, -1)
    cov = np.cov(coords)
    evals, evecs = np.linalg.eigh(cov)

    # Sort eigenvalues descending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Get angle of the major axis
    phi = np.arctan2(evecs[1, 0], evecs[0, 0])  # radians
    return phi

def quasar_int_sparse(x_coords, y_coords, radius, theta_max, N, wavelength=8.06e-5, quasar_pos=[0,0]):
    """
    Modified quasar intensity function to work with sparse coordinates
    """
    const = 2*h*c**2
    theta_step = 2*theta_max/N #need N as a param because we don't have the full grid here
    
    beta_x, beta_y = jnp.array(quasar_pos)*theta_step
    beta_vals = jnp.sqrt((x_coords-beta_x)**2 + (y_coords-beta_y)**2)
    
    def T(theta):
        theta_min = theta_500_791#*.001
        return T_500*((theta+theta_min)/radius)**(-.75)
    
    B_of_theta = (1/(jnp.exp((h*c)/(wavelength*K*T(beta_vals)))-1))*(const/(wavelength**5))
    
    return B_of_theta

def sparse_quasar_image(beta, radius, theta_max, beta_min, quasar_int_func, **kwargs):
    """
    beta_min: minimum beta value to consider
    Returns:
        sparse_image: scipy.sparse.coo_matrix
        coords: tuple of (row_indices, col_indices)
        values: intensity values at those coordinates
    """
    N = beta[0].shape[0]
    
    beta_vals = jnp.sqrt(beta[0]**2 + beta[1]**2)
    
    # define 'interesting' regions
    mask = beta_vals < beta_min
    row_indices, col_indices = jnp.where(mask)
    selected_beta = (beta[0][mask], beta[1][mask])
    
    # Create a small grid just for the selected points
    selected_grid = jnp.zeros((2, len(row_indices)))
    selected_grid = jnp.stack([
    beta[0][row_indices, col_indices],
    beta[1][row_indices, col_indices]], axis=0)
    values = quasar_int_func(selected_grid[0], selected_grid[1], radius, theta_max,N, **kwargs)
    sparse_image = coo_matrix((values, (row_indices, col_indices)), shape=(N, N))
    
    return sparse_image, (row_indices, col_indices), values
    
def parameter_uncertainties(F, theta_step, printout=True, numstars=1):
    """
    Compute parameter uncertainties from the Fisher matrix.
    https://supernova.lbl.gov/~evlinder/InfoMatrixNotes.pdf

    Uses Jacobi preconditioning before inversion: scale F to a correlation
    matrix (unit diagonal), invert, then unscale.  This reduces the condition
    number by the spread in Fisher diagonal elements and preserves float64
    precision throughout.
    """
    F = np.array(F, dtype=np.float64)

    # Jacobi preconditioning: d_i = sqrt(F[i,i])
    d = np.sqrt(np.maximum(np.diag(F), 0.0))
    d[d == 0] = 1.0
    F_corr = F / np.outer(d, d)          # correlation matrix — diagonal = 1, off-diag in [-1, 1]
    C_corr = np.linalg.inv(F_corr)       # invert the well-conditioned version
    cov_matrix = C_corr / np.outer(d, d) # unscale to physical covariance

    uncertainties = np.sqrt(np.abs(np.diag(cov_matrix)))
    if printout:
        print('Source radius uncertainty [rad]: ', uncertainties[0])
        for i in range(numstars):
            print('Lensing star ', i+1, 'mass uncertainty [M_0]: ', uncertainties[1+i])
        for j in range(numstars):
            print('Lensing star ', j+1, 'x position uncertainty [rad]: ', uncertainties[1+numstars+j]*theta_step)
        for k in range(numstars):
            print('Lensing star ', k+1, 'y position uncertainty [rad]: ', uncertainties[1+2*numstars+k]*theta_step)

    return cov_matrix, uncertainties


def place_star_radial(star_masses, fraction, angle=np.pi/4, image_size=7000, phi_0=-2.078586999806823, theta_max=2e-10):
    """
    Cleanly handle units and everything so we can give a fraction f and this will return the indicies we should put a star of mass m at 
    so that it is at f*theta_E for that star, and a 7000 x 7000 image (by default) 
    default angle is along the 45 in the second quadrant (arb)
    """
    N_step = image_size
    theta_step = 2 * theta_max / image_size

    # Einstein radius (radians)
    theta_E = jnp.array([
        jnp.sqrt(4 * G * (star_masses * u.solMass.to(u.gram)) / c**2 * D_ls / (D_s * D_l))
    ])

    # radial coordinate in radians
    r_vals = fraction * theta_E

    # angular coordinate
    theta = phi_0 + np.pi / 2
    phi = np.array((np.pi - theta - angle))

    # meshgrid (keeps same output shape as before)
    rr, pp = np.meshgrid(r_vals, phi, indexing='ij')

    # Cartesian (radians)
    x = rr * np.cos(pp)
    y = rr * np.sin(pp)

    positions_plot = np.stack([x.flatten(), y.flatten()], axis=1)

    # map [-theta_max, +theta_max] → [0, image_size]  (same shape as before)
    return ((positions_plot / (2 * theta_max)) + 0.5) * image_size


def likelihood_static(params, inputs, num_stars, data_obs, sigma, sparse, ellipse_pts, pad_frac,
                      sampled=False, beta_min=0.9e-11):
    """
    Log-likelihood for II observations at fixed (static) lens position.

    params: (r_s, m_l, x, y) — source radius in mas, lens mass in M_sun, lens x/y position in mas
    inputs: (totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l, DLS, g_params)
    num_stars: 0 (null/macro-only) or 1
    data_obs: correlation values sampled at ellipse_pts
    sigma: per-point uncertainty on correlation
    sparse: always True in current usage
    ellipse_pts: (v_indices, u_indices) pixel-index arrays into the correlation image
    pad_frac: zero-padding fraction applied before FFT
    beta_min: sparse-image cutoff radius in radians
    """
    if sampled:
        uv_tracks, totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l, DLS, g_params = inputs
    else:
        totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l, DLS, g_params = inputs

    source_rad, m_stars, xpos, ypos = params
    theta_step = 2 * theta_max / len(theta[0])

    if num_stars == 0:
        m_stars = jnp.array([])
        xpos = jnp.array([])
        ypos = jnp.array([])

    source_rad = source_rad / rad_to_arcsec  # mas → radians
    m_stars = jnp.array(m_stars)
    xpos = jnp.array(xpos) / rad_to_arcsec   # mas → radians
    ypos = jnp.array(ypos) / rad_to_arcsec

    arr_rand_idx = jnp.array((xpos, ypos)).T if len(xpos) > 0 else np.empty((0, 2))
    star_params = (m_stars, Ds, D_l, DLS, theta_max, arr_rand_idx)

    if sparse:
        _, beta_sparse = beta(g_params, star_params, theta)
        image, (indicesx, indicesy), vals = sparse_quasar_image(
            beta_sparse, source_rad, theta_max, beta_min, quasar_int_sparse)
        theta_step = 2 * theta_max / beta_sparse.shape[-1]
        _, _, model_values = calcC_sparse(
            pad_image(image, int(beta_sparse.shape[-1] * pad_frac)),
            theta_step, box_size=box_size, ellipse=False)
    else:
        _, image, _, _ = increase_res(intensity_fun, source_rad, g_params, star_params, theta)
        theta_step = 2 * theta_max / len(image[0])
        model_values = calcC(
            pad_image(image, len(image[0]) * pad_frac), theta_step,
            box_size=box_size, center_ind=center_ind, ellipse=False)

    model_values = model_values[ellipse_pts[0], ellipse_pts[1]]

    residuals = (model_values - data_obs) ** 2 / sigma ** 2
    II_term = -0.5 * jnp.sum(jnp.abs(residuals))
    return jnp.real(II_term), jnp.real(II_term)


def fisher_three_pt(params, inputs, num_stars, data_obs, sigma, delta, ll_func, elps_pts,
                    pad_frac=10, sparse=False, sampled=False):
    """
    Fisher information matrix via 4-point mixed-partial stencil.

    params: (r_s, [m_l], [x], [y]) in units: mas, M_sun, mas, mas
    delta:  [delta_r, delta_m, delta_x, delta_y] — perturbation per parameter type
    ll_func: callable with signature ll_func(params, inputs, num_stars, data_obs, sigma,
                                             sparse, elps_pts, pad_frac, sampled)

    NOTE on diagonal elements: when i == j the effective step is 2*delta[i], not delta[i],
    because both the i and j perturbations are applied to the same parameter.  The formula
    computes (f(x+2h) - 2f(x) + f(x-2h)) / (4h²), which is correct but uses a coarser step
    than intended.  For highest accuracy on diagonal elements use a dedicated 3-point stencil.

    NOTE on units: the Fisher matrix mixes mas (rows 0,2,3) and M_sun (row 1).  Invert only
    after rescaling to a single unit system (see rescale_fisher_to_radians in the notebook).
    """
    totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l, DLS, g_params = inputs

    if num_stars == 0:
        params = [params[0], [], [], []]

    source_rad, m_stars, xpos, ypos = params
    flat_params = [source_rad] + list(m_stars) + list(xpos) + list(ypos)

    delta = [delta[0]] + delta[1:] * num_stars
    n_params = len(flat_params)

    fisher_full = np.zeros((n_params, n_params))
    fisher_II   = np.zeros((n_params, n_params))

    def rebuild_params(flat):
        return [flat[0],
                flat[1:num_stars + 1],
                flat[num_stars + 1:2 * num_stars + 1],
                flat[2 * num_stars + 1:]]

    def call(p):
        return ll_func(rebuild_params(p), inputs, num_stars, data_obs, sigma,
                       sparse, elps_pts, pad_frac, sampled)

    # Evaluate at unperturbed point once — reused for every diagonal element.
    ll_0, II_0 = call(flat_params)

    for i in range(n_params):
        for j in range(i, n_params):
            dtheta_i = delta[i]
            dtheta_j = delta[j]

            if i == j:
                # 3-point stencil: (f(x+h) - 2f(x) + f(x-h)) / h²
                # The old 4-point loop applied both the i and j perturbations to the same
                # parameter when i==j, giving an effective step of 2h and a denominator of
                # 4h² — correct in principle but twice as coarse as intended.
                params_p = copy.deepcopy(flat_params)
                params_m = copy.deepcopy(flat_params)
                params_p[i] += dtheta_i
                params_m[i] -= dtheta_i

                ll_p, II_p = call(params_p)
                ll_m, II_m = call(params_m)

                fisher_full[i, i] = -(ll_p - 2 * ll_0 + ll_m) / dtheta_i ** 2
                fisher_II[i, i]   = -(II_p - 2 * II_0 + II_m) / dtheta_i ** 2
            else:
                # 4-point cross-partial stencil for off-diagonals
                params_pp = copy.deepcopy(flat_params)
                params_pm = copy.deepcopy(flat_params)
                params_mp = copy.deepcopy(flat_params)
                params_mm = copy.deepcopy(flat_params)

                params_pp[i] += dtheta_i;  params_pp[j] += dtheta_j
                params_pm[i] += dtheta_i;  params_pm[j] -= dtheta_j
                params_mp[i] -= dtheta_i;  params_mp[j] += dtheta_j
                params_mm[i] -= dtheta_i;  params_mm[j] -= dtheta_j

                ll_pp, II_pp = call(params_pp)
                ll_pm, II_pm = call(params_pm)
                ll_mp, II_mp = call(params_mp)
                ll_mm, II_mm = call(params_mm)

                def cross_deriv(a_pp, a_pm, a_mp, a_mm):
                    return (a_pp - a_pm - a_mp + a_mm) / (4 * dtheta_i * dtheta_j)

                fisher_full[i, j] = -jnp.sum(cross_deriv(ll_pp, ll_pm, ll_mp, ll_mm))
                fisher_II[i, j]   = -jnp.sum(cross_deriv(II_pp, II_pm, II_mp, II_mm))

                fisher_full[j, i] = fisher_full[i, j]
                fisher_II[j, i]   = fisher_II[i, j]

    return fisher_full, fisher_II


def model_C(params, inputs, num_stars, ellipse_pts, sparse=True, pad_frac=0,
            beta_min=0.9e-11):
    """
    Return model correlation values at ellipse_pts for given params.
    Same parameter/input convention as likelihood_static.
    """
    totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l, DLS, g_params = inputs
    source_rad, m_stars, xpos, ypos = params

    if num_stars == 0:
        m_stars = jnp.array([])
        xpos    = jnp.array([])
        ypos    = jnp.array([])

    source_rad = source_rad / rad_to_arcsec
    m_stars    = jnp.array(m_stars)
    xpos       = jnp.array(xpos) / rad_to_arcsec
    ypos       = jnp.array(ypos) / rad_to_arcsec

    arr_rand_idx = jnp.array((xpos, ypos)).T if len(xpos) > 0 else np.empty((0, 2))
    star_params  = (m_stars, Ds, D_l, DLS, theta_max, arr_rand_idx)

    _, beta_sparse = beta(g_params, star_params, theta)
    image, _, _ = sparse_quasar_image(
        beta_sparse, source_rad, theta_max, beta_min, quasar_int_sparse)
    theta_step = 2 * theta_max / beta_sparse.shape[-1]
    _, _, corr = calcC_sparse(
        pad_image(image, int(beta_sparse.shape[-1] * pad_frac)),
        theta_step, box_size=box_size, ellipse=False)
    return corr[ellipse_pts[0], ellipse_pts[1]]


def fisher_gram(params, inputs, num_stars, sigma, delta, elps_pts,
                pad_frac=0, beta_min=0.9e-11):
    """
    Fisher information matrix F = Jᵀ J / σ² via central differences of model_C.

    Guaranteed positive semi-definite. Requires 2*n_params model evaluations.

    params: (r_s, [m_l], [x], [y]) in units: mas, M_sun, mas, mas
    sigma:  array of per-point correlation uncertainties (length = n_obs)
    delta:  [delta_r, delta_m, delta_x, delta_y] — perturbation per parameter type
    """
    source_rad, m_stars, xpos, ypos = params
    flat_params = [source_rad] + list(m_stars) + list(xpos) + list(ypos)
    delta_flat  = ([delta[0]]
                   + [delta[1]] * num_stars
                   + [delta[2]] * num_stars
                   + [delta[3]] * num_stars)
    n_params    = len(flat_params)

    def rebuild(flat):
        return [flat[0],
                flat[1:num_stars + 1],
                flat[num_stars + 1:2 * num_stars + 1],
                flat[2 * num_stars + 1:]]

    def eval_C(flat):
        # float64: JAX defaults to float32; explicit cast avoids rounding errors in J^T J
        return np.array(model_C(rebuild(flat), inputs, num_stars, elps_pts,
                                True, pad_frac, beta_min), dtype=np.float64)

    n_obs = len(eval_C(flat_params))
    J     = np.zeros((n_obs, n_params), dtype=np.float64)

    for i in range(n_params):
        p_plus  = list(flat_params); p_plus[i]  += delta_flat[i]
        p_minus = list(flat_params); p_minus[i] -= delta_flat[i]
        J[:, i] = (eval_C(p_plus) - eval_C(p_minus)) / (2 * delta_flat[i])

    sigma_arr = np.array(sigma, dtype=np.float64)
    return J.T @ (J / sigma_arr[:, None] ** 2)