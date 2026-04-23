from II_funcs import *

def compute_magnification(params, inputs, num_stars, totalI_obs, beta_min=0.9e-11, **kwargs):
    source_rad, m_stars, xpos, ypos = params
    theta, theta_max, Ds, D_l, DLS, g_params = inputs
    xpos = jnp.array(xpos)/(rad_to_arcsec) #given in mas, convert to radians, above was for a pixelated earlier code version
    ypos =jnp.array(ypos)/(rad_to_arcsec)
    
    star_params = (jnp.array(m_stars), Ds, D_l, DLS, theta_max, jnp.array((xpos, ypos)))
    _, beta_sparse = beta(g_params, star_params, theta)
    
    image, _, _ = sparse_quasar_image(beta_sparse, source_rad/rad_to_arcsec, theta_max, 0.9e-11, quasar_int_sparse )
    total_I_complex = image.data.sum()
    return (total_I_complex / totalI_obs) - 1.0

def compute_jacobian(params_init, model_func, delta, inputs, num_stars, totalI_obs):
    #flat = np.asarray(flat_params, dtype=float)
    npar = 4 #flat.size
    y0 = 1 #model_func(params_init, inputs, num_stars, totalI_obs)
    #m = y0.size
    J = np.zeros(npar, dtype=float)
    source_rad, m_stars, xpos, ypos = params_init
    flat_params = [source_rad] + list(m_stars) + list(xpos) + list(ypos)
    #perturbed_params=jnp.asarray(flat_params)+jnp.asarray(delta)
    for i in range(npar):
        perturbed = flat_params.copy()
        perturbed[i] += delta[i]
        yi = model_func(perturbed, inputs, num_stars, totalI_obs)
        J[i] = (yi - y0) / delta[i]
    return J

#define image specific constants, these don't change within this file
theta_max = 1.5e-10 # (half) size of image
image_size=8000
N_step = image_size
g_params=(kappa_bulk, gamma1, gamma2)
theta_step = 2 * theta_max / image_size
vec_theta = np.linspace(-theta_max, theta_max, N_step)
arr_theta = np.asarray(np.meshgrid(vec_theta, vec_theta, indexing='xy'))
beta_min = 0.9e-11
rad = theta_500  
rad_mas=rad*rad_to_arcsec
g_params = (kappa_bulk, gamma1, gamma2)
sigma_C = error_C(mag=15.76, wavelength=570*1e-7, R = 2e4, sigma_t=3e-12, A=10*(500**2)*np.pi, t_obs= 8 * 3600)


arr_rands = jnp.array([[(300-239), (500-68)]])
star_params = (jnp.asarray([0]), D_s, D_l, D_ls, theta_max, arr_rands)
alpha_0, beta_0 = beta(g_params, star_params, arr_theta)
pad_frac = 0
image_0, _,_ = sparse_quasar_image(beta_0, rad, theta_max, beta_min, quasar_int_sparse)
intensity_unlensed=image_0.data.sum()
u_full,v_full, corr_0_full=calcC_sparse(image_0, theta_step, ellipse=False)
upts_8k, vpts_8k, corr_0 = calcC_sparse(image_0, theta_step, ellipse=True)
elps_pts=jnp.asarray((vpts_8k, upts_8k)) #reversed again, as we do in the corr_C function
phi_0=-get_rot_angle(beta_0)


# even more grid making for when we eventually get final results, not in use right now
# these blocks do define some good constants to have (like phi) so I've left them for now
dists=[0.05, .1, .2, .4, .6, .8, 1, 1.2, 1.4, 1.6]
phi = np.linspace(0,np.pi/2, 5) 
max_y = (np.array(place_star_radial(jnp.asarray(0.4), 7.8, angle=-np.pi/2,#phi,
                                   image_size=image_size, phi_0=phi_0, theta_max=theta_max))/image_size-0.5)*theta_step*image_size

max_x = (np.array(place_star_radial(jnp.asarray(0.4), -7.8, angle=0,
                                   image_size=image_size, phi_0=phi_0, theta_max=theta_max))/image_size-0.5)*theta_step*image_size
ex = max_x[0] #get the vectors
ey = max_y[0]

x_vals = np.linspace(0, 1, 12)
y_vals = -np.linspace(0, 1, 12)
positions_0, positions_1= np.meshgrid(x_vals, y_vals, indexing="xy") #unrotated
positions_plot= (positions_0[..., None] * ex + positions_1[..., None] * ey).reshape(-1,2) #rotate it
positions=positions_plot/theta_step+(image_size/2)

if __name__ == "__main__":
    theta_max = 1.5e-10 # (half) size of image
    image_size=8000
    N_step = image_size
    g_params=(kappa_bulk, gamma1, gamma2)
    theta_step = 2 * theta_max / image_size
    vec_theta = np.linspace(-theta_max, theta_max, N_step)
    arr_theta = np.asarray(np.meshgrid(vec_theta, vec_theta, indexing='xy'))
    beta_min = 0.9e-11
    rad = theta_500
    rad_mas=rad*rad_to_arcsec
    g_params = (kappa_bulk, gamma1, gamma2)
    mag_error = 0.101

    M1 = 0.4 #* u.solMass.to(u.g)
    star1 = [M1]
    m_stars = [.4]
    #processed_positions = set()
    #with open("position_check.txt", "a") as f:
    #        f.write(f"{positions}\n")
    #try:
    #    with open("m04_78tE_mag.txt", "r") as f:
    #        for line in f:
    #            if line.strip() and line.startswith('['):
    #                tokens = line.split('\t')[0].strip('[]').split()
    #                if len(tokens) >= 2:
    #                    pos_tuple = (float(tokens[0]), float(tokens[1]))
    #                    processed_positions.add(pos_tuple)
    #except FileNotFoundError:
    #    pass 
        
    for i in range(len(positions)):
        pos = positions[i]
        pos_arr = np.array([float(pos[0]), float(pos[1])])
    
        # skip if already processed (with tolerance)
        #if any(np.allclose(pos_arr, p, atol=1e-6) for p in processed_positions):
        #    print(f"Skipping {pos_arr} (already processed)")
        #    continue
        arr_rands = jnp.array([[pos[0]*theta_step, pos[1]*theta_step]])
        star_params = (jnp.asarray(star1), D_s, D_l, D_ls, theta_max, arr_rands)
        alpha_, beta_ = beta(g_params, star_params, arr_theta)
        image_new, indices, vals = sparse_quasar_image(beta_, rad, theta_max, beta_min, quasar_int_sparse)
        total_I_0=image_new.data.sum()
        #_,_,corr_lensed = calcC_sparse(image_new, theta_step, box_size=100, ellipse=False)
        #corr_lensed=corr_lensed[vpts_8k, upts_8k]
        #err_c_full = jnp.full_like(corr_lensed, (sigma_C/jnp.sqrt(10)))
        params_initial = [(rad_mas), (jnp.asarray(m_stars)), arr_rands.T[0]*rad_to_arcsec, arr_rands.T[1]*rad_to_arcsec]
        delta=[rad_mas*1e-4 , 1e-5, .05*theta_step*rad_to_arcsec, .01*theta_step*rad_to_arcsec]
        input_test=[arr_theta, theta_max, D_s, D_l, D_ls, g_params]
        J_mag = compute_jacobian(params_initial, compute_magnification, delta, input_test, 1, total_I_0)
        F_mag = np.outer(J_mag, J_mag) / mag_error**2
        # pseudo-inverse to handle possible rank-deficiency
        #w, V = np.linalg.eigh(F_total)
        #tol = 1e-20 * np.max(w)
        #w_inv = np.array([1.0 / wi if wi > tol else 0.0 for wi in w])
        #cov_combined = (V * w_inv) @ V.T
        mag_here=(total_I_0 / intensity_unlensed) - 1.0
        chi2_val= chi2(mag_here, 0, mag_error)
    
        with open("m04_78tE_mag.txt", "a") as f:
            f.write(f"{pos}\t{F_mag}\n")
        with open("m04_78tE_chi2_mag.txt", "a") as f:
            f.write(f"{pos}\t{chi2_val}\n")

