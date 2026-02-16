from II_funcs import *

def compute_magnification(params, inputs, num_stars, totalI_obs, beta_min=0.9e-11, **kwargs):
    """
    params: source radius in mas, lens mass in solar masses, and lens position in mas
    inputs: 
            theta array - points we're using for image generation
            theta_max - size of the sky we're looking at
            Ds - source distance in cm
            D_l - lens distance in cm
            DLS - distance from source to lens
                ** all angular_diameter_distances, so DLS =/= Ds-D-l
            g_params - macrolensing paramters (kappa, shear1, shear2)
    num_stars: 0 or 1
    totalI_obs: sum of pixels in truth image
    returns magnification, the sum of intensity in an image generated with the parameters input here divided by the truth (observed) intensity 
    """
    
    source_rad, m_stars, xpos, ypos = params
    theta, theta_max, Ds, D_l, DLS, g_params = inputs
    xpos = jnp.array(xpos)/(rad_to_arcsec) #given in mas, convert to radians, above was for a pixelated earlier code version
    ypos =jnp.array(ypos)/(rad_to_arcsec)
    
    star_params = (jnp.array(m_stars), Ds, D_l, DLS, theta_max, jnp.array((xpos, ypos)))
    _, beta_sparse = beta(g_params, star_params, theta)
    
    image, _, _ = sparse_quasar_image(beta_sparse, source_rad/rad_to_arcsec, theta_max, 0.9e-11, quasar_int_sparse )
    total_I_complex = image.data.sum()
    return (total_I_complex / totalI_obs)

def compute_jacobian(params_init, model_func, delta, inputs, num_stars, totalI_obs):
    """
    params_init: source radius in mas, lens mass in solar masses, and lens position in mas
    model_func: compute magnification 
    delta: step size, requires a small amount of tuning for close vs far lenses
    inputs: other model inputs (see compute_magnification)
    num_stars: almost always 1, could be 0
    totalI_obs: sum of pixels in truth image

    returns J, the gradient of the model prediction. J_i = d (mu)/d theta_i for paramters theta
    """
    npar = 4 
    y0 = 1 # always will be one, define to save on computation time
    J = np.zeros(npar, dtype=float)
    source_rad, m_stars, xpos, ypos = params_init
    flat_params = [source_rad] + list(m_stars) + list(xpos) + list(ypos)
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
rad_mas=rad*rad_to_arcsec # really should be rad to micro arcsec, slight misnomer
g_params = (kappa_bulk, gamma1, gamma2)
sigma_C = error_C(mag=16.5, wavelength=500*1e-7, R = 2e4, sigma_t=3e-12, A=10*(500**2)*np.pi, t_obs= 8 * 3600)


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


#max of 7.2 thetaE should be more than far enough to see the full area we would be able to resolve
# define the points we will consider, along what angles etc
phi = np.linspace(0,np.pi/2, 5) 
max_y = (np.array(place_star_radial(jnp.asarray(0.1), 7.2, angle=-np.pi/2,#phi,
                                   image_size=image_size, phi_0=phi_0, theta_max=theta_max))/image_size-0.5)*theta_step*image_size

max_x = (np.array(place_star_radial(jnp.asarray(0.1), -7.2, angle=0,
                                   image_size=image_size, phi_0=phi_0, theta_max=theta_max))/image_size-0.5)*theta_step*image_size
ex = max_x[0] #get the vectors
ey = max_y[0]

x_vals = np.linspace(0, 1, 10)
y_vals = -np.linspace(0, 1, 10)

# build the grids of x and y vals projected along ex, ey 
positions_0, positions_1= np.meshgrid(x_vals, y_vals, indexing="xy") #unrotated
positions_plot= (positions_0[..., None] * ex + positions_1[..., None] * ey).reshape(-1,2) #rotate it
positions=positions_plot/theta_step+(image_size/2)

if __name__ == "__main__":
    # we use  a slightly smaller image here since we don't need to resolve things in uv space, so constraints are looser
    theta_max = 1.5e-10 # (half) size of image
    image_size=8000

    # define what doesn't change
    N_step = image_size
    g_params=(kappa_bulk, gamma1, gamma2)
    theta_step = 2 * theta_max / image_size
    vec_theta = np.linspace(-theta_max, theta_max, N_step)
    arr_theta = np.asarray(np.meshgrid(vec_theta, vec_theta, indexing='xy'))
    #
    #threshold for resolving points in the sparse image we generate
    beta_min = 0.9e-11
    rad = theta_500
    rad_mas=rad*rad_to_arcsec #micro arcseconds
    g_params = (kappa_bulk, gamma1, gamma2)
    mag_error = 0.101

    M1 = 0.1 # mass in M_0
    star1 = [M1]
    m_stars = [.1]

    # include this if this is already partially run
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

        # again, include if some positions were run previously 
        # skip if already processed (with tolerance)
        #if any(np.allclose(pos_arr, p, atol=1e-6) for p in processed_positions):
        #    print(f"Skipping {pos_arr} (already processed)")
        #    continue

        #generate observed values
        arr_rands = jnp.array([[pos[0]*theta_step, pos[1]*theta_step]])
        star_params = (jnp.asarray(star1), D_s, D_l, D_ls, theta_max, arr_rands)
        alpha_, beta_ = beta(g_params, star_params, arr_theta)
        image_new, indices, vals = sparse_quasar_image(beta_, rad, theta_max, beta_min, quasar_int_sparse)
        total_I_0=image_new.data.sum()
        
        params_initial = [(rad_mas), (jnp.asarray(m_stars)), arr_rands.T[0]*rad_to_arcsec, arr_rands.T[1]*rad_to_arcsec]
        delta=[rad_mas*1e-4 , 1e-5, .05*theta_step*rad_to_arcsec, .01*theta_step*rad_to_arcsec]
        input_test=[arr_theta, theta_max, D_s, D_l, D_ls, g_params]

        #compute gradient 
        J_mag = compute_jacobian(params_initial, compute_magnification, delta, input_test, 1, total_I_0)
        #compute fisher from gradient
        F_mag = np.outer(J_mag, J_mag) / mag_error**2

        #also calcaulte magnification for completeness
        mag_here=(total_I_0 / intensity_unlensed)
        chi2_val= chi2(mag_here, 1, mag_error)
    
        with open("m01_72tE_mag.txt", "a") as f:
            f.write(f"{pos}\t{F_mag}\n")
        with open("m01_72tE_chi2_mag.txt", "a") as f:
            f.write(f"{pos}\t{chi2_val}\n")

