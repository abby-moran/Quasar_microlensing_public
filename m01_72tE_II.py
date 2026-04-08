from II_funcs import *

def likelihood_static(params, inputs, num_stars, data_obs, sigma, sparse, ellipse_pts, pad_frac,
                      sampled=False, beta_min = 0.9e-11):
    """
    Returns the log-likelihood for given parameters. Handles null tests where num_stars = 0.
    Inputs:
        inputs:
            totalI_obs: measured intensity (sum)
            box_size: old, we don't use this anymore
            theta: array we'll warp to make beta
            intensity_fun: intensity function we will use to make our image from beta grid
    params: pertrubed parameters we're attempting to constrain (r_s, m_l, x, y)
        r_s in mas
        m_l in solar masses
        x,y in mas
    num_stars: 0 or 1 for us right now
    data_obs: correlation function values sampled at ellipse_pts
    sigma: error on correlation values
    sparse: always on at this point
    pad_frac: how much we pad our image when we pass it to uv space 
    sampled: old, always False at this point
    beta_min: I never touch this param anymore, cuttoff for beta when we do sparse image creation
    
    """
    #this is old, sampled is always False for us, I probably should remove this functionality at some point
    if sampled:
         uv_tracks,totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l,DLS, g_params = inputs
    else:
        totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l , DLS, g_params =inputs
    
    source_rad, m_stars, xpos, ypos = params
    theta_step = 2 * theta_max / len(theta[0])
    
    # Handle null case
    if num_stars == 0:
        m_stars = jnp.array([])  # Empty array for null test
        xpos = jnp.array([])
        ypos = jnp.array([])

    # Ensure arrays are valid
    source_rad=source_rad/rad_to_arcsec #given in mas, convert to radians
    m_stars = jnp.array(m_stars)
    
    #xpos =  (jnp.round(jnp.array(xpos)/(theta_step*rad_to_arcsec)).astype(int))#given in mas, convert to radians then pixels
    #ypos = (jnp.round(jnp.array(ypos)/(theta_step*rad_to_arcsec)).astype(int))
    xpos=jnp.array(xpos)/(rad_to_arcsec) #given in mas, convert to radians, above was for a pixelated earlier code version
    ypos =jnp.array(ypos)/(rad_to_arcsec)

    # Create star parameters
    arr_rand_idx = jnp.array((xpos, ypos)).T if len(xpos) > 0 else np.empty((0, 2))
    star_params = (m_stars, Ds, D_l, DLS, theta_max, (arr_rand_idx))
    
    if sparse:
        _, beta_sparse =beta(g_params, star_params, arr_theta)
        image, (indicesx, indicesy), vals = sparse_quasar_image(beta_sparse, source_rad, theta_max, beta_min, quasar_int_sparse)
        theta_step=2*theta_max/beta_sparse.shape[-1]
        _,_, model_values  = calcC_sparse(pad_image(image, int(beta_sparse.shape[-1]*pad_frac)), theta_step, 
                                     box_size=box_size, ellipse=False) #get full correlation, sample later

    else: #rarely are we using this part
        _, image, _, _ = increase_res(intensity_fun, source_rad, g_params, star_params, theta,)
        theta_step_hr = 2 * theta_max / len(image[0])
        theta_step= 2 * theta_max / len(image[0])

        model_values = calcC(pad_image(image, len(image[0])*pad_frac), theta_step_hr,
                              box_size=box_size, center_ind=center_ind, ellipse=False)
   
    #totalI_complex = jnp.sum(image.data.astype(jnp.complex128))  # complex, used for mag term only
    model_values=model_values[ellipse_pts[0], ellipse_pts[1]]  #sample the same as we on 'observed' data
   
    
    #magnification=(totalI_complex/totalI_obs)-1 
    #mag_error = 0.101  # fractional flux uncertainty from 0.11 mag
    #mag_term = -0.5 * ((magnification) / mag_error)**2 

    residuals = (model_values - data_obs)**2 / sigma**2
    II_term = -0.5 * jnp.sum(jnp.abs(residuals))
    log_likelihood = II_term  #+ mag_term
    return jnp.real(log_likelihood), jnp.real(II_term)#, jnp.real(mag_term)
def fisher_three_pt(params, inputs, num_stars, data_obs, sigma, delta, ll_func, elps_pts, pad_frac=10, sparse=False, 
           sampled=False):
    """
    Calculate the Fisher information matrix.
    Handles 0 to infinity stars by dynamically adjusting parameters.
    This is my old function which uses only the three point central difference to get the 2nd derivative. 
    
    This is also pretty slow on Jax because of how I copy the parameters, but it is the most stable when used with the discrete 
    alpha function I was using before with the II term
    Inputs:
        inputs:
            totalI_obs: measured intensity (sum)
            box_size: old, we don't use this anymore
            theta: array we'll warp to make beta
            intensity_fun: intensity function we will use to make our image from beta grid
    params: un-perturbed parameters we used to make data_obs 
        r_s in mas
        m_l in solar masses
        x,y in mas
    delta: how much we perturb each parameter by
    """
    
    totalI_obs, box_size, theta, intensity_fun, theta_max, Ds, D_l , DLS,  g_params =inputs
    theta_step = 2 * theta_max / len(theta[0])
    
    # Ensure `params` has the correct structure
    if num_stars == 0:
        params = [params[0], [], [], []]  # Add empty lists for m_stars, xpos, ypos

    # Flatten parameters for iteration
    source_rad, m_stars, xpos, ypos = params
    flat_params = [source_rad] + list(m_stars) + list(xpos) + list(ypos)
    
    # Adjust delta structure
    # just using one delta for both in this version of the code, might need to change that if we go back to these lower
    # order derivatives
    delta = [delta[0]] + delta[1:] * num_stars
    n_params = len(flat_params)
    
    fisher_full = np.zeros((n_params, n_params))
    fisher_II = np.zeros((n_params, n_params))
    fisher_mag = np.zeros((n_params, n_params))
    
    # Iterate over parameter pDLS,airs
    for i in range(n_params):
        for j in range(i, n_params):
            # Perturb parameters
            dtheta_i = delta[i]
            dtheta_j = delta[j]

            # Deep copy of parameters
            params_pp = copy.deepcopy(flat_params)
            params_pm = copy.deepcopy(flat_params)
            params_mp = copy.deepcopy(flat_params)
            params_mm = copy.deepcopy(flat_params)

            # Add perturbations
            params_pp[i] += dtheta_i
            params_pm[i] += dtheta_i
            params_mp[i] -= dtheta_i
            params_mm[i] -= dtheta_i

            params_pp[j] += dtheta_j
            params_pm[j] -= dtheta_j
            params_mp[j] += dtheta_j
            params_mm[j] -= dtheta_j
            
            def rebuild_params(flat):
                source_rad = flat[0]
                m_stars = flat[1:num_stars + 1]
                xpos = flat[num_stars + 1:2 * num_stars + 1]
                ypos = flat[2 * num_stars + 1:]
                return [source_rad, m_stars, xpos, ypos]

            params_pp = rebuild_params(params_pp)
            params_pm = rebuild_params(params_pm)
            params_mp = rebuild_params(params_mp)
            params_mm = rebuild_params(params_mm)
            
            ll_pp, II_pp= ll_func(params_pp, inputs, num_stars, data_obs, sigma, sparse, elps_pts, pad_frac, sampled)
            ll_pm, II_pm = ll_func(params_pm, inputs, num_stars, data_obs, sigma, sparse, elps_pts, pad_frac, sampled)
            ll_mp, II_mp= ll_func(params_mp, inputs, num_stars, data_obs, sigma, sparse, elps_pts, pad_frac, sampled)
            ll_mm, II_mm = ll_func(params_mm, inputs, num_stars, data_obs, sigma, sparse, elps_pts, pad_frac, sampled)            
            
            # Compute second derivatives for each term
            def second_deriv(a_pp, a_pm, a_mp, a_mm):
                return (a_pp - a_pm - a_mp + a_mm) / (4 * dtheta_i * dtheta_j)

            #store the negative of each derivative
            fisher_full[i, j] = -jnp.sum(second_deriv(ll_pp, ll_pm, ll_mp, ll_mm))
            fisher_II[i, j] = -jnp.sum(second_deriv(II_pp, II_pm, II_mp, II_mm))
            #fisher_mag[i, j] = -jnp.sum(second_deriv(mag_pp, mag_pm, mag_mp, mag_mm))
            
            # Symmetrize
            fisher_full[j, i] = fisher_full[i, j]
            fisher_II[j, i] = fisher_II[i, j]
            #fisher_mag[j, i] = fisher_mag[i, j]

    return fisher_full, fisher_II#, fisher_mag

#define image specific constants, these don't change within this file
theta_max = 2e-10 # (half) size of image
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
sigma_C = error_C(mag=15.76, wavelength=570*1e-7, R = 2e4, sigma_t=3e-12, A=10*(500**2)*np.pi, t_obs= 8 * 3600)


#get unlensed rotation angle, correlation points
arr_rands = jnp.array([[(300-239), (500-68)]])
star_params = (jnp.asarray([0]), D_s, D_l, D_ls, theta_max, arr_rands)
alpha_0, beta_0 = beta(g_params, star_params, arr_theta)
pad_frac = 0 #padding turns out to not be needed for these params
image_0, _,_ = sparse_quasar_image(beta_0, rad, theta_max, beta_min, quasar_int_sparse)
u_full,v_full, corr_0_full=calcC_sparse(image_0, theta_step, ellipse=False)
upts_8k, vpts_8k, corr_0 = calcC_sparse(image_0, theta_step, ellipse=True)
elps_pts=jnp.asarray((vpts_8k, upts_8k)) #reversed again, as we do in the corr_C function
phi_0=-get_rot_angle(beta_0)


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
    theta_max = 2e-10 # (half) size of image
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

    M1 = 0.1 #* u.solMass.to(u.g)
    star1 = [M1]
    m_stars = [.1]
    
    for i in range(len(positions)):
        pos = positions[i]
        pos_arr = np.array([float(pos[0]), float(pos[1])])
    
        arr_rands = jnp.array([[pos[0]*theta_step, pos[1]*theta_step]])
        star_params = (jnp.asarray(star1), D_s, D_l, D_ls, theta_max, arr_rands)
        alpha_, beta_ = beta(g_params, star_params, arr_theta)
        image_new, indices, vals = sparse_quasar_image(beta_, rad, theta_max, beta_min, quasar_int_sparse)
        total_I_0=image_new.data.sum()
        _,_,corr_lensed = calcC_sparse(image_new, theta_step, box_size=100, ellipse=False)
        corr_lensed=corr_lensed[vpts_8k, upts_8k]
        err_c_full = jnp.full_like(corr_lensed, (sigma_C/jnp.sqrt(10)))
        input_test = [total_I_0, 100, arr_theta, sparse_quasar_image, theta_max, D_s, D_l, D_ls, g_params]
        params_initial = [rad_mas, [jnp.asarray(m_stars)], arr_rands.T[0]*rad_to_arcsec, arr_rands.T[1]*rad_to_arcsec]
        delta=[theta_step*rad_to_arcsec/2, (2e-5), .01*theta_step*rad_to_arcsec, .01*theta_step*rad_to_arcsec]
        
        fisher_II = fisher_three_pt(params_initial, input_test, 1, corr_lensed, err_c_full,
                   delta, likelihood_static, jnp.asarray((vpts_8k, upts_8k)), pad_frac=0, sparse=True)
        
        cov_matrix_f, uncertainties_f = parameter_uncertainties(fisher_II[0], theta_step, numstars=1, printout=False)
        chi2_val= chi2(corr_lensed, corr_0, err_c_full[0])
        delta_text='delta=0'
        #delta_text is from a check to make sure delta values aren't all over the place (they aren't)

        # a little bit of delta tuning is needed, need a few values for very close/very far points, particualrly at this low a mass
        if jnp.isnan(jnp.array(uncertainties_f,  dtype=jnp.float64)).any():
            print('nan at ', pos)
            delta=[theta_step*rad_to_arcsec/2, (1e-4), 2*theta_step*rad_to_arcsec, 2*theta_step*rad_to_arcsec]
            fisher_II = fisher_three_pt(params_initial, input_test, 1, corr_lensed, err_c_full,
                       delta, likelihood_static, jnp.asarray((vpts_8k, upts_8k)), pad_frac=0, sparse=True)
            cov_matrix_f, uncertainties_f = parameter_uncertainties(fisher_II[0], theta_step, numstars=1, printout=False)
            delta_text='delta=1'
        if jnp.isnan(jnp.array(uncertainties_f,  dtype=jnp.float64)).any():
            print('nan at ', pos)
            delta=[theta_step*rad_to_arcsec/2, (2e-5), 1*theta_step*rad_to_arcsec, 1*theta_step*rad_to_arcsec]
            fisher_II = fisher_three_pt(params_initial, input_test, 1, corr_lensed, err_c_full,
                       delta, likelihood_static, jnp.asarray((vpts_8k, upts_8k)), pad_frac=0, sparse=True)
            cov_matrix_f, uncertainties_f = parameter_uncertainties(fisher_II[0], theta_step, numstars=1, printout=False)
            delta_text='delta=2'
        
        if jnp.isnan(jnp.array(uncertainties_f,  dtype=jnp.float64)).any():
            print('nan at ', pos)
            delta=[theta_step*rad_to_arcsec/2, (2e-4), .3*theta_step*rad_to_arcsec, .4*theta_step*rad_to_arcsec]
            fisher_II = fisher_three_pt(params_initial, input_test, 1, corr_lensed, err_c_full,
                       delta, likelihood_static, jnp.asarray((vpts_8k, upts_8k)), pad_frac=0, sparse=True)
            cov_matrix_f, uncertainties_f = parameter_uncertainties(fisher_II[0], theta_step, numstars=1, printout=False)
            delta_text='delta=3'
        
        with open("m01_72tE_II.txt", "a") as f:
            f.write(f"{pos}\t{uncertainties_f[0]:.6e}\t"
                    f"{uncertainties_f[1]:.6e}\t{uncertainties_f[2]:.6e}\t"
                    f"{uncertainties_f[3]:.6e}\t{cov_matrix_f}\n")
        with open("m01_72tE_chi2_II.txt", "a") as f:
            f.write(f"{pos}\t{chi2_val}\n")
