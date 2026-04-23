from II_funcs import *

def compute_jacobian(params_init, model_func, delta, inputs, num_stars):
    npar = 4 
    source_rad, m_stars, xpos, ypos = params_init
    flat_params = [source_rad] + list(m_stars) + list(xpos) + list(ypos)
    y0 = model_func(flat_params.copy(), inputs, num_stars)
    J=[]
    for i in range(npar): 
        perturbed = flat_params.copy() 
        perturbed[i] += delta[i] 
        yi = model_func(perturbed, inputs, num_stars) 
        J.append(np.array([(yi - y0) / delta[i]]))
    return J, y0
    
def compute_centroid(params, inputs, num_stars, beta_min=0.9e-11, **kwargs):
    source_rad, m_stars, xpos, ypos = params
    #xpos=xpos[0]
    #ypos=ypos[0]
    theta, theta_max, Ds, D_l, DLS, g_params = inputs
    xpos = jnp.array(xpos)/(rad_to_arcsec) #given in mas, convert to radians, above was for a pixelated earlier code version
    ypos =jnp.array(ypos)/(rad_to_arcsec)
    
    star_params = (jnp.array(m_stars), Ds, D_l, DLS, theta_max, jnp.array((xpos, ypos)))
    _, beta_sparse = beta(g_params, star_params, theta)
    
    image, _, _ = sparse_quasar_image(beta_sparse, source_rad/rad_to_arcsec, theta_max, 0.9e-11, quasar_int_sparse)
    image=image.toarray()
    #test_grid=np.meshgrid(np.linspace(0,1, 8000), np.linspace(0,1, 8000))
    cent_x =  np.sum(image * theta[0]) / np.sum(image)# -4000
    cent_y = np.sum(image * theta[1]) / np.sum(image)# -4000
    return np.array([cent_x, cent_y])

#define image specific constants, these don't change within this file
theta_max =2e-10 # (half) size of image
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


#get unlensed rotation angle, correlation points
arr_rands = jnp.array([[(300-239), (500-68)]])
star_params = (jnp.asarray([0]), D_s, D_l, D_ls, theta_max, arr_rands)
alpha_0, beta_0 = beta(g_params, star_params, arr_theta)
pad_frac = 0
image_0, _,_ = sparse_quasar_image(beta_0, rad, theta_max, beta_min, quasar_int_sparse)
intensity_unlensed=image_0.data.sum()
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
    theta_max =2e-10 # (half) size of image
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
    processed_positions = set()
        
    for i in range(len(positions)):
        pos = positions[i]
        pos_arr = np.array([float(pos[0]), float(pos[1])])
    
        ## skip if already processed (with tolerance)
        #if any(np.allclose(pos_arr, p, atol=1e-6) for p in processed_positions):
        #    print(f"Skipping {pos_arr} (already processed)")
        #    continue
        arr_rands = jnp.array([[pos[0]*theta_step, pos[1]*theta_step]])
        star_params = (jnp.asarray(star1), D_s, D_l, D_ls, theta_max, arr_rands)
        alpha_, beta_ = beta(g_params, star_params, arr_theta)
        image_new, indices, vals = sparse_quasar_image(beta_, rad, theta_max, beta_min, quasar_int_sparse)
        total_I_0=image_new.data.sum()
        params_initial = [(rad_mas), (jnp.asarray(m_stars)), arr_rands.T[0]*rad_to_arcsec, arr_rands.T[1]*rad_to_arcsec]
        delta=[rad_mas*1e-3 , 3e-3, .05*theta_step*rad_to_arcsec, 2*theta_step*rad_to_arcsec]
        input_test=[arr_theta, theta_max, D_s, D_l, D_ls, g_params]
        J_mag, cent0 = compute_jacobian(params_initial, compute_centroid, delta, input_test, 1)
        
        with open("m04_78tE_cent.txt", "a") as f:
            f.write(f"{pos}\t{J_mag}\t{cent0}\n")
        

