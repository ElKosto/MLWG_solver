import numpy as np
# from numba import njit

# def transfer_matrix_TM_layer_seconds_approachs(n, d, theta_inc_rad, n_core, wavelength):

# @njit
def TM_boundary(n, d, theta_prev, n_prev, wavelength):
    """
    Calculate the transfer matrix for a single layer.
    ET polarization is assumed 
    Parameters:
        n: Refractive index of the layer.
        theta: Angle of incidence (in radians).
        wavelength: Wavelength of incident light (in um).
        des not need d as a parameter - done to have same input as in propagation
    Returns:
        Matrix representing the transfer matrix of the layer bound (saleh).
    """
    # Find the new refraction angle using snell's law assuming n is complex
    theta = np.arcsin(n_prev / n * np.sin(theta_prev),dtype='complex')
    n_1  = n_prev*np.cos(theta_prev,dtype='complex')
    n_2 = n*np.cos(theta,dtype='complex')
    matrix = np.array([[ n_1 + n_2 , n_2-n_1],
                       [ n_2-n_1, n_1 + n_2]],dtype='complex')/2/n_2
    return matrix,theta

# @njit
def TM_propagation(n, d, theta_prev, n_prev, wavelength):
    """
    Calculate the transfer matrix for the propagation in a layer.
    """
    theta = np.arcsin(n_prev / n * np.sin(theta_prev),dtype='complex')
    k = 2 * np.pi * n * np.cos(theta+0j) / wavelength
    phase = k * d
    zero = 0j if isinstance(theta, float) else np.zeros_like(theta)
    matrix = np.array([[ np.exp(-1j*phase), zero],[ zero, np.exp(+1j*phase)]],dtype='complex')

    return matrix


def mult_rules(a, b):
    """
    Check if 'a' is a 2x2 matrix for direct matrix multiplication
    Parameters:
        a,b - tensors of two possible kinds 2x2 or 2x2xNN where NN is arbitrary
    """
    if a.shape == (2, 2):
        # Perform 2D matrix multiplication using Einstein summation convention
        return np.einsum('ij,jl->il', a, b)
    else:
        # Assume 'a' is 3D (2x2x??), perform matrix multiplication slice-by-slice
        return np.einsum('ijk,jlk->ilk', a, b)


# @njit
def TM_multilayer(n_list, d_list, theta_in, n_core, wavelength):
    """
    Calculate the transfer matrix for a multilayer film.
    """
    if len(n_list)!= len(d_list):
        print("error n_list and d_list are not of the same size")
        
    M,theta = TM_boundary(n_list[0], d_list[0], theta_in, n_core, wavelength)
    M_prop = TM_propagation(n_list[0], d_list[0], theta_in, n_core, wavelength)
    M = mult_rules(M_prop,M)
    for ii in range(1,len(n_list)):
        M_bound,theta = TM_boundary(n_list[ii], d_list[ii], theta, n_list[ii-1], wavelength)
        M = mult_rules(M_bound,M)
        M_prop = TM_propagation(n_list[ii], d_list[ii], theta_in, n_list[ii-1], wavelength)
        M = mult_rules(M_prop,M)
        
    if len(n_list) == 1:
        M_bound, theta = TM_boundary(1., 1., theta, n_list[-1], wavelength)
        M_prop = TM_propagation(1., 10., theta, n_list[-1], wavelength)
        M = mult_rules(M_prop,mult_rules(M_bound,M))
    else:
        M_bound, theta = TM_boundary(1., 1., theta, n_list[-1], wavelength)
        M_prop = TM_propagation(1., 10., theta, n_list[-1], wavelength)
        M = mult_rules(M_prop,mult_rules(M_bound,M))
    return M

# @njit
def compute_reflection_transmission(n_list, d_list, theta, n_core, wavelength):
    """
    Compute reflection and transmission coefficients from a multilayer dielectric film.
    """
    M = TM_multilayer(n_list, d_list, theta, n_core, wavelength)
    t =  (M[0,0]*M[1,1] - M[0,1]*M[1,0])/M[1,1]
    r =  - M[1,0]/M[1,1]
    return r,t

