
import numpy as np
from .ML_coating import compute_reflection_transmission
from scipy.optimize import brentq
# from numba import njit,prange


# @njit
def optim_ml_pwg(neff, n_core, n_sub, n_ml, w, w_ml, wavelength, m, polar):
    k_0 = 2 * np.pi / wavelength 
    beta = neff * k_0
    theta = np.arccos(neff/n_core) ### theta is not in the incidence plane!
    h = k_0**2 * n_core**2 - beta**2
    q = beta**2 - k_0**2 * n_sub**2
    ### reflection phase from the substrate
    if polar == "TE": 
        phi_SUB = 2*np.arctan( np.sqrt(q/h))
    elif polar == "TM":
        phi_SUB = 2*np.arctan((n_core/n_sub)**2 * np.sqrt(q/h))
    ### reflection phase the multilayer cladding
    r, t = compute_reflection_transmission(n_ml, w_ml, np.pi/2-theta, n_core, wavelength, polar)
    # print(np.pi/2-theta,wavelength,' in')
    phi_ML = np.angle(r)
    # print(phui_SUB, wavelength, ' reflection')
    return np.sqrt(h) * w*2 - m*2*np.pi - phi_SUB - phi_ML


# Function to solve for h given a guess for neff
# @njit
def optim_asymmetric_pwg(neff, n_core, n_2, n_3, w, wavelength, m, polar):
    k_0 = 2 * np.pi / wavelength 
    beta = neff * k_0
    h = k_0**2 * n_core**2 - beta**2
    q = lambda n_x : beta**2 - k_0**2 * n_x**2
    if polar == "TE":
        phui_SUB = 2*np.arctan( np.sqrt(q(n_2)/h))
        phi_CLAD = 2*np.arctan( np.sqrt(q(n_3)/h))
    elif polar == "TM":## the right experession is take from wiki
        phui_SUB = 2*np.arctan((n_core/n_2)**2 * np.sqrt(q(n_2)/h))
        phi_CLAD = 2*np.arctan((n_core/n_3)**2 * np.sqrt(q(n_3)/h))
    return np.sqrt(h) * w*2 - m*2*np.pi - phui_SUB - phi_CLAD


# Function to solve for h given a guess for neff
# @njit
def optim_symmetric_pwg(neff, n_1, n_2, w, wavelength ,m, polar):
    k_0 = 2 * np.pi / wavelength 
    beta = neff * k_0
    h = k_0**2 * n_1**2 - beta**2
    q = beta**2 - k_0**2 * n_2**2
    if polar == "TE": 
        return np.sqrt(h) * w/2 - m*np.pi/2 - np.arctan( np.sqrt(q/h))
    elif polar == "TM":
        return np.sqrt(h) * w/2 - m*np.pi/2 - np.arctan((n_1/n_2)**2 * np.sqrt(q/h))

    
# @njit
def func_ML(neff, *args):
    return optim_ml_pwg(neff, *args)
# @njit
### calculate the simple version of the asymmtric thick cladding
def func(neff, *args):
    return optim_asymmetric_pwg(neff, *args)



# @njit(parallel=True)
def calc_n_eff_ML(wavelength, n_core, n_sub, n_clad, d, d_clad, m, polar):
    neff = np.zeros_like(wavelength)
    for ii in range(len(wavelength)):
        if type(n_core) == np.ndarray:
            neff[ii] = brentq(func_ML, n_sub[ii]+0.01, n_core[ii]-0.00001, args=(n_core[ii], n_sub[ii], n_clad[ii], d, d_clad, wavelength[ii], m), maxiter=5000)
        else:
            neff[ii] = brentq(func_ML, n_sub+0.01, n_core-0.00001, args=(n_core, n_sub, n_clad, d, d_clad, wavelength[ii], m), maxiter=5000)
    return neff

# @njit(parallel=True)
def calc_n_eff(wavelength, n_core, n_sub, n_clad, d, m, polar):
    neff = np.zeros_like(wavelength)
    for ii in range(len(wavelength)):
        if type(n_core) == np.ndarray:
            neff[ii] = brentq(func, max(n_sub[ii],n_clad[ii])+0.01, n_core[ii]-0.00001, args=(n_core[ii], n_sub[ii], n_clad[ii], d, wavelength[ii], m), maxiter=5000)
        else:
            neff[ii] = brentq(func, max(n_sub,n_clad)+0.01, n_core-0.00001, args=(n_core, n_sub, n_clad, d, wavelength[ii], m), maxiter=5000)
    return neff

 
def find_zero_crossings(func, range_of_n_eff, args):
    """
    Identifies intervals where a function crosses zero between two points.
    
    Parameters:
    - func: The function to evaluate, expected to change signs across zero crossings.
    - n_sub: Lower bound of the search interval (e.g., substrate refractive index).
    - n_core: Upper bound of the search interval (e.g., core refractive index).
    - args: Additional arguments to pass to the function.
    - num_points: Number of points to evaluate within the interval for detecting sign changes.
    
    Returns:
    - intervals: A list of tuples, each representing an interval (start, end) where a zero crossing is likely.
    """
    # Generate a linear space between n_sub and n_core

    # Evaluate the function at each point in the linear space
    func_values = func(range_of_n_eff, *args)
    
    # Identify where the function changes sign
    sign_changes = np.diff(np.sign(func_values)) != 0
    
    # Find indices just before the sign changes (potential zero crossings)
    indices = np.where(sign_changes)[0]
    
    # Generate intervals based on indices where sign changes occur##  commented part kicks the sharp changes
    intervals = [(range_of_n_eff[ii], range_of_n_eff[ii+1])for ii in indices if abs(func_values[ii]-func_values[ii+1])<1]
    
    return intervals



def calc_n_eff_general(wavelength, n_core, n_sub, n_clad, d, m, polar, d_clad=np.inf):
    """
    Calculates the effective refractive index (neff) for each wavelength in a given range, considering 
    the refractive indices of the core, substrate, and cladding, as well as the geometrical dimensions 
    of the waveguide and the mode number.
    
    Parameters:
    - wavelength: An array of wavelengths for which to calculate neff.
    - n_core: Refractive index of the core, can be a constant value or an array matching the wavelength array.
    - n_sub: Refractive index of the substrate, can be a constant value or an array matching the wavelength array.
    - n_clad: Refractive index of the cladding, can be a constant value or an array matching the wavelength array.
    - d: Thickness of the core layer.
    - d_clad: Thickness of the cladding layer.
    - m: Mode number for which to calculate the effective refractive index.
    - polar: TE or TM polarization
    Returns:
    - neff: An array of the calculated effective refractive indices for each wavelength.
    """
    # neff = np.zeros_like(wavelength)
    neff = []
    for ii in range(len(wavelength)):
        # Prepare args for func_ML and find_zero_crossings
        if isinstance(n_core, np.ndarray):
            if  isinstance(d_clad,np.ndarray):
                args = (n_core[ii], n_sub[ii], n_clad[ii], d, d_clad, wavelength[ii], m, polar)
                range_n = np.linspace(max(n_sub[ii],n_clad[ii][0])+0.01, n_core[ii]-0.01,1000)
            else:
                args = (n_core[ii], n_sub[ii], n_clad[ii], d, wavelength[ii], m, polar)
                range_n = np.linspace(max(n_sub[ii],n_clad[ii])+0.01, n_core[ii]-0.01,1000)
     
        else:
            if  isinstance(d_clad,np.ndarray):
                args = (n_core, n_sub, n_clad, d, d_clad, wavelength[ii], m, polar)
                range_n = np.linspace(max(n_sub,n_clad[0])+0.01, n_core-0.01,10000)
            else:
                args = (n_core, n_sub, n_clad, d, wavelength[ii], m, polar)
                range_n = np.linspace(max(n_sub,n_clad)+0.01, n_core-0.01,10000)
            
        
        # Use find_zero_crossings to suggest search intervals
        suggested_intervals = find_zero_crossings(func_ML if isinstance(n_clad[0], np.ndarray) else func, range_n, args)
        n_local = []
        # For each suggested interval, attempt to find a root
        for start, end in suggested_intervals:
            try:
                n_local.append(brentq(func_ML if isinstance(n_clad[0], np.ndarray) else func, start, end, args=args, maxiter=5000))
                # break  # Exit the loop if a root is successfully found
            except ValueError:
                # Handle cases where brentq fails to find a root in the current interval
                print('No guided mode found!')
                continue
        neff.append(n_local)
    
    return neff




