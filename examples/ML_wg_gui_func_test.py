import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from Solve_PWML.src.core.mode_solver import calc_n_eff_ML
from Solve_PWML.src.core.mode_solver import calc_n_eff


lambda_0 = np.linspace(0.2,5, 1000)  # Free-space wavelength in micro-meters  
k_0_arr = 2 * np.pi / lambda_0  # Free-space wave number

# Waveguide parameters
n_1 = 2.12  # Refractive index of the core
n_2 = 1.45   # Refractive index of the substrate and cladding (assuming symmetric waveguide)
n_clad = [1.45, 1.4, 1.1]  # refractive indices of each layer


w = 0.5 # Width of the waveguide core in micrometers
w_clad = [0.1, .1, .1]    # thicknesses of each layer in um

neff_guess =2.1# Initial guess for neff
m = 0 # mode number

n_eff = np.array(calc_n_eff_ML (lambda_0, n_1, n_2, n_clad, w, w_clad, m))
n_eff_simple = np.array(calc_n_eff (lambda_0,n_1, n_2, w, m))

