#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:36:06 2024

@author: alexey
"""

import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd() + '/../../MLWG_solver')

from refractiveindex import RefractiveIndexMaterial

# from src.core.mode_solver import calc_n_eff_ML
# from src.core.mode_solver import calc_n_eff
from src.visualization.GUI import run_gui_simple
from src.core.mode_solver import optim_ml_pwg



lambda_0 = np.linspace(0.4, 3., 500)  # Free-space wavelength in micro-meters  

# Waveguide parameters
n_core = 2.1
n_sub = 1.7607843331457966
n_clad =  np.array([1.4820739955470303,2.1409150049633197])
w = 0.5 # Width of the waveguide core in micrometers
w_clad =np.array([0.1,0.1]) # thicknesses of each layer in um

m = 0 # mode number


# n_eff = np.array(calc_n_eff_ML (np.linspace(0.9, 1.1,3), n_1, n_2, n_clad, w, w_clad, m))

run_gui_simple(lambda_0, n_core, n_sub, n_clad, w, w_clad, m)




# %% tests
## this test shows how the function to minimize lookds like
xx = np.linspace(0.01,3,1000)
### optim_ml_pwg(neff, n_core, n_sub, n_ml, w, w_ml, wavelength, m):
plt.figure()
for ii in range(5):
    for jj in range(len(xx)):
        plt.plot(xx[jj],optim_ml_pwg(xx[jj], n_core, n_sub, n_clad, w, w_clad, 3.7, ii),'-')
plt.axhline(y=0, color='black', linestyle='--')
plt.axvline(x=n_core, color='black', linestyle='-')