import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd() + '/../../Solve_PWML')
sys.path.append(os.getcwd() + '/../../../')

from refractiveindex import RefractiveIndexMaterial

from src.core.mode_solver import calc_n_eff_ML,calc_n_eff,calc_n_eff_general
from src.core.mode_solver import find_zero_crossings,optim_ml_pwg
from src.visualization.GUI import run_gui,run_gui_simple,run_gui_general
from src.utils.help_functs import construct_clad
from src.utils.help_functs import dispersion_calc_splineine


import pyle

import time
# import cProfile
# import pstats
# import timeit

#### check that the simplet version works well

# Import materials
ta2o5d = pyle.mat.IBSTa2O5d_28()
sio2d = pyle.mat.IBSSiO2d_28()
sapphire = pyle.mat.Sapphire()

SP = RefractiveIndexMaterial(shelf='main', book='Al2O3', page='Malitson')
LN = RefractiveIndexMaterial(shelf='main', book='LiNbO3', page='Zelmon-o')
CF = RefractiveIndexMaterial(shelf='main', book='CaF2', page='Malitson') #mg fluoride
SiO2 = RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
TaO = RefractiveIndexMaterial(shelf='main', book='Ta2O5', page='Gao')

lambda_0 = np.linspace(0.5, 1.8, 500,dtype=np.float64)  # Free-space wavelength in micro-meters  

# Waveguide parameters
n_core = LN.get_refractive_index(lambda_0*1e3)#2.12  # Refractive index of the core in [nm] as in the database
n_sub = SP.get_refractive_index(lambda_0*1e3)  # Refractive index of the substrate and cladding in [nm]

# Refractive index of layers and cladding in [nm]
# l1 = SiO2.get_refractive_index(lambda_0*1e3)
# l2 = TaO.get_refractive_index(lambda_0*1e3)

l1 = sio2d.get_n(lambda_0*1e-6)
l2 = ta2o5d.get_n(lambda_0*1e-6)
n_clad = construct_clad(l1,l2,l1,l2)

 

w = .5 # Width of the waveguide core in micrometers
# w_clad = np.array([0.1]*np.size(n_clad,axis=1)) # thicknesses of each layer in um
w_clad = np.array([0.1,1.79,0.1,0.4])
m = 0 # mode number

# run_gui(lambda_0, n_core, n_sub, n_clad, w, w_clad, m)

# run_gui_general(lambda_0, n_core, n_sub, n_clad, w, w_clad, m)

######### various tests

# a,b,c = dispersion_calc_splineine(lambda_0,l2)
# plt.figure()
# plt.plot(lambda_0,l1)
# plt.plot(lambda_0,l2)
# plt.plot(lambda_0,n_core)
# plt.plot(lambda_0,n_sub)
# plt.plot(lambda_0,b)
# plt.plot(lambda_0,c)

# plt.figure()
# plt.plot(lambda_0,vp)
# plt.plot(lambda_0,n_eff)
# plt.plot(lambda_0,n_eff_simple)


run_gui(lambda_0, n_core, n_sub, n_clad, w, w_clad, m)
#%% 
## %% TEST
## this test shows how the function to minimize lookds like
xx = np.linspace(1.5,2.3,1000)
# #### optim_ml_pwg(neff, n_core, n_sub, n_ml, w, w_ml, wavelength, m):
plt.figure()
for ii in range(3):           
    # for jj in range(len(xx)):
    plt.plot(xx,optim_ml_pwg(xx, n_core[0], n_sub[0], n_clad[0], w, w_clad, lambda_0[0], ii),'k')
    args = (n_core[0], n_sub[0], n_clad[0], w, w_clad, lambda_0[0], ii)
    intevals = find_zero_crossings(optim_ml_pwg,xx,args)
    try:
        for itr in intevals:
            plt.axvline(itr[0])
            plt.axvline(itr[1])
    except:
        pass
plt.axhline(y=0, color='black', linestyle='--')
plt.axvline(x=n_core[0], color='r', linestyle='-')

#%% test n_effective calculations
# def calc_n_eff_general(wavelength, n_core, n_sub, n_clad, d, d_clad, dm):
# from src.core.mode_solver import calc_n_eff_general
    
# aa = calc_n_eff_general(lambda_0,n_core,n_sub, n_clad, w, m,w_clad)

# cols = ['k','r','g','b','m']

# plt.figure()
# for ii,ai in enumerate(aa):
#     for jj,aj in enumerate(ai):
#         plt.plot(lambda_0[ii],aj,'.', color = cols[jj])

