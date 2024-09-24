#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:33:38 2024

@author: alexey
"""

import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd() + '/../../MLWG_solver')
 
from refractiveindex import RefractiveIndexMaterial

from src.core.mode_solver import calc_n_eff_ML,calc_n_eff,calc_n_eff_general
from src.core.mode_solver import find_zero_crossings,optim_ml_pwg,optim_asymmetric_pwg
from src.visualization.GUI import run_gui,run_gui_simple,run_gui_general
 
from src.visualization.GUI import run_gui_general
from src.utils.help_functs import construct_clad

 
SP = RefractiveIndexMaterial(shelf='main', book='Al2O3', page='Malitson')
LN = RefractiveIndexMaterial(shelf='main', book='LiNbO3', page='Zelmon-o')
CF = RefractiveIndexMaterial(shelf='main', book='CaF2', page='Malitson') #mg fluoride
SiO2 = RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
TaO = RefractiveIndexMaterial(shelf='main', book='Ta2O5', page='Gao')

lambda_0 = np.linspace(.5, 1.8, 500,dtype=np.float64)  # Free-space wavelength in micro-meters  

# Waveguide parameters
n_core = LN.get_refractive_index(lambda_0*1e3)#2.12  # Refractive index of the core in [nm] as in the database
n_sub = SP.get_refractive_index(lambda_0*1e3)  # Refractive index of the substrate and cladding in [nm]

# Refractive index of layers and cladding in [nm]
l1 = SiO2.get_refractive_index(lambda_0*1e3)
l2 = TaO.get_refractive_index(lambda_0*1e3)
 
n_clad = construct_clad(l1,l2)



w = 0.4 # Width of the waveguide core in micrometers
w_clad = np.array([.1,.1])
m = 0 # mode number

run_gui_general(lambda_0, n_core, n_sub, n_clad, w, w_clad, m, 'TM')


