#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:54:00 2024

@author: alexey
"""
import numpy as np
from scipy.constants import c
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

### a simple function that give correct cladding refracive index input
construct_clad = lambda *arrays: np.array([[arr[ii] for arr in arrays] for ii in range(len(arrays[0]))])

reshape_n_eff = lambda xdat, ydat: np.array([(x, y) for x, ys in zip(xdat, ydat) for y in ys])

#### this function is needed to separate the lasso-selected areas, sort as x data goes
def segment_data(data_x,data_y):
    # Ensure there are enough elements to determine the step and to segment
    if len(data_x) < 2:
        return [data_x]  # Not enough data to segment

    segments_x = []
    segments_y = []
    ii_prev_seg = 0
    current_segment = [data_x[0]]

    # Determine the step size from the first two elements
    step = data_x[1] - data_x[0]

    # Start from the second element since the first is already in the current segment
    for ii in range(1, len(data_x)):
        expected_value = current_segment[-1] + step
        if abs(data_x[ii] - expected_value) < 0.001:
            # The current value follows the expected sequence; add it to the current segment
            current_segment.append(data_x[ii])
        else:
            # The current value deviates from the expected sequence; segment here
            segments_x.append(current_segment)
            segments_y.append(data_y[ii_prev_seg:ii])
            ii_prev_seg = ii
            current_segment = [data_x[ii]]

    # Add the last segment
    segments_x.append(current_segment)
    segments_y.append(data_y[ii_prev_seg:])
    return segments_x,segments_y

def dispersion_calc(lambda_arr, n_eff):
    '''
    Parameters
    ----------
    lambda_arr : array of wavelength
        DESCRIPTION.
    n_eff : array of effective refractive indexes
        DESCRIPTION.

    Returns phase velocity, group velocity, and group velocity dispersion
    in 
    -------
    Note all this is a function of wavelength
    '''
    n_eff = np.array(n_eff)
    lambda_arr_m = lambda_arr*1e-3
    c_norm = c*1e-12
    dn_dl = np.gradient(n_eff, lambda_arr_m[1]-lambda_arr_m[0])
    d2n_dl2 = np.gradient(dn_dl, lambda_arr_m[1]-lambda_arr_m[0])
    betta_1 = (n_eff-lambda_arr_m*dn_dl)/c_norm
    betta_2 = (lambda_arr_m**3 * d2n_dl2)/c_norm**2/(2*np.pi)
    
    return c_norm/n_eff, 1/betta_1, betta_2

def dispersion_calc_splineine(lambda_arr, n_eff):
    """
    Calculate dispersion properties using spline interpolation for derivatives.
    Note all this is a function of wavelength!
    Parameters
    ----------
    lambda_arr : array
        Array of wavelengths (in meters).
    n_eff : array
        Array of effective refractive indexes.

    Returns
    -------
    phase_velocity : array
        Phase velocity in [mm/fs]
    group_velocity : array
        Group velocity [mm/fs]
    gvd : array
        Group velocity dispersion (GVD) [mm^2/fs]
    """
    
    lambda_arr_m = lambda_arr * 1e-3  # Convert wavelength to millimeters
    c_norm = c*1e-12 # [mm/fs]
    # Create a spline of n_eff as a function of lambda_arr_m
    spline = UnivariateSpline(lambda_arr_m, n_eff, k=4, s=0)
    # plt.figure()
    # plt.plot(n_eff-spline(lambda_arr_m))
    # plt.figure()
    # plt.plot(n_eff)
    # plt.plot(spline(lambda_arr_m))
    
    # First derivative dn/dlambda
    dn_dl = spline.derivative(1)
    
    # Second derivative d^2n/dlambda^2
    d2n_dl2 = spline.derivative(2)
    
    # Compute beta_1, beta_2 using the derivatives from the spline
    beta_1 = (n_eff - lambda_arr_m * dn_dl(lambda_arr_m)) / c_norm
    beta_2 = (lambda_arr_m**3 * d2n_dl2(lambda_arr_m)) / c_norm**2 / (2 * np.pi)
    
    # Compute phase velocity and group velocity
    phase_velocity = c_norm / n_eff
    group_velocity = 1 / beta_1
    
    return phase_velocity, group_velocity, beta_2