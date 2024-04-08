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
sys.path.append(os.getcwd() + '/../../../')

from refractiveindex import RefractiveIndexMaterial

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering

from src.core.mode_solver import calc_n_eff_ML,calc_n_eff,calc_n_eff_general
from src.core.mode_solver import find_zero_crossings,optim_ml_pwg,optim_asymmetric_pwg
from src.visualization.GUI import run_gui,run_gui_simple,run_gui_general
# from src.utils.help_functs import construct_clad
# from src.utils.help_functs import dispersion_calc_splineine


from src.visualization.GUI import run_gui_general
from src.utils.help_functs import construct_clad
import pyle
import time

# Import materials
ta2o5d = pyle.mat.IBSTa2O5d_28()
sio2d = pyle.mat.IBSSiO2d_28()
sapphire = pyle.mat.Sapphire()
lno = pyle.mat.LiNbO3()


SP = RefractiveIndexMaterial(shelf='main', book='Al2O3', page='Malitson')
LN = RefractiveIndexMaterial(shelf='main', book='LiNbO3', page='Zelmon-o')
CF = RefractiveIndexMaterial(shelf='main', book='CaF2', page='Malitson') #mg fluoride
SiO2 = RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
TaO = RefractiveIndexMaterial(shelf='main', book='Ta2O5', page='Gao')

lambda_0 = np.linspace(.5, 1.8, 500,dtype=np.float64)  # Free-space wavelength in micro-meters  

# Waveguide parameters
# n_core = LN.get_refractive_index(lambda_0*1e3)#2.12  # Refractive index of the core in [nm] as in the database
# n_sub = SP.get_refractive_index(lambda_0*1e3)  # Refractive index of the substrate and cladding in [nm]

# Refractive index of layers and cladding in [nm]
# l1 = SiO2.get_refractive_index(lambda_0*1e3)
# l2 = TaO.get_refractive_index(lambda_0*1e3)

n_core = lno.get_n(lambda_0*1e-6)
n_sub = sapphire.get_n(lambda_0*1e-6)

l1 = sio2d.get_n(lambda_0*1e-6)
l2 = ta2o5d.get_n(lambda_0*1e-6)
n_clad = construct_clad(l1,l2)



w = 0.5 # Width of the waveguide core in micrometers
w_clad = np.array([1.1,1.])
m = 0 # mode number

run_gui_general(lambda_0, n_core, n_sub, n_clad, w, w_clad, m)


# 
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

# 
# %% TEST
# this test shows how the function to minimize lookds like
xx = np.linspace(1.5,2.3,1000)
#### optim_ml_pwg(neff, n_core, n_sub, n_ml, w, w_ml, wavelength, m):
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
    
n_eff_mv = calc_n_eff_general(lambda_0,n_core,n_sub, n_clad, w, m,w_clad)

cols = ['k','r','g','b','m']

plt.figure()
for ii,ai in enumerate(n_eff_mv):
    for jj,aj in enumerate(ai):
        plt.plot(lambda_0[ii],aj,'.', color = cols[jj])

#%% test separate into lines

# ##Flatten the data into a list of (x, y) tuples
# points = [(x, y) for x, ys in zip(lambda_0, n_eff_mv) for y in ys]
# # Convert your data into numpy array if it isn't one already
# X = np.array(points)

# # # Scale the data
# X_scaled = StandardScaler().fit_transform(X)



# ### Initialize DBSCAN with some parameters
# ###### DBSCAN
# ### Extract the cluster labels
# db = DBSCAN(eps=0.08, min_samples=2).fit(X_scaled)
# labels = db.labels_
# # Number of clusters in labels, ignoring noise if present
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# unique_labels = set(labels)

# plt.figure()
# # Plot up the results
# for k in unique_labels:
#     if k == -1:
#         # Black used for noise
#         col = 'k'
#     else:
#         col = plt.cm.Spectral(float(k) / n_clusters_)

#     class_member_mask = (labels == k)

#     xy = X[class_member_mask]
#     plt.plot(xy[:, 0], xy[:, 1], '.')

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()


# ### Initialize OPTICS with some parameters

# optics_clust = OPTICS(min_samples=3, min_cluster_size=0.01,eps=0.1,cluster_method='dbscan')
# optics_clust.fit(X_scaled)



# ###### OPTICS
# # Extract the cluster labels
# labels = optics_clust.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# # Plot the clusters and noise (if any)
# for k in set(labels):
#     class_member_mask = (labels == k)
#     if k == -1:
#         # Black used for noise.
#         plt.plot(X[class_member_mask, 0], X[class_member_mask, 1], 'k+', markersize=2)
#     else:
#         plt.plot(X[class_member_mask, 0], X[class_member_mask, 1], 'o', markerfacecolor=plt.cm.nipy_spectral(k / n_clusters_))

# plt.title(f'Estimated number of clusters: {n_clusters_}')
# plt.show()







###### spectral scan - can be the right solution

# Assuming you want to try dividing the data into 3 clusters
# n_clusters = 13
# sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
# labels = sc.fit_predict(X_scaled)
# # Plot the data points, color-coded by cluster label
# for label in np.unique(labels):
#     plt.scatter(X_scaled[labels == label, 0], X_scaled[labels == label, 1], label=f'Cluster {label}')

# plt.title(f'Spectral Clustering Results with {n_clusters} Clusters')
# plt.legend()
# plt.show()

#%% single linkage 
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# # Given your example data format
# x_values = lambda_0
# y_values = n_eff_mv

# # Flatten the data into a list of (x, y) tuples
# data = np.array([(x, y) for x, ys in zip(x_values, y_values) for y in ys])
# # Perform hierarchical/agglomerative clustering using the single linkage method
# Z = linkage(data, method='single')
# # Plot dendrogram
# # plt.figure(figsize=(10, 7))
# # plt.title("Hierarchical Clustering Dendrogram (single linkage)")
# # dendrogram(Z, above_threshold_color='y', orientation='top')
# # plt.show()
# # Define the cutoff as a distance
# distance_cutoff = 0.5 # You may need to adjust this based on your dendrogram
# clusters = fcluster(Z, distance_cutoff, criterion='distance')

# # Or define the cutoff as the number of clusters you want
# num_clusters = 11 # Adjust this to the number of clusters you need
# clusters = fcluster(Z, num_clusters, criterion='maxclust')

# # clusters now contains the labels for each point
# # print(clusters)

# # Plot the original data points with the color labels
# for i in range(1, num_clusters + 1):
#     plt.scatter(data[clusters == i, 0], data[clusters == i, 1], label=f'Cluster {i}')

# plt.title('Hierarchical Clustering Results (single linkage)')
# plt.legend()
# plt.show()

####### LASSO tool 

# from matplotlib import colors as mcolors
# from matplotlib import path
# from matplotlib.collections import RegularPolyCollection
# from matplotlib.widgets import Lasso


# class LassoManager:
#     def __init__(self, ax, data):
#         self.ax = ax
#         self.canvas = ax.figure.canvas
#         self.data = data

#         # Use scatter for plotting
#         self.scatter = ax.scatter(data[:, 0], data[:, 1], color='tab:blue', s=10)

#         # Lasso selector attributes
#         self.lasso_selector = LassoSelector(ax, onselect=self.on_select)
#         self.selected_index = np.zeros(len(data), dtype=bool)

#     def on_select(self, verts):
#         path = Path(verts)
#         self.selected_index = path.contains_points(self.data)

#         # Update colors based on selection
#         colors = ['red' if sel else 'tab:blue' for sel in self.selected_index]
#         self.scatter.set_facecolors(colors)
#         self.canvas.draw_idle()

# # Example usage
# np.random.seed(0)
# data = np.random.rand(100, 2)

# fig, ax = plt.subplots()
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0, 1)

# lasso_manager = LassoManager(ax, np.array(points))

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import LassoSelector, Button
# from matplotlib.path import Path

# class LassoManager:
#     def __init__(self, ax, data):
#         self.ax = ax
#         self.canvas = ax.figure.canvas
#         self.data = data
#         self.add_data = False
#         # Initial scatter plot
#         self.scatter = ax.scatter(data[:, 0], data[:, 1], color='tab:blue', s=10)

#         # Lasso selector attributes
#         self.lasso_selector = LassoSelector(ax, onselect=self.on_select)
#         self.selected_index = np.zeros(len(data), dtype=bool)

#         # Add button - the button axes must be in figure-relative coordinates
#         self.button_ax = self.ax.figure.add_axes([0.0, 0.0, 0.1, 0.05])
#         self.button = Button(self.button_ax, 'Add', color='lightgray')
#         self.button.on_clicked(self.add_to_selection)

#     def on_select(self, verts):
#         path = Path(verts)
#         # Update selected_index based on lasso selection
#         currently_selected = path.contains_points(self.data)
#         if self.add_data:
#             self.selected_index = np.logical_or(self.selected_index, currently_selected)
#         else:
#             self.selected_index =  currently_selected

#         self.update_plot_colors()

#     def add_to_selection(self, event):
#         if self.add_data:
#             self.add_data = False
#         else:
#             self.add_data = True
       

#     def update_plot_colors(self):
#         # Update scatter plot colors based on the selection
#         colors = ['red' if sel else 'tab:blue' for sel in self.selected_index]
#         self.scatter.set_facecolors(colors)
#         self.canvas.draw_idle()

# # Example usage
# np.random.seed(0)
# data = np.random.rand(100, 2)

# fig, ax = plt.subplots()
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0, 1)

# lasso_manager = LassoManager(ax,  np.array(points))

# plt.show()

