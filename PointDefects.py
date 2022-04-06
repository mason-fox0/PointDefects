#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:34:13 2022

@author: Mason Fox
Written for the final project of MATH 578 at UTK - Spring 2022

Purpose: Simulate the diffusion of point defects (vacancies and interstitials) in a material subject to irradiation damage.
"""

import numpy as np
import matplotlib.pyplot as plt

#sim parameters
plot_freq = 25 #wait this many time iterations before plotting

#material parameters
dpar = 0.5     #displacements per atom/sec
sinkStrength_i = 0
sinkStrength_v = 0
K_IV = 0.01
D_i = 2
D_v = 0.5

#define geometry - rectangular slab
xmin = 0        #meters
xmax = 1    #meters
ymin = 0        #meters
ymax = 10        #meters
numXnodes = 11
numYnodes = 251

#time discretization
t_start = 0
t_end = 100   #seconds
numdT = 2001
t, stepT = np.linspace(t_start, t_end, numdT, retstep=True)



#if () #TODO - stability check

#spatial discretization/mesh
ci = np.zeros((numXnodes, numYnodes, numdT), dtype=float)
cv = np.zeros((numXnodes, numYnodes, numdT), dtype=float)
x, stepX = np.linspace(xmin, xmax, num=numXnodes, retstep=True)
y, stepY = np.linspace(ymin, ymax, num=numYnodes, retstep=True)

#BCs
ci[:,0,:] = 0   #dirichlet
cv[0,:,:] = 0   #dirichlet
ci[:,numYnodes-1, :] = 0
cv[:,numYnodes-1, :] = 0
ci[numXnodes-1,:, :] = 0
cv[numXnodes-1,:, :] = 0

#plotting set up
fig = plt.figure()
ax = plt.axes()
#TODO: fix axis scaling/labels


plt.pcolor(ci[:,:,0]) #show initial plot

#print parameters and run sim
print("Simulation Parameters *****************")
print("Simulation Time: ", t_end)
print("Time step: ", stepT)
print("X step: ", stepX)
print("Y step: ", stepY)
print("Plot Update Freq: ", plot_freq, " iterations")

print("Material Parameters ********************")
print("Vacancy Diffusion Coefficient: ", D_v)
print("Interstitial Diffusion Coefficient: ", D_i)
print("Recombination Coefficient (K_IV): ", K_IV)
print("Displacements per Atom / sec: ", dpar)
print("****************************************\n")      

#Forward Time Centered Space (FTCS) 
for t_iter in range(0, numdT-1):
    for x_iter in range(0, numXnodes-1):
        for y_iter in range(0, numYnodes-1):
            
                #compute component terms
                gen = dpar
                recomb = K_IV * ci[x_iter, y_iter, t_iter] * cv[x_iter, y_iter, t_iter]
                
                #interstitial terms
                laplacian_i = ci[x_iter+1, y_iter, t_iter] - 2 * ci[x_iter, y_iter, t_iter] + ci[x_iter-1, y_iter, t_iter] + ci[x_iter, y_iter+1, t_iter] - 2*ci[x_iter, y_iter, t_iter] + ci[x_iter, y_iter-1, t_iter]
                sink_i = sinkStrength_i * D_i * ci[x_iter, y_iter, t_iter]
                
                #vacancy terms
                laplacian_v = cv[x_iter+1, y_iter, t_iter] - 2 * cv[x_iter, y_iter, t_iter] + cv[x_iter-1, y_iter, t_iter] + cv[x_iter, y_iter+1, t_iter] - 2*cv[x_iter, y_iter, t_iter] + cv[x_iter, y_iter-1, t_iter]
                sink_v = sinkStrength_v * D_v * cv[x_iter, y_iter, t_iter]
        
                #update next time step
                ci[x_iter, y_iter, t_iter + 1] = stepT * D_i * laplacian_i + gen - recomb - sink_i
                cv[x_iter, y_iter, t_iter + 1] = stepT * D_v * laplacian_v + gen - recomb - sink_v

            
    if (t_iter % plot_freq == 0 or t_iter == numdT-2):
        plt.pcolor(ci[:,:,t_iter])
        #filename = "ci_",t_iter,".png" #this is a tuple, saveplot() doesn't like it
        #TODO: export plot as image
        
    #TODO: stop loop early if converged
        
print("Done!")