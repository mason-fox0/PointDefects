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

#plot parameters
plot_freq = 20 #wait this many time iterations before plotting

#material parameters
dpar = 2     #displacements per atom/sec
sinkStrength_i = 0.1
sinkStrength_v = 0.05
K_IV = 0.25
D_i = 0.5
D_v = 0.5

#define geometry - rectangular slab
xmin = 0        #meters
xmax = 1    #meters
ymin = 0        #meters
ymax = 5        #meters
numdX = 5
numdY = 25

#time discretization
t_start = 0
t_end = 1000    #seconds
numdT = 10000    #TODO - bug with overflows in recomb based on t_end and numdT (stability?)
stepT = (t_end - t_start)/numdT

#spatial discretization/mesh
ci = np.zeros((numdX, numdY, numdT), dtype=float)
cv = np.zeros((numdX, numdY, numdT), dtype=float)
stepX = (xmax - xmin)/numdX
stepY = (ymax - ymin)/numdY

#BCs
#TODO

#plotting set up
fig = plt.figure()
ax = plt.axes(projection='3d')


#print parameters and run sim
print("Simulation Parameters *****************")
print("Simulation Time: ", t_end)
print("Time step: ", stepT)
print("delta X: ", stepX)
print("delta Y: ", stepY, "\n")
print("Plot Update Freq: ", plot_freq, " iterations")


print("Material Parameters ********************")
print("Vacancy Diffusion Coefficient: ", D_v)
print("Interstitial Diffusion Coefficient: ", D_i)
print("Recombination Coefficient (K_IV): ", K_IV)
print("")
print("****************************************\n")      

#Forward Time Centered Space (FTCS) 
#TODO: Fix time and space intervals/steps
for t_iter in range(1, numdT-1):
    for x_iter in range(1, numdX-1):
        for y_iter in range(1, numdY-1):
            
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
            
            #for debug only
            #print("Interstitial: ", ci[x_iter, y_iter, t_iter + 1])
            #print("Vacancy: ", cv[x_iter, y_iter, t_iter + 1])
            
    #if (t_iter % plot_freq == 0):
        #TODO - 3D plot of stuff

print("Final Values: ")
print("C_v: ", cv[x_iter, y_iter, t_iter + 1])
print("C_i: ", ci[x_iter, y_iter, t_iter + 1])