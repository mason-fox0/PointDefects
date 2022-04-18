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
from matplotlib.colors import Normalize as nm


def main():
    #sim parameters
    plot_freq = 100  #wait this many time iterations before plotting
    
    #TODO: select realistic parameters
    #material parameters
    flux = 1e10    #particles/cm^2*s
    sinkStrength_i = 0
    sinkStrength_v = 0
    K_IV = 0.001
    D_i = 0.01
    D_v = 0.01
    
    displacement_cross_section = 100 * 1e-24   #cm^-2; 316 Stainless Steel, 1 MeV neutron (Iwamoto et. al, 2013, Fig 5) - https://doi.org/10.1080/00223131.2013.851042 
    density = 7.99  #g/cm^3; 316 stainless steel
    mass_num = 56 #iron
    atomic_density = density * 6.022e23 / mass_num    #atoms/cm^3
    macro_disp_cross_section = atomic_density * displacement_cross_section #cm^-1 ; probability of interaction per unit length traveled
    E_threshold = 40    #eV
    E_incident = 1e6    #eV
    
    #define geometry - rectangular slab
    xmin = 0        #meters
    xmax = 0.5    #meters
    ymin = 0        #meters
    ymax = 1        #meters
    numXnodes = 11
    numYnodes = 21
    
    #time discretization
    t_start = 0
    t_end = 86400  #seconds
    numdT = 864001
    t, stepT = np.linspace(t_start, t_end, numdT, retstep=True)
    
    #spatial discretization/mesh
    ci = np.zeros((numXnodes, numYnodes, numdT), dtype=float)
    cv = np.zeros((numXnodes, numYnodes, numdT), dtype=float)
    x, stepX = np.linspace(xmin, xmax, num=numXnodes, retstep=True)
    y, stepY = np.linspace(ymin, ymax, num=numYnodes, retstep=True)
    
    
    #BCs
    ci[:,0,:] = 0   #dirichlet
    cv[0,:,:] = 0
    ci[0,:,:] = 0
    cv[:,0,:] = 0
    ci[:,numYnodes-1, :] = 0
    cv[:,numYnodes-1, :] = 0
    ci[numXnodes-1,:, :] = 0
    cv[numXnodes-1,:, :] = 0
    
    #print parameters and run sim
    print("Simulation Parameters *****************")
    print("Simulation Time: ", t_end)
    print("Time step: ", stepT)
    print("X step: ", stepX)
    print("Y step: ", stepY)
    print("Plot Update Freq: ", plot_freq, " iterations")
    
    #check_stability(stepT, stepX, stepY, max(D_i,D_v))  #disabled... overly restrictive
    
    print("Material Parameters ********************")
    print("Vacancy Diffusion Coefficient: ", D_v)
    print("Interstitial Diffusion Coefficient: ", D_i)
    print("Recombination Coefficient (K_IV): ", K_IV)
    print("Radiation Flux: ", flux)
    print("****************************************\n")      
    
    #TODO: implement a more stable scheme
    #Forward Time Centered Space (FTCS) 
    for t_iter in range(0, numdT-1):
        for x_iter in range(0, numXnodes-1):
            for y_iter in range(0, numYnodes-1):
                
                    #compute component terms
                    gen = flux_to_DPA(flux, displacement_cross_section, mass_num, E_incident, E_threshold) #displacement generates both a vacancy and interstitial
                    recomb = compute_recomb(ci, cv, K_IV, x_iter, y_iter, t_iter)
                    
                    #interstitial terms
                    laplacian_i = compute_laplacian(ci, x_iter, y_iter, t_iter)
                    sink_i = compute_sink(ci, sinkStrength_i, D_i, x_iter, y_iter, t_iter)
                    
                    #vacancy terms
                    laplacian_v = compute_laplacian(cv, x_iter, y_iter, t_iter)
                    sink_v = compute_sink(cv, sinkStrength_v, D_v, x_iter, y_iter, t_iter)
            
                    #update next time step
                    ci[x_iter, y_iter, t_iter + 1] = stepT * D_i * laplacian_i + gen - recomb - sink_i + ci[x_iter, y_iter, t_iter]
                    cv[x_iter, y_iter, t_iter + 1] = stepT * D_v * laplacian_v + gen - recomb - sink_v + cv[x_iter, y_iter, t_iter]

            
        if (t_iter % plot_freq == 0 or t_iter == numdT-2): #plot and save png    
            #TODO: edit colorbars to not mess up formatting
            plt.figure(figsize = (8,4))
            conc = np.transpose(ci[:,:,t_iter]) #transpose to make x/y axis plot correctly
            time = "".join(["Time: ", str(r'{:.3f}'.format(t_iter*stepT)), " sec"]) #tuple -> string, keep following zeros in time string
            plt.suptitle(time)
            plt.subplot(1,2,1)
            
            plt.pcolormesh(conc,edgecolors='none', norm=nm(), shading='gouraud')
            plt.title("Interstitial Conc. (m^-3)")
            plt.colorbar()
            
            plt.subplot(1,2,2)
            conc = np.transpose(cv[:,:,t_iter]) #transpose to make x/y axis plot correctly
            
            plt.pcolormesh(conc,edgecolors='none', norm=nm(), shading='gouraud')
            plt.title("Vacancy Conc. (m^-3)")
            plt.colorbar()
            
            filename = "".join(["PointDefects",str(t_iter),".png"]) #tuple -> string
            plt.savefig(filename, dpi = 300)
            plt.show()
            
    print("Done!")

def flux_to_DPA(flux, micro_cs, mass_num, E_neutron, threshold_energy): #calculates displacements per atom per second using Kinchin-Pease model, assumes monoenergetic incident neutrons perpendicular to surface
    transf_param = 4.0*(1*mass_num) / (1+mass_num)**2
    return flux * micro_cs * mass_num * transf_param * E_neutron / (4.0 * threshold_energy) #K-P model; Source: Olander, Motta: LWR materials Vol 1. Ch 12, eqn 12.77

def compute_laplacian(func, x, y, t):
    return func[x+1, y, t] - 2 * func[x, y, t] + func[x-1, y, t] + func[x, y+1, t] - 2*func[x, y, t] + func[x, y-1, t]
    
def compute_sink(func, strength, D, x, y, t):
    return strength * D * func[x, y, t]

def compute_recomb(c1, c2, KIV, x, y, t):
    return KIV * c1[x,y,t] * c2[x,y,t]

def check_stability(dt, dx, dy, coeff):
    #for FTCS scheme, by von Neumann stability analysis. Considers only diffusion terms
    
    stability_cond = 1/((2*coeff) * (dx**-2 + dy**-2))
    
    if (dt > stability_cond):
        raise Exception('Unstable, select a time step less than ', str(stability_cond))
    return

#run
if __name__ == "__main__":
    main()
