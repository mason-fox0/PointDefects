#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:34:13 2022

@author: Mason Fox
Written for the final project of MATH 578 at UTK - Spring 2022
Purpose: Simulate the diffusion of point defects (vacancies and interstitials) in a material subject to irradiation damage.

Inputs: plotting parameters, radiation flux density, incident radiation energy, temperature, material properties (macro and atomic scale), threshold energies, material geometry, number of spatial nodes, number of time points desired.
Outputs: Vacancy and Interstitial concentrations plotted as a function of space

Numerical Method: Crank-Nicholson Adams Bashforth (2-step) IMEX Scheme
"""

import math
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as nm


def main():
    ####inputs
    #plot parameters
    plot_freq = 1  #wait this many time iterations before plotting
    fig_size = (8,4) #size of saved plots; (x,y) in inches
    fig_dpi = 300 #pixels per inch (affects plot quality and file size)
    
    
    #sim time
    t_start = 0
    t_end = 4  #seconds
    numTnodes = 201    #includes t=0, t = t_end;   number of time steps = numTnodes - 1
    
    
    #geometry size - square
    side_length = 1 #cm
    numSpatialNodes = 11
    
    #set material properties
    flux = 1e6    #particles/cm^2*s
    displacement_cross_section = 100 * 1e-24   #cm^-2; 316 Stainless Steel, 1 MeV neutron (Iwamoto et. al, 2013, Fig 5) - https://doi.org/10.1080/00223131.2013.851042 
    density = 7.99  #g/cm^3; 316 stainless steel
    mass_num = 56 #iron
    adjacent_lattice_sites = 6 #steel is Face-Centered Cubic
    recomb_num = 100    #normally 50-500 depending on material, structure, and interstitial config (# energy permissible combinations of jumps for recombination)
    lattice_parameter = 3.57    #Angstroms (cm ^-8)
    E_v_jump_threshold = 0.93    #ev/atom, energy required for vacancy to relocate to another site in lattice structure
    E_i_jump_threshold = 2.1     #ev/atom, energy required for interstitial to diffuse #TODO: find a real value
    E_disp_threshold = 10    #eV, energy required to produce a vacancy-interstitial pair.
    E_incident = 1e6    #eV (assumes monoenergetic incident radiation)
    temperature = 600   #K
    ####end inputs
    
    
    
    
    ####setup
    #calculate macro material parameters
    atomic_density = density * 6.022e23 / mass_num   #atoms/cm^3
    sinkStrength_i = 0
    sinkStrength_v = 0
    D_i = compute_diff(lattice_parameter, adjacent_lattice_sites, mass_num, E_i_jump_threshold, temperature)
    D_v = compute_diff(lattice_parameter, adjacent_lattice_sites, mass_num, E_v_jump_threshold, temperature)
    K_IV = recomb_num * (D_i + D_v) / (lattice_parameter)**2
    
    
    #define geometry
    numXnodes = numSpatialNodes #note: this includes zero and final spatial point
    numYnodes = numSpatialNodes #force same spatial step sizes for simplicity of CN method
    xmin = 0                #cm
    xmax = side_length      #cm
    ymin = 0                #cm
    ymax = side_length       #cm
    
    
    #time discretization
    t, stepT = np.linspace(t_start, t_end, numTnodes, retstep=True)
    
    
    #spatial discretization/mesh
    ci = np.zeros((numXnodes, numYnodes, numTnodes), dtype=float) #initial values all zero
    cv = np.zeros((numXnodes, numYnodes, numTnodes), dtype=float)
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
    
    #print important parameters
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
    print("Radiation Flux: ", flux)
    print("****************************************\n")
    
    
    
    
    #solver
    m = numXnodes-3
    k = numYnodes-3 #excludes BC terms
    
    alpha = [D_v * stepT / (2 * stepX**2), D_i * stepT / (2*stepX**2)]  #combine leading coefficient in CN method for simpler notation. (Note: alpha different for vacancy/interstitial)
    
    #build (m*k)x(m*k) vacancy matrix
    firstrow = np.zeros((m*k)*(m*k)) #characterize toeplitz matrix (constant diag matrix) with first row, first column (these should be the same)
    firstrow[0] = (1-4*alpha[0])
    firstrow[1] = alpha[0]
    firstrow[m] = alpha[0]
    A_v = sp.linalg.toeplitz(firstrow) #build matrix
    A_v[:,numYnodes-1] = 0
    A_v[numXnodes-1,:] = 0
    
    #build interstitial matrix
    firstrow = np.zeros((m*k)*(m*k)) #characterize toeplitz matrix with first row, first column (these should be the same)
    firstrow[0] = (1-4*alpha[1])
    firstrow[1] = alpha[1]
    firstrow[m] = alpha[1]
    A_i = sp.linalg.toeplitz(firstrow) #build matrix
    
    gen = int(atomic_density * flux_to_DPA(flux, displacement_cross_section, mass_num, E_incident, E_disp_threshold))
    
    #start first timestep
    cv[:,:,1] = stepT*gen
    ci[:,:,1] = stepT*gen
    
    for t_iter in range(2, numTnodes):
        F_v = np.zeros((m*k)*(m*k)) #reset
        b_v = np.zeros((m*k)*(m*k))
        b_v[0] = stepT*gen #start with euler's method
        
        F_i = np.zeros((m*k)*(m*k))
        b_i = np.zeros((m*k)*(m*k))
        b_i[0] = stepT*gen #start with euler's method (note b_i[0] = zero, as well as other terms for j=1)
        
        j=1 #counter for terms in b matrix
        for x_iter in range(1, numXnodes-1): #not inclusive of final iteration (numXnodes)
            for y_iter in range(1, numYnodes-1):
                
                F_v[j] = 3/2 * dcdt(cv, ci, gen, D_v, K_IV, sinkStrength_v, x_iter, y_iter, t_iter-1) - 1/2 * dcdt(cv, ci, gen, D_v, K_IV, sinkStrength_v, x_iter, y_iter, t_iter-2)
                b_v[j] = stepT * F_v[j] + (1-4*alpha[0]) * cv[x_iter, y_iter, t_iter-1] + alpha[0] * cv[x_iter+1, y_iter, t_iter-1] + alpha[0] * cv[x_iter-1, y_iter, t_iter-1] + alpha[0] * cv[x_iter, y_iter-1, t_iter-1] + alpha[0] * cv[x_iter, y_iter+1, t_iter-1]
                
                F_i[j] = 3/2 * dcdt(ci, cv, gen, D_i, K_IV, sinkStrength_i, x_iter, y_iter, t_iter-1) - 1/2 * dcdt(ci, cv, gen, D_i, K_IV, sinkStrength_i, x_iter, y_iter, t_iter-2)
                b_i[j] = stepT * F_v[j] + (1-4*alpha[1]) * ci[x_iter, y_iter, t_iter-1] + alpha[1] * ci[x_iter+1, y_iter, t_iter-1] + alpha[1] * ci[x_iter-1, y_iter, t_iter-1] + alpha[1] * ci[x_iter, y_iter-1, t_iter-1] + alpha[0] * ci[x_iter, y_iter+1, t_iter-1]
                
                j=j+1
                
                
        x_v, exitcode_v = sp.sparse.linalg.minres(A_v, b_v, check=True) #solve system with efficient iterative method
        x_i, exitcode_i = sp.sparse.linalg.minres(A_i, b_i, check=True)
        
        if (exitcode_v or exitcode_i != 0):
            raise RuntimeError('Failed to converge!')
        
        #disallow negative concentrations
        x_i = np.where(x_i<0, 0, x_i) #for all negative elements in vector, replace with zero
        x_v = np.where(x_v<0, 0, x_v)
        
        #map x to concentration row by row
        first = 1
        for i in range(1, k+1):
            last = first+k
            cv[i, 1:m+1, t_iter] = x_v[first:last]
            ci[i, 1:m+1, t_iter] = x_i[first:last]
            
            first = last + 1
              
        if (t_iter % plot_freq == 0 or t_iter == numTnodes-1): #plot and save png
            plot_and_save(ci, cv, numXnodes-1, numYnodes-1, t_iter, stepT, fig_size, fig_dpi)
            
    print("Done!")
    
    
    
    
def flux_to_DPA(flux, micro_cs, mass_num, E_neutron, threshold_energy): #calculates displacements per atom per second using Kinchin-Pease model, assumes monoenergetic incident neutrons perpendicular to surface
    transf_param = 4.0*(1*mass_num) / (1+mass_num)**2
    return flux * micro_cs * mass_num * transf_param * E_neutron / (4.0 * threshold_energy) #K-P model; Source: Olander, Motta: LWR materials Vol 1. Ch 12, eqn 12.77

def compute_diff(lat_param, num_adj_sites, mass_num, E_jump, temp): #compute diffusion coefficients using einstein diffusion formula
    k_Boltzmann = 8.617333e-5 #eV/K
    conv_fac = 9.64853e27 #convert eV/(angstroms^2 * amu) to s^-2
    vib_freq = (1 / math.sqrt(2)) * math.sqrt(E_jump/(mass_num*lat_param**2)*conv_fac)
    
    return (1.0/6) * (lat_param*1e-8)**2 * num_adj_sites * vib_freq * math.exp(-E_jump/(k_Boltzmann*temp)) #cm^2/s

def compute_sink(func, strength, D, x, y, t):
    return strength * D * func[x, y, t]

def compute_recomb(c1, c2, KIV, x, y, t):
    return KIV * int(c1[x,y,t]) * int(c2[x,y,t]) #take advantage of no max int size to prevent overflow

def dcdt(c1, c2, gen, D, KIV, sinkstrength, x, y, t): #note: this function excludes diffusion
    return gen - compute_recomb(c1, c2, KIV, x, y, t) - compute_sink(c1, sinkstrength, D, x, y, t)

def plot_and_save(ci, cv, xmax, ymax, t, time_step, figsize, imgdpi):
    plt.figure(figsize = (8,4))
    conc = ci[:,:,t]
    time = "".join(["Time: ", str(r'{:.3f}'.format(t*time_step)), " sec"]) #tuple -> string, keep following zeros after decimal point
    plt.suptitle(time)
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.subplot(1,2,1)
    
    plt.pcolormesh(conc,edgecolors='none', norm=nm())        #plot without shading
    plt.title("Interstitial Conc. (cm^-3)")
    plt.colorbar()
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.subplot(1,2,2)
    
    conc = cv[:,:,t]   
    plt.pcolormesh(conc,edgecolors='none', norm=nm())   #plot without shading
    plt.title("Vacancy Conc. (cm^-3)")
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.colorbar()
            
    filename = "".join(["PointDefects",str(t),".png"]) #tuple -> string
    plt.savefig(filename, dpi=imgdpi)
    plt.show()


#run
if __name__ == "__main__":
    main()
