#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:34:13 2022

@author: Mason Fox
Written for the final project of MATH 578 at UTK - Spring 2022
Purpose: Simulate the diffusion of point defects (vacancies and interstitials) in a material subject to irradiation damage.

Inputs: plotting frequency, radiation flux density, incident radiation energy, temperature, material properties (macro and atomic scale), threshold energies, material geometry, number of spatial nodes, number of time points desired.
Outputs: Vacancy and Interstitial concentrations plotted as a function of space

Numerical Method: Crank-Nicholson Adams Bashforth IMEX Scheme
"""

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as nm


def main():
    #sim parameters
    plot_freq = 1  #wait this many time iterations before plotting
    
    #set material properties
    flux = 1e8    #particles/cm^2*s
    displacement_cross_section = 100 * 1e-24   #cm^-2; 316 Stainless Steel, 1 MeV neutron (Iwamoto et. al, 2013, Fig 5) - https://doi.org/10.1080/00223131.2013.851042 
    density = 7.99  #g/cm^3; 316 stainless steel
    mass_num = 56 #iron
    adjacent_lattice_sites = 6 #steel is Face-Centered Cubic
    recomb_num = 100    #normally 50-500 depending on material, structure, and interstitial config (# energy permissible combinations of jumps for recombination)
    lattice_parameter = 3.57    #Angstroms (cm ^-8)
    E_v_jump_threshold = 0.93    #ev/atom, energy required for vacancy to relocate to another site in lattice structure
    E_i_jump_threshold = 2.1      #ev/atom, energy required for interstitial to diffuse #TODO: find a real value
    E_disp_threshold = 40    #eV, energy required to produce a vacancy-interstitial pair.
    E_incident = 1e6    #eV (assumes monoenergetic incident radiation)
    temperature = 800   #K
    
    
    #calculate macro material parameters
    atomic_density = density * 6.022e23 / mass_num    #atoms/cm^3
    macro_disp_cross_section = atomic_density * displacement_cross_section #cm^-1 ; probability of interaction per unit length traveled
    sinkStrength_i = 1e-4
    sinkStrength_v = 1e-4
    D_i = compute_diff(lattice_parameter, adjacent_lattice_sites, mass_num, E_i_jump_threshold, temperature)
    D_v = compute_diff(lattice_parameter, adjacent_lattice_sites, mass_num, E_v_jump_threshold, temperature)
    K_IV = recomb_num * (D_i + D_v) / (lattice_parameter)**2
    
    #define geometry - square
    s = 1 #cm (side length of square domain)
    thickness = 0.1 #cm
    
    numXnodes = 101 #note: this includes zero and final time points
    numYnodes = numXnodes #force same spatial step sizes for simplicity
    xmin = 0        #cm
    xmax = s    #cm
    ymin = 0        #cm
    ymax = s        #cm
    num_atoms = (ymax-ymin)*(xmax-xmin)*thickness #doesn't do anything at the moment
    
    
    #time discretization
    t_start = 0
    t_end = 10  #seconds
    numdT = 101
    t, stepT = np.linspace(t_start, t_end, numdT, retstep=True)
    
    #spatial discretization/mesh
    ci = np.zeros((numXnodes, numYnodes, numdT), dtype=float) #initial values all zero
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
    
    print("Material Parameters ********************")
    print("Vacancy Diffusion Coefficient: ", D_v)
    print("Interstitial Diffusion Coefficient: ", D_i)
    print("Recombination Coefficient (K_IV): ", K_IV)
    print("Radiation Flux: ", flux)
    print("****************************************\n")
    
   ############################################################
       
    gen = atomic_density * flux_to_DPA(flux, displacement_cross_section, mass_num, E_incident, E_disp_threshold)
    
    alpha = [D_v * stepT / (2 * stepX**2), D_i * stepT / (2*stepX**2)]  #combine leading coefficient in CN method for simpler notation. (Note: alpha different for vacancy/interstitial)
    m = numXnodes-2
    k = numYnodes-2 #excludes BC terms
    
    #build (m*k)x(m*k) vacancy matrix (store as sparse for memory savings)
    ones = np.ones(m*k) #vector filled with ones to build diagonals
    diag_entries = np.array([(1-4*alpha[0])*ones, alpha[0]*ones, alpha[0]*ones, alpha[0]*ones, alpha[0]*ones]) #need main diag, two adjacent diags, and two additional diags
    offsets = np.array([0, -1, 1, k, -k])
    A_v = sp.dia_array((diag_entries, offsets), shape=(m,k)) #construct array in sparse form
    
    
    #build interstitial matrix
    diag_entries = np.array([(1-4*alpha[1])*ones, alpha[1]*ones, alpha[1]*ones, alpha[1]*ones, alpha[1]*ones]) #need main diag, two adjacent diags, and two additional diags
    offsets = np.array([0, -1, 1, k, -k])
    A_i = sp.dia_array((diag_entries, offsets), shape=(m,k)) #construct array in sparse form
    
    for t_iter in range(1, numdT-1):
        for x_iter in range(0, numXnodes-1):
            for y_iter in range(0, numYnodes-1):
                
                recomb = compute_recomb(ci, cv, K_IV, x_iter, y_iter, t_iter-1)
                
                #TODO: code AB2 here
                b_v = np.zeros(m)
                
        x = sp.linalg.spsolve(A_v, b_v)
        print(x)
            
        if (t_iter % plot_freq == 0 or t_iter == numdT-2): #plot and save png
            plt.figure(figsize = (8,4))
            conc = np.transpose(ci[:,:,t_iter]) #transpose to make x/y axis plot correctlyi
            time = "".join(["Time: ", str(r'{:.3f}'.format(t_iter*stepT)), " sec"]) #tuple -> string, keep following zeros after decimal point
            plt.suptitle(time)
            plt.subplot(1,2,1)
            
            plt.pcolormesh(conc,edgecolors='none', norm=nm(), shading='gouraud')
            plt.title("Interstitial Conc. (cm^-3)")
            plt.colorbar()
            
            plt.subplot(1,2,2)
            conc = np.transpose(cv[:,:,t_iter]) #transpose to make x/y axis plot correctly
            
            plt.pcolormesh(conc,edgecolors='none', norm=nm(), shading='gouraud')
            plt.title("Vacancy Conc. (cm^-3)")
            plt.colorbar()
            
            filename = "".join(["PointDefects",str(t_iter),".png"]) #tuple -> string
            plt.savefig(filename, dpi = 300)
            plt.show()
            
    print("Done!")

##############functions
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
    return KIV * c1[x,y,t] * c2[x,y,t]

#############

#run
if __name__ == "__main__":
    main()
