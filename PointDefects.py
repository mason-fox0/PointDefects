#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:34:13 2022

@author: Mason Fox
Written for the final project of MATH 578 at UTK - Spring 2022
Purpose: Simulate the diffusion of point defects (vacancies and interstitials) in a material subject to irradiation damage.

Inputs: plotting frequency, radiation flux density, incident radiation energy, temperature, material properties (macro and atomic scale), threshold energies, material geometry, number of spatial nodes, number of time points desired.
Outputs: Vacancy and Interstitial concentrations plotted as a function of space

Numerical Method: Finite difference method with adams-bashforth 3 step method
"""

import math
import numpy as np
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
    E_jump_threshold = 0.93    #ev/atom, energy required for vacancy to relocate to another site in lattice structure
    E_disp_threshold = 40    #eV, energy required to produce a vacancy-interstitial pair.
    E_incident = 1e6    #eV (assumes monoenergetic, could expand to read tabulated data or cross-section data libs)
    temperature = 1200   #K
    
    
    #calculate macro material parameters
    atomic_density = density * 6.022e23 / mass_num    #atoms/cm^3
    macro_disp_cross_section = atomic_density * displacement_cross_section #cm^-1 ; probability of interaction per unit length traveled
    sinkStrength_i = 0 #TODO
    sinkStrength_v = 0
    D_i = compute_diff('i', lattice_parameter, adjacent_lattice_sites, mass_num, E_jump_threshold, temperature)
    D_v = compute_diff('v', lattice_parameter, adjacent_lattice_sites, mass_num, E_jump_threshold, temperature)
    K_IV = recomb_num * (D_i + D_v) / (lattice_parameter)**2
    
    #define geometry - rectangular slab
    xmin = 0        #cm
    xmax = 0.5    #cm
    ymin = 0        #cm
    ymax = 1        #cm
    thickness = 0.5 #cm
    numXnodes = 6
    numYnodes = 11
    num_atoms = (ymax-ymin)*(xmax-xmin)*thickness
    
    #time discretization
    t_start = 0
    t_end = 10  #seconds
    numdT = 1001
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
    
    #check_stability(stepT, stepX, stepY, max(D_i,D_v))
    
    print("Material Parameters ********************")
    print("Vacancy Diffusion Coefficient: ", D_v)
    print("Interstitial Diffusion Coefficient: ", D_i)
    print("Recombination Coefficient (K_IV): ", K_IV)
    print("Radiation Flux: ", flux)
    print("****************************************\n")
    
   ############################################################
       
    gen = atomic_density * flux_to_DPA(flux, displacement_cross_section, mass_num, E_incident, E_disp_threshold)
    
    
    #TODO solver - need an implicit or IMEX method
    for t_iter in range(1, numdT-1):
        for x_iter in range(0, numXnodes-1):
            for y_iter in range(0, numYnodes-1):
                
                recomb = compute_recomb(ci, cv, K_IV, x_iter, y_iter, t_iter-1)
                
                
            
        if (t_iter % plot_freq == 0 or t_iter == numdT-2): #plot and save png    
            plt.figure(figsize = (8,4))
            conc = np.transpose(ci[:,:,t_iter]) #transpose to make x/y axis plot correctly
            time = "".join(["Time: ", str(r'{:.3f}'.format(t_iter*stepT)), " sec"]) #tuple -> string, keep following zeros in time string
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

def compute_diff(typeChar, lat_param, num_adj_sites, mass_num, E_jump, temp): #compute diffusion coefficients using einstein diffusion formula
    k_Boltzmann = 8.617333e-5 #eV/K
    conv_fac = 9.64853e27 #convert eV/(angstroms^2 * amu) to s^-2
    vib_freq = (1 / math.sqrt(2)) * math.sqrt(E_jump/(mass_num*lat_param**2)*conv_fac)
    
    if (typeChar == 'v' or typeChar == 'V'): #use vacancy formula
        return (1.0/6) * (lat_param*1e-8)**2 * num_adj_sites * vib_freq * math.exp(-E_jump/(k_Boltzmann*temp)) #cm^2/s
    elif (typeChar == 'i' or typeChar == 'I'): #use interstitial formula
        return (1.0/6) * (lat_param*1e-8)**2 * num_adj_sites * vib_freq * math.exp(-E_jump/(k_Boltzmann*temp)) #TODO: make work for interstitial
    else:
        raise Exception("Invalid type character. Choose i or v")

def compute_laplacian(func, x, y, t):
    return func[x+1, y, t] - 2 * func[x, y, t] + func[x-1, y, t] + func[x, y+1, t] - 2*func[x, y, t] + func[x, y-1, t]
    
def compute_sink(func, strength, D, x, y, t):
    return strength * D * func[x, y, t]

def compute_recomb(c1, c2, KIV, x, y, t):
    return round(KIV * c1[x,y,t] * c2[x,y,t])

def compute_dcdt(conc, difCoef, generation, recombination, sinkStrength, x, y, t):
    #compute concentration balance - Olander, Motta Ch. 13
    return difCoef * compute_laplacian(conc, x, y, t) + generation - recombination - compute_sink(conc, sinkStrength, difCoef, x, y, t) 

def check_stability(dt, dx, dy, coeff):
    #for FTCS scheme, by von Neumann stability analysis. Considers only diffusion terms
    #TODO update for new scheme
    
    stability_cond = 1/((2*coeff) * (dx**-2 + dy**-2))
    
    if (dt > stability_cond):
        raise Exception('Unstable, select a time step less than ', str(stability_cond))
    return

#run
if __name__ == "__main__":
    main()
