import swiftsimio as sw
import swiftgalaxy
import numpy as np
import h5py
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import csv
import time
import math
from swiftsimio import cosmo_quantity, cosmo_array
from operator import attrgetter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from graphformat import set_rc_params
from generate_schechter import _leja_schechter
from read_dataset_directories_colibre import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================



""" Unyts guide:

This applies to both halo_catalogue and particle data (.stars):

.in_units(u.km/u.s)         -> converts, makes a copy, keeps units as unyt array
.in_units(u.dimensionless)  -> converts, makes a copy, keeps units as unyt array
.to_physical()              -> converts to physical, makes a copy, keeps units as unyt array
.to(u.km/u.s)               -> same as .in_units(u.km/u.s)
    
.convert_to_units('kpc')    -> call function to convert array without copy, keeps units as unyt array
  or .convert_to_units(u.kpc)       but this function is bugged as it keeps the old units, despite converting
                                    values
    
.value                      -> makes a copy, converts to np array (removes units)
.to_value(u.km/u.s)         -> converts, makes a copy, converts to np array (removes units)
    
7*u.Msun                    -> applies a unit to a value

"""
#------------------
# Returns mass function of given value
def _mass_function(simulation_run = ['L100_m6', 'L0025N0752'], 
                   simulation_type = ['THERMAL_AGN_m6', 'THERMAL_AGN_m5'],
                    snapshot_no   = [127, 123],        # available for L100_m6: 127, 119, 114, 102, 092
                   #=====================================
                   # Graph settings
                   mass_type = 'stellar_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / gas_mass_in_cold_dense_gas
                                                   #  molecular_hydrogen_mass / atomic_hydrogen_mass / molecular_and_atomic_hydrogen_mass ]
                   centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                     min_stelmass     = 1,
                     max_stelmass     = 10**15,
                   aperture = 'exclusive_sphere_50kpc', 
                   #----------
                   add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                   format_z          = False,        # For variations in z graphs
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'pdf',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
    #-----------------
    # Add SOAP data
    if len(snapshot_no) > len(simulation_run):
        simulation_run_list = simulation_run * len(snapshot_no)
        simulation_type_list = simulation_type * len(snapshot_no)
        snapshot_no_list = snapshot_no
    else:
        simulation_run_list = simulation_run
        simulation_type_list = simulation_type
        snapshot_no_list = snapshot_no
        
    
    run_name_list = []
    redshift_list = []
    for sim_run_i, sim_type_i, snap_i in zip(simulation_run_list, simulation_type_list, snapshot_no_list):
        # Load data
        simulation_dir = sim_run_i + '/' + sim_type_i
        soap_dir       = colibre_base_path + simulation_dir + "/SOAP/"
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
    
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0]
        #lookbacktime
        if run_name not in run_name_list:
            run_name_list.append(run_name)
        redshift_list.append(z)
        print('\n', run_name, snap_i)
    
    
        # Get mass data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
        central_sat = attrgetter('input_halos.is_central')(data)
        if mass_type == 'molecular_and_atomic_hydrogen_mass':
            HI_mass = attrgetter('%s.%s'%(aperture, 'atomic_hydrogen_mass'))(data)
            H2_mass = attrgetter('%s.%s'%(aperture, 'molecular_hydrogen_mass'))(data)
            quantity_mass = HI_mass + H2_mass
            HI_mass = 0
            H2_mass = 0      
        else:
            quantity_mass = attrgetter('%s.%s'%(aperture, mass_type))(data)
        
        # convert in place
        stellar_mass.convert_to_units('Msun')
        quantity_mass.convert_to_units('Msun')

        # Mask for central/satellite and no DM-only galaxies
        if centrals_or_satellites == 'centrals':
            candidates = np.argwhere(np.logical_and.reduce([stellar_mass > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0), 
                                                            stellar_mass < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0), 
                                                            central_sat == cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                            quantity_mass > cosmo_quantity(0, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0)])).squeeze()
            quantity_mass = quantity_mass[candidates]
            print('Masking only centrals: ', len(quantity_mass))
        elif centrals_or_satellites == 'satellites':
            candidates = np.argwhere(np.logical_and.reduce([stellar_mass > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0), 
                                                            stellar_mass < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0), 
                                                            central_sat == cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                            quantity_mass > cosmo_quantity(0, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0)])).squeeze()
            quantity_mass = quantity_mass[candidates]
            print('Masking only satellites: ', len(quantity_mass))
        else:
            candidates = np.argwhere(np.logical_and.reduce([stellar_mass > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0), 
                                                            stellar_mass < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0), 
                                                            quantity_mass > cosmo_quantity(0, u.Msun, comoving=True, scale_factor= data.metadata.a, scale_exponent=0)])).squeeze()
            quantity_mass = quantity_mass[candidates]
            print('Masking all galaxies: ', len(quantity_mass))
            
    
        #---------------
        # Histograms   
        hist_bin_width = 0.2
        if mass_type == 'stellar_mass':
            lower_mass_limit = 10**6.2
        else:
            lower_mass_limit = 10**6
        upper_mass_limit = 10**13
    
        hist_masses, bin_edges =  np.histogram(np.log10(quantity_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**3
        hist_masses = hist_masses/hist_bin_width        # density
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(np.log10(quantity_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_err = (np.sqrt(hist_n)/(box_size)**3)/hist_bin_width
        
        
        """hist_masses, _ = np.histogram((hist_bin_width/2)+np.floor(np.log10(quantity_mass)/hist_bin_width)*hist_bin_width , bins=np.arange(np.log10(lower_mass_limit)+(hist_bin_width/2), np.log10(upper_mass_limit), hist_bin_width))
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**3
        hist_masses = hist_masses/hist_bin_width
        hist_bins   = np.arange(np.log10(lower_mass_limit)+(hist_bin_width/2), np.log10(upper_mass_limit)-hist_bin_width, hist_bin_width)"""
    
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask_finite = np.isfinite(np.log10(hist_masses))
        hist_masses = hist_masses[hist_mask_finite]
        bin_midpoints   = bin_midpoints[hist_mask_finite]
        hist_err    = hist_err[hist_mask_finite]
        hist_n      = hist_n[hist_mask_finite]
        
        # Find bins with more than 10 entry
        hist_mask_n = hist_n > 1
    
    
        #-----------
        # Plotting error for n >= 10, and single dots for others
        if format_z:
            label_i = '$z=%.2f$'%z
            cmap = plt.get_cmap('Blues')
            norm = ((1-np.sqrt(z))+0.3)/1.3
            linecol = cmap(norm)
        else:
            label_i = run_name
            linecol = 'C0'
        axs.fill_between(10**bin_midpoints[hist_mask_n], 10**(np.log10(hist_masses[hist_mask_n]-hist_err[hist_mask_n])), 10**(np.log10(hist_masses[hist_mask_n]+hist_err[hist_mask_n])), alpha=0.4, fc=linecol)
        lines = axs.plot(10**bin_midpoints[hist_mask_n], 10**np.log10(hist_masses[hist_mask_n]), label='%s'%label_i, ls='-', linewidth=1.2, c=linecol, marker='o', ms=1)
        axs.plot(10**bin_midpoints[~hist_mask_n], 10**np.log10(hist_masses[~hist_mask_n]), 'o', color=lines[0].get_color(), ms=2)
        
    #-----------------
    # Add observations
    axs.plot([],[])    # skip C0
    if add_observational:
        if mass_type == 'stellar_mass':
            """
            Pick observations we want to add
            """
            add_leja2020    = True
            add_driver2022  = True
            
            if add_leja2020:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Leja_2020.hdf5'%obs_dir, 'r') as file:
                    
                    """ # A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['multi_file_metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                    
                    obs_mass     = file['x/z000.200_values'][:] * u.Unit(file['x/z000.200_values'].attrs['units'])
                    obs_fraction = file['y/z000.200_values'][:] * u.Unit(file['y/z000.200_values'].attrs['units'])
                    obs_fraction_scatter = file['y/z000.200_scatter'][:] * u.Unit(file['y/z000.200_scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Leja+20 ($z=0.2$)', ls='-', linewidth=1, alpha=0.8, zorder=-20)
            if add_driver2022:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Driver2022.hdf5'%obs_dir, 'r') as file:
                    
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_mass     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_fraction = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_fraction_scatter = file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Driver+22 (GAMA-DR4) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)              
        if mass_type == 'molecular_hydrogen_mass':
            """
            Pick observations we want to add
            """
            add_andreani2020    = False
            add_fletcher2021    = True     # most robust where sample bias has been taken into account
            add_lagos2015       = False
            #add_guo2023 = True
            
            if add_andreani2020:
                logm = np.arange(6.5, 10.2, 0.25)
                phi      = np.array([-2.36, -2.14, -2.02, -1.96, -1.93, -1.94, -1.98, -2.04, -2.12, -2.24, -2.40, -2.64, -3.78, -5.2, -6.00])
                #phi_err  = np.array([])
                
                #phi_err_lower = 10**phi - (10**(phi-phi_err))
                #phi_err_upper = (10**(phi+phi_err)) - 10**phi
                
                axs.plot(10**logm, 10**phi, ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20, label='Andreani+20 (HRS) ($z=0.0$)')
                #axs.errorbar(10**logm, 10**phi, yerr=np.array([phi_err_lower, phi_err_upper]), label='Andreani+20 (HRS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)                
            if add_fletcher2021:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyH2MassFunction/Fletcher2021.hdf5'%obs_dir, 'r') as file:
                    
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_mass     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_fraction = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_fraction_scatter = file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')/1.36     # divide to account for x1.36 He correction in observations
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Fletcher+21 (xCOLD GASS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)           
            if add_lagos2015:
                print('lagos not available')
        if mass_type == 'atomic_hydrogen_mass':
            """
            Pick observations we want to add
            """
            add_zwaan_2005      = True
            add_xi_2022         = True
            add_guo_2023        = True
            add_ma2024_north    = True
            add_ma2024_south    = True
            
            if add_zwaan_2005:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyHIMassFunction/Zwaan2005.hdf5'%obs_dir, 'r') as file:
                    
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_mass     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_fraction = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_fraction_scatter = file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Zwaan+05 (HIPASS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20) 
            if add_xi_2022:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyHIMassFunction/Xi2022.hdf5'%obs_dir, 'r') as file:
                    
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_mass     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_fraction = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_fraction_scatter = file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Xi+22 (AUDS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)
            if add_guo_2023:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyHIMassFunction/Guo2023.hdf5'%obs_dir, 'r') as file:
                    
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_mass     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_fraction = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_fraction_scatter = file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Guo+23 (ALFALFA) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)
            if add_ma2024_north:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyHIMassFunction/Ma2024_north.hdf5'%obs_dir, 'r') as file:
                    
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_mass     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_fraction = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_fraction_scatter = file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Ma+24 (FASHI North) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)
                
                
                
                """logm = np.arange(7.1, 11, 0.2)
                phi      = np.array([1.118e-1, 1.191e-1, 9.678e-2, 5.811e-2, 6.102e-2, 4.724e-2, 4.227e-2, 3.746e-2, 3.078e-2, 3.057e-2, 2.569e-2, 2.129e-2, 1.494e-2, 1.020e-2, 5.525e-3, 2.358e-3, 8.044e-4, 1.754e-4, 2.709e-5, 4.523e-6])
                phi_err  = np.array([0.255e-1, 0.204e-1, 1.408e-2, 0.758e-2, 0.595e-2, 0.388e-2, 0.285e-2, 0.227e-2, 0.143e-2, 0.109e-2, 0.078e-2, 0.054e-2, 0.034e-2, 0.022e-2, 0.130e-3, 0.067e-3, 0.327e-4, 0.133e-4, 0.506e-5, 2.095e-6])
                
                axs.errorbar(10**logm, phi, yerr=phi_err, label='Ma+24 (FASHI North) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)                
                """
            if add_ma2024_south:
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyHIMassFunction/Ma2024_south.hdf5'%obs_dir, 'r') as file:
                    
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_mass     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_fraction = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_fraction_scatter = file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                obs_mass = obs_mass.to('Msun')
                obs_size = obs_fraction.to('Mpc**(-3)')
                obs_size_scatter = obs_fraction_scatter.to('Mpc**(-3)')
                
                axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Ma+24 (FASHI South) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)
                
                
                
                """logm = np.arange(7.3, 11, 0.2)
                phi      = np.array([2.192e-2, 8.694e-2, 6.954e-2, 3.903e-2, 4.067e-2, 3.796e-2, 2.687e-2, 3.667e-2, 2.139e-2, 2.085e-2, 1.641e-2, 1.155e-2, 8.062e-3, 4.463e-3, 2.085e-3, 7.612e-4, 1.386e-4, 1.055e-5, 3.448e-6])
                phi_err  = np.array([1.676e-2, 2.899e-2, 1.831e-2, 1.787e-2, 0.860e-2, 0.774e-2, 0.380e-2, 0.398e-2, 0.197e-2, 0.167e-2, 0.102e-2, 0.069e-2, 0.463e-3, 0.251e-3, 0.144e-3, 0.670e-4, 0.230e-4, 0.611e-5, 3.434e-6])
                                
                axs.errorbar(10**logm, phi, yerr=phi_err, label='Ma+24 (FASHI South) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20)                
                """
                
    #-----------
    # Axis formatting
    if mass_type == 'stellar_mass':
        plt.xlim(10**6, 10**12)
        plt.ylim(10**(-6), 1)
    else:
        plt.xlim(10**6, 10**11)
        plt.ylim(10**(-6), 1)
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))

    dict_aperture = {'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_xlabel = {'stellar_mass': '$M_*$ (%s)'%dict_aperture[aperture], 
                   'gas_mass': '$M_{\mathrm{gas}}$ (%s)'%dict_aperture[aperture],
                   'star_forming_gas_mass': '$M_{\mathrm{gas}}$ (%s)'%dict_aperture[aperture],
                   'molecular_hydrogen_mass': '$M_{\mathrm{H_{2}}}$ (%s)'%dict_aperture[aperture],
                   'atomic_hydrogen_mass': '$M_{\mathrm{I}}$ (%s)'%dict_aperture[aperture],
                   'molecular_gas_mass': '$M_{\mathrm{mol}}$ (incl. He, %s)'%dict_aperture[aperture],
                   'molecular_and_atomic_hydrogen_mass': '$M_{\mathrm{HI+H_{2}}}$ (%s)'%dict_aperture[aperture],
                   'molecular_and_atomic_gas_mass': '$M_{\mathrm{HI+H_{mol}}}$ (%s)'%dict_aperture[aperture],
                   'gas_mass_in_cold_dense_gas': '$M_{\mathrm{cold,dense}}$ (%s)'%dict_aperture[aperture]
                   }
    plt.xlabel(r'%s [M$_{\odot}$]'%(dict_xlabel[mass_type]))
    plt.ylabel(r'dn/dlog$_{10}$($M$) [cMpc$^{-3}$]')
      
    #-----------  
    # Annotations
    if not format_z:
        if mass_type == 'stellar_mass':
            plt.text(10**10.7, 0.4, '${z=%.2f}$' %z, fontsize=7)
        else:
            plt.text(10**10, 0.4, '${z=%.2f}$' %z, fontsize=7)
    if centrals_or_satellites != 'both':
        axs.set_title(r'%s' %(centrals_or_satellites), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '%s'%('redshift_' if format_z else '') + aperture + '_' + centrals_or_satellites + '_' + savefig_txt
        
        plt.savefig("%s/mass_functions/%s%s_%s_%smassfunction_%s.%s" %(fig_dir, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600) 
        print("\n  SAVED: %s/mass_functions/%s%s_%s_%smassfunction_%s.%s" %(fig_dir, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
    
    



#============================================
# Example to mimic pipeline:
"""_mass_function(simulation_run = ['L100_m6', 'L0025N0752'], 
                 simulation_type = ['THERMAL_AGN_m6', 'THERMAL_AGN_m5'],
                 snapshot_no     = [127, 123],        
                mass_type = 'stellar_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                 centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                 aperture = 'exclusive_sphere_50kpc', 
                savefig = True)"""
                

#-------------------------
# Stellar mass function in 50pkpc, all, centrals, satellites
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                mass_type = 'stellar_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                 centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                mass_type = 'stellar_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                 centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                mass_type = 'stellar_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                 centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                savefig = True)
                

#-------------------------
# Cold and dense (T<10**4.5K, n>0.1 cm**-1) function in 50pkpc, all, centrals, satellites
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
# Redshift
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],      
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],       
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)


#------------------------
# H1 mass function in 50pkpc, all, centrals, satellites 
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                    mass_type = 'atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                    mass_type = 'atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],        
                    mass_type = 'atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
# Redshift
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],        
                    mass_type = 'atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],      
                    mass_type = 'atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],       
                    mass_type = 'atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)


#----------------------
# H2 mass function in 50pkpc, all, centrals, satellites 
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],     
                    mass_type = 'molecular_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],  
                    mass_type = 'molecular_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],    
                    mass_type = 'molecular_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
# Redshift
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],        
                    mass_type = 'molecular_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],      
                    mass_type = 'molecular_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],       
                    mass_type = 'molecular_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
                    
#----------------------
# HI + H2 mass function in 50pkpc, all, centrals, satellites 
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],     
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],  
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127],    
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
# Redshift
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],        
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],      
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)
_mass_function(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [127, 119, 114, 102, 92],       
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ stellar_mass / gas_mass / star_forming_gas_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                     format_z = True,
                    savefig = True)







