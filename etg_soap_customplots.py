import swiftsimio as sw
import swiftgalaxy
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import h5py
import csv
import time
import math
import astropy.stats
import pandas as pd
from swiftsimio import cosmo_quantity, cosmo_array
from operator import attrgetter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patheffects as mpe
import matplotlib as mpl
import cmasher
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from highlight_text import fig_text
from astropy.stats import binom_conf_interval
from matplotlib.lines import Line2D
from graphformat import set_rc_params
from generate_schechter import _leja_schechter
from read_dataset_directories_colibre import _assign_directories
from load_soap_sample import _load_soap_sample
from packaging.version import Version
assert Version(sw.__version__) >= Version("9.0.2")
assert Version(swiftgalaxy.__version__) >= Version("1.2.0")


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================
outline=mpe.withStroke(linewidth=1.8, foreground='black', alpha=0.9)
outline2=mpe.withStroke(linewidth=1.4, foreground='black', alpha=0.9)




#-----------------
# Returns stelmass - H2 mass
def _stelmass_H2_H2frac_r50H2(csv_samples = [], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                     use_projected = False,
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                     
    outline=mpe.withStroke(linewidth=1.5, foreground='black', alpha=0.9)
    outline2=mpe.withStroke(linewidth=1.1, foreground='black', alpha=0.9)
    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[8.3, 2.3], sharex=False, sharey=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
    dict_label1 = {'L100_m6': 'L100m6', 
                   'L200_m6': 'L200m6',
				   'L200_m7': 'L200m7'}
    dict_label2 = {'THERMAL_AGN_m6': '',
				   'THERMAL_AGN_m7': '',
                   'HYBRID_AGN_m6': 'h'}
    dict_colors = {'THERMAL_AGN_m6': 'C1',
				   'THERMAL_AGN_m7': 'C5',
                   'HYBRID_AGN_m6': 'C2'}

    _stelmass_H2mass = True
    _stelmass_H2frac = True
    _stelmass_r50H2 = True
    for iii, csv_sample_i in enumerate(csv_samples):
        
        soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = csv_sample_i)
    
        #-----------------
        # Add SOAP data
        simulation_run  = sample_input['simulation_run']
        simulation_type = sample_input['simulation_type']
        snapshot_no     = sample_input['snapshot_no']
        simulation_dir  = sample_input['simulation_dir']
        soap_catalogue_file = sample_input['soap_catalogue_file']
        data = sw.load(f'%s'%soap_catalogue_file)

        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0].to(u.Mpc)


        #-------------------------------
        # Get essential SOAP data for analysis
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()

        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        H2_mass.convert_to_units('Msun')
        H2_mass.convert_to_physical()

        u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
        r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
        u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        mag_plot = u_mag - r_mag

        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
        
        #================================
        # Calculated values
        H2_mass_fraction = np.divide(H2_mass, stellar_mass)
        
        if use_projected:
            r50_h2 = attrgetter('%s.%s'%(('projected_aperture_50kpc_projz' if aperture_h2 == 'exclusive_sphere_50kpc' else 'projected_aperture_10kpc_projz'), 'half_mass_radius_molecular_hydrogen'))(data)[soap_indicies_sample]
            r50_h2.convert_to_units('kpc')
            r50_h2.convert_to_physical()
        else:
            r50_h2 = attrgetter('%s.%s'%(aperture_h2, 'half_mass_radius_molecular_hydrogen'))(data)[soap_indicies_sample]
            r50_h2.convert_to_units('kpc')
            r50_h2.convert_to_physical()
        
        
        #==========================================================
        # Useful masks
        mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
        mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
        mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
        mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4



        #==========================================================
        # Plot 1
        if _stelmass_H2mass:
            # Define inputs
            with np.errstate(divide='ignore'):
                X_VALUE_1 = np.log10(stellar_mass)
                Y_VALUE_1 = np.log10(H2_mass)

        
            #------------
            # Median line of detected ETGs
            if add_median_line:
                #-----------------
                # Define binning parameters
                hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
                #------------------------------------------
                ### ETG (excl. FR) sample
                """# Feed median, gas is found under H2_mass
                X_MEDIAN = np.log10(stellar_mass[mask_SRs])
                Y_MEDIAN = H2_mass[mask_SRs]
        
                # Compute statistics in each bin
                medians = []
                lower_1sigma = []
                upper_1sigma = []
                bin_centers = []
                bins_n = []
                for i in range(len(hist_bins) - 1):
                    mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                    y_bin = Y_MEDIAN.value[mask]
        
                    # Remove <107 H2 mass from sample
                    y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
                    # Append bin count
                    bins_n.append(len(y_bin))
            
                    if len(y_bin) >= 1:  # Ensure the bin contains data
                        medians.append(np.median(y_bin))
                        lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                        upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                    else:
                        medians.append(math.nan)
                        lower_1sigma.append(math.nan)
                        upper_1sigma.append(math.nan)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
                # Convert bin centers back to linear space for plotting
                bin_centers = np.array(bin_centers)
                bins_n = np.array(bins_n)
                medians_log = np.log10(medians)
                medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
                lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
                upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))
        
                ax1.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                line1 = ax1.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=0.8, label='excl. FRs, $\mathrm{H_2}>10^7$', zorder=20, path_effects=[outline], alpha=0.9)
                ax1.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                ax1.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                #ax1.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
                """
        
                #------------------------------------------
                ### ETG (incl. FR) sample
                # Feed median, gas is found under H2_mass
                X_MEDIAN = np.log10(stellar_mass)
                Y_MEDIAN = H2_mass
        
                # Compute statistics in each bin
                medians = []
                lower_1sigma = []
                upper_1sigma = []
                bin_centers = []
                bins_n = []
                for i in range(len(hist_bins) - 1):
                    mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                    y_bin = Y_MEDIAN.value[mask]
        
                    # Remove <107 H2 mass from sample
                    y_bin = y_bin[H2_mass[mask].value > h2_detection_limit]
            
                    # Append bin count
                    bins_n.append(len(y_bin))
            
                    if len(y_bin) >= 1:  # Ensure the bin contains data
                        medians.append(np.median(y_bin))
                        lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                        upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                    else:
                        medians.append(math.nan)
                        lower_1sigma.append(math.nan)
                        upper_1sigma.append(math.nan)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
                # Convert bin centers back to linear space for plotting
                bin_centers = np.array(bin_centers)
                bins_n = np.array(bins_n)
                medians_log = np.log10(medians)
                medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
                lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
                upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))
                
                if iii == 1:
                    ax1.plot(np.flip(bin_centers), np.flip(medians_log), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                    line2 = ax1.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.8, label='%s%s, $\mathrm{H_2}>10^7$'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
                    ax1.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    ax1.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                else:
                    ax1.plot(np.flip(bin_centers), np.flip(medians_log), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                    line2 = ax1.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.8, label='%s%s, $\mathrm{H_2}>10^7$'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
                    #ax1.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    #ax1.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    
                    ax1.fill_between(np.flip(bin_centers), np.flip(lower_1sigma_masked), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], alpha=0.3, zorder=3, linewidth=0)
                    
    
            #-----------------
            # Add observations
            if add_observational:
                """
                Pick observations we want to add
                """
                add_Davis2019    = True
        
                if add_Davis2019:      # ($z=0.0$)
                    # Load the observational data
                    with h5py.File('%s/Davis2019_ATLAS3D.hdf5'%obs_dir, 'r') as file:
                        obs_names_1 = file['data/Galaxy/values'][:]
                        obs_H2_1       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                        obs_Mstar_1    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                        obs_mask_1     = file['data/det_mask/values'][:]
                        obs_isvirgo_1  = file['data/Virgo/values'][:]
                        obs_iscentral_1  = file['data/BCG/values'][:]
                
                    with h5py.File('%s/Davis2019_MASSIVE.hdf5'%obs_dir, 'r') as file:
                        obs_names_2 = file['data/Galaxy/values'][:]
                        obs_H2_2       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                        obs_Mstar_2    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                        obs_mask_2     = file['data/det_mask/values'][:]
                        obs_iscluster_2  = file['data/Cluster/values'][:]
                        obs_iscentral_2  = file['data/BCG/values'][:]
            
                    obs_names_1 = np.array(obs_names_1)
                    obs_mask_1  = np.array(obs_mask_1, dtype=bool)
                    obs_isvirgo_1  = np.array(obs_isvirgo_1, dtype=bool)
                    obs_iscentral_1  = np.array(obs_iscentral_1, dtype=bool)
                    obs_H2_1.to('Msun')
                    obs_Mstar_1.to('Msun')
            
                    obs_names_1 = obs_names_1[obs_mask_1]
                    obs_H2_1    = obs_H2_1[obs_mask_1]
                    obs_Mstar_1 = obs_Mstar_1[obs_mask_1]
                    obs_isvirgo_1 = obs_isvirgo_1[obs_mask_1]
                    obs_iscentral_1 = obs_iscentral_1[obs_mask_1]
            
                    #--------------------------------------------
            
                    obs_names_2 = np.array(obs_names_2)
                    obs_mask_2  = np.array(obs_mask_2, dtype=bool)
                    obs_iscluster_2  = np.array(obs_iscluster_2, dtype=bool)
                    obs_iscentral_2  = np.array(obs_iscentral_2, dtype=bool)
                    obs_H2_2.to('Msun')
                    obs_Mstar_2.to('Msun')
            
                    obs_names_2 = obs_names_2[obs_mask_2]
                    obs_H2_2    = obs_H2_2[obs_mask_2]
                    obs_Mstar_2 = obs_Mstar_2[obs_mask_2]
                    obs_iscluster_2 = obs_iscluster_2[obs_mask_2]
                    obs_iscentral_2 = obs_iscentral_2[obs_mask_2]
            
                    #--------------------------------------------
            
                    if 'central' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[obs_iscentral_1]
                        obs_H2_1    = obs_H2_1[obs_iscentral_1]

                        obs_Mstar_2 = obs_Mstar_2[obs_iscentral_2]
                        obs_H2_2    = obs_H2_2[obs_iscentral_2]
                    if 'satellite' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[~obs_iscentral_1]
                        obs_H2_1    = obs_H2_1[~obs_iscentral_1]

                        obs_Mstar_2 = obs_Mstar_2[~obs_iscentral_2]
                        obs_H2_2    = obs_H2_2[~obs_iscentral_2]
                    if 'cluster' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[obs_isvirgo_1]
                        obs_H2_1    = obs_H2_1[obs_isvirgo_1]

                        obs_Mstar_2 = obs_Mstar_2[obs_iscluster_2]
                        obs_H2_2    = obs_H2_2[obs_iscluster_2]
                    if 'group' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[~obs_isvirgo_1]
                        obs_H2_1    = obs_H2_1[~obs_isvirgo_1]

                        obs_Mstar_2 = obs_Mstar_2[~obs_iscluster_2]
                        obs_H2_2    = obs_H2_2[~obs_iscluster_2]

                    print('Sample length Davis+19 ATLAS3D:   %s'%len(obs_Mstar_1))
                    print('Sample length Davis+19 MASSIVE:   %s'%len(obs_Mstar_2))
            
            
                    obs_x = np.append(np.array(obs_Mstar_1), np.array(obs_Mstar_2))
                    obs_y = np.append(np.array(obs_H2_1), np.array(obs_H2_2))
                    ax1.scatter(obs_x, obs_y, marker='o', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+19')
                    ax1.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=0.8, c='grey')
                    #ax1.text(np.log10(0.82*(10**11.41)), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                    #ax1.text(np.log10(0.82*(10**11.41)), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
                    #ax1.axvline(-0.9832+1.1*np.log10(10**11.41), ls='--', linewidth=0.8, c='grey')
                    #ax1.text(-0.9832+1.1*np.log10(10**11.41), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                    #ax1.text(-0.9832+1.1*np.log10(10**11.41), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
                    #ax1.scatter(obs_Mstar_1, obs_H2_1, marker='o', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (ATLAS$^{\mathrm{3D}}$)')
                    #ax1.scatter(obs_Mstar_2, obs_H2_2, marker='s', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (MASSIVE)')
            
        # Plot 2
        if _stelmass_H2frac:
            # Define inputs
            with np.errstate(divide='ignore'):
                X_VALUE_1 = np.log10(stellar_mass)
                Y_VALUE_1 = np.log10(H2_mass_fraction)
                C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
                C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
                S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
                
            #------------
            # Median line of detected ETGs
            if add_median_line:
                #-----------------
                # Define binning parameters
                hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
                #------------------------------------------
                ### ETG (excl. FR) sample
                """# Feed median, gas is found under H2_mass
                X_MEDIAN = np.log10(stellar_mass[mask_SRs])
                Y_MEDIAN = H2_mass_fraction[mask_SRs]
        
                # Compute statistics in each bin
                medians = []
                lower_1sigma = []
                upper_1sigma = []
                bin_centers = []
                bins_n = []
                for i in range(len(hist_bins) - 1):
                    mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                    y_bin = Y_MEDIAN.value[mask]
        
                    # Remove <107 H2 mass from sample
                    y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
                    # Append bin count
                    bins_n.append(len(y_bin))
            
                    if len(y_bin) >= 1:  # Ensure the bin contains data
                        medians.append(np.median(y_bin))
                        lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                        upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                    else:
                        medians.append(math.nan)
                        lower_1sigma.append(math.nan)
                        upper_1sigma.append(math.nan)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
                # Convert bin centers back to linear space for plotting
                bin_centers = np.array(bin_centers)
                bins_n = np.array(bins_n)
                medians_log = np.log10(medians)
                medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
                lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
                upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))
        
                ax2.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                line1 = ax2.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=0.8, label='excl. FRs, $\mathrm{H_2}>10^7$', zorder=20, path_effects=[outline], alpha=0.9)
                ax2.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                ax2.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                #ax2.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
                """
        
        
                #------------------------------------------
                ### ETG (incl. FR) sample
                # Feed median, gas is found under H2_mass
                X_MEDIAN = np.log10(stellar_mass)
                Y_MEDIAN = H2_mass_fraction
        
                # Compute statistics in each bin
                medians = []
                lower_1sigma = []
                upper_1sigma = []
                bin_centers = []
                bins_n = []
                for i in range(len(hist_bins) - 1):
                    mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                    y_bin = Y_MEDIAN.value[mask]
        
                    # Remove <107 H2 mass from sample
                    y_bin = y_bin[H2_mass[mask].value > h2_detection_limit]
            
                    # Append bin count
                    bins_n.append(len(y_bin))
            
                    if len(y_bin) >= 1:  # Ensure the bin contains data
                        medians.append(np.median(y_bin))
                        lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                        upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                    else:
                        medians.append(math.nan)
                        lower_1sigma.append(math.nan)
                        upper_1sigma.append(math.nan)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
                # Convert bin centers back to linear space for plotting
                bin_centers = np.array(bin_centers)
                bins_n = np.array(bins_n)
                medians_log = np.log10(medians)
                medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
                lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
                upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))
        
                if iii == 1:
                    ax2.plot(np.flip(bin_centers), np.flip(medians_log), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                    line2 = ax2.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.8, label='%s%s, $\mathrm{H_2}>10^7$'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
                    ax2.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    ax2.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                else:
                    ax2.plot(np.flip(bin_centers), np.flip(medians_log), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                    line2 = ax2.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.8, label='%s%s, $\mathrm{H_2}>10^7$'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
                    #ax2.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    #ax2.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    
                    ax2.fill_between(np.flip(bin_centers), np.flip(lower_1sigma_masked), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], alpha=0.3, zorder=3, linewidth=0)
                
    
            #-----------------
            # Add observations
            if add_observational:
                """
                Pick observations we want to add
                """
                add_Davis2019    = True
        
                if add_Davis2019:      # ($z=0.0$)
                    # Load the observational data
                    with h5py.File('%s/Davis2019_ATLAS3D.hdf5'%obs_dir, 'r') as file:
                        obs_names_1 = file['data/Galaxy/values'][:]
                        obs_H2_1       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                        obs_Mstar_1    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                        obs_mask_1     = file['data/det_mask/values'][:]
                        obs_isvirgo_1  = file['data/Virgo/values'][:]
                        obs_iscentral_1  = file['data/BCG/values'][:]
                
                    with h5py.File('%s/Davis2019_MASSIVE.hdf5'%obs_dir, 'r') as file:
                        obs_names_2 = file['data/Galaxy/values'][:]
                        obs_H2_2       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                        obs_Mstar_2    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                        obs_mask_2     = file['data/det_mask/values'][:]
                        obs_iscluster_2  = file['data/Cluster/values'][:]
                        obs_iscentral_2  = file['data/BCG/values'][:]
            
                    obs_names_1 = np.array(obs_names_1)
                    obs_mask_1  = np.array(obs_mask_1, dtype=bool)
                    obs_isvirgo_1  = np.array(obs_isvirgo_1, dtype=bool)
                    obs_iscentral_1  = np.array(obs_iscentral_1, dtype=bool)
                    obs_H2_1.to('Msun')
                    obs_Mstar_1.to('Msun')
            
                    obs_names_1 = obs_names_1[obs_mask_1]
                    obs_H2_1    = obs_H2_1[obs_mask_1]
                    obs_Mstar_1 = obs_Mstar_1[obs_mask_1]
                    obs_isvirgo_1 = obs_isvirgo_1[obs_mask_1]
                    obs_iscentral_1 = obs_iscentral_1[obs_mask_1]
            
                    #--------------------------------------------
            
                    obs_names_2 = np.array(obs_names_2)
                    obs_mask_2  = np.array(obs_mask_2, dtype=bool)
                    obs_iscluster_2  = np.array(obs_iscluster_2, dtype=bool)
                    obs_iscentral_2  = np.array(obs_iscentral_2, dtype=bool)
                    obs_H2_2.to('Msun')
                    obs_Mstar_2.to('Msun')
            
                    obs_names_2 = obs_names_2[obs_mask_2]
                    obs_H2_2    = obs_H2_2[obs_mask_2]
                    obs_Mstar_2 = obs_Mstar_2[obs_mask_2]
                    obs_iscluster_2 = obs_iscluster_2[obs_mask_2]
                    obs_iscentral_2 = obs_iscentral_2[obs_mask_2]
            
                    #--------------------------------------------
            
                    if 'central' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[obs_iscentral_1]
                        obs_H2_1    = obs_H2_1[obs_iscentral_1]

                        obs_Mstar_2 = obs_Mstar_2[obs_iscentral_2]
                        obs_H2_2    = obs_H2_2[obs_iscentral_2]
                    if 'satellite' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[~obs_iscentral_1]
                        obs_H2_1    = obs_H2_1[~obs_iscentral_1]

                        obs_Mstar_2 = obs_Mstar_2[~obs_iscentral_2]
                        obs_H2_2    = obs_H2_2[~obs_iscentral_2]
                    if 'cluster' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[obs_isvirgo_1]
                        obs_H2_1    = obs_H2_1[obs_isvirgo_1]

                        obs_Mstar_2 = obs_Mstar_2[obs_iscluster_2]
                        obs_H2_2    = obs_H2_2[obs_iscluster_2]
                    if 'group' in sample_input['name_of_preset']:
                        obs_Mstar_1 = obs_Mstar_1[~obs_isvirgo_1]
                        obs_H2_1    = obs_H2_1[~obs_isvirgo_1]

                        obs_Mstar_2 = obs_Mstar_2[~obs_iscluster_2]
                        obs_H2_2    = obs_H2_2[~obs_iscluster_2]

                    print('Sample length Davis+19 ATLAS3D:   %s'%len(obs_Mstar_1))
                    print('Sample length Davis+19 MASSIVE:   %s'%len(obs_Mstar_2))
            
            
                    obs_x = np.append(np.array(obs_Mstar_1), np.array(obs_Mstar_2))
                    obs_y = np.append(np.array(obs_H2_1), np.array(obs_H2_2))
                    obs_s = obs_y
            
                    # Find fraction fgas = Mgas / Mgas + M*
                    #obs_y = obs_y - np.log10((10**obs_y) + (10**obs_x))
                    obs_y = obs_y - obs_x   # (in 10** space same as obs_y/obs_x = HI/M*)
            
                    ax2.scatter(obs_x, obs_y, marker='o', s=-0.7+(obs_s-5)**1.8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+19')
                    ax2.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=0.8, c='grey')
                    #ax2.text(np.log10(0.82*(10**11.41)), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                    #ax2.text(np.log10(0.82*(10**11.41)), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
                    #ax2.axvline(-0.9832+1.1*np.log10(10**11.41), ls='--', linewidth=0.8, c='grey')
                    #ax2.text(-0.9832+1.1*np.log10(10**11.41), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                    #ax2.text(-0.9832+1.1*np.log10(10**11.41), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
                    #ax2.scatter(obs_Mstar_1, obs_H2_1, marker='o', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (ATLAS$^{\mathrm{3D}}$)')
                    #ax2.scatter(obs_Mstar_2, obs_H2_2, marker='s', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (MASSIVE)')
            
        # Plot 3
        if _stelmass_r50H2:
            # Define inputs
            with np.errstate(divide='ignore'):
                X_VALUE_1 = np.log10(stellar_mass)
        
                Y_VALUE_1 = np.log10(r50_h2)
        
                C_VALUE_1 = np.zeros(len(stellar_mass))
                C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
                #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
                
            #------------
            # Median line of all ETGs for excl. FR and incl. FR
            if add_median_line:
                #-----------------
                # Define binning parameters
                hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
                #------------------------------------------
                ### ETG (excl. FR) sample
                """# Feed median, gas is found under H2_mass
                X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
                Y_MEDIAN_1 = r50_h2[mask_SRs]
        
                # Compute statistics in each bin
                medians = []
                lower_1sigma = []
                upper_1sigma = []
                bin_centers = []
                bins_n = []
                for i in range(len(hist_bins) - 1):
                    mask = (X_MEDIAN_1 >= hist_bins[i]) & (X_MEDIAN_1 < hist_bins[i + 1])
                    y_bin = Y_MEDIAN_1.value[mask]
        
                    # Remove <107 H2 mass from sample
                    y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
                    # Append bin count
                    bins_n.append(len(y_bin))
            
                    if len(y_bin) >= 1:  # Ensure the bin contains data
                        medians.append(np.median(y_bin))
                        lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                        upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                    else:
                        medians.append(math.nan)
                        lower_1sigma.append(math.nan)
                        upper_1sigma.append(math.nan)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
                # Convert bin centers back to linear space for plotting
                bin_centers = np.array(bin_centers)
                bins_n = np.array(bins_n)
                medians_log = np.log10(medians)
                medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
                lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
                upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))
        
                ax3.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                line1 = ax3.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=0.8, label='excl. FRs,\n$\mathrm{H_2}>10^7$', zorder=20, path_effects=[outline], alpha=0.9)
                ax3.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                ax3.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                #ax3.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
                """
        
                #------------------------------------------    
                # Compute statistics in each bin
                
                ### ETG (incl. FR) sample
                # Feed median, gas is found under H2_mass
                X_MEDIAN_2 = np.log10(stellar_mass)
                Y_MEDIAN_2 = r50_h2
                
                medians = []
                lower_1sigma = []
                upper_1sigma = []
                bin_centers = []
                bins_n = []
                for i in range(len(hist_bins) - 1):
                    mask = (X_MEDIAN_2 >= hist_bins[i]) & (X_MEDIAN_2 < hist_bins[i + 1])
                    y_bin = Y_MEDIAN_2.value[mask]
        
                    # Remove <107 H2 mass from sample
                    y_bin = y_bin[H2_mass[mask].value > h2_detection_limit]
            
                    # Append bin count
                    bins_n.append(len(y_bin))
            
                    if len(y_bin) >= 1:  # Ensure the bin contains data
                        medians.append(np.median(y_bin))
                        lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                        upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                    else:
                        medians.append(math.nan)
                        lower_1sigma.append(math.nan)
                        upper_1sigma.append(math.nan)
                        bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
                # Convert bin centers back to linear space for plotting
                bin_centers = np.array(bin_centers)
                bins_n = np.array(bins_n)
                medians_log = np.log10(medians)
                medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
                lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
                upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))
        
                if iii == 1:
                    ax3.plot(np.flip(bin_centers), np.flip(medians_log), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                    line2 = ax3.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.8, label='%s%s, $\mathrm{H_2}>10^7$'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
                    ax3.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    ax3.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                else:
                    ax3.plot(np.flip(bin_centers), np.flip(medians_log), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=0.8, zorder=10, path_effects=[outline], alpha=0.9)
                    line2 = ax3.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.8, label='%s%s, $\mathrm{H_2}>10^7$'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
                    #ax3.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    #ax3.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
                    
                    ax3.fill_between(np.flip(bin_centers), np.flip(lower_1sigma_masked), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], alpha=0.3, zorder=3, linewidth=0)
                
            #-----------------
            # Add observations
            if add_observational:
                """
                Pick observations we want to add
                """
                add_Davis2013 = True
        
                if add_Davis2013:      # ($z=0.0$)
                    # Load the observational data
                    with h5py.File('%s/Davis2019_ATLAS3D.hdf5'%obs_dir, 'r') as file:
                        obs_names_1 = file['data/Galaxy/values'][:]
                        obs_Mstar_1    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                        obs_H2_1       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                        obs_Re         = file['data/Re/values'][:] * u.Unit(file['data/Re/values'].attrs['units'])
                        obs_mask_1     = file['data/det_mask/values'][:]
                        obs_isvirgo_1  = file['data/Virgo/values'][:]
                        obs_iscentral_1  = file['data/BCG/values'][:]
            
            
                    obs_names_1 = np.array(obs_names_1)
                    obs_Mstar_1 = np.array(obs_Mstar_1)
                    obs_mask_1  = np.array(obs_mask_1, dtype=bool)
                    obs_isvirgo_1  = np.array(obs_isvirgo_1, dtype=bool)
                    obs_iscentral_1  = np.array(obs_iscentral_1, dtype=bool)
                    obs_H2_1.to('Msun')
                    obs_Re.to('kpc')
            
                    obs_names_duplicate = obs_names_1
                    obs_Mstar_duplicate = obs_Mstar_1
            
                    # Pull out only detected systems
                    obs_names_1 = obs_names_1[obs_mask_1]
                    obs_Mstar_1 = obs_Mstar_1[obs_mask_1]
                    obs_isvirgo_1 = obs_isvirgo_1[obs_mask_1]
                    obs_iscentral_1 = obs_iscentral_1[obs_mask_1]
                    obs_H2_1 = obs_H2_1[obs_mask_1]
                    obs_Re = obs_Re[obs_mask_1]
                    obs_mask_1 = obs_mask_1[obs_mask_1]
            
                    with h5py.File('%s/Davis2013_ATLAS3D_CO_extent.hdf5'%obs_dir, 'r') as file:
                        obs_names_2 = file['data/Galaxy/values'][:]
                        obs_Re_CO   = file['data/R_CO/values'][:] * u.Unit(file['data/R_CO/values'].attrs['units'])
                        obs_Lk_ratios  = file['data/log_R_CO_L_Ks/values'][:]
            
                    obs_names_2 = np.array(obs_names_2)
                    obs_Re_CO.to('kpc')
            
            
                    #--------------------------------------------
            
                    if 'central' in sample_input['name_of_preset']:
                        obs_names_1 = obs_names_1[obs_iscentral_1]
                        obs_H2_1    = obs_H2_1[obs_iscentral_1]
                        obs_Mstar_1 = obs_Mstar_1[obs_iscentral_1]
                        obs_Re      = obs_Re[obs_iscentral_1]
                        obs_mask_1  = obs_mask_1[obs_iscentral_1]
                    if 'satellite' in sample_input['name_of_preset']:
                        obs_names_1 = obs_names_1[~obs_iscentral_1]
                        obs_H2_1    = obs_H2_1[~obs_iscentral_1]
                        obs_Mstar_1 = obs_Mstar_1[~obs_iscentral_1]
                        obs_Re      = obs_Re[~obs_iscentral_1]
                        obs_mask_1  = obs_mask_1[~obs_iscentral_1]
                    if 'cluster' in sample_input['name_of_preset']:
                        obs_names_1 = obs_names_1[obs_isvirgo_1]
                        obs_H2_1    = obs_H2_1[obs_isvirgo_1]
                        obs_Mstar_1 = obs_Mstar_1[obs_isvirgo_1]
                        obs_Re      = obs_Re[obs_isvirgo_1]
                        obs_mask_1  = obs_mask_1[obs_isvirgo_1]
                    if 'group' in sample_input['name_of_preset']:
                        obs_names_1 = obs_names_1[~obs_isvirgo_1]
                        obs_H2_1    = obs_H2_1[~obs_isvirgo_1]
                        obs_Mstar_1 = obs_Mstar_1[~obs_isvirgo_1]
                        obs_Re      = obs_Re[~obs_isvirgo_1]
                        obs_mask_1  = obs_mask_1[~obs_isvirgo_1]
                 
                
                    # Match galaxy names to assign Re_CO
                    obs_x = []
                    obs_y = []
                    obs_s = []
                    obs_x2 = []
                    obs_y2 = []
                    obs_s2 = []
                    for name_i in obs_names_2:
                        if name_i in obs_names_1:
                            mask_name_1 = np.where(name_i == obs_names_1)[0]
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                            obs_x.append(obs_Mstar_1[mask_name_1])
                            obs_y.append(obs_Re_CO[mask_name_2])
                            obs_s.append(obs_H2_1[mask_name_1])
                        else:
                            if 'central' not in sample_input['name_of_preset']:
                                print('cannot find %s in 260 ATLAS3D sample'%name_i)
                                if name_i == b'NGC 4550':
                                    mask_name_1 = np.where(name_i == obs_names_duplicate)[0]
                                    mask_name_2 = np.where(name_i == obs_names_2)[0]
                                    h2_i = u.array.unyt_array([np.log10((10**7)/1.32)], units=u.Msun)

                                    obs_x.append(obs_Mstar_duplicate[mask_name_1])
                                    obs_y.append(obs_Re_CO[mask_name_2])
                                    obs_s.append(h2_i)
                                    print('  included NGC 4550')
                                if name_i == b'NGC 4292':
                                    mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                                    obs_Lk_ratio_i = 10**obs_Lk_ratios[mask_name_2] 
                                    obs_Re_CO_i = obs_Re_CO[mask_name_2]
                        
                                    L_k_i = (1/obs_Lk_ratio_i) * obs_Re_CO_i
                                    logMsun_i = np.log10(0.82 * L_k_i)
                                    h2_i = u.array.unyt_array([np.log10((10**7.74)/1.32)], units=u.Msun)
                        
                                    obs_x.append(logMsun_i)
                                    obs_y.append(obs_Re_CO_i)
                                    obs_s.append(h2_i)
                                    print('  included NGC 4292 manually with log Msun: %.2f' %logMsun_i)
                                if name_i == b'NGC 2697':
                                    mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                                    obs_Lk_ratio_i = 10**obs_Lk_ratios[mask_name_2] 
                                    obs_Re_CO_i = obs_Re_CO[mask_name_2]
                        
                                    L_k_i = (1/obs_Lk_ratio_i) * obs_Re_CO_i
                                    logMsun_i = np.log10(0.82 * L_k_i)
                                    h2_i = u.array.unyt_array([np.log10((10**8.61)/1.32)], units=u.Msun)
                        
                                    obs_x.append(logMsun_i)
                                    obs_y.append(obs_Re_CO_i)
                                    obs_s.append(h2_i)
                                    print('  included NGC 2697 manually with log Msun: %.2f' %logMsun_i)
                    

                    obs_x = np.array(obs_x)
                    obs_y = np.array(obs_y)
                    obs_s = np.array(obs_s)
                    assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
            

                    print('Sample length Davis+13 ATLAS3D (detected) + 3 extra:   %s'%(len(obs_x)+len(obs_x2)))
            
                    ax3.scatter(obs_x, np.log10(obs_y), marker='o', s=-0.7+(obs_s-5)**1.8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+13')
                    #ax3.scatter(obs_x2, np.log10(obs_y2), marker='o', s=6, alpha=1, zorder=5, facecolors='none', linewidths=0.3,  edgecolors='r', label='Davis+13\n(non-det.)')
            
                    ##ax3.axvline(-0.9832+1.1*np.log10(10**11.41), ls='--', linewidth=0.8, c='grey')
            
            
    #==========================================================
    # Axis formatting
    # Plot 1
    if _stelmass_H2mass:
        # Axis formatting
        ax1.set_xlim(9.5, 12.5)
        ax1.set_ylim(6, 11)
        #ax1.set_xscale("log")
        #ax1.set_yscale("log")
        #plt.yticks(np.arange(-5, -1.4, 0.5))
        #plt.xticks(np.arange(9.5, 12.5, 0.5))
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='minor')
        ax1.tick_params(axis='y', which='minor')
        dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                         'exclusive_sphere_30kpc': '30 pkpc', 
                         'exclusive_sphere_50kpc': '50 pkpc'}
        dict_aperture_h2 = {'exclusive_sphere_3kpc': '3 pkpc',
                            'exclusive_sphere_10kpc': '10 pkpc',
                            'exclusive_sphere_30kpc': '30 pkpc', 
                            'exclusive_sphere_50kpc': '50 pkpc'}
        ax1.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
        ax1.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ [M$_{\odot}$]')
    

        #-----------  
        # Annotations
        title_dict = {'all_galaxies': 'All galaxies',
                      'all_galaxies_centrals': 'All central galaxies',
                      'all_galaxies_satellites': 'All satellite galaxies',
                      'all_ETGs': 'ETGs',
                      'all_ETGs_centrals': 'ETGs, centrals',
                      'all_ETGs_satellites': 'ETGs, satellites',
                      'all_ETGs_cluster': 'ETGs, cluster',
                      'all_ETGs_groupfield': 'ETGs, group/field',
                      'all_ETGs_plus_redspiral': "ETGs (incl. FRs)",
                      'all_ETGs_plus_redspiral_centrals': "ETGs (incl. FRs), centrals",
                      'all_ETGs_plus_redspiral_satellites': "ETGs (incl. FRs), satellites",
                      'all_ETGs_plus_redspiral_cluster': 'ETGs (incl. FRs), cluster',
                      'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs (incl. FRs), cluster centrals',
                      'all_ETGs_plus_redspiral_groupfield': 'ETGs (incl. FRs), group/field',
                      'all_LTGs': 'LTGs',
                      'all_LTGs_excl_redspiral': 'LTGs'
                      }
        text_title = r'%s%s'%(title_dict[sample_input['name_of_preset']], title_text_in)
        ax1.set_title(r'%s' %(text_title), size=7, x=0.25, y=1.02, pad=3, c='k')
    
    
        #-----------
        # Legend
        handles, labels = ax1.get_legend_handles_labels()
        handles = [handles[0], handles[2], handles[1]]
        labels = [labels[0], labels[2], labels[1]]
        ax1.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1)
        
        #-----------
        # other
        #plt.tight_layout()
    
    # Plot 2
    if _stelmass_H2frac:
        # Axis formatting
        ax2.set_xlim(9.5, 12.5)
        ax2.set_ylim(-5, 0)
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        #plt.yticks(np.arange(-5, -1.4, 0.5))
        #plt.xticks(np.arange(9.5, 12.5, 0.5))
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='minor')
        ax2.tick_params(axis='y', which='minor')
        dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                         'exclusive_sphere_30kpc': '30 pkpc', 
                         'exclusive_sphere_50kpc': '50 pkpc'}
        dict_aperture_h2 = {'exclusive_sphere_3kpc': '3 pkpc',
                            'exclusive_sphere_10kpc': '10 pkpc',
                            'exclusive_sphere_30kpc': '30 pkpc', 
                            'exclusive_sphere_50kpc': '50 pkpc'}
        ax2.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
        ax2.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}/M_*$')
    

        #-----------  
        # Annotations
        title_dict = {'all_galaxies': 'All galaxies',
                      'all_galaxies_centrals': 'All central galaxies',
                      'all_galaxies_satellites': 'All satellite galaxies',
                      'all_ETGs': 'ETGs',
                      'all_ETGs_centrals': 'ETGs, centrals',
                      'all_ETGs_satellites': 'ETGs, satellites',
                      'all_ETGs_cluster': 'ETGs, cluster',
                      'all_ETGs_groupfield': 'ETGs, group/field',
                      'all_ETGs_plus_redspiral': "ETGs (incl. FRs)",
                      'all_ETGs_plus_redspiral_centrals': "ETGs (incl. FRs), centrals",
                      'all_ETGs_plus_redspiral_satellites': "ETGs (incl. FRs), satellites",
                      'all_ETGs_plus_redspiral_cluster': 'ETGs (incl. FRs), cluster',
                      'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs (incl. FRs), cluster centrals',
                      'all_ETGs_plus_redspiral_groupfield': 'ETGs (incl. FRs), group/field',
                      'all_LTGs': 'LTGs',
                      'all_LTGs_excl_redspiral': 'LTGs'
                      }
        text_title = r'%s%s'%(title_dict[sample_input['name_of_preset']], title_text_in)
        ax2.set_title(r'%s' %(text_title), size=7, x=0.25, y=1.02, pad=3, c='k')
             
        #-----------
        # Legend
        handles, labels = ax1.get_legend_handles_labels()
        #handles_in = [handles[0], handles[2], handles[1]]
        #labels_in = [labels[0], labels[2], labels[1]]
        handles_in = [handles[1]]
        labels_in = [labels[1]]
        first_legend = ax2.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1)
        
        scat_list = []
        for i, h2_i in enumerate([7, 8, 9]):
            s = -0.7+(h2_i-5)**1.8
            if i == 0:
                scat_i = ax2.scatter([-2], [-2], c='r', marker='o', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, edgecolor='none')
                scat_list.append(scat_i)
            else:
                scat_i = ax2.scatter([-2], [-2], c='r', marker='o', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
                scat_list.append(scat_i)
                
        second_legend = ax2.legend(handles=scat_list, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='right', markerfirst=False, handlelength=1.3, markerscale=1)
        # Manually add the first legend back to the plot
        ax2.add_artist(first_legend)
    
    # Plot 3
    if _stelmass_r50H2:
        # Axis formatting
        ax3.set_xlim(9.5, 12.5)
        ax3.set_ylim(-0.5, 2)
        ax3.minorticks_on()
        ax3.tick_params(axis='x', which='minor')
        ax3.tick_params(axis='y', which='minor')
        dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                         'exclusive_sphere_30kpc': '30 pkpc', 
                         'exclusive_sphere_50kpc': '50 pkpc'}
        dict_aperture_h2 = {'exclusive_sphere_3kpc': '3 pkpc',
                            'exclusive_sphere_10kpc': '10 pkpc',
                            'exclusive_sphere_30kpc': '30 pkpc', 
                            'exclusive_sphere_50kpc': '50 pkpc'}
        ax3.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
        ax3.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]', labelpad=-1)
        if use_projected:
            ax3.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}^{\mathrm{proj}}$ [pkpc]')
    

        #-----------  
        # Annotations
        title_dict = {'all_galaxies': 'All galaxies',
                      'all_galaxies_centrals': 'All central galaxies',
                      'all_galaxies_satellites': 'All satellite galaxies',
                      'all_ETGs': 'ETGs',
                      'all_ETGs_centrals': 'ETGs, centrals',
                      'all_ETGs_satellites': 'ETGs, satellites',
                      'all_ETGs_cluster': 'ETGs, cluster',
                      'all_ETGs_groupfield': 'ETGs, group/field',
                      'all_ETGs_plus_redspiral': "ETGs (incl. FRs)",
                      'all_ETGs_plus_redspiral_centrals': "ETGs (incl. FRs), centrals",
                      'all_ETGs_plus_redspiral_satellites': "ETGs (incl. FRs), satellites",
                      'all_ETGs_plus_redspiral_cluster': 'ETGs (incl. FRs), cluster',
                      'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs (incl. FRs), cluster centrals',
                      'all_ETGs_plus_redspiral_groupfield': 'ETGs (incl. FRs), group/field',
                      'all_LTGs': 'LTGs',
                      'all_LTGs_excl_redspiral': 'LTGs'
                      }
        text_title = r'%s%s'%(title_dict[sample_input['name_of_preset']], title_text_in)
        ax3.set_title(r'%s' %(text_title), size=7, x=0.25, y=1.02, pad=3, c='k')
            
        #-----------
        # Legend
        handles, labels = ax3.get_legend_handles_labels()
        #handles_in = [handles[0], handles[2], handles[1]]
        #labels_in = [labels[0], labels[2], labels[1]]
        handles_in = [handles[1]]
        labels_in = [labels[1]]
        ax3.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1)
        
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/multi_plots/LXXX_m6_THERMAL_HYBRID_%s_Mstar_MH2_H2frac_r50H2%s_%s_H2ap%s_%s.%s" %(fig_dir, sample_input['snapshot_no'], ('proj' if use_projected else ''), sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/multi_plots/LXXX_m6_THERMAL_HYBRID_%s_Mstar_MH2_H2frac_r50H2%s_%s_H2ap%s_%s.%s" %(fig_dir, sample_input['snapshot_no'], ('proj' if use_projected else ''), sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#----------------
# Returns stelmass - kappaco stars
def _etg_stelmass_kappaco(csv_samples = [], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   add_detection_hist = True,
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    

    #---------------------------
    # Graph initialising and base formatting
    if add_detection_hist:
        fig = plt.figure(figsize=(10/2.5, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(1, 3),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.5)
        # Create the Axes.
        axs     = fig.add_subplot(gs[1])
        ax_hist = fig.add_subplot(gs[0])
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
                        
    #---------------------------
    # Extract data from samples:
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
    dict_label1 = {'L100_m6': 'L100m6', 
                   'L200_m6': 'L200m6',
				   'L200_m7': 'L200m7'}
    dict_label2 = {'THERMAL_AGN_m6': '',
				   'THERMAL_AGN_m7': '',
                   'HYBRID_AGN_m6': 'h'}
    dict_colors = {'THERMAL_AGN_m6': 'C1',
				   'THERMAL_AGN_m7': 'C5',
                   'HYBRID_AGN_m6': 'C2'}
    for iii, csv_sample_i in enumerate(csv_samples):
        
        soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = csv_sample_i)
    
        #-----------------
        # Add SOAP data
        simulation_run  = sample_input['simulation_run']
        simulation_type = sample_input['simulation_type']
        snapshot_no     = sample_input['snapshot_no']
        simulation_dir  = sample_input['simulation_dir']
        soap_catalogue_file = sample_input['soap_catalogue_file']
        data = sw.load(f'%s'%soap_catalogue_file)

        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0].to(u.Mpc)
        
        
        #-------------------------------
        # Get essential SOAP data for analysis
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
    
        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        H2_mass.convert_to_units('Msun')
        H2_mass.convert_to_physical()

        u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
        r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
        u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        mag_plot = u_mag - r_mag

        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
        
    
        #================================
        # Calculated values
        
        
        #==========================================================
        # Useful masks
        mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
        mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
        mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
        mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
    
        
        #==========================================================
        # Plot scatter or hexbin
    
        # Define inputs
        with np.errstate(divide='ignore'):
            X_VALUE_1 = np.log10(stellar_mass)
            Y_VALUE_1 = kappa_stars
            C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
            C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
            #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
        if sample_input['simulation_type'] == 'HYBRID_AGN_m6':
            if scatter_or_hexbin == 'scatter_old':
                ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
                cb = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
                axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
            if scatter_or_hexbin == 'scatter_kappa':
                # Define new colormap for high kappa and low kappa
                colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
                colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
                colors_combined = np.vstack((colors1, colors2))
                mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
        
                # Normalise colormap
                vmin = 0
                vmax = 0.8
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
        

                C_VALUE_1 = kappa_stars
        
                # Plot scatters
                sc_points = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c=C_VALUE_1[mask_h2], s=4, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
                axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], c=C_VALUE_1[~mask_h2], s=4, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0.2, edgecolor='k')
            if scatter_or_hexbin == 'scatter_new':
                ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
                C_VALUE_1 = np.log10(H2_mass)
        
                # Define default colormap for H2
                norm = colors.Normalize(vmin=6.5, vmax=10)      ###
        
                # Create custom colormap with grey at the bottom
                viridis = mpl.cm.viridis
                newcolors = viridis(np.linspace(0, 1, 9))
                newcolors[:1, :] = colors.to_rgba('grey')
                newcmp = ListedColormap(newcolors)
        
                # Plot detections as filled viridis circles and squares
                axs.scatter(X_VALUE_1[mask_h2_SRs], Y_VALUE_1[mask_h2_SRs], c=C_VALUE_1[mask_h2_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
                cb = axs.scatter(X_VALUE_1[mask_h2_FRs], Y_VALUE_1[mask_h2_FRs], c=C_VALUE_1[mask_h2_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
                # Plot non-detections as empty grey circles and squares
                axs.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
                axs.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
            if scatter_or_hexbin == 'hexbin_count':
                ### Plots hexbin showing number of galaxies in bin
        
                cmap = cmasher.jungle_r
                newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
                # hexbin for all values
                extent = (9.5, 12.5, 0, 1)                     ###
                gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
                #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
                # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
                cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
            if scatter_or_hexbin == 'hexbin_H2':  
                ### Plots hexbin showing median log10 H2_mass
        
                # Define default colormap for H2
                norm = colors.Normalize(vmin=6.5, vmax=10)      ###
        
                # Create custom colormap with grey at the bottom
                viridis = mpl.cm.viridis
                newcolors = viridis(np.linspace(0, 1, 7))
                newcolors[:1, :] = colors.to_rgba('grey')
                newcmp = ListedColormap(newcolors)
        
                # hexbin for all values
                extent = (9.5, 12.5, 0, 1)                     ###
                gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
                #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
                # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
                #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
                axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
                # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
                cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
            
        
        
        #------------
        # Median line of all ETGs for excl. FR and incl. FR
        if add_median_line:
            #-----------------
            # Define binning parameters
            hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
            
            #------------------------------------------    
            # Compute statistics in each bin
            
            ### ETG (incl. FR) sample
            # Feed median, gas is found under H2_mass
            X_MEDIAN_2 = np.log10(stellar_mass)
            Y_MEDIAN_2 = kappa_stars
            
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            bins_n = []
            for i in range(len(hist_bins) - 1):
                mask = (X_MEDIAN_2 >= hist_bins[i]) & (X_MEDIAN_2 < hist_bins[i + 1])
                y_bin = Y_MEDIAN_2.value[mask]
        
                # Remove <107 H2 mass from sample
                #y_bin = y_bin[H2_mass[mask].value > h2_detection_limit]
            
                # Append bin count
                bins_n.append(len(y_bin))
            
                if len(y_bin) >= 1:  # Ensure the bin contains data
                    medians.append(np.median(y_bin))
                    lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                    upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                    bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                else:
                    medians.append(math.nan)
                    lower_1sigma.append(math.nan)
                    upper_1sigma.append(math.nan)
                    bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
            # Convert bin centers back to linear space for plotting
            bin_centers = np.array(bin_centers)
            bins_n = np.array(bins_n)
            medians_log = np.log10(medians)
            medians_masked = np.ma.masked_where(bins_n < 10, medians)
            lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
            upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
            
            axs.plot(np.flip(bin_centers), np.flip(medians), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            line2 = axs.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=1, label='%s%s'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
            axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
            
        #-----------------
        # Add observations
        if add_observational:
            """
            Pick observations we want to add
            """
            print('no obs available')
    
    
        #-----------------
        # Add detection hist for excl. FR and incl. FR
        if add_detection_hist:
            # we want the fraction within a bin, not normalised
            bin_width = 0.05
            hist_bins = np.arange(0, 1.1, bin_width)  # Binning edges
        
            #==========================================
            ### ETG (incl. FR) sample
            bin_n, _           = np.histogram(Y_VALUE_1, bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2], bins=hist_bins)
            mask_positive_n     = bin_n > 0
            bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]

            # returns a % upper and a % lower
            bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
            bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
            bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
            # step
            mask_positive_n_extra = list(mask_positive_n)
            if True in mask_positive_n_extra:
                last_true_index = len(mask_positive_n_extra) - 1 - mask_positive_n_extra[::-1].index(True)
        
            mask_positive_n_extra.insert(last_true_index + 1, True)
            mask_positive_n_extra = np.array(mask_positive_n_extra)
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor=dict_colors[sample_input['simulation_type']], linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor=dict_colors[sample_input['simulation_type']], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(0, 1)
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    axs.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h2 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    if not add_detection_hist:
        axs.set_ylabel(r'$\kappa_{\mathrm{co}}^{*}$')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'$\kappa_{\mathrm{co}}^{*}$')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(0, 1)
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_kappa':
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H2':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$]', extend='both')
      
    
    #-----------  
    # Annotations
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs (incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs (incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs (incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs (incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs (incl. FRs), cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs (incl. FRs), group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6',
					  'L200_m7': 'L200m7'}
    title_type_dict = {'THERMAL_AGN_m6': '',
					   'THERMAL_AGN_m7': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3",
						'L200m7': "red"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.107, y=0.957, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    else:
        fig_text(x=0.133, y=0.938, ha='left', s=text_title, fontsize=7, ax=axs, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    
    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if add_median_line or add_observational:
        if scatter_or_hexbin == 'scatter_new':
            handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3])]
            labels = [labels[5], labels[4], labels[0], labels[1]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[0], handles[1]]
            labels = [labels[0], labels[1]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
        
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/multi_plots/LXXX_m6_THERMAL_HYBRID_%s_Mstar_kappaco_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/multi_plots/LXXX_m6_THERMAL_HYBRID_%s_Mstar_kappaco_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()

#----------------
# Returns stelmass - eta kin = Msun^1/3 / sigma, Msun^1/3 km-1 s , (using Victor's r50)
def _etg_stelmass_etakin(csv_samples = [], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                     aperture_sigma = 'r50',     # [ exclusive_sphere_50kpc / r50 ]
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = False,
                   add_detection_hist = True,
                   use_interpolation = False,
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    

    #---------------------------
    # Graph initialising and base formatting
    if add_detection_hist:
        fig = plt.figure(figsize=(10/2.5, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(1, 3),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.5)
        # Create the Axes.
        axs     = fig.add_subplot(gs[1])
        ax_hist = fig.add_subplot(gs[0])
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
                        
    #---------------------------
    # Extract data from samples:
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
    dict_label1 = {'L100_m6': 'L100m6', 
                   'L200_m6': 'L200m6',
				   'L200_m7': 'L200m7'}
    dict_label2 = {'THERMAL_AGN_m6': '',
				   'THERMAL_AGN_m7': '',
                   'HYBRID_AGN_m6': 'h'}
    dict_colors = {'THERMAL_AGN_m6': 'C1',
				   'THERMAL_AGN_m7': 'C5',
                   'HYBRID_AGN_m6': 'C2'}
    for iii, csv_sample_i in enumerate(csv_samples):
        
        soap_indicies_sample, trackID_sample, sample_input = _load_soap_sample(sample_dir, csv_sample = csv_sample_i)
    
        #-----------------
        # Add SOAP data
        simulation_run  = sample_input['simulation_run']
        simulation_type = sample_input['simulation_type']
        snapshot_no     = sample_input['snapshot_no']
        simulation_dir  = sample_input['simulation_dir']
        soap_catalogue_file = sample_input['soap_catalogue_file']
        data = sw.load(f'%s'%soap_catalogue_file)

        # Get Victor morphology data
        if not use_interpolation:
            if (simulation_run == 'L200_m6') & (simulation_type == 'THERMAL_AGN_m6'):
                soap_catalogue_file = '/home/cosmos/c22048063/COLIBRE/outputs/complete_morphology_metrics/L0200N3008/Thermal/SOAP_uncompressed/halo_properties_0127.hdf5'
            elif (simulation_run == 'L100_m6') & (simulation_type == 'HYBRID_AGN_m6'):
                soap_catalogue_file = '/home/cosmos/c22048063/COLIBRE/outputs/complete_morphology_metrics/L0100N1504/Hybrid/SOAP_uncompressed/halo_properties_0127.hdf5'
            elif (simulation_run == 'L100_m6') & (simulation_type == 'THERMAL_AGN_m6'):
                soap_catalogue_file = '/home/cosmos/c22048063/COLIBRE/outputs/complete_morphology_metrics/L0100N1504/Thermal/SOAP_uncompressed/halo_properties_0127.hdf5'
            elif (simulation_run == 'L200_m7') & (simulation_type == 'THERMAL_AGN_m7'):
                soap_catalogue_file = '/home/cosmos/c22048063/COLIBRE/outputs/complete_morphology_metrics/L0200N1504/Thermal/SOAP_uncompressed/halo_properties_0127.hdf5'
            else:
                raise Exception('Victor morphology data not available for %s %s'%(simulation_run, simulation_type))
            data_morph = sw.load(f'%s'%soap_catalogue_file)
            #print(dir(data_morph.exclusive_sphere_3xhalfmassradiusstars))
            #print(' ')
            #print(dir(data_morph))
    
            # Get new soap indicies
            trackID_morph = data_morph.input_halos_hbtplus.track_id
            soap_indicies_sample_morph = np.where(np.in1d(trackID_morph, trackID_sample))[0]       # limits to sample, but not in same order
            trackID_morph_sample = trackID_morph[soap_indicies_sample_morph]
            index_sort = np.where(trackID_morph_sample==trackID_sample[:,None])[1]                 # re-orderes to make consistent with [data]
            
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0].to(u.Mpc)
            
                   
        #-------------------------------
        # Get essential SOAP data for analysis
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
    
        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        H2_mass.convert_to_units('Msun')
        H2_mass.convert_to_physical()

        u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
        r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
        u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        mag_plot = u_mag - r_mag

        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
    
    
        #================================
        # Calculated values
    
        # Stellar velocity dispersion
        def _compute_vel_disp(aperture_veldisp='50kpc'):        # Msun^1/3 / sigma, Msun^1/3 km-1 s
            stellar_vel_disp_matrix = (attrgetter('exclusive_sphere_%s.stellar_velocity_dispersion_matrix'%(aperture_veldisp))(data))[soap_indicies_sample]
            stellar_vel_disp_matrix.convert_to_units(u.km**2 / u.s**2)
            stellar_vel_disp_matrix.convert_to_physical()
            stellar_vel_disp = np.sqrt((stellar_vel_disp_matrix[:,0] + stellar_vel_disp_matrix[:,1] + stellar_vel_disp_matrix[:,2])/3)

            return stellar_vel_disp
            
        # Eta_kin
        def _compute_eta_kin(stellar_vel_disp, aperture_etakin='50'):        # Msun^1/3 / sigma, Msun^1/3 km-1 s    
            stellar_mass_kin = attrgetter('exclusive_sphere_%skpc.%s'%(aperture_etakin, 'stellar_mass'))(data)[soap_indicies_sample]
            stellar_mass_kin.convert_to_units('Msun')
            stellar_mass_kin.convert_to_physical()
    
            eta_kin = np.divide(np.cbrt(stellar_mass_kin), stellar_vel_disp)
        
        
            #K_band_L = (attrgetter('exclusive_sphere_%skpc.stellar_luminosity'%(aperture))(sg.halo_catalogue)).squeeze()[0]
            #multiply K_band_L by 3631 Janskys to convert to units of 10^−23 erg s−1
            #eta_kin = np.cbrt(K_band_L)/stellar_vel_disp
        
        
            return eta_kin
    
        if (aperture_sigma == 'r50') & (use_interpolation == False):
            stellar_vel_disp_matrix = (attrgetter('exclusive_sphere_halfmassradiusstars.stellar_velocity_dispersion_matrix')(data_morph))[soap_indicies_sample_morph][index_sort]
            stellar_vel_disp_matrix.convert_to_units(u.km**2 / u.s**2)
            stellar_vel_disp_matrix.convert_to_physical()
            stellar_vel_disp_plot = np.sqrt((stellar_vel_disp_matrix[:,0] + stellar_vel_disp_matrix[:,1] + stellar_vel_disp_matrix[:,2])/3)
            
            print('using r50 not interpolating')
        if use_interpolation:
            # Old interpolation
            r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
            r50.convert_to_units('kpc')
            r50.convert_to_physical()
        
            stellar_vel_disp50 = _compute_vel_disp(aperture_veldisp='50kpc')
            stellar_vel_disp30 = _compute_vel_disp(aperture_veldisp='30kpc')
            stellar_vel_disp10 = _compute_vel_disp(aperture_veldisp='10kpc')
            stellar_vel_disp3 = _compute_vel_disp(aperture_veldisp='3kpc')
            stellar_vel_disp1 = _compute_vel_disp(aperture_veldisp='1kpc')
            stellar_vel_disp_2r50 = _compute_vel_disp(aperture_veldisp='2xhalfmassradiusstars')
        
            stellar_vel_disp_plot = []
            for i,_ in tqdm(enumerate(r50)):
                r50_i = r50[i]
            
                # Make array of available radii, mask sort them
                x = cosmo_array(np.array([1, 3, 10, 30, 50, 2*r50_i.value]), u.kpc, comoving=False, scale_factor=data.metadata.a, scale_exponent=1)
                mask_sort = np.argsort(x)
                x = x[mask_sort]
            
                # Repeat for y, then sort
                y = np.array([stellar_vel_disp1[i], stellar_vel_disp3[i], stellar_vel_disp10[i], stellar_vel_disp30[i], stellar_vel_disp50[i], stellar_vel_disp_2r50[i]])
                y = y[mask_sort]
            
                yinterp = np.interp(r50_i, x, y)
                stellar_vel_disp_plot.append(yinterp)
            stellar_vel_disp_plot = np.array(stellar_vel_disp_plot)
            stellar_vel_disp_plot = cosmo_array(stellar_vel_disp_plot, (u.km / u.s), comoving=False, scale_factor=data.metadata.a, scale_exponent=0)      
        else:
            stellar_vel_disp_plot = _compute_vel_disp(aperture_veldisp='50kpc')
        eta_kin_plot = _compute_eta_kin(stellar_vel_disp_plot, aperture_etakin='50')
    
    
        #==========================================================
        # Useful masks
        mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
        mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
        mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
        mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4


        #==========================================================
        # Plot scatter or hexbin
    
        # Define inputs
        with np.errstate(divide='ignore'):
            X_VALUE_1 = np.log10(stellar_mass)
            Y_VALUE_1 = np.log10(eta_kin_plot)
            C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
            C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
            #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
        if iii == 1:
            if scatter_or_hexbin == 'scatter_old':
                ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
                cb = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
                axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
            if scatter_or_hexbin == 'scatter_kappa':
                # Define new colormap for high kappa and low kappa
                colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
                colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
                colors_combined = np.vstack((colors1, colors2))
                mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
        
                # Normalise colormap
                vmin = 0
                vmax = 0.8
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
        

                C_VALUE_1 = kappa_stars
        
                # Plot scatters
                sc_points = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c=C_VALUE_1[mask_h2], s=4, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
                axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], c=C_VALUE_1[~mask_h2], s=4, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0.2, edgecolor='k')
            if scatter_or_hexbin == 'scatter_new':
                ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
                C_VALUE_1 = np.log10(H2_mass)
        
                # Define default colormap for H2
                norm = colors.Normalize(vmin=6.5, vmax=10)      ###
        
                # Create custom colormap with grey at the bottom
                viridis = mpl.cm.viridis
                newcolors = viridis(np.linspace(0, 1, 9))
                newcolors[:1, :] = colors.to_rgba('grey')
                newcmp = ListedColormap(newcolors)
        
                # Plot detections as filled viridis circles and squares
                axs.scatter(X_VALUE_1[mask_h2_SRs], Y_VALUE_1[mask_h2_SRs], c=C_VALUE_1[mask_h2_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
                cb = axs.scatter(X_VALUE_1[mask_h2_FRs], Y_VALUE_1[mask_h2_FRs], c=C_VALUE_1[mask_h2_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
                # Plot non-detections as empty grey circles and squares
                axs.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
                axs.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
            if scatter_or_hexbin == 'hexbin_count':
                ### Plots hexbin showing number of galaxies in bin
        
                cmap = cmasher.jungle_r
                newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
                # hexbin for all values
                extent = (9.5, 12.5, 1, 2)                     ###
                gridsize = (27,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
                #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
                # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
                cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
            if scatter_or_hexbin == 'hexbin_H2':  
                ### Plots hexbin showing median log10 H2_mass
        
                # Define default colormap for H2
                norm = colors.Normalize(vmin=6.5, vmax=10)      ###
        
                # Create custom colormap with grey at the bottom
                viridis = mpl.cm.viridis
                newcolors = viridis(np.linspace(0, 1, 7))
                newcolors[:1, :] = colors.to_rgba('grey')
                newcmp = ListedColormap(newcolors)
        
                # hexbin for all values
                extent = (9.5, 12.5, 1, 2)                     ###
                gridsize = (27,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
                #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
                # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
                #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
                axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
                # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
                cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
            
        
    
        #------------
        # Median line of all ETGs for excl. FR and incl. FR
        if add_median_line:
            #-----------------
            # Define binning parameters
            hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
            #------------------------------------------    
            # Compute statistics in each bin
        
            ### ETG (incl. FR) sample
            # Feed median, gas is found under H2_mass
            X_MEDIAN_2 = np.log10(stellar_mass)
            Y_MEDIAN_2 = eta_kin_plot
        
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            bins_n = []
            for i in range(len(hist_bins) - 1):
                mask = (X_MEDIAN_2 >= hist_bins[i]) & (X_MEDIAN_2 < hist_bins[i + 1])
                y_bin = Y_MEDIAN_2.value[mask]
        
                # Remove <107 H2 mass from sample
                #y_bin = y_bin[H2_mass[mask].value > h2_detection_limit]
            
                # Append bin count
                bins_n.append(len(y_bin))
            
                if len(y_bin) >= 1:  # Ensure the bin contains data
                    medians.append(np.median(y_bin))
                    lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
                    upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
                    bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
                else:
                    medians.append(math.nan)
                    lower_1sigma.append(math.nan)
                    upper_1sigma.append(math.nan)
                    bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
            # Convert bin centers back to linear space for plotting
            bin_centers = np.array(bin_centers)
            bins_n = np.array(bins_n)
            medians_log = np.log10(medians)
            medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
            lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
            upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))
            
            axs.plot(np.flip(bin_centers), np.flip(medians), color=dict_colors[sample_input['simulation_type']],   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            line2 = axs.plot(np.flip(bin_centers), np.flip(medians_masked), color=dict_colors[sample_input['simulation_type']], linewidth=1, label='%s%s, $\mathrm{H_2}>10^7$'%(dict_label1[sample_input['simulation_run']], dict_label2[sample_input['simulation_type']]), zorder=20, path_effects=[outline], alpha=0.9)
            axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color=dict_colors[sample_input['simulation_type']], linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
            
        
        #-----------------
        # Add observations
        if add_observational & iii == 0:
            """
            Pick observations we want to add
            """
            add_Davis2019    = True
        
            if add_Davis2019:      # ($z=0.0$)
                # Load the observational data
                with h5py.File('%s/Davis2019_ATLAS3D.hdf5'%obs_dir, 'r') as file:
                    obs_names_1 = file['data/Galaxy/values'][:]
                    obs_H2_1       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                    obs_Mstar_1    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                    obs_sig_1      = file['data/Sig_e/values'][:] * u.Unit(file['data/Sig_e/values'].attrs['units'])
                    obs_mask_1     = file['data/det_mask/values'][:]
                    obs_isvirgo_1  = file['data/Virgo/values'][:]
                    obs_iscentral_1  = file['data/BCG/values'][:]
                
                with h5py.File('%s/Davis2019_MASSIVE.hdf5'%obs_dir, 'r') as file:
                    obs_names_2 = file['data/Galaxy/values'][:]
                    obs_H2_2       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                    obs_Mstar_2    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                    obs_sig_2      = file['data/Sig_e/values'][:] * u.Unit(file['data/Sig_e/values'].attrs['units'])
                    obs_mask_2     = file['data/det_mask/values'][:]
                    obs_iscluster_2  = file['data/Cluster/values'][:]
                    obs_iscentral_2  = file['data/BCG/values'][:]
            
                obs_names_1 = np.array(obs_names_1)
                obs_mask_1  = np.array(obs_mask_1, dtype=bool)
                obs_isvirgo_1  = np.array(obs_isvirgo_1, dtype=bool)
                obs_iscentral_1  = np.array(obs_iscentral_1, dtype=bool)
                obs_H2_1.to('Msun')
                obs_Mstar_1.to('Msun')
                obs_sig_1 = np.log10(obs_sig_1)
            
            
                #--------------------------------------------
            
                obs_names_2 = np.array(obs_names_2)
                obs_mask_2  = np.array(obs_mask_2, dtype=bool)
                obs_iscluster_2  = np.array(obs_iscluster_2, dtype=bool)
                obs_iscentral_2  = np.array(obs_iscentral_2, dtype=bool)
                obs_H2_2.to('Msun')
                obs_Mstar_2.to('Msun')
                obs_sig_2 = np.log10(obs_sig_2)
            
            
                #--------------------------------------------
            
                if 'central' in sample_input['name_of_preset']:
                    obs_Mstar_1 = obs_Mstar_1[obs_iscentral_1]
                    obs_H2_1    = obs_H2_1[obs_iscentral_1]
                    obs_sig_1   = obs_sig_1[obs_iscentral_1]
                    obs_mask_1  = obs_mask_1[obs_iscentral_1]

                    obs_Mstar_2 = obs_Mstar_2[obs_iscentral_2]
                    obs_H2_2    = obs_H2_2[obs_iscentral_2]
                    obs_sig_2   = obs_sig_2[obs_iscentral_2]
                    obs_mask_2  = obs_mask_2[obs_iscentral_2]
                if 'satellite' in sample_input['name_of_preset']:
                    obs_Mstar_1 = obs_Mstar_1[~obs_iscentral_1]
                    obs_H2_1    = obs_H2_1[~obs_iscentral_1]
                    obs_sig_1   = obs_sig_1[~obs_iscentral_1]
                    obs_mask_1  = obs_mask_1[~obs_iscentral_1]

                    obs_Mstar_2 = obs_Mstar_2[~obs_iscentral_2]
                    obs_H2_2    = obs_H2_2[~obs_iscentral_2]
                    obs_sig_2   = obs_sig_2[~obs_iscentral_2]
                    obs_mask_2  = obs_mask_2[~obs_iscentral_2]
                if 'cluster' in sample_input['name_of_preset']:
                    obs_Mstar_1 = obs_Mstar_1[obs_isvirgo_1]
                    obs_H2_1    = obs_H2_1[obs_isvirgo_1]
                    obs_sig_1   = obs_sig_1[obs_isvirgo_1]
                    obs_mask_1  = obs_mask_1[obs_isvirgo_1]

                    obs_Mstar_2 = obs_Mstar_2[obs_iscluster_2]
                    obs_H2_2    = obs_H2_2[obs_iscluster_2]
                    obs_sig_2   = obs_sig_2[obs_iscluster_2]
                    obs_mask_2  = obs_mask_2[obs_iscluster_2]
                if 'group' in sample_input['name_of_preset']:
                    obs_Mstar_1 = obs_Mstar_1[~obs_isvirgo_1]
                    obs_H2_1    = obs_H2_1[~obs_isvirgo_1]
                    obs_sig_1   = obs_sig_1[~obs_isvirgo_1]
                    obs_mask_1  = obs_mask_1[~obs_isvirgo_1]

                    obs_Mstar_2 = obs_Mstar_2[~obs_iscluster_2]
                    obs_H2_2    = obs_H2_2[~obs_iscluster_2]
                    obs_sig_2   = obs_sig_2[~obs_iscluster_2]
                    obs_mask_2  = obs_mask_2[~obs_iscluster_2]

                print('Sample length Davis+19 ATLAS3D:  det:  %s   non-det: %s'%(len(obs_Mstar_1[obs_mask_1]), len(obs_Mstar_1[~obs_mask_1])))
                print('Sample length Davis+19 MASSIVE:  det:  %s   non-det: %s'%(len(obs_Mstar_2[obs_mask_2]), len(obs_Mstar_2[~obs_mask_2])))
            
            
                obs_x = np.append(np.array(obs_Mstar_1), np.array(obs_Mstar_2))
                obs_y = np.append(np.array(obs_sig_1), np.array(obs_sig_2))
                obs_s = np.append(np.array(obs_H2_1), np.array(obs_H2_2))
                obs_det = np.append(np.array(obs_mask_1), np.array(obs_mask_2))
            
                # calculate log eta kin = 1/3 * logM* - log sigma = 1/3 * obs_x - obs_y
                obs_y = (1/3*(obs_x)) - obs_y
            
                axs.scatter(obs_x[obs_det], obs_y[obs_det], marker='o', s=-0.7+(obs_s[obs_det]-5)**1.8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+19\n(det.)')
                axs.scatter(obs_x[~obs_det], obs_y[~obs_det], marker='o', s=6, alpha=1, zorder=5, facecolors='none', linewidths=0.3,  edgecolors='r', label='Davis+19\n(non-det.)')
            
                axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
                #axs.text(np.log10(0.82*(10**11.41)), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                #axs.text(np.log10(0.82*(10**11.41)), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
                #axs.axvline(-0.9832+1.1*np.log10(10**11.41), ls='--', linewidth=1, c='grey')
                #axs.text(-0.9832+1.1*np.log10(10**11.41), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                #axs.text(-0.9832+1.1*np.log10(10**11.41), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
                #axs.scatter(obs_Mstar_1, obs_H2_1, marker='o', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (ATLAS$^{\mathrm{3D}}$)')
                #axs.scatter(obs_Mstar_2, obs_H2_2, marker='s', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (MASSIVE)')
        
    
        #-----------------
        # Add detection hist for excl. FR and incl. FR
        if add_detection_hist:
            # we want the fraction within a bin, not normalised
            bin_width = 0.1
            hist_bins = np.arange(0.5, 2, bin_width)  # Binning edges
        
            #==========================================
            ### ETG (incl. FR) sample
            bin_n, _           = np.histogram(Y_VALUE_1, bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2], bins=hist_bins)
            mask_positive_n     = bin_n > 0
            bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]

            # returns a % upper and a % lower
            bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
            bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
            bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
            # step
            mask_positive_n_extra = list(mask_positive_n)
            if True in mask_positive_n_extra:
                last_true_index = len(mask_positive_n_extra) - 1 - mask_positive_n_extra[::-1].index(True)
        
            mask_positive_n_extra.insert(last_true_index + 1, True)
            mask_positive_n_extra = np.array(mask_positive_n_extra)
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor=dict_colors[sample_input['simulation_type']], linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor=dict_colors[sample_input['simulation_type']], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
            #==========================================
            # Obs sample
            if add_observational & iii == 0:
                # obs_x     = log10 M*
                # obs_y     = ellip
                # obs_det = mask for detected systems
            
                bin_n, _           = np.histogram(obs_y, bins=hist_bins)
                bin_n_detected, _  = np.histogram(obs_y[obs_det], bins=hist_bins)
                mask_positive_n     = bin_n > 0
                bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]

                # returns a % upper and a % lower
                bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
                bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
                bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
                # step
                mask_positive_n_extra = list(mask_positive_n)
                if True in mask_positive_n_extra:
                    last_true_index = len(mask_positive_n_extra) - 1 - mask_positive_n_extra[::-1].index(True)
        
                mask_positive_n_extra.insert(last_true_index + 1, True)
                mask_positive_n_extra = np.array(mask_positive_n_extra)
                #ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
                ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='none', linewidth=0, alpha=0.4, label='Davis+19\n(det.)') 
            
        
                # errorbar
                hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
                ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='r', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
      
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(0.9, 2.1)
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    axs.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h2 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    dict_ylabel     = {'exclusive_sphere_50kpc': '$\sigma _{\mathrm{*}}$', 
                       'r50': '$\sigma _{\mathrm{50}}$'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    if not add_detection_hist:
        axs.set_ylabel(r'log$_{10}$ $\eta _{\mathrm{kin}}$ [M$_{\odot}^{1/3}$ km$^{-1}$ s]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $\eta _{\mathrm{kin}}$ [M$_{\odot}^{1/3}$ km$^{-1}$ s]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(0.9, 2.1)
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_kappa':
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H2':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$]', extend='both')
      
    
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    
    axs.text(0.05, 0.95, '↑ less $\sigma _{\mathrm{50}}$', fontsize=6, color='grey', horizontalalignment='left', verticalalignment='top', transform = axs.transAxes, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round', alpha=0.2))
    axs.text(0.05, 0.05, '↓ more $\sigma _{\mathrm{50}}$', fontsize=6, color='grey', horizontalalignment='left', verticalalignment='bottom', transform = axs.transAxes, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round', alpha=0.2))

    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs (incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs (incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs (incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs (incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs (incl. FRs), cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs (incl. FRs), group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6',
					  'L200_m7': 'L200m7'}
    title_type_dict = {'THERMAL_AGN_m6': '',
					   'THERMAL_AGN_m7': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3",
						'L200m7': "red"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.107, y=0.957, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    else:
        fig_text(x=0.133, y=0.938, ha='left', s=text_title, fontsize=7, ax=axs, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    
    
    
    #-----------
    # Legend
    axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='L100m6, $\mathrm{H_2}>10^7$', zorder=20, path_effects=[outline], alpha=0.9)
    axs.plot([0,1], [-20,-21], color='C2', linewidth=1, label='L100m6h, $\mathrm{H_2}>10^7$', zorder=20, path_effects=[outline], alpha=0.9)
    handles, labels = axs.get_legend_handles_labels()

    #handles_in = [handles[-2], handles[-1]]
    #labels_in = [labels[-2], labels[-1]]
    #first_legend = axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    second_legend = axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1, handler_map={tuple: HandlerTuple(ndivide=None)})
    #axs.add_artist(first_legend)
    
    ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/multi_plots/LXXX_m6_THERMAL_HYBRID_%s_Mstar_etakin_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/multi_plots/LXXX_m6_THERMAL_HYBRID_%s_Mstar_etakin_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()






#================================================================================



#=====================================
# Plots 3 graphs with only median lines: stelmass - H2, stelmass - H2/M*, stelmass - r50H2 projected
"""_stelmass_H2_H2frac_r50H2(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                           use_projected = True,
                         showfig       = False,
                         savefig       = True)"""


#--------------
# Plot stelmass - kappaco stars
_etg_stelmass_kappaco(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                        aperture_h2 = 'exclusive_sphere_50kpc',
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)

#---------------              
# Returns stelmass - eta kin = Msun^1/3 / sigma, Msun^1/3 km-1 s , (r50 from victor)
_etg_stelmass_etakin(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                        aperture_h2 = 'exclusive_sphere_50kpc',
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)











