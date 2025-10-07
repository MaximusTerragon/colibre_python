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
#-----------------
# Returns stelmass - H1 mass
def _etg_stelmass_h1mass(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h1 = 'exclusive_sphere_50kpc', 
                   h1_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter / scatter_new / hexbin_count / hexbin_H1 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   print_typical_galaxy = True,       # median galaxy properties of detected ETG
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    
    H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
    H1_mass.convert_to_units('Msun')
    H1_mass.convert_to_physical()

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
    mask_h1      = H1_mass > cosmo_quantity(h1_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h1_SRs  = np.logical_and.reduce([mask_h1, mask_SRs])       # detection kappa < 0.4
    mask_h1_FRs  = np.logical_and.reduce([mask_h1, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h1, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h1, ~mask_SRs])      # non-detection kappa > 0.4
    
    
    #==========================================================
    # Print H1 detection
    print('--------------------------------')
    print('Sample length:   %s     Excl. kappa>0.4:    %s' %(len(stellar_mass), len(stellar_mass[mask_SRs])))
    print('FRACTION OF SAMPLE ABOVE H1 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
    if print_fdet:
        # Total sample
        for h1_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
            # Including FRs
            mask_h1_i = H1_mass > cosmo_quantity(h1_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
            f_detected_i = len(stellar_mass[mask_h1_i])/len(stellar_mass) 
            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h1_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
            print('  >  H1 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h1_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h1_i]), len(stellar_mass)))
        
            # Excluding FRs (ignore if LTG sample)
            if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                mask_h1_i = H1_mass > cosmo_quantity(h1_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h1_SRs_i = np.logical_and.reduce([mask_h1_i, mask_SRs_i])
                f_detected_i = len(stellar_mass[mask_h1_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h1_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h1_SRs_i]), len(stellar_mass[mask_SRs_i])))
                
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h1]), np.log10(H1_mass[mask_h1])) 
    print('\nSpearman incl. FR:    M* - h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h1_SRs]), np.log10(H1_mass[mask_h1_SRs])) 
    print('Spearman excl. FR:    M* - h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))


    #-----------------
    # Print median properties of H1-detected galaxy
    if print_typical_galaxy:
        H1_mass_fraction = np.divide(H1_mass, H1_mass + stellar_mass)
        sfr50 = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)[soap_indicies_sample]
        sfr50.convert_to_units(u.Msun/u.yr)
        sfr50.convert_to_physical()
        ssfr50 = np.divide(sfr50, stellar_mass)
    
        print('--------------------------------')
        print('MEDIAN VALUES OF HI-DETECTED GALAXY sample:   %s' %sample_input['name_of_preset'])
        print('    Median stelmass:             %.3e'%np.median(stellar_mass[mask_h1]))
        print('      Median H1mass:             %.3e'%np.median(H1_mass[mask_h1]))
        print('      Median H1mass fraction:    %.3e'%np.median(H1_mass_fraction[mask_h1]))
        #print('      Median SFR:                %.3e'%np.median(sfr50[mask_h1]))
        #print('      Median sSFR:               %.3e'%np.median(ssfr50[mask_h1]))
        print('      Median kappaco:            %.3f'%np.median(kappa_stars[mask_h1]))
        print('      Median u-r:                %.3f'%np.median(mag_plot[mask_h1]))
        
        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
            print('--------------------------------')
            print('MEDIAN VALUES OF HI-DETECTED GALAXY sample:   only kappa < 0.4')
            print('    Median stelmass:             %.3e'%np.median(stellar_mass[mask_h1_SRs]))
            print('      Median H1mass:             %.3e'%np.median(H1_mass[mask_h1_SRs]))
            print('      Median H1mass fraction:    %.3e'%np.median(H1_mass_fraction[mask_h1_SRs]))
            #print('      Median SFR:                %.3e'%np.median(sfr50[mask_h1_SRs]))
            #print('      Median sSFR:               %.3e'%np.median(ssfr50[mask_h1_SRs]))
            print('      Median kappaco:            %.3f'%np.median(kappa_stars[mask_h1_SRs]))
            print('      Median u-r:                %.3f'%np.median(mag_plot[mask_h1_SRs]))
        
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(H1_mass)
        C_VALUE_1 = kappa_stars
        S_VALUE_1 = (np.log10(H1_mass)-(np.log10(h1_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H1
        cb = axs.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c='r', s=S_VALUE_1[mask_h1], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H1_mass)
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=11)
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # Plot detections as filled viridis circles and squares
        axs.scatter(X_VALUE_1[mask_h1_SRs], Y_VALUE_1[mask_h1_SRs], c=C_VALUE_1[mask_h1_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
        cb = axs.scatter(X_VALUE_1[mask_h1_FRs], Y_VALUE_1[mask_h1_FRs], c=C_VALUE_1[mask_h1_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
        # Plot non-detections as empty grey circles and squares
        axs.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
        axs.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
    if scatter_or_hexbin == 'hexbin_count':
        ### Plots hexbin showing number of galaxies in bin
        
        cmap = cmasher.jungle_r
        newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
        # hexbin for all values
        extent = (9.5, 12.5, 6, 11)                     ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H1':  
        ### Plots hexbin showing median log10 H1_mass
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (9.5, 12.5, 6, 11)                     ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN = H1_mass[mask_SRs]
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
            y_bin = Y_MEDIAN.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask_SRs][mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN = np.log10(stellar_mass)
        Y_MEDIAN = H1_mass
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
            y_bin = Y_MEDIAN.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
    #-----------------
    # Add observations
    if add_observational:
        """
        Pick observations we want to add
        """
        add_serra2012    = True
        
        if add_serra2012:      # ($z=0.0$)
            # Load the observational data
            with h5py.File('%s/Serra2012_ATLAS3D_HI.hdf5'%obs_dir, 'r') as file:
                obs_names_1 = file['data/Galaxy/values'][:]
                obs_y       = file['data/log_H1/values'][:] * u.Unit(file['data/log_H1/values'].attrs['units'])
                obs_mask_1  = file['data/det_mask/values'][:]

            with h5py.File('%s/Cappellari2011_masses.hdf5'%obs_dir, 'r') as file:
                obs_names_2 = file['data/Galaxy/values'][:]
                obs_mstar  = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                obs_virgo  = file['data/Virgo/values'][:]
            
            obs_names_1 = np.array(obs_names_1)
            obs_y = obs_y.to('Msun')
            obs_mask_1  = np.array(obs_mask_1, dtype=bool)
            obs_names_1 = obs_names_1[obs_mask_1]
            obs_y       = obs_y[obs_mask_1]
            
            obs_names_2 = np.array(obs_names_2)
            obs_mstar = obs_mstar.to('Msun')
            
            # Match galaxy names to get mass (log)
            obs_x = []
            obs_isvirgo = []
            for name_i in obs_names_1:
                mask_name = np.argwhere(name_i == obs_names_2).squeeze()
                obs_x.append(obs_mstar[mask_name])
                obs_isvirgo.append(obs_virgo[mask_name])
            
            obs_isvirgo = np.array(obs_isvirgo, dtype=bool)
            obs_x = np.array(obs_x)
            assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
            
            label_extra = ''
            if 'cluster' in sample_input['name_of_preset']:
                obs_x = obs_x[obs_isvirgo]
                obs_y = obs_y[obs_isvirgo]
                label_extra = '\n(Virgo)'
            if 'group' in sample_input['name_of_preset']:
                obs_x = obs_x[~obs_isvirgo]
                obs_y = obs_y[~obs_isvirgo]
                label_extra = '\n(field/group)'
            
            
            print('Sample length Serra+12 ATLAS3D:   %s'%len(obs_y))
            axs.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Serra+12%s'%(label_extra))
            
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(6, 11)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    axs.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h1 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{I}}}$ [M$_{\odot}$]')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H1':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$]', extend='min')
      
    
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_MH1/%s_%s_%s_Mstar_MH1_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_MH1/%s_%s_%s_Mstar_MH1_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - H1 mass fraction ( H1 / H1 + M* )
def _etg_stelmass_h1massfraction(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h1 = 'exclusive_sphere_50kpc', 
                   h1_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter / scatter_new / hexbin_count / hexbin_H1 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    
    H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
    H1_mass.convert_to_units('Msun')
    H1_mass.convert_to_physical()

    u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
    r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
    u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    mag_plot = u_mag - r_mag

    central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

    kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
    
    #================================
    # Calculated values
    H1_mass_fraction = np.divide(H1_mass, H1_mass + stellar_mass)
    
    
    #==========================================================
    # Useful masks
    mask_h1      = H1_mass > cosmo_quantity(h1_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h1_SRs  = np.logical_and.reduce([mask_h1, mask_SRs])       # detection kappa < 0.4
    mask_h1_FRs  = np.logical_and.reduce([mask_h1, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h1, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h1, ~mask_SRs])      # non-detection kappa > 0.4
    
    
    #==========================================================
    # Print H1 detection
    print('--------------------------------')
    print('Sample length:   %s     Excl. kappa>0.4:    %s' %(len(stellar_mass), len(stellar_mass[mask_SRs])))
    print('FRACTION OF SAMPLE ABOVE H1 FRACTION:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
    if print_fdet:
        # Total sample
        for h1_detection_limit_i in [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]:
            # Including FRs
            mask_h1_i = H1_mass_fraction > cosmo_quantity(h1_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
            f_detected_i = len(stellar_mass[mask_h1_i])/len(stellar_mass) 
            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h1_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
            print('  >  f_H1 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h1_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h1_i]), len(stellar_mass)))
        
            # Excluding FRs (ignore if LTG sample)
            if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                mask_h1_i = H1_mass_fraction > cosmo_quantity(h1_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h1_SRs_i = np.logical_and.reduce([mask_h1_i, mask_SRs_i])
                f_detected_i = len(stellar_mass[mask_h1_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h1_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h1_SRs_i]), len(stellar_mass[mask_SRs_i])))
                
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h1]), np.log10(H1_mass_fraction[mask_h1])) 
    print('\nSpearman incl. FR:    M* - f_h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h1_SRs]), np.log10(H1_mass_fraction[mask_h1_SRs])) 
    print('Spearman excl. FR:    M* - f_h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(H1_mass_fraction)
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H1_mass.value > 0] = np.log10(H1_mass[H1_mass.value > 0])
        S_VALUE_1 = (np.log10(H1_mass)-(np.log10(h1_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H1
        cb = axs.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c='r', s=S_VALUE_1[mask_h1], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H1_mass)
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=11)
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # Plot detections as filled viridis circles and squares
        axs.scatter(X_VALUE_1[mask_h1_SRs], Y_VALUE_1[mask_h1_SRs], c=C_VALUE_1[mask_h1_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
        cb = axs.scatter(X_VALUE_1[mask_h1_FRs], Y_VALUE_1[mask_h1_FRs], c=C_VALUE_1[mask_h1_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
        # Plot non-detections as empty grey circles and squares
        axs.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
        axs.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
    if scatter_or_hexbin == 'hexbin_count':
        ### Plots hexbin showing number of galaxies in bin
        
        cmap = cmasher.jungle_r
        newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
        # hexbin for all values
        extent = (9.5, 12.5, -5, 0)                     ###
        gridsize = (25,12)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H1':  
        ### Plots hexbin showing median log10 H1_mass
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (9.5, 12.5, -5, 0)                     ###
        gridsize = (25,12)                              ###     25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN = H1_mass_fraction[mask_SRs]
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
            y_bin = Y_MEDIAN.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask_SRs][mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN = np.log10(stellar_mass)
        Y_MEDIAN = H1_mass_fraction
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
            y_bin = Y_MEDIAN.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
    #-----------------
    # Add observations
    if add_observational:
        """
        Pick observations we want to add
        """
        add_serra2012    = True
        
        if add_serra2012:      # ($z=0.0$)
            # Load the observational data
            with h5py.File('%s/Serra2012_ATLAS3D_HI.hdf5'%obs_dir, 'r') as file:
                obs_names_1 = file['data/Galaxy/values'][:]
                obs_HI       = file['data/log_H1/values'][:] * u.Unit(file['data/log_H1/values'].attrs['units'])
                obs_mask_1  = file['data/det_mask/values'][:]

            with h5py.File('%s/Cappellari2011_masses.hdf5'%obs_dir, 'r') as file:
                obs_names_2 = file['data/Galaxy/values'][:]
                obs_mstar  = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                obs_virgo  = file['data/Virgo/values'][:]
            
            obs_names_1 = np.array(obs_names_1)
            obs_HI = obs_HI.to('Msun')
            obs_mask_1  = np.array(obs_mask_1, dtype=bool)
            obs_names_1 = obs_names_1[obs_mask_1]
            obs_HI      = obs_HI[obs_mask_1]
            
            obs_names_2 = np.array(obs_names_2)
            obs_mstar = obs_mstar.to('Msun')
            
            # Match galaxy names to get mass (log)
            obs_x = []
            obs_isvirgo = []
            for name_i in obs_names_1:
                mask_name = np.argwhere(name_i == obs_names_2).squeeze()
                obs_x.append(obs_mstar[mask_name])
                obs_isvirgo.append(obs_virgo[mask_name])
            
            obs_isvirgo = np.array(obs_isvirgo, dtype=bool)
            obs_x = np.array(obs_x)            
            assert len(obs_x) == len(obs_HI), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_HI))
            
            # Find fraction fgas = Mgas / Mgas + M*
            obs_y = obs_HI.value - np.log10((10**obs_HI.value) + (10**obs_x))
            
            label_extra = ''
            if 'cluster' in sample_input['name_of_preset']:
                obs_x = obs_x[obs_isvirgo]
                obs_y = obs_y[obs_isvirgo]
                label_extra = '\n(Virgo)'
            if 'group' in sample_input['name_of_preset']:
                obs_x = obs_x[~obs_isvirgo]
                obs_y = obs_y[~obs_isvirgo]
                label_extra = '\n(field/group)'
            
            print('Sample length Serra+12 ATLAS3D:   %s'%len(obs_y))
            axs.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Serra+12%s'%(label_extra))
            
        
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-5, 0)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    axs.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h1 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'log$_{10}$ $f_{\mathrm{H_{I}}}$')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H1':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$]', extend='min')
      
    
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_fracH1/%s_%s_%s_Mstar_fracH1_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_fracH1/%s_%s_%s_Mstar_fracH1_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Plots above plots as double
def _etg_stelmass_h1mass_double(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h1 = 'exclusive_sphere_50kpc', 
                   h1_detection_limit = 10**7,
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):

    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig = plt.figure(figsize=(10/3, 5))
    gs  = fig.add_gridspec(2, 1,  height_ratios=(1, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.1)
    # Create the Axes.
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])  
    
    
    
    
    #=====================================================
    # Plot top plot as H2 mass
    scatter_or_hexbin       = 'hexbin_count'
    
    plot_top = True
    if plot_top:
        #---------------------------
        # Extract data from samples:
        dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                       'all_ETGs': 'ETGs (excl. FRs)',
                       'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
        dict_colors = {'all_galaxies': 'k',
                       'all_ETGs': 'C0',
                       'all_ETGs_plus_redspiral': 'C1'}
        dict_ls     = {'all_galaxies': '-',
                       'all_ETGs': '--',
                       'all_ETGs_plus_redspiral': '-.'}
        dict_ms     = {'all_galaxies': 'o',
                       'all_ETGs': 's',
                       'all_ETGs_plus_redspiral': 'D'}
                   
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
        box_size = data.metadata.boxsize[0]


        #-------------------------------
        # Get essential SOAP data for analysis
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
    
        H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        H1_mass.convert_to_units('Msun')
        H1_mass.convert_to_physical()

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
        mask_h1      = H1_mass > cosmo_quantity(h1_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_h1_SRs  = np.logical_and.reduce([mask_h1, mask_SRs])       # detection kappa < 0.4
        mask_h1_FRs  = np.logical_and.reduce([mask_h1, ~mask_SRs])      # detection kappa > 0.4
        mask_X_SRs  = np.logical_and.reduce([~mask_h1, mask_SRs])       # non-detection kappa < 0.4
        mask_X_FRs  = np.logical_and.reduce([~mask_h1, ~mask_SRs])      # non-detection kappa > 0.4

    
    
        #==========================================================
        # Plot scatter or hexbin
    
        # Define inputs
        with np.errstate(divide='ignore'):
            X_VALUE_1 = np.log10(stellar_mass)
            Y_VALUE_1 = np.log10(H1_mass)
            C_VALUE_1 = kappa_stars
            S_VALUE_1 = (np.log10(H1_mass)-(np.log10(h1_detection_limit)-1))**2.5
    
    
        if scatter_or_hexbin == 'scatter_old':
            ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H1
            cb_1 = ax_top.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c='r', s=S_VALUE_1[mask_h1], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
            ax_top.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
        if scatter_or_hexbin == 'scatter_new':
            ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
            C_VALUE_1 = np.log10(H1_mass)
        
            # Define default colormap for H1
            norm = colors.Normalize(vmin=6.5, vmax=11)
        
            # Create custom colormap with grey at the bottom
            viridis = mpl.cm.viridis
            newcolors = viridis(np.linspace(0, 1, 9))
            newcolors[:1, :] = colors.to_rgba('grey')
            newcmp = ListedColormap(newcolors)
        
            # Plot detections as filled viridis circles and squares
            ax_top.scatter(X_VALUE_1[mask_h1_SRs], Y_VALUE_1[mask_h1_SRs], c=C_VALUE_1[mask_h1_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
            cb_1 = ax_top.scatter(X_VALUE_1[mask_h1_FRs], Y_VALUE_1[mask_h1_FRs], c=C_VALUE_1[mask_h1_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
            # Plot non-detections as empty grey circles and squares
            ax_top.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
            ax_top.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
        if scatter_or_hexbin == 'hexbin_count':
            ### Plots hexbin showing number of galaxies in bin
        
            cmap = cmasher.jungle_r
            newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
            # hexbin for all values
            extent = (9.5, 12.5, 6, 11)                     ###
            gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_top.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_1 = ax_top.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
        if scatter_or_hexbin == 'hexbin_H1':  
            ### Plots hexbin showing median log10 H1_mass
        
            # Define default colormap for H1
            norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
            # Create custom colormap with grey at the bottom
            viridis = mpl.cm.viridis
            newcolors = viridis(np.linspace(0, 1, 9))
            newcolors[:1, :] = colors.to_rgba('grey')
            newcmp = ListedColormap(newcolors)
        
            # hexbin for all values
            extent = (9.5, 12.5, 6, 11)                     ###
            gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            #cb_1 = ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
            ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_1 = ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
        #------------
        # Median line of detected ETGs
        if add_median_line:
            #-----------------
            # Define binning parameters
            hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
            #------------------------------------------
            ### ETG (excl. FR) sample
            # Feed median, gas is found under H1_mass
            X_MEDIAN = np.log10(stellar_mass[mask_SRs])
            Y_MEDIAN = H1_mass[mask_SRs]
        
            # Compute statistics in each bin
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            bins_n = []
            for i in range(len(hist_bins) - 1):
                mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                y_bin = Y_MEDIAN.value[mask]
        
                # Remove <107 H1 mass from sample
                y_bin = y_bin[H1_mass[mask_SRs][mask].value > h1_detection_limit]
            
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
        
            ax_top.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_top.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
            #------------------------------------------
            ### ETG (incl. FR) sample
            # Feed median, gas is found under H1_mass
            X_MEDIAN = np.log10(stellar_mass)
            Y_MEDIAN = H1_mass
        
            # Compute statistics in each bin
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            bins_n = []
            for i in range(len(hist_bins) - 1):
                mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                y_bin = Y_MEDIAN.value[mask]
        
                # Remove <107 H1 mass from sample
                y_bin = y_bin[H1_mass[mask].value > h1_detection_limit]
            
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
        
            ax_top.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_top.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
        #-----------------
        # Add observations
        if add_observational:
            """
            Pick observations we want to add
            """
            add_serra2012    = True
        
            if add_serra2012:      # ($z=0.0$)
                # Load the observational data
                with h5py.File('%s/Serra2012_ATLAS3D_HI.hdf5'%obs_dir, 'r') as file:
                    obs_names_1 = file['data/Galaxy/values'][:]
                    obs_y       = file['data/log_H1/values'][:] * u.Unit(file['data/log_H1/values'].attrs['units'])
                    obs_mask_1  = file['data/det_mask/values'][:]

                with h5py.File('%s/Cappellari2011_masses.hdf5'%obs_dir, 'r') as file:
                    obs_names_2 = file['data/Galaxy/values'][:]
                    obs_mstar  = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                    obs_virgo  = file['data/Virgo/values'][:]
            
                obs_names_1 = np.array(obs_names_1)
                obs_y = obs_y.to('Msun')
                obs_mask_1  = np.array(obs_mask_1, dtype=bool)
                obs_names_1 = obs_names_1[obs_mask_1]
                obs_y       = obs_y[obs_mask_1]
            
                obs_names_2 = np.array(obs_names_2)
                obs_mstar = obs_mstar.to('Msun')
            
                # Match galaxy names to get mass (log)
                obs_x = []
                obs_isvirgo = []
                for name_i in obs_names_1:
                    mask_name = np.argwhere(name_i == obs_names_2).squeeze()
                    obs_x.append(obs_mstar[mask_name])
                    obs_isvirgo.append(obs_virgo[mask_name])
            
                obs_isvirgo = np.array(obs_isvirgo, dtype=bool)
                obs_x = np.array(obs_x)
                assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
            
                label_extra = ''
                if 'cluster' in sample_input['name_of_preset']:
                    obs_x = obs_x[obs_isvirgo]
                    obs_y = obs_y[obs_isvirgo]
                    label_extra = '\n(Virgo)'
                if 'group' in sample_input['name_of_preset']:
                    obs_x = obs_x[~obs_isvirgo]
                    obs_y = obs_y[~obs_isvirgo]
                    label_extra = '\n(field/group)'
            
            
                print('Sample length Serra+12 ATLAS3D:   %s'%len(obs_y))
                ax_top.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Serra+12%s'%(label_extra))
            
        
        

    #=====================================================
    # Plot bottom plot as H2 mass fraction
    scatter_or_hexbin       = 'hexbin_H1'
    
    plot_bottom = True
    if plot_bottom:
        #---------------------------
        # Extract data from samples:
        dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                       'all_ETGs': 'ETGs (excl. FRs)',
                       'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
        dict_colors = {'all_galaxies': 'k',
                       'all_ETGs': 'C0',
                       'all_ETGs_plus_redspiral': 'C1'}
        dict_ls     = {'all_galaxies': '-',
                       'all_ETGs': '--',
                       'all_ETGs_plus_redspiral': '-.'}
        dict_ms     = {'all_galaxies': 'o',
                       'all_ETGs': 's',
                       'all_ETGs_plus_redspiral': 'D'}
                   
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
        box_size = data.metadata.boxsize[0]


        #-------------------------------
        # Get essential SOAP data for analysis
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
    
        H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        H1_mass.convert_to_units('Msun')
        H1_mass.convert_to_physical()

        u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
        r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
        u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        mag_plot = u_mag - r_mag

        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
    
        #================================
        # Calculated values
        H1_mass_fraction = np.divide(H1_mass, H1_mass + stellar_mass)
    
    
        #==========================================================
        # Useful masks
        mask_h1      = H1_mass > cosmo_quantity(h1_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_h1_SRs  = np.logical_and.reduce([mask_h1, mask_SRs])       # detection kappa < 0.4
        mask_h1_FRs  = np.logical_and.reduce([mask_h1, ~mask_SRs])      # detection kappa > 0.4
        mask_X_SRs  = np.logical_and.reduce([~mask_h1, mask_SRs])       # non-detection kappa < 0.4
        mask_X_FRs  = np.logical_and.reduce([~mask_h1, ~mask_SRs])      # non-detection kappa > 0.4
    
    
        #==========================================================
        # Plot scatter or hexbin
    
        # Define inputs
        with np.errstate(divide='ignore'):
            X_VALUE_1 = np.log10(stellar_mass)
            Y_VALUE_1 = np.log10(H1_mass_fraction)
            C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
            C_VALUE_1[H1_mass.value > 0] = np.log10(H1_mass[H1_mass.value > 0])
            S_VALUE_1 = (np.log10(H1_mass)-(np.log10(h1_detection_limit)-1))**2.5
    
    
        if scatter_or_hexbin == 'scatter_old':
            ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H1
            cb_2 = ax_bot.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c='r', s=S_VALUE_1[mask_h1], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
            ax_bot.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
        if scatter_or_hexbin == 'scatter_new':
            ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
            C_VALUE_1 = np.log10(H1_mass)
        
            # Define default colormap for H1
            norm = colors.Normalize(vmin=6.5, vmax=11)
        
            # Create custom colormap with grey at the bottom
            viridis = mpl.cm.viridis
            newcolors = viridis(np.linspace(0, 1, 9))
            newcolors[:1, :] = colors.to_rgba('grey')
            newcmp = ListedColormap(newcolors)
        
            # Plot detections as filled viridis circles and squares
            ax_bot.scatter(X_VALUE_1[mask_h1_SRs], Y_VALUE_1[mask_h1_SRs], c=C_VALUE_1[mask_h1_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
            cb_2 = ax_bot.scatter(X_VALUE_1[mask_h1_FRs], Y_VALUE_1[mask_h1_FRs], c=C_VALUE_1[mask_h1_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
            # Plot non-detections as empty grey circles and squares
            ax_bot.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
            ax_bot.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
        if scatter_or_hexbin == 'hexbin_count':
            ### Plots hexbin showing number of galaxies in bin
        
            cmap = cmasher.jungle_r
            newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
            # hexbin for all values
            extent = (9.5, 12.5, -5, 0)                     ###
            gridsize = (25,12)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_2 = ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
        if scatter_or_hexbin == 'hexbin_H1':  
            ### Plots hexbin showing median log10 H1_mass
        
            # Define default colormap for H1
            norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
            # Create custom colormap with grey at the bottom
            viridis = mpl.cm.viridis
            newcolors = viridis(np.linspace(0, 1, 9))
            newcolors[:1, :] = colors.to_rgba('grey')
            newcmp = ListedColormap(newcolors)
        
            # hexbin for all values
            extent = (9.5, 12.5, -5, 0)                     ###
            gridsize = (25,12)                              ###     25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            #cb_2 = ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
            ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_2 = ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
        #------------
        # Median line of detected ETGs
        if add_median_line:
            #-----------------
            # Define binning parameters
            hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
            #------------------------------------------
            ### ETG (excl. FR) sample
            # Feed median, gas is found under H1_mass
            X_MEDIAN = np.log10(stellar_mass[mask_SRs])
            Y_MEDIAN = H1_mass_fraction[mask_SRs]
        
            # Compute statistics in each bin
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            bins_n = []
            for i in range(len(hist_bins) - 1):
                mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                y_bin = Y_MEDIAN.value[mask]
        
                # Remove <107 H1 mass from sample
                y_bin = y_bin[H1_mass[mask_SRs][mask].value > h1_detection_limit]
            
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
        
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_bot.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
            #------------------------------------------
            ### ETG (incl. FR) sample
            # Feed median, gas is found under H1_mass
            X_MEDIAN = np.log10(stellar_mass)
            Y_MEDIAN = H1_mass_fraction
        
            # Compute statistics in each bin
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            bins_n = []
            for i in range(len(hist_bins) - 1):
                mask = (X_MEDIAN >= hist_bins[i]) & (X_MEDIAN < hist_bins[i + 1])
                y_bin = Y_MEDIAN.value[mask]
        
                # Remove <107 H1 mass from sample
                y_bin = y_bin[H1_mass[mask].value > h1_detection_limit]
            
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
        
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_bot.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
        #-----------------
        # Add observations
        if add_observational:
            """
            Pick observations we want to add
            """
            add_serra2012    = True
        
            if add_serra2012:      # ($z=0.0$)
                # Load the observational data
                with h5py.File('%s/Serra2012_ATLAS3D_HI.hdf5'%obs_dir, 'r') as file:
                    obs_names_1 = file['data/Galaxy/values'][:]
                    obs_HI       = file['data/log_H1/values'][:] * u.Unit(file['data/log_H1/values'].attrs['units'])
                    obs_mask_1  = file['data/det_mask/values'][:]

                with h5py.File('%s/Cappellari2011_masses.hdf5'%obs_dir, 'r') as file:
                    obs_names_2 = file['data/Galaxy/values'][:]
                    obs_mstar  = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                    obs_virgo  = file['data/Virgo/values'][:]
            
                obs_names_1 = np.array(obs_names_1)
                obs_HI = obs_HI.to('Msun')
                obs_mask_1  = np.array(obs_mask_1, dtype=bool)
                obs_names_1 = obs_names_1[obs_mask_1]
                obs_HI      = obs_HI[obs_mask_1]
            
                obs_names_2 = np.array(obs_names_2)
                obs_mstar = obs_mstar.to('Msun')
            
                # Match galaxy names to get mass (log)
                obs_x = []
                obs_isvirgo = []
                for name_i in obs_names_1:
                    mask_name = np.argwhere(name_i == obs_names_2).squeeze()
                    obs_x.append(obs_mstar[mask_name])
                    obs_isvirgo.append(obs_virgo[mask_name])
            
                obs_isvirgo = np.array(obs_isvirgo, dtype=bool)
                obs_x = np.array(obs_x)            
                assert len(obs_x) == len(obs_HI), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_HI))
            
                # Find fraction fgas = Mgas / Mgas + M*
                obs_y = obs_HI.value - np.log10((10**obs_HI.value) + (10**obs_x))
            
                label_extra = ''
                if 'cluster' in sample_input['name_of_preset']:
                    obs_x = obs_x[obs_isvirgo]
                    obs_y = obs_y[obs_isvirgo]
                    label_extra = '\n(Virgo)'
                if 'group' in sample_input['name_of_preset']:
                    obs_x = obs_x[~obs_isvirgo]
                    obs_y = obs_y[~obs_isvirgo]
                    label_extra = '\n(field/group)'
            
                print('Sample length Serra+12 ATLAS3D:   %s'%len(obs_y))
                ax_bot.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Serra+12%s'%(label_extra))


    #=====================================================
    ### Graph formatting
    
    #-----------
    # Axis formatting
    ax_top.set_xlim(9.5, 12.5)
    ax_top.set_ylim(6, 11)
    ax_top.minorticks_on()
    ax_top.tick_params(axis='x', which='minor')
    ax_top.tick_params(axis='y', which='minor')
    ax_top.set_xticklabels([])
    ax_top.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{I}}}$ [M$_{\odot}$]')
    
    ax_bot.set_xlim(9.5, 12.5)
    ax_bot.set_ylim(-5, 0)
    ax_bot.minorticks_on()
    ax_bot.tick_params(axis='x', which='minor')
    ax_bot.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h1 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    ax_bot.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    ax_bot.set_ylabel(r'log$_{10}$ $f_{\mathrm{H_{I}}}$')


	#-----------
    # colorbar
    fig.colorbar(cb_1, ax=ax_top, label='Number of galaxies')
    fig.colorbar(cb_2, ax=ax_bot, label='Median log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$]', extend='min')
      
    
    #-----------  
    # Annotations
    #ax_top.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = ax_top.transAxes)
    ##ax_top.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = ax_top.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    else:
        fig_text(x=0.109, y=0.93, ha='left', s=text_title, fontsize=7, ax=ax_top, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    
    #ax_top.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = ax_top.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        ax_top.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        ax_top.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_MH1/%s_%s_%s_Mstar_H1both_%s_H1ap%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_MH1/%s_%s_%s_Mstar_H1both_%s_H1ap%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
    

#-----------------
# Returns stelmass - H2 mass
def _etg_stelmass_h2mass(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   print_typical_galaxy = True,       # median galaxy properties of detected ETG
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    # Print H2 detection
    print('--------------------------------')
    print('Sample length:   %s     Excl. kappa>0.4:    %s' %(len(stellar_mass), len(stellar_mass[mask_SRs])))
    print('FRACTION OF SAMPLE ABOVE H2 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
    if print_fdet:
        # Total sample
        for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
            # Including FRs
            mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
            f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
            print('  >  H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
            # Excluding FRs (ignore if LTG sample)
            if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
                
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h2]), np.log10(H2_mass[mask_h2])) 
    print('\nSpearman incl. FR:    M* - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h2_SRs]), np.log10(H2_mass[mask_h2_SRs])) 
    print('Spearman excl. FR:    M* - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #-----------------
    # Print median properties of H2-detected galaxy
    if print_typical_galaxy:
        H2_mass_fraction = np.divide(H2_mass, H2_mass + stellar_mass)
        sfr50 = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)[soap_indicies_sample]
        sfr50.convert_to_units(u.Msun/u.yr)
        sfr50.convert_to_physical()
        ssfr50 = np.divide(sfr50, stellar_mass)
    
        print('--------------------------------')
        print('MEDIAN VALUES OF HI-DETECTED GALAXY sample:   %s' %sample_input['name_of_preset'])
        print('    Median stelmass:             %.3e'%np.median(stellar_mass[mask_h2]))
        print('      Median H2mass:             %.3e'%np.median(H2_mass[mask_h2]))
        print('      Median H2mass fraction:    %.3e'%np.median(H2_mass_fraction[mask_h2]))
        #print('      Median SFR:                %.3e'%np.median(sfr50[mask_h2]))
        #print('      Median sSFR:               %.3e'%np.median(ssfr50[mask_h2]))
        print('      Median kappaco:            %.3f'%np.median(kappa_stars[mask_h2]))
        print('      Median u-r:                %.3f'%np.median(mag_plot[mask_h2]))
        
        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
            print('--------------------------------')
            print('MEDIAN VALUES OF HI-DETECTED GALAXY sample:   only kappa < 0.4')
            print('    Median stelmass:             %.3e'%np.median(stellar_mass[mask_h2_SRs]))
            print('      Median H2mass:             %.3e'%np.median(H2_mass[mask_h2_SRs]))
            print('      Median H2mass fraction:    %.3e'%np.median(H2_mass_fraction[mask_h2_SRs]))
            #print('      Median SFR:                %.3e'%np.median(sfr50[mask_h2_SRs]))
            #print('      Median sSFR:               %.3e'%np.median(ssfr50[mask_h2_SRs]))
            print('      Median kappaco:            %.3f'%np.median(kappa_stars[mask_h2_SRs]))
            print('      Median u-r:                %.3f'%np.median(mag_plot[mask_h2_SRs]))
        
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(H2_mass)
        C_VALUE_1 = kappa_stars
        S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
        cb = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H2_mass)
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)
        
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
        extent = (9.5, 12.5, 6, 11)                     ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H2':  
        ### Plots hexbin showing median log10 H2_mass
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (9.5, 12.5, 6, 11)                     ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
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
            axs.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+19')
            axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
            #axs.text(np.log10(0.82*(10**11.41)), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
            #axs.text(np.log10(0.82*(10**11.41)), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
            #axs.scatter(obs_Mstar_1, obs_H2_1, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (ATLAS$^{\mathrm{3D}}$)')
            #axs.scatter(obs_Mstar_2, obs_H2_2, marker='s', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (MASSIVE)')
            
            
        
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(6, 11)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
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
    axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ [M$_{\odot}$]')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H2':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$]', extend='min')
      
    
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_MH2/%s_%s_%s_Mstar_MH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_MH2/%s_%s_%s_Mstar_MH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - H2 mass fraction ( H2 / H2 + M* )
def _etg_stelmass_h2massfraction(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    H2_mass_fraction = np.divide(H2_mass, H2_mass + stellar_mass)
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
    
    
    #==========================================================
    # Print H2 detection
    print('--------------------------------')
    print('Sample length:   %s     Excl. kappa>0.4:    %s' %(len(stellar_mass), len(stellar_mass[mask_SRs])))
    print('FRACTION OF SAMPLE ABOVE H2 FRACTION:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
    if print_fdet:
        # Total sample
        for h2_detection_limit_i in [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]:
            # Including FRs
            mask_h2_i = H2_mass_fraction > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
            f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
            print('  >  f_H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
            # Excluding FRs (ignore if LTG sample)
            if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                mask_h2_i = H2_mass_fraction > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
                
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h2]), np.log10(H2_mass_fraction[mask_h2])) 
    print('\nSpearman incl. FR:    M* - f_h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(np.log10(stellar_mass[mask_h2_SRs]), np.log10(H2_mass_fraction[mask_h2_SRs])) 
    print('Spearman excl. FR:    M* - f_h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(H2_mass_fraction)
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
        cb = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H2_mass)
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)
        
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
        extent = (9.5, 12.5, -5, 0)                     ###
        gridsize = (25,12)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        extent = (9.5, 12.5, -5, 0)                     ###
        gridsize = (25,12)                              ###     25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
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
            
            # Find fraction fgas = Mgas / Mgas + M*
            obs_y = obs_y - np.log10((10**obs_y) + (10**obs_x))
            
            axs.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+19')
            axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
            #axs.text(np.log10(0.82*(10**11.41)), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
            #axs.text(np.log10(0.82*(10**11.41)), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
            #axs.scatter(obs_Mstar_1, obs_H2_1, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (ATLAS$^{\mathrm{3D}}$)')
            #axs.scatter(obs_Mstar_2, obs_H2_2, marker='s', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (MASSIVE)')
    
        
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-5, 0)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
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
    axs.set_ylabel(r'log$_{10}$ $f_{\mathrm{H_{2}}}$')


	#-----------
    # colorbar
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
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_fracH2/%s_%s_%s_Mstar_fracH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_fracH2/%s_%s_%s_Mstar_fracH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Plots above plots as double
def _etg_stelmass_h2mass_double(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):

    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig = plt.figure(figsize=(10/3, 5))
    gs  = fig.add_gridspec(2, 1,  height_ratios=(1, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.1)
    # Create the Axes.
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])  
    
    
    
    
    #=====================================================
    # Plot top plot as H2 mass
    scatter_or_hexbin       = 'hexbin_count'
    
    plot_top = True
    if plot_top:
        #---------------------------
        # Extract data from samples:
        dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                       'all_ETGs': 'ETGs (excl. FRs)',
                       'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
        dict_colors = {'all_galaxies': 'k',
                       'all_ETGs': 'C0',
                       'all_ETGs_plus_redspiral': 'C1'}
        dict_ls     = {'all_galaxies': '-',
                       'all_ETGs': '--',
                       'all_ETGs_plus_redspiral': '-.'}
        dict_ms     = {'all_galaxies': 'o',
                       'all_ETGs': 's',
                       'all_ETGs_plus_redspiral': 'D'}
                   
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
        box_size = data.metadata.boxsize[0]


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
            Y_VALUE_1 = np.log10(H2_mass)
            C_VALUE_1 = kappa_stars
            S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
        if scatter_or_hexbin == 'scatter_old':
            ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
            cb_1 = ax_top.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
            ax_top.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
        if scatter_or_hexbin == 'scatter_new':
            ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
            C_VALUE_1 = np.log10(H2_mass)
        
            # Define default colormap for H2
            norm = colors.Normalize(vmin=6.5, vmax=11)
        
            # Create custom colormap with grey at the bottom
            viridis = mpl.cm.viridis
            newcolors = viridis(np.linspace(0, 1, 9))
            newcolors[:1, :] = colors.to_rgba('grey')
            newcmp = ListedColormap(newcolors)
        
            # Plot detections as filled viridis circles and squares
            ax_top.scatter(X_VALUE_1[mask_h2_SRs], Y_VALUE_1[mask_h2_SRs], c=C_VALUE_1[mask_h2_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
            cb_1 = ax_top.scatter(X_VALUE_1[mask_h2_FRs], Y_VALUE_1[mask_h2_FRs], c=C_VALUE_1[mask_h2_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
            # Plot non-detections as empty grey circles and squares
            ax_top.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
            ax_top.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
        if scatter_or_hexbin == 'hexbin_count':
            ### Plots hexbin showing number of galaxies in bin
        
            cmap = cmasher.jungle_r
            newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
            # hexbin for all values
            extent = (9.5, 12.5, 6, 11)                     ###
            gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_top.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_1 = ax_top.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
        if scatter_or_hexbin == 'hexbin_H2':  
            ### Plots hexbin showing median log10 H2_mass
        
            # Define default colormap for H2
            norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
            # Create custom colormap with grey at the bottom
            viridis = mpl.cm.viridis
            newcolors = viridis(np.linspace(0, 1, 9))
            newcolors[:1, :] = colors.to_rgba('grey')
            newcmp = ListedColormap(newcolors)
        
            # hexbin for all values
            extent = (9.5, 12.5, 6, 11)                     ###
            gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            #cb_1 = ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
            ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_1 = ax_top.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
        #------------
        # Median line of detected ETGs
        if add_median_line:
            #-----------------
            # Define binning parameters
            hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
            #------------------------------------------
            ### ETG (excl. FR) sample
            # Feed median, gas is found under H2_mass
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
        
            ax_top.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_top.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
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
        
            ax_top.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_top.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_top.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
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
                ax_top.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+19')
                ax_top.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
                #ax_top.text(np.log10(0.82*(10**11.41)), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                #ax_top.text(np.log10(0.82*(10**11.41)), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
                #ax_top.scatter(obs_Mstar_1, obs_H2_1, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (ATLAS$^{\mathrm{3D}}$)')
                #ax_top.scatter(obs_Mstar_2, obs_H2_2, marker='s', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (MASSIVE)')
            

    #=====================================================
    # Plot bottom plot as H2 mass fraction
    scatter_or_hexbin       = 'hexbin_H2'
    
    plot_bottom = True
    if plot_bottom:
        #---------------------------
        # Extract data from samples:
        dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                       'all_ETGs': 'ETGs (excl. FRs)',
                       'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
        dict_colors = {'all_galaxies': 'k',
                       'all_ETGs': 'C0',
                       'all_ETGs_plus_redspiral': 'C1'}
        dict_ls     = {'all_galaxies': '-',
                       'all_ETGs': '--',
                       'all_ETGs_plus_redspiral': '-.'}
        dict_ms     = {'all_galaxies': 'o',
                       'all_ETGs': 's',
                       'all_ETGs_plus_redspiral': 'D'}
                   
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
        box_size = data.metadata.boxsize[0]


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
        H2_mass_fraction = np.divide(H2_mass, H2_mass + stellar_mass)
    
    
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
            Y_VALUE_1 = np.log10(H2_mass_fraction)
            C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
            C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
            S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
        if scatter_or_hexbin == 'scatter_old':
            ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
            cb_2 = ax_bot.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
            ax_bot.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
        if scatter_or_hexbin == 'scatter_new':
            ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
            C_VALUE_1 = np.log10(H2_mass)
        
            # Define default colormap for H2
            norm = colors.Normalize(vmin=6.5, vmax=11)
        
            # Create custom colormap with grey at the bottom
            viridis = mpl.cm.viridis
            newcolors = viridis(np.linspace(0, 1, 9))
            newcolors[:1, :] = colors.to_rgba('grey')
            newcmp = ListedColormap(newcolors)
        
            # Plot detections as filled viridis circles and squares
            ax_bot.scatter(X_VALUE_1[mask_h2_SRs], Y_VALUE_1[mask_h2_SRs], c=C_VALUE_1[mask_h2_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
            cb_2 = ax_bot.scatter(X_VALUE_1[mask_h2_FRs], Y_VALUE_1[mask_h2_FRs], c=C_VALUE_1[mask_h2_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
            # Plot non-detections as empty grey circles and squares
            ax_bot.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
            ax_bot.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
        if scatter_or_hexbin == 'hexbin_count':
            ### Plots hexbin showing number of galaxies in bin
        
            cmap = cmasher.jungle_r
            newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
            # hexbin for all values
            extent = (9.5, 12.5, -5, 0)                     ###
            gridsize = (25,12)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_2 = ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
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
            extent = (9.5, 12.5, -5, 0)                     ###
            gridsize = (25,12)                              ###     25 by default, then multiply by axis_x_range/axis_y_range
        
            #ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            #cb_2 = ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
            ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
            # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
            cb_2 = ax_bot.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
        #------------
        # Median line of detected ETGs
        if add_median_line:
            #-----------------
            # Define binning parameters
            hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
            #------------------------------------------
            ### ETG (excl. FR) sample
            # Feed median, gas is found under H2_mass
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
        
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_bot.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
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
        
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            ax_bot.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
            #ax_bot.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
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
            
                # Find fraction fgas = Mgas / Mgas + M*
                obs_y = obs_y - np.log10((10**obs_y) + (10**obs_x))
            
                ax_bot.scatter(obs_x, obs_y, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+19')
                ax_bot.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
                #ax_bot.text(np.log10(0.82*(10**11.41)), 11, 'ATLAS$^{\mathrm{3D}}$', fontsize=5, c='grey', rotation=90, ha='left', va='top')
                #ax_bot.text(np.log10(0.82*(10**11.41)), 11, 'MASSIVE', fontsize=5, c='grey', rotation=270, ha='right', va='top')
            
                #ax_bot.scatter(obs_Mstar_1, obs_H2_1, marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (ATLAS$^{\mathrm{3D}}$)')
                #ax_bot.scatter(obs_Mstar_2, obs_H2_2, marker='s', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='D+19 (MASSIVE)')
    
        
        
    #=====================================================
    ### Graph formatting
    
    #-----------
    # Axis formatting
    ax_top.set_xlim(9.5, 12.5)
    ax_top.set_ylim(6, 11)
    ax_top.minorticks_on()
    ax_top.tick_params(axis='x', which='minor')
    ax_top.tick_params(axis='y', which='minor')
    ax_top.set_xticklabels([])
    ax_top.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ [M$_{\odot}$]')
    
    
    ax_bot.set_xlim(9.5, 12.5)
    ax_bot.set_ylim(-5, 0)
    ax_bot.minorticks_on()
    ax_bot.tick_params(axis='x', which='minor')
    ax_bot.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h2 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    ax_bot.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    ax_bot.set_ylabel(r'log$_{10}$ $f_{\mathrm{H_{2}}}$')


	#-----------
    # colorbar
    fig.colorbar(cb_1, ax=ax_top, label='Number of galaxies')
    fig.colorbar(cb_2, ax=ax_bot, label='Median log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$]', extend='both')
      
    
    #-----------  
    # Annotations
    #ax_top.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = ax_top.transAxes)
    ##ax_top.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = ax_top.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    else:
        fig_text(x=0.109, y=0.93, ha='left', s=text_title, fontsize=7, ax=ax_top, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    
    #ax_top.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = ax_top.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        ax_top.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        ax_top.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_MH2/%s_%s_%s_Mstar_H2_both_%s_H2ap%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_MH2/%s_%s_%s_Mstar_H2both_%s_H2ap%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns soap aperture - detection rate for 3 samples
def _aperture_fdet(sim_box_size_name_1 = 'L100_m6',
                   sim_type_name_1     = 'THERMAL_AGN_m6',
                   sim_box_size_name_2 = 'L100_m6',
                   sim_type_name_2     = 'HYBRID_AGN_m6',
                     snapshot_name     = '127',
                   title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                   aperture_h2_list = ['exclusive_sphere_3kpc', 'exclusive_sphere_10kpc', 'exclusive_sphere_30kpc', 'exclusive_sphere_50kpc'],        # list of apertures to consider
                   h2_detection_limit = 10**7,
                     plot_all = True,
                     plot_centrals = True,
                     plot_satellites = True,
                   #---------------
                   print_fdet         = True,
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
    fig = plt.figure(figsize=(10/3, 2.8))
    gs  = fig.add_gridspec(1, 2,  width_ratios=(1, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.2)
    # Create the Axes.
    ax_left = fig.add_subplot(gs[0])
    ax_rig = fig.add_subplot(gs[1])
    
               
    #---------------------------
    # Extract data from samples:
    plot_top = True
    if plot_top:
        #===========================================================
        # All ETGs
        if plot_all:
            soap_indicies_sample,  _, sample_input = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral'%(sim_box_size_name_1, sim_type_name_1, snapshot_name))  
            print(sample_input['name_of_preset'])
        
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
            box_size = data.metadata.boxsize[0]
    
    
            #-------------------------------
            # Get stelmass, molecular hydrogen, and magnitude data
            stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
            stellar_mass.convert_to_units('Msun')
            stellar_mass.convert_to_physical()
            stellar_mass = stellar_mass[soap_indicies_sample]
        
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - r_mag
            mag_plot = mag_plot[soap_indicies_sample]
    
            central_sat = attrgetter('input_halos.is_central')(data)
            central_sat = central_sat[soap_indicies_sample]
        
            kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
            kappa_stars = kappa_stars[soap_indicies_sample]
        
            #-----------------------------------
            # Loop over desired H2 apertures and add
            aperture_x_list     = []
            fdet_list_incl           = []
            fdet_list_excl           = []
            fdet_err_lower_list_incl = []
            fdet_err_lower_list_excl = []
            fdet_err_upper_list_incl = []
            fdet_err_upper_list_excl = []
            for aperture_h2_i in aperture_h2_list:
                H2_mass = attrgetter('%s.%s'%(aperture_h2_i, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
                H2_mass.convert_to_units('Msun')
                H2_mass.convert_to_physical()
            
            
                #==========================================================
                # Useful masks
                mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
                mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
                mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
                mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
            
                #----------------
                # Print H2 detection
                print('--------------------------------')
                print('Sample length:   %s     Excl. kappa>0.4:    %s   aperture: %s' %(len(stellar_mass), len(stellar_mass[mask_SRs]), aperture_h2_i))
                print('FRACTION OF SAMPLE ABOVE H2 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
                if print_fdet:
                    # Total sample
                    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
                        # Including FRs
                        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
                        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                        print('  >  H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
                        # Excluding FRs (ignore if LTG sample)
                        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                            mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                            f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                            print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
            

                #--------------
                # Append fractions
                x_dict = {'exclusive_sphere_3kpc': 3, 
                          'exclusive_sphere_10kpc': 10,
                          'exclusive_sphere_30kpc': 30, 
                          'exclusive_sphere_50kpc': 50}
                aperture_x_list.append(x_dict[aperture_h2_i])

                #--------------
                # Find detection rate of sample among all ETGs and Kappa<0.4 ETGs for a given H2_aperture
            
            
                # ETG including FRs
                f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_incl.append(f_detected)
                fdet_err_lower_list_incl.append(f_detected_err_lower)
                fdet_err_upper_list_incl.append(f_detected_err_upper)
            
            
                # ETG excluding FRs
                f_detected = len(stellar_mass[mask_h2_SRs])/len(stellar_mass[mask_SRs]) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs]), n=len(stellar_mass[mask_SRs]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_excl.append(f_detected)
                fdet_err_lower_list_excl.append(f_detected_err_lower)
                fdet_err_upper_list_excl.append(f_detected_err_upper)


            #------------
            # PLOT EACH LINE FOR EACH SAMPLE:
            # Plot several lines: ETG excl FR, ETG incl FR
            ax_left.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_excl), yerr=[100*np.array(fdet_err_lower_list_excl), 100*np.array(fdet_err_upper_list_excl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C0', 
                            ecolor='C0', 
                            ls='-', 
                            label= 'ETGs (excl. FRs)',
                            )
            ax_left.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_incl), yerr=[100*np.array(fdet_err_lower_list_incl), 100*np.array(fdet_err_upper_list_incl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C1', 
                            ecolor='C1', 
                            ls='-', 
                            )
        # Central ETGs
        if plot_centrals:
            # Central galaxies
            soap_indicies_sample,  _, sample_input = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_centrals'%(sim_box_size_name_1, sim_type_name_1, snapshot_name))  
            print(sample_input['name_of_preset'])
        
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
            box_size = data.metadata.boxsize[0]
    
    
            #-------------------------------
            # Get stelmass, molecular hydrogen, and magnitude data
            stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
            stellar_mass.convert_to_units('Msun')
            stellar_mass.convert_to_physical()
            stellar_mass = stellar_mass[soap_indicies_sample]
        
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - r_mag
            mag_plot = mag_plot[soap_indicies_sample]
    
            central_sat = attrgetter('input_halos.is_central')(data)
            central_sat = central_sat[soap_indicies_sample]
        
            kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
            kappa_stars = kappa_stars[soap_indicies_sample]
        
            #-----------------------------------
            # Loop over desired H2 apertures and add
            aperture_x_list     = []
            fdet_list_incl           = []
            fdet_list_excl           = []
            fdet_err_lower_list_incl = []
            fdet_err_lower_list_excl = []
            fdet_err_upper_list_incl = []
            fdet_err_upper_list_excl = []
            for aperture_h2_i in aperture_h2_list:
                H2_mass = attrgetter('%s.%s'%(aperture_h2_i, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
                H2_mass.convert_to_units('Msun')
                H2_mass.convert_to_physical()
            
            
                #==========================================================
                # Useful masks
                mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
                mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
                mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
                mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
            
                #----------------
                # Print H2 detection
                print('--------------------------------')
                print('Sample length:   %s     Excl. kappa>0.4:    %s   aperture: %s' %(len(stellar_mass), len(stellar_mass[mask_SRs]), aperture_h2_i))
                print('FRACTION OF SAMPLE ABOVE H2 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
                if print_fdet:
                    # Total sample
                    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
                        # Including FRs
                        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
                        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                        print('  >  H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
                        # Excluding FRs (ignore if LTG sample)
                        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                            mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                            f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                            print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
            

                #--------------
                # Append fractions
                x_dict = {'exclusive_sphere_3kpc': 3, 
                          'exclusive_sphere_10kpc': 10,
                          'exclusive_sphere_30kpc': 30, 
                          'exclusive_sphere_50kpc': 50}
                aperture_x_list.append(x_dict[aperture_h2_i])

                #--------------
                # Find detection rate of sample among all ETGs and Kappa<0.4 ETGs for a given H2_aperture
            
            
                # ETG including FRs
                f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_incl.append(f_detected)
                fdet_err_lower_list_incl.append(f_detected_err_lower)
                fdet_err_upper_list_incl.append(f_detected_err_upper)
            
            
                # ETG excluding FRs
                f_detected = len(stellar_mass[mask_h2_SRs])/len(stellar_mass[mask_SRs]) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs]), n=len(stellar_mass[mask_SRs]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_excl.append(f_detected)
                fdet_err_lower_list_excl.append(f_detected_err_lower)
                fdet_err_upper_list_excl.append(f_detected_err_upper)


            #------------
            # PLOT EACH LINE FOR EACH SAMPLE:
            # Plot several lines: ETG excl FR, ETG incl FR
            ax_left.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_excl), yerr=[100*np.array(fdet_err_lower_list_excl), 100*np.array(fdet_err_upper_list_excl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C0', 
                            ecolor='C0', 
                            ls='-.', 
                            )
            ax_left.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_incl), yerr=[100*np.array(fdet_err_lower_list_incl), 100*np.array(fdet_err_upper_list_incl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C1', 
                            ecolor='C1', 
                            ls='-.',
                            )
        # Satelliet ETGs
        if plot_satellites:
            #===========================================================
            # Satellite galaxies
            soap_indicies_sample,  _, sample_input = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_satellites'%(sim_box_size_name_1, sim_type_name_1, snapshot_name))  
            print(sample_input['name_of_preset'])
        
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
            box_size = data.metadata.boxsize[0]
    
    
            #-------------------------------
            # Get stelmass, molecular hydrogen, and magnitude data
            stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
            stellar_mass.convert_to_units('Msun')
            stellar_mass.convert_to_physical()
            stellar_mass = stellar_mass[soap_indicies_sample]
        
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - r_mag
            mag_plot = mag_plot[soap_indicies_sample]
    
            central_sat = attrgetter('input_halos.is_central')(data)
            central_sat = central_sat[soap_indicies_sample]
        
            kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
            kappa_stars = kappa_stars[soap_indicies_sample]
        
            #-----------------------------------
            # Loop over desired H2 apertures and add
            aperture_x_list     = []
            fdet_list_incl           = []
            fdet_list_excl           = []
            fdet_err_lower_list_incl = []
            fdet_err_lower_list_excl = []
            fdet_err_upper_list_incl = []
            fdet_err_upper_list_excl = []
            for aperture_h2_i in aperture_h2_list:
                H2_mass = attrgetter('%s.%s'%(aperture_h2_i, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
                H2_mass.convert_to_units('Msun')
                H2_mass.convert_to_physical()
            
            
                #==========================================================
                # Useful masks
                mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
                mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
                mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
                mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
            
                #----------------
                # Print H2 detection
                print('--------------------------------')
                print('Sample length:   %s     Excl. kappa>0.4:    %s   aperture: %s' %(len(stellar_mass), len(stellar_mass[mask_SRs]), aperture_h2_i))
                print('FRACTION OF SAMPLE ABOVE H2 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
                if print_fdet:
                    # Total sample
                    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
                        # Including FRs
                        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
                        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                        print('  >  H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
                        # Excluding FRs (ignore if LTG sample)
                        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                            mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                            f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                            print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
            

                #--------------
                # Append fractions
                x_dict = {'exclusive_sphere_3kpc': 3, 
                          'exclusive_sphere_10kpc': 10,
                          'exclusive_sphere_30kpc': 30, 
                          'exclusive_sphere_50kpc': 50}
                aperture_x_list.append(x_dict[aperture_h2_i])

                #--------------
                # Find detection rate of sample among all ETGs and Kappa<0.4 ETGs for a given H2_aperture
            
            
                # ETG including FRs
                f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_incl.append(f_detected)
                fdet_err_lower_list_incl.append(f_detected_err_lower)
                fdet_err_upper_list_incl.append(f_detected_err_upper)
            
            
                # ETG excluding FRs
                f_detected = len(stellar_mass[mask_h2_SRs])/len(stellar_mass[mask_SRs]) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs]), n=len(stellar_mass[mask_SRs]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_excl.append(f_detected)
                fdet_err_lower_list_excl.append(f_detected_err_lower)
                fdet_err_upper_list_excl.append(f_detected_err_upper)


            #------------
            # PLOT EACH LINE FOR EACH SAMPLE:
            # Plot several lines: ETG excl FR, ETG incl FR
            ax_left.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_excl), yerr=[100*np.array(fdet_err_lower_list_excl), 100*np.array(fdet_err_upper_list_excl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C0', 
                            ecolor='C0', 
                            ls=(0, (1, 1)), 
                            )
            ax_left.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_incl), yerr=[100*np.array(fdet_err_lower_list_incl), 100*np.array(fdet_err_upper_list_incl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C1', 
                            ecolor='C1', 
                            ls=(0, (1, 1)), 
                            )
             
    plot_bottom = True
    if plot_bottom:
        #===========================================================
        # All ETGs
        if plot_all:
            soap_indicies_sample,  _, sample_input = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral'%(sim_box_size_name_2, sim_type_name_2, snapshot_name))  
            print(sample_input['name_of_preset'])
        
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
            box_size = data.metadata.boxsize[0]
    
    
            #-------------------------------
            # Get stelmass, molecular hydrogen, and magnitude data
            stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
            stellar_mass.convert_to_units('Msun')
            stellar_mass.convert_to_physical()
            stellar_mass = stellar_mass[soap_indicies_sample]
        
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - r_mag
            mag_plot = mag_plot[soap_indicies_sample]
    
            central_sat = attrgetter('input_halos.is_central')(data)
            central_sat = central_sat[soap_indicies_sample]
        
            kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
            kappa_stars = kappa_stars[soap_indicies_sample]
        
            #-----------------------------------
            # Loop over desired H2 apertures and add
            aperture_x_list     = []
            fdet_list_incl           = []
            fdet_list_excl           = []
            fdet_err_lower_list_incl = []
            fdet_err_lower_list_excl = []
            fdet_err_upper_list_incl = []
            fdet_err_upper_list_excl = []
            for aperture_h2_i in aperture_h2_list:
                H2_mass = attrgetter('%s.%s'%(aperture_h2_i, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
                H2_mass.convert_to_units('Msun')
                H2_mass.convert_to_physical()
            
            
                #==========================================================
                # Useful masks
                mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
                mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
                mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
                mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
            
                #----------------
                # Print H2 detection
                print('--------------------------------')
                print('Sample length:   %s     Excl. kappa>0.4:    %s   aperture: %s' %(len(stellar_mass), len(stellar_mass[mask_SRs]), aperture_h2_i))
                print('FRACTION OF SAMPLE ABOVE H2 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
                if print_fdet:
                    # Total sample
                    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
                        # Including FRs
                        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
                        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                        print('  >  H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
                        # Excluding FRs (ignore if LTG sample)
                        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                            mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                            f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                            print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
            

                #--------------
                # Append fractions
                x_dict = {'exclusive_sphere_3kpc': 3, 
                          'exclusive_sphere_10kpc': 10,
                          'exclusive_sphere_30kpc': 30, 
                          'exclusive_sphere_50kpc': 50}
                aperture_x_list.append(x_dict[aperture_h2_i])

                #--------------
                # Find detection rate of sample among all ETGs and Kappa<0.4 ETGs for a given H2_aperture
            
            
                # ETG including FRs
                f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_incl.append(f_detected)
                fdet_err_lower_list_incl.append(f_detected_err_lower)
                fdet_err_upper_list_incl.append(f_detected_err_upper)
            
            
                # ETG excluding FRs
                f_detected = len(stellar_mass[mask_h2_SRs])/len(stellar_mass[mask_SRs]) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs]), n=len(stellar_mass[mask_SRs]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_excl.append(f_detected)
                fdet_err_lower_list_excl.append(f_detected_err_lower)
                fdet_err_upper_list_excl.append(f_detected_err_upper)


            #------------
            # PLOT EACH LINE FOR EACH SAMPLE:
            # Plot several lines: ETG excl FR, ETG incl FR
            ax_rig.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_excl), yerr=[100*np.array(fdet_err_lower_list_excl), 100*np.array(fdet_err_upper_list_excl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C0', 
                            ecolor='C0', 
                            ls='-', 
                            label= 'ETGs (excl. FRs)',
                            )
            ax_rig.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_incl), yerr=[100*np.array(fdet_err_lower_list_incl), 100*np.array(fdet_err_upper_list_incl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C1', 
                            ecolor='C1', 
                            ls='-', 
                            label= 'ETGs (incl. FRs)',
                            )
        # Central ETGs
        if plot_centrals:
            # Central galaxies
            soap_indicies_sample,  _, sample_input = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_centrals'%(sim_box_size_name_2, sim_type_name_2, snapshot_name))  
            print(sample_input['name_of_preset'])
        
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
            box_size = data.metadata.boxsize[0]
    
    
            #-------------------------------
            # Get stelmass, molecular hydrogen, and magnitude data
            stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
            stellar_mass.convert_to_units('Msun')
            stellar_mass.convert_to_physical()
            stellar_mass = stellar_mass[soap_indicies_sample]
        
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - r_mag
            mag_plot = mag_plot[soap_indicies_sample]
    
            central_sat = attrgetter('input_halos.is_central')(data)
            central_sat = central_sat[soap_indicies_sample]
        
            kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
            kappa_stars = kappa_stars[soap_indicies_sample]
        
            #-----------------------------------
            # Loop over desired H2 apertures and add
            aperture_x_list     = []
            fdet_list_incl           = []
            fdet_list_excl           = []
            fdet_err_lower_list_incl = []
            fdet_err_lower_list_excl = []
            fdet_err_upper_list_incl = []
            fdet_err_upper_list_excl = []
            for aperture_h2_i in aperture_h2_list:
                H2_mass = attrgetter('%s.%s'%(aperture_h2_i, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
                H2_mass.convert_to_units('Msun')
                H2_mass.convert_to_physical()
            
            
                #==========================================================
                # Useful masks
                mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
                mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
                mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
                mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
            
                #----------------
                # Print H2 detection
                print('--------------------------------')
                print('Sample length:   %s     Excl. kappa>0.4:    %s   aperture: %s' %(len(stellar_mass), len(stellar_mass[mask_SRs]), aperture_h2_i))
                print('FRACTION OF SAMPLE ABOVE H2 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
                if print_fdet:
                    # Total sample
                    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
                        # Including FRs
                        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
                        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                        print('  >  H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
                        # Excluding FRs (ignore if LTG sample)
                        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                            mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                            f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                            print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
            

                #--------------
                # Append fractions
                x_dict = {'exclusive_sphere_3kpc': 3, 
                          'exclusive_sphere_10kpc': 10,
                          'exclusive_sphere_30kpc': 30, 
                          'exclusive_sphere_50kpc': 50}
                aperture_x_list.append(x_dict[aperture_h2_i])

                #--------------
                # Find detection rate of sample among all ETGs and Kappa<0.4 ETGs for a given H2_aperture
            
            
                # ETG including FRs
                f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_incl.append(f_detected)
                fdet_err_lower_list_incl.append(f_detected_err_lower)
                fdet_err_upper_list_incl.append(f_detected_err_upper)
            
            
                # ETG excluding FRs
                f_detected = len(stellar_mass[mask_h2_SRs])/len(stellar_mass[mask_SRs]) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs]), n=len(stellar_mass[mask_SRs]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_excl.append(f_detected)
                fdet_err_lower_list_excl.append(f_detected_err_lower)
                fdet_err_upper_list_excl.append(f_detected_err_upper)


            #------------
            # PLOT EACH LINE FOR EACH SAMPLE:
            # Plot several lines: ETG excl FR, ETG incl FR
            ax_rig.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_excl), yerr=[100*np.array(fdet_err_lower_list_excl), 100*np.array(fdet_err_upper_list_excl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C0', 
                            ecolor='C0', 
                            ls='-.', 
                            )
            ax_rig.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_incl), yerr=[100*np.array(fdet_err_lower_list_incl), 100*np.array(fdet_err_upper_list_incl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C1', 
                            ecolor='C1', 
                            ls='-.',
                            )
        # Satelliet ETGs
        if plot_satellites:
            #===========================================================
            # Satellite galaxies
            soap_indicies_sample,  _, sample_input = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_satellites'%(sim_box_size_name_2, sim_type_name_2, snapshot_name))  
            print(sample_input['name_of_preset'])
        
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
            box_size = data.metadata.boxsize[0]
    
    
            #-------------------------------
            # Get stelmass, molecular hydrogen, and magnitude data
            stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
            stellar_mass.convert_to_units('Msun')
            stellar_mass.convert_to_physical()
            stellar_mass = stellar_mass[soap_indicies_sample]
        
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - r_mag
            mag_plot = mag_plot[soap_indicies_sample]
    
            central_sat = attrgetter('input_halos.is_central')(data)
            central_sat = central_sat[soap_indicies_sample]
        
            kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
            kappa_stars = kappa_stars[soap_indicies_sample]
        
            #-----------------------------------
            # Loop over desired H2 apertures and add
            aperture_x_list     = []
            fdet_list_incl           = []
            fdet_list_excl           = []
            fdet_err_lower_list_incl = []
            fdet_err_lower_list_excl = []
            fdet_err_upper_list_incl = []
            fdet_err_upper_list_excl = []
            for aperture_h2_i in aperture_h2_list:
                H2_mass = attrgetter('%s.%s'%(aperture_h2_i, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
                H2_mass.convert_to_units('Msun')
                H2_mass.convert_to_physical()
            
            
                #==========================================================
                # Useful masks
                mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
                mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
                mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
                mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
            
                #----------------
                # Print H2 detection
                print('--------------------------------')
                print('Sample length:   %s     Excl. kappa>0.4:    %s   aperture: %s' %(len(stellar_mass), len(stellar_mass[mask_SRs]), aperture_h2_i))
                print('FRACTION OF SAMPLE ABOVE H2 MASS:      %s  %s  %s' %(sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset']))
                if print_fdet:
                    # Total sample
                    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
                        # Including FRs
                        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
                        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                        print('  >  H2 > %.1e:      %.3f (-%.3f + %.3f),  count: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
                        # Excluding FRs (ignore if LTG sample)
                        if sample_input['name_of_preset'] not in ['all_LTGs', 'all_LTGs_excl_redspiral']:
                            mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_SRs_i    = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
                            mask_h2_SRs_i = np.logical_and.reduce([mask_h2_i, mask_SRs_i])
                            f_detected_i = len(stellar_mass[mask_h2_SRs_i])/len(stellar_mass[mask_SRs_i]) 
                            f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs_i]), n=len(stellar_mass[mask_SRs_i]), confidence_level= 0.68269, interval='jeffreys')
                            f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
                            f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
                            print('     only kappa<0.4:  %.3f (-%.3f + %.3f), count: %s / %s'%(f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_SRs_i]), len(stellar_mass[mask_SRs_i])))
            

                #--------------
                # Append fractions
                x_dict = {'exclusive_sphere_3kpc': 3, 
                          'exclusive_sphere_10kpc': 10,
                          'exclusive_sphere_30kpc': 30, 
                          'exclusive_sphere_50kpc': 50}
                aperture_x_list.append(x_dict[aperture_h2_i])

                #--------------
                # Find detection rate of sample among all ETGs and Kappa<0.4 ETGs for a given H2_aperture
            
            
                # ETG including FRs
                f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_incl.append(f_detected)
                fdet_err_lower_list_incl.append(f_detected_err_lower)
                fdet_err_upper_list_incl.append(f_detected_err_upper)
            
            
                # ETG excluding FRs
                f_detected = len(stellar_mass[mask_h2_SRs])/len(stellar_mass[mask_SRs]) 
                f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2_SRs]), n=len(stellar_mass[mask_SRs]), confidence_level= 0.68269, interval='jeffreys')
                f_detected_err_lower = f_detected - f_detected_err[0]
                f_detected_err_upper = f_detected_err[1] - f_detected
            
                fdet_list_excl.append(f_detected)
                fdet_err_lower_list_excl.append(f_detected_err_lower)
                fdet_err_upper_list_excl.append(f_detected_err_upper)


            #------------
            # PLOT EACH LINE FOR EACH SAMPLE:
            # Plot several lines: ETG excl FR, ETG incl FR
            ax_rig.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_excl), yerr=[100*np.array(fdet_err_lower_list_excl), 100*np.array(fdet_err_upper_list_excl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C0', 
                            ecolor='C0', 
                            ls=(0, (1, 1)), 
                            )
            ax_rig.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list_incl), yerr=[100*np.array(fdet_err_lower_list_incl), 100*np.array(fdet_err_upper_list_incl)], xerr=None, capsize=2, lw=0.9, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, path_effects=[outline2],
                            color='C1', 
                            ecolor='C1', 
                            ls=(0, (1, 1)), 
                            )
                            
        

    #-----------
    # Axis formatting
    ax_left.set_xlim(0, 55)
    ax_left.set_ylim(0, 100)
    ax_left.minorticks_on()
    #ax_left.set_yticks([20, 40, 60, 80])
    ax_left.set_ylabel('Percentage of sub-sample with\n' + r'log$_{10}$ $M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot} > 7$ [\%]')
    #ax_left.set_ylabel('testtest', color='w')
    ax_left.set_xlabel(r'SOAP aperture radius [pkpc]', x=1)
    

    ax_rig.set_xlim(0, 55)
    ax_rig.set_ylim(0, 100)
    #ax_rig.set_yticks([20, 40, 60, 80])
    ax_rig.minorticks_on()
    ax_rig.set_yticklabels([])
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    #ax_rig.set_xlabel(r'SOAP aperture radius [pkpc]')
    #ax_rig.set_ylabel('testtest', color='w')
    #fig.supylabel('Percentage of sub-sample with\n' + r'$M_{\mathrm{H_{2}}} > 10^{7}$ M$_{\odot}$) [\%]', fontsize=8)
    
    #-----------
    # Title
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<%s><..><%s>'%(run_name_title, title_text_in)
    """fig_text(x=0.195, y=0.935, ha='left', s=text_title, fontsize=7, ax=axs,
        highlight_textprops=[
            {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
            {"color": "white"},
            {"color": "black"}
        ])"""
    ax_left.set_title(r'L200m6%s' %(title_text_in), size=7, x=0.17, y=1.02, pad=3, c=title_color_dict['L200m6'], bbox={"edgecolor": title_color_dict['L200m6'], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})
    ax_rig.set_title(r'L100m6h%s' %(title_text_in), size=7, x=0.19, y=1.02, pad=3, c=title_color_dict['L100m6h'], bbox={"edgecolor": title_color_dict['L100m6h'], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})
    
    
    #-----------
    # Legend
    ax_rig.plot([-1, -2], [-1, -2], ls='-', linewidth=1, c='k', zorder=-3, path_effects=[outline], label='all ETGs')
    ax_rig.plot([-1, -2], [-1, -2], ls='-.', linewidth=1, c='k', zorder=-3, path_effects=[outline], label='centrals')
    ax_rig.plot([-1, -2], [-1, -2], ls=(0, (1, 1)), linewidth=1, c='k', zorder=-3, path_effects=[outline], label='satellites')
    
    handles, labels = ax_rig.get_legend_handles_labels()
    handles = [handles[3], handles[4], handles[0], handles[1], handles[2]]
    labels = [labels[3], labels[4], labels[0], labels[1], labels[2]]
    ax_rig.legend(handles, labels, ncol=1, loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/aperture_fdet/LXXX_both_%s_fdet_h2aperture_%s.%s" %(fig_dir, sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/aperture_fdet/LXXX_both_%s_yfdet_h2aperture_%s.%s" %(fig_dir, sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns m200c - h2mass
def _etg_m200c_h2mass(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   print_typical_galaxy = True,       # median galaxy properties of detected ETG
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    # Extra / Calculated values
    m200c_ref = (data.spherical_overdensity_200_crit.total_mass)
    m200c_ref.convert_to_units('Msun')
    m200c_ref.convert_to_physical()
    
    HostFOFId_ref   = (data.input_halos_hbtplus.host_fofid)
    central_sat_ref = attrgetter('input_halos.is_central')(data)
    
    # Create dataframe with m200c as an empty array of zeros
    df = pd.DataFrame(data={'m200c_ref': m200c_ref,
                            'IsCentral': central_sat_ref,
                            'HostFOFId': HostFOFId_ref,
                            'm200c': m200c_ref
                        })
    
    # Build lookup, but EXCLUDE centrals with HostFOFId == -1
    central_lookup = (
        df.loc[(df["IsCentral"] == 1) & (df["HostFOFId"] != -1), ["HostFOFId", "m200c"]]
        .drop_duplicates("HostFOFId")   # just in case
        .set_index("HostFOFId")["m200c"]
        )

    # Fill satellites: map only if HostFOFId != -1, else keep their current m200c
    df["m200c"] = df.apply(
        lambda row: central_lookup.get(row["HostFOFId"], row["m200c"]),
        axis=1
    )
    m200c = df['m200c'][soap_indicies_sample]
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
    
    
    #==========================================================
    # Spearman with gas mass
    res = scipy.stats.spearmanr(np.log10(m200c[mask_h2]), np.log10(H2_mass[mask_h2])) 
    print('\nSpearman incl. FR:    M200c - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(np.log10(m200c[mask_h2_SRs]), np.log10(H2_mass[mask_h2_SRs])) 
    print('Spearman excl. FR:    M200c - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(m200c)
        Y_VALUE_1 = np.log10(H2_mass)
        C_VALUE_1 = kappa_stars
        S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
        cb = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H2_mass)
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)
        
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
        extent = (10.5, 15.5, 6, 11)                     ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H2':  
        ### Plots hexbin showing median log10 H2_mass
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (10.5, 15.5, 6, 11)                     ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(10.5, 16.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(m200c[mask_SRs])
        X_MEDIAN_2 = np.log10(m200c)
        Y_MEDIAN_1 = H2_mass[mask_SRs]
        Y_MEDIAN_2 = H2_mass
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
    #-----------------
    # Add observations
    if add_observational:
        """
        Pick observations we want to add
        """
        print('no obs available')
            
        
        
    #-----------
    # Axis formatting
    axs.set_xlim(10.5, 15.5)
    axs.set_ylim(6, 11)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
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
    axs.set_xlabel(r'log$_{10}$ $M_{\mathrm{200c}}$ [M$_{\odot}$]')
    axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ [M$_{\odot}$]')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H2':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$]', extend='min')
      
    
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3])]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/m200c_MH2/%s_%s_%s_m200c_MH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/m200c_MH2/%s_%s_%s_m200c_MH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - m200c
def _etg_stelmass_m200c(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    m200c_ref = (data.spherical_overdensity_200_crit.total_mass)
    m200c_ref.convert_to_units('Msun')
    m200c_ref.convert_to_physical()
    
    HostFOFId_ref   = (data.input_halos_hbtplus.host_fofid)
    central_sat_ref = attrgetter('input_halos.is_central')(data)
    
    # Create dataframe with m200c as an empty array of zeros
    df = pd.DataFrame(data={'m200c_ref': m200c_ref,
                            'IsCentral': central_sat_ref,
                            'HostFOFId': HostFOFId_ref,
                            'm200c': m200c_ref
                        })
    
    # Build lookup, but EXCLUDE centrals with HostFOFId == -1
    central_lookup = (
        df.loc[(df["IsCentral"] == 1) & (df["HostFOFId"] != -1), ["HostFOFId", "m200c"]]
        .drop_duplicates("HostFOFId")   # just in case
        .set_index("HostFOFId")["m200c"]
        )

    # Fill satellites: map only if HostFOFId != -1, else keep their current m200c
    df["m200c"] = df.apply(
        lambda row: central_lookup.get(row["HostFOFId"], row["m200c"]),
        axis=1
    )
    m200c = df['m200c'][soap_indicies_sample]
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(np.log10(m200c[mask_h2]), np.log10(H2_mass[mask_h2])) 
    print('\nSpearman incl. FR:    m200c - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(np.log10(m200c[mask_h2_SRs]), np.log10(H2_mass[mask_h2_SRs])) 
    print('Spearman excl. FR:    m200c - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(m200c)
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, 10.5, 15.5)                     ###
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
        extent = (9.5, 12.5, 10.5, 15.5)                     ###
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = np.log10(m200c[mask_SRs])
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = np.log10(m200c)
        
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
            #y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
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
        """medians_log = np.log10(medians)
        medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))"""
        medians_log = medians
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        """medians_log = np.log10(medians)
        medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))"""
        medians_log = medians
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        bin_width = 0.25
        hist_bins = np.arange(10, 16, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(10.5, 15.5)
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
        axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{200c}}$ [M$_{\odot}$]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $M_{\mathrm{200c}}$ [M$_{\odot}$]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(10.5, 15.5)
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
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_m200c/%s_%s_%s_Mstar_m200c_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_m200c/%s_%s_%s_Mstar_m200c_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - u-r
def _etg_stelmass_u_r(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = False,
                   add_detection_hist = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(mag_plot[mask_h2], np.log10(H2_mass[mask_h2])) 
    print('\nSpearman incl. FR:    u-r - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(mag_plot[mask_h2_SRs], np.log10(H2_mass[mask_h2_SRs])) 
    print('Spearman excl. FR:    u-r - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = mag_plot
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
        
    
    
    
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
        extent = (9.5, 12.5, 0.5, 3.5)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        extent = (9.5, 12.5, 0.5, 3.5)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = mag_plot[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = mag_plot
        
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
            #y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
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
        """medians_log = np.log10(medians)
        medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))"""
        medians_log = medians
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        """medians_log = np.log10(medians)
        medians_masked = np.ma.masked_where(bins_n < 10, np.log10(medians))
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(lower_1sigma))
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, np.log10(upper_1sigma))"""
        medians_log = medians
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        bin_width = 0.1
        hist_bins = np.arange(0.5, 3.2, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(0.8, 3.3)
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
        axs.set_ylabel(r'$u^{*} - r^{*}$')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(0.8, 3.3)
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
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=False)
    
    
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_u_r/%s_%s_%s_Mstar_u-r_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_u_r/%s_%s_%s_Mstar_u-r_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - SFR
def _etg_stelmass_sfr(soap_indicies_sample=[], sample_input=[], title_text_in = '',
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    sfr50 = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)[soap_indicies_sample]
    sfr50.convert_to_units(u.Msun/u.yr)
    sfr50.convert_to_physical()
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(sfr50[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    SFR - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(sfr50[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    SFR - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(sfr50)
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, -4, 3)                     ###
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
        extent = (9.5, 12.5, -4, 3)                     ###
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = sfr50[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = sfr50
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        bin_width = 0.25
        hist_bins = np.arange(-4.2, 4, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-4, 3)
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
        axs.set_ylabel(r'log$_{10}$ SFR [M$_{\odot}$ yr$^{-1}$]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ SFR [M$_{\odot}$ yr$^{-1}$]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-4, 3)
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
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
        
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_SFR/%s_%s_%s_Mstar_SFR_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_SFR/%s_%s_%s_Mstar_SFR_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - sSFR
def _etg_stelmass_ssfr(soap_indicies_sample=[], sample_input=[], title_text_in = '',
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    sfr50 = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)[soap_indicies_sample]
    sfr50.convert_to_units(u.Msun/u.yr)
    sfr50.convert_to_physical()
    
    ssfr50 = np.divide(sfr50, stellar_mass)
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(ssfr50[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    sSFR - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(ssfr50[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    sSFR - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(ssfr50)
        
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, -16, -6)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        extent = (9.5, 12.5, -16, -6)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = ssfr50[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = ssfr50
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        bin_width = 0.25
        hist_bins = np.arange(-16, 7, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-15, -7)
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
        axs.set_ylabel(r'log$_{10}$ sSFR [yr$^{-1}$]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ sSFR [yr$^{-1}$]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-15, -7)
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
    #axs.axhspan(-11, -17, color='grey', alpha=0.3, zorder=-30, edgecolor=None)
    axs.axhline(-11, color='k', zorder=40, ls='--', lw=1)
    ax_hist.axhline(-11, color='k', zorder=40, ls='--', lw=1)
    axs.text(12.45, -11.05, 'quiescent', color='k', zorder=40, fontsize=7, horizontalalignment='right', verticalalignment='top')
    axs.text(12.45, -10.95, 'star-forming', color='k', zorder=40, fontsize=7, horizontalalignment='right', verticalalignment='bottom')
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
        
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_sSFR/%s_%s_%s_Mstar_sSFR_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_sSFR/%s_%s_%s_Mstar_sSFR_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - t_dep = H2_mass / SFR
def _etg_stelmass_tdep(soap_indicies_sample=[], sample_input=[], title_text_in = '',
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    sfr50 = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)[soap_indicies_sample]
    sfr50.convert_to_units(u.Msun/u.yr)
    sfr50.convert_to_physical()
    
    #tdep50 = np.zeros(len(stellar_mass))    
    #tdep50[H2_mass.value > 0] = np.divide(H2_mass[H2_mass.value > 0], sfr50[H2_mass.value > 0])
    # array of 0 0 0, tdep values, and some -inf f0r sfr = 0
    tdep50 = np.divide(H2_mass, sfr50)
    tdep50.convert_to_units(u.Gyr)
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(tdep50[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    t_dep - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(tdep50[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    t_dep - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    print('\nMax tdep in sample:  %.3f Gyr' %np.max(tdep50[np.isfinite(tdep50)]))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(tdep50)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, -4, 4)                     ###
        gridsize = (25,23)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        extent = (9.5, 12.5, -4, 4)                     ###
        gridsize = (25,23)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = tdep50[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = tdep50
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        bin_width = 0.2
        hist_bins = np.arange(-4, 4, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-2, 3)
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
        axs.set_ylabel(r'log$_{10}$ $\tau_{\mathrm{dep}}$ [Gyr]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $\tau_{\mathrm{dep}}$ [Gyr]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-2, 3)
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
    #axs.axhspan(np.log10(13.8), -4, color='grey', alpha=0.3, zorder=-30)
    axs.axhline(np.log10(13.8), color='k', zorder=40, ls='--', lw=1)
    ax_hist.axhline(np.log10(13.8), color='k', zorder=40, ls='--', lw=1)
    axs.text(12.45, np.log10(13.8)+0.02, '13.8 Gyr', color='k', zorder=40, fontsize=7, horizontalalignment='right', verticalalignment='bottom')
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        if 'LTG' not in sample_input['name_of_preset']:
            axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
            handles, labels = axs.get_legend_handles_labels()
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_tdep/%s_%s_%s_Mstar_tdep_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_tdep/%s_%s_%s_Mstar_tdep_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - r50
def _etg_stelmass_r50(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    r50 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    r50 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(r50)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, -1, 3)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        extent = (9.5, 12.5, -1, 3)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = r50[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = r50
        
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
            #y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    #-----------------
    # Add observations
    if add_observational:
        """
        Pick observations we want to add
        """
        add_Young2011 = True
        
        if add_Young2011:      # ($z=0.0$)
            # Load the observational data
            with h5py.File('%s/Davis2019_ATLAS3D.hdf5'%obs_dir, 'r') as file:
                obs_names_1 = file['data/Galaxy/values'][:]
                obs_H2_1       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                obs_Mstar_1    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                obs_Re         = file['data/Re/values'][:] * u.Unit(file['data/Re/values'].attrs['units'])
                obs_mask_1     = file['data/det_mask/values'][:]
                obs_isvirgo_1  = file['data/Virgo/values'][:]
                obs_iscentral_1  = file['data/BCG/values'][:]
            
            obs_names_1 = np.array(obs_names_1)
            obs_mask_1  = np.array(obs_mask_1, dtype=bool)
            obs_isvirgo_1  = np.array(obs_isvirgo_1, dtype=bool)
            obs_iscentral_1  = np.array(obs_iscentral_1, dtype=bool)
            obs_H2_1.to('Msun')
            obs_Mstar_1.to('Msun')
            obs_Re.to('kpc')
            
            #--------------------------------------------
            
            if 'central' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[obs_iscentral_1]
                obs_H2_1    = obs_H2_1[obs_iscentral_1]
                obs_Re      = obs_Re[obs_iscentral_1]
                obs_mask_1  = obs_mask_1[obs_iscentral_1]
            if 'satellite' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[~obs_iscentral_1]
                obs_H2_1    = obs_H2_1[~obs_iscentral_1]
                obs_Re      = obs_Re[~obs_iscentral_1]
                obs_mask_1  = obs_mask_1[~obs_iscentral_1]
            if 'cluster' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[obs_isvirgo_1]
                obs_H2_1    = obs_H2_1[obs_isvirgo_1]
                obs_Re      = obs_Re[obs_isvirgo_1]
                obs_mask_1  = obs_mask_1[obs_isvirgo_1]
            if 'group' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[~obs_isvirgo_1]
                obs_H2_1    = obs_H2_1[~obs_isvirgo_1]
                obs_Re      = obs_Re[~obs_isvirgo_1]
                obs_mask_1  = obs_mask_1[~obs_isvirgo_1]

            print('Sample length Young+11 ATLAS3D (detected):   %s'%len(obs_Mstar_1[obs_mask_1]))
            print('Sample length Cappellari+11 ATLAS3D (non-det):    %s'%len(obs_Mstar_1[~obs_mask_1]))
            
            
            obs_x = np.array(obs_Mstar_1)
            obs_y = np.array(obs_Re)
            axs.scatter(obs_x[obs_mask_1], np.log10(obs_y[obs_mask_1]), marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Young+11 (det.)')
            axs.scatter(obs_x[~obs_mask_1], np.log10(obs_y[~obs_mask_1]), marker='^', s=6, alpha=1, zorder=5, facecolors='none', linewidths=0.3,  edgecolors='r', label='Cappellari+11 (non-det.)')
            #axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
        
            
    #-----------------
    # Add detection hist for excl. FR and incl. FR
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.2
        hist_bins = np.arange(-1, 3, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        #==========================================
        # Obs sample
        if add_observational:
            # obs_x     = log10 M*
            # obs_y     = Re
            # obs_mask_1 = mask for detected systems
            
            bin_n, _           = np.histogram(np.log10(obs_y), bins=hist_bins)
            bin_n_detected, _  = np.histogram(np.log10(obs_y)[obs_mask_1], bins=hist_bins)
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
            ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='none', linewidth=0, alpha=0.4, label='Young+11\n(det.)') 
            
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='r', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
  
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-0.5, 2)
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
        axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50}}$ [pkpc]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $r_{\mathrm{50}}$ [pkpc]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-0.5, 2)
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
    # add resolution limits
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    if add_observational:
        handles_in.append(handles[-2])
        labels_in.append(labels[-2])
        handles_in.append(handles[-1])
        labels_in.append(labels[-1])
    axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
    if add_detection_hist:
        if add_observational:
            ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
            
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_r50/%s_%s_%s_Mstar_r50_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_r50/%s_%s_%s_Mstar_r50_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - r50 HI, detected only
def _etg_stelmass_r50H1(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h1 = 'exclusive_sphere_50kpc', 
                   h1_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H1',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H1 ]
                     add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   add_detection_hist = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()

    H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
    H1_mass.convert_to_units('Msun')
    H1_mass.convert_to_physical()

    u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
    r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
    u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    mag_plot = u_mag - r_mag

    central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

    kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
    
    
    #================================
    # Calculated values
    r50_h1 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_atomic_hydrogen'))(data)[soap_indicies_sample]
    r50_h1.convert_to_units('kpc')
    r50_h1.convert_to_physical()
    
    
    #==========================================================
    # Useful masks
    mask_h1      = H1_mass > cosmo_quantity(h1_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h1_SRs  = np.logical_and.reduce([mask_h1, mask_SRs])       # detection kappa < 0.4
    mask_h1_FRs  = np.logical_and.reduce([mask_h1, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h1, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h1, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50_h1[mask_h1], H1_mass[mask_h1]) 
    print('\nSpearman incl. FR:    r50_h1 - h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50_h1[mask_h1_SRs], H1_mass[mask_h1_SRs])
    print('Spearman excl. FR:    r50_h1 - h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        
        Y_VALUE_1 = np.log10(r50_h1)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H1_mass.value > 0] = np.log10(H1_mass[H1_mass.value > 0])
        #S_VALUE_1 = (np.log10(H1_mass)-(np.log10(h1_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H1
        cb = axs.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c='r', s=S_VALUE_1[mask_h1], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
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
        sc_points = axs.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c=C_VALUE_1[mask_h1], s=4, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], c=C_VALUE_1[~mask_h1], s=4, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0.2, edgecolor='k')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H1_mass)
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=10)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # Plot detections as filled viridis circles and squares
        axs.scatter(X_VALUE_1[mask_h1_SRs], Y_VALUE_1[mask_h1_SRs], c=C_VALUE_1[mask_h1_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
        cb = axs.scatter(X_VALUE_1[mask_h1_FRs], Y_VALUE_1[mask_h1_FRs], c=C_VALUE_1[mask_h1_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
        # Plot non-detections as empty grey circles and squares
        axs.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
        axs.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
    if scatter_or_hexbin == 'hexbin_count':
        ### Plots hexbin showing number of galaxies in bin
        
        cmap = cmasher.jungle_r
        newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
        # hexbin for all values
        extent = (9.5, 12.5, -1, 3)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H1':  
        ### Plots hexbin showing median log10 H1_mass
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (9.5, 12.5, -1, 3)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], C=C_VALUE_1[mask_h1], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], C=C_VALUE_1[mask_h1], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of all ETGs for excl. FR and incl. FR
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = r50_h1[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = r50_h1
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN_1 >= hist_bins[i]) & (X_MEDIAN_1 < hist_bins[i + 1])
            y_bin = Y_MEDIAN_1.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask_SRs][mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN_2 >= hist_bins[i]) & (X_MEDIAN_2 < hist_bins[i + 1])
            y_bin = Y_MEDIAN_2.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        bin_width = 0.2
        hist_bins = np.arange(-1, 3, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h1_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
        #==========================================
        ### ETG (incl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1, bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h1], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-0.5, 2)
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    axs.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h1 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    if not add_detection_hist:
        axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{I}}}$ [pkpc]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{I}}}$ [pkpc]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-0.5, 2)
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{I}}>10^{7}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_kappa':
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H1':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$]', extend='min')
      
    
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
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
    if add_detection_hist:
        if add_observational:
            ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_r50H1/%s_%s_%s_Mstar_r50HI_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_r50H1/%s_%s_%s_Mstar_r50HI_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - r50 H2, detected only
def _etg_stelmass_r50H2(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   add_detection_hist = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    r50_h2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_molecular_hydrogen'))(data)[soap_indicies_sample]
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
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50_h2[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    r50_h2 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50_h2[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    r50_h2 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        
        Y_VALUE_1 = np.log10(r50_h2)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, -1, 3)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
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
        extent = (9.5, 12.5, -1, 3)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of all ETGs for excl. FR and incl. FR
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = r50_h2[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = r50_h2
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
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
            for name_i in obs_names_2:
                if name_i in obs_names_1:
                    mask_name_1 = np.where(name_i == obs_names_1)[0]
                    mask_name_2 = np.where(name_i == obs_names_2)[0]
                    obs_x.append(obs_Mstar_1[mask_name_1])
                    obs_y.append(obs_Re_CO[mask_name_2])
                else:
                    if 'central' not in sample_input['name_of_preset']:
                        print('cannot find %s in 260 ATLAS3D sample'%name_i)
                        if name_i == b'NGC 4550':
                            mask_name_1 = np.where(name_i == obs_names_duplicate)[0]
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                            obs_x.append(obs_Mstar_duplicate[mask_name_1])
                            obs_y.append(obs_Re_CO[mask_name_2])
                            print('  included NGC 4550')
                        if name_i == b'NGC 4292':
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                            obs_Lk_ratio_i = 10**obs_Lk_ratios[mask_name_2] 
                            obs_Re_CO_i = obs_Re_CO[mask_name_2]
                        
                            L_k_i = (1/obs_Lk_ratio_i) * obs_Re_CO_i
                            logMsun_i = np.log10(0.82 * L_k_i)
                        
                            obs_x.append(logMsun_i)
                            obs_y.append(obs_Re_CO_i)
                            print('  included NGC 4292 manually with log Msun: %.2f' %logMsun_i)
                        if name_i == b'NGC 2697':
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                            obs_Lk_ratio_i = 10**obs_Lk_ratios[mask_name_2] 
                            obs_Re_CO_i = obs_Re_CO[mask_name_2]
                        
                            L_k_i = (1/obs_Lk_ratio_i) * obs_Re_CO_i
                            logMsun_i = np.log10(0.82 * L_k_i)
                        
                            obs_x.append(logMsun_i)
                            obs_y.append(obs_Re_CO_i)
                            print('  included NGC 4292 manually with log Msun: %.2f' %logMsun_i)
                    

            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
            assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
            

            print('Sample length Davis+13 ATLAS3D (detected):   %s'%len(obs_x))
            
            axs.scatter(obs_x, np.log10(obs_y), marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+13')
            #axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
            
    #-----------------
    # Add detection hist for excl. FR and incl. FR
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.2
        hist_bins = np.arange(-1, 3, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-0.5, 2)
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
        axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-0.5, 2)
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
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    if add_observational:
        handles_in.append(handles[-1])
        labels_in.append(labels[-1])
    axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)

    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_r50H2/%s_%s_%s_Mstar_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_r50H2/%s_%s_%s_Mstar_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#-----------------
# Returns stelmass - r50_H2 / r50
def _etg_stelmass_r50r50H2(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   add_detection_hist = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    r50_h2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_molecular_hydrogen'))(data)[soap_indicies_sample]
    r50_h2.convert_to_units('kpc')
    r50_h2.convert_to_physical()
    
    r50_ratio = np.divide(r50_h2, r50)
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(stellar_mass[mask_h2], r50_ratio[mask_h2]) 
    print('\nSpearman incl. FR:    M* - r50/r50 H2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(stellar_mass[mask_h2_SRs], r50_ratio[mask_h2_SRs])
    print('Spearman excl. FR:    M* - r50/r50 H2  rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        
        Y_VALUE_1 = np.log10(r50_ratio)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, -2, 2)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
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
        extent = (9.5, 12.5, -2, 2)                     ###
        gridsize = (25,20)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of all ETGs for excl. FR and incl. FR
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = r50_ratio[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = r50_ratio
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
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
                R_CO_Re_ratio = file['data/R_CO_Re_ratio/values'][:]
            
            obs_names_2 = np.array(obs_names_2)
            R_CO_Re_ratio = np.array(R_CO_Re_ratio)
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
            for name_i in obs_names_2:
                if name_i in obs_names_1:
                    mask_name_1 = np.where(name_i == obs_names_1)[0]
                    mask_name_2 = np.where(name_i == obs_names_2)[0]
                    obs_x.append(obs_Mstar_1[mask_name_1])
                    obs_y.append(R_CO_Re_ratio[mask_name_2])
                else:
                    if 'central' not in sample_input['name_of_preset']:
                        print('cannot find %s in 260 ATLAS3D sample'%name_i)
                        if name_i == b'NGC 4550':
                            mask_name_1 = np.where(name_i == obs_names_duplicate)[0]
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                            obs_x.append(obs_Mstar_duplicate[mask_name_1])
                            obs_y.append(R_CO_Re_ratio[mask_name_2])
                            print('  included NGC 4550')
                        if name_i == b'NGC 4292':
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                            obs_Lk_ratio_i = 10**obs_Lk_ratios[mask_name_2] 
                            obs_Re_CO_i = obs_Re_CO[mask_name_2]
                        
                            L_k_i = (1/obs_Lk_ratio_i) * obs_Re_CO_i
                            logMsun_i = np.log10(0.82 * L_k_i)
                            
                            obs_x.append(logMsun_i)
                            obs_y.append(R_CO_Re_ratio[mask_name_2])
                            print('  included NGC 4292 manually with log ratio: %.2f' %logMsun_i)
                        if name_i == b'NGC 2697':
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                            obs_Lk_ratio_i = 10**obs_Lk_ratios[mask_name_2] 
                            obs_Re_CO_i = obs_Re_CO[mask_name_2]
                        
                            L_k_i = (1/obs_Lk_ratio_i) * obs_Re_CO_i
                            logMsun_i = np.log10(0.82 * L_k_i)
                        
                            obs_x.append(logMsun_i)
                            obs_y.append(R_CO_Re_ratio[mask_name_2])
                            print('  included NGC 4292 manually with log ratio: %.2f' %logMsun_i)
                    

            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
            assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
            

            print('Sample length Davis+13 ATLAS3D (detected):   %s'%len(obs_x))
            
            axs.scatter(obs_x, np.log10(obs_y), marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+13')
            #axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
            
    #-----------------
    # Add detection hist for excl. FR and incl. FR
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.2
        hist_bins = np.arange(-1, 3, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-1.5, 1.5)
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
        axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}/r_{\mathrm{50}}$ [pkpc]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}/r_{\mathrm{50}}$ [pkpc]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-1.5, 1.5)
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
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    if add_observational:
        handles_in.append(handles[-1])
        labels_in.append(labels[-1])
    axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)

    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_r50_ratio/%s_%s_%s_Mstar_r50_ratio_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_r50_ratio/%s_%s_%s_Mstar_r50_ratio_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()

#-----------------
# returns r50 - r50 HI, detected only
def _etg_r50_r50H1(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h1 = 'exclusive_sphere_50kpc', 
                   h1_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H1',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H1 ]
                     add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   add_detection_hist = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()

    H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
    H1_mass.convert_to_units('Msun')
    H1_mass.convert_to_physical()

    u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
    r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
    u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    mag_plot = u_mag - r_mag

    central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

    kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
    
    
    #================================
    # Calculated values
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    r50_h1 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_atomic_hydrogen'))(data)[soap_indicies_sample]
    r50_h1.convert_to_units('kpc')
    r50_h1.convert_to_physical()
    
    
    #==========================================================
    # Useful masks
    mask_h1      = H1_mass > cosmo_quantity(h1_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h1_SRs  = np.logical_and.reduce([mask_h1, mask_SRs])       # detection kappa < 0.4
    mask_h1_FRs  = np.logical_and.reduce([mask_h1, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h1, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h1, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50[mask_h1], r50_h1[mask_h1]) 
    print('\nSpearman incl. FR:    r50 - r50_h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50[mask_h1_SRs], r50_h1[mask_h1_SRs])
    print('Spearman excl. FR:    r50 - r50_h1 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(r50)
        
        Y_VALUE_1 = np.log10(r50_h1)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H1_mass.value > 0] = np.log10(H1_mass[H1_mass.value > 0])
        #S_VALUE_1 = (np.log10(H1_mass)-(np.log10(h1_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H1
        cb = axs.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c='r', s=S_VALUE_1[mask_h1], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
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
        sc_points = axs.scatter(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], c=C_VALUE_1[mask_h1], s=4, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h1], Y_VALUE_1[~mask_h1], c=C_VALUE_1[~mask_h1], s=4, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0.2, edgecolor='k')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H1_mass)
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=10)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # Plot detections as filled viridis circles and squares
        axs.scatter(X_VALUE_1[mask_h1_SRs], Y_VALUE_1[mask_h1_SRs], c=C_VALUE_1[mask_h1_SRs], s=4.5, cmap=newcmp, norm=norm, marker='o', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
        cb = axs.scatter(X_VALUE_1[mask_h1_FRs], Y_VALUE_1[mask_h1_FRs], c=C_VALUE_1[mask_h1_FRs], s=4.5, cmap=newcmp, norm=norm, marker='s', alpha=0.65, linewidths=0, edgecolor='none', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
        # Plot non-detections as empty grey circles and squares
        axs.scatter(X_VALUE_1[mask_X_SRs], Y_VALUE_1[mask_X_SRs], s=4.5, marker='o', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa<0.4')
        axs.scatter(X_VALUE_1[mask_X_FRs], Y_VALUE_1[mask_X_FRs], s=4.5, marker='s', alpha=0.65, linewidths=0.4, edgecolor='grey', facecolor='none', label='undet. kappa>0.4')
    if scatter_or_hexbin == 'hexbin_count':
        ### Plots hexbin showing number of galaxies in bin
        
        cmap = cmasher.jungle_r
        newcmp = cmasher.get_sub_cmap(cmap, 0.1, 0.75)
        
        # hexbin for all values
        extent = (-1, 2, -1, 2)                      ###
        gridsize = (35,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H1':  
        ### Plots hexbin showing median log10 H1_mass
        
        # Define default colormap for H1
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (-1, 2, -1, 2)                      ###
        gridsize = (35,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], C=C_VALUE_1[mask_h1], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h1], Y_VALUE_1[mask_h1], C=C_VALUE_1[mask_h1], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of all ETGs for excl. FR and incl. FR
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(-0.5, 3, 0.1)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN_1 = np.log10(r50[mask_SRs])
        Y_MEDIAN_1 = r50_h1[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H1_mass
        X_MEDIAN_2 = np.log10(r50)
        Y_MEDIAN_2 = r50_h1
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN_1 >= hist_bins[i]) & (X_MEDIAN_1 < hist_bins[i + 1])
            y_bin = Y_MEDIAN_1.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask_SRs][mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (X_MEDIAN_2 >= hist_bins[i]) & (X_MEDIAN_2 < hist_bins[i + 1])
            y_bin = Y_MEDIAN_2.value[mask]
        
            # Remove <107 H1 mass from sample
            y_bin = y_bin[H1_mass[mask].value > h1_detection_limit]
            
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        hist_bins = np.arange(-1, 2, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h1_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
        #==========================================
        ### ETG (incl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1, bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h1], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(-0.3, 1.5)
    axs.set_ylim(-0.5, 2)
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    axs.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h1 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'log$_{10}$ $r_{\mathrm{50}}$ (%s) [pkpc]'%dict_aperture[aperture])
    if not add_detection_hist:
        axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{I}}}$ [pkpc]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{I}}}$ [pkpc]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-0.5, 2.0)
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{I}}>10^{7}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_kappa':
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H1':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_I}}$ [M$_{\odot}$]', extend='min')
      
    
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
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    if add_observational:
        handles_in.append(handles[-1])
        labels_in.append(labels[-1])
    axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    
            
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/r50_r50H1/%s_%s_%s_r50_r50H1_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/r50_r50H1/%s_%s_%s_r50_r50H1_%s_H1ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h1, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# returns r50 - r50 H2, detected only
def _etg_r50_r50H2(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   add_detection_hist = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    r50_h2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_molecular_hydrogen'))(data)[soap_indicies_sample]
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
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50[mask_h2], r50_h2[mask_h2]) 
    print('\nSpearman incl. FR:    r50 - r50_h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50[mask_h2_SRs], r50_h2[mask_h2_SRs])
    print('Spearman excl. FR:    r50 - r50_h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(r50)
        
        Y_VALUE_1 = np.log10(r50_h2)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (-1, 2, -1, 2)                      ###
        gridsize = (35,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
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
        extent = (-1, 2, -1, 2)                      ###
        gridsize = (35,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of all ETGs for excl. FR and incl. FR
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(-0.5, 3, 0.1)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(r50[mask_SRs])
        Y_MEDIAN_1 = r50_h2[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(r50)
        Y_MEDIAN_2 = r50_h2
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
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
                obs_H2_1       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                obs_Re_1       = file['data/Re/values'][:] * u.Unit(file['data/Re/values'].attrs['units'])
                obs_mask_1     = file['data/det_mask/values'][:]
                obs_isvirgo_1  = file['data/Virgo/values'][:]
                obs_iscentral_1  = file['data/BCG/values'][:]
            
            obs_names_1 = np.array(obs_names_1)
            obs_mask_1  = np.array(obs_mask_1, dtype=bool)
            obs_isvirgo_1  = np.array(obs_isvirgo_1, dtype=bool)
            obs_iscentral_1  = np.array(obs_iscentral_1, dtype=bool)
            obs_H2_1.to('Msun')
            obs_Re_1.to('kpc')
            
            obs_names_duplicate = obs_names_1
            obs_Re_duplicate = obs_Re_1
            
            # Pull out only detected systems
            obs_names_1 = obs_names_1[obs_mask_1]
            obs_isvirgo_1 = obs_isvirgo_1[obs_mask_1]
            obs_iscentral_1 = obs_iscentral_1[obs_mask_1]
            obs_Re_1 = obs_Re_1[obs_mask_1]
            obs_H2_1 = obs_H2_1[obs_mask_1]
            obs_mask_1 = obs_mask_1[obs_mask_1]
            
            with h5py.File('%s/Davis2013_ATLAS3D_CO_extent.hdf5'%obs_dir, 'r') as file:
                obs_names_2 = file['data/Galaxy/values'][:]
                obs_Re_CO   = file['data/R_CO/values'][:] * u.Unit(file['data/R_CO/values'].attrs['units'])
                obs_ratios   = file['data/R_CO_Re_ratio/values'][:]
            
            obs_Re_CO.to('kpc')
            
            #--------------------------------------------
            
            if 'central' in sample_input['name_of_preset']:
                obs_names_1 = obs_names_1[obs_iscentral_1]
                obs_H2_1    = obs_H2_1[obs_iscentral_1]
                obs_Re_1      = obs_Re_1[obs_iscentral_1]
                obs_mask_1  = obs_mask_1[obs_iscentral_1]
            if 'satellite' in sample_input['name_of_preset']:
                obs_names_1 = obs_names_1[~obs_iscentral_1]
                obs_H2_1    = obs_H2_1[~obs_iscentral_1]
                obs_Re_1      = obs_Re_1[~obs_iscentral_1]
                obs_mask_1  = obs_mask_1[~obs_iscentral_1]
            if 'cluster' in sample_input['name_of_preset']:
                obs_names_1 = obs_names_1[obs_isvirgo_1]
                obs_H2_1    = obs_H2_1[obs_isvirgo_1]
                obs_Re_1      = obs_Re_1[obs_isvirgo_1]
                obs_mask_1  = obs_mask_1[obs_isvirgo_1]
            if 'group' in sample_input['name_of_preset']:
                obs_names_1 = obs_names_1[~obs_isvirgo_1]
                obs_H2_1    = obs_H2_1[~obs_isvirgo_1]
                obs_Re_1      = obs_Re_1[~obs_isvirgo_1]
                obs_mask_1  = obs_mask_1[~obs_isvirgo_1]
                
                
            # Match galaxy names to assign Re_CO
            obs_x = []
            obs_y = []
            for name_i in obs_names_2:
                if name_i in obs_names_1:
                    mask_name_1 = np.where(name_i == obs_names_1)[0]
                    mask_name_2 = np.where(name_i == obs_names_2)[0]
                    obs_x.append(obs_Re_1[mask_name_1])
                    obs_y.append(obs_Re_CO[mask_name_2])
                else:
                    print('cannot find %s in 260 ATLAS3D sample'%name_i)
                    if 'central' not in sample_input['name_of_preset']:
                        if name_i == b'NGC 4550':
                            mask_name_1 = np.where(name_i == obs_names_duplicate)[0]
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                            obs_x.append(obs_Re_duplicate[mask_name_1])
                            obs_y.append(obs_Re_CO[mask_name_2])
                            print('  included NGC 4550')
                        if name_i == b'NGC 4292':
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                            obs_ratio_i = obs_ratios[mask_name_2] 
                            obs_Re_CO_i = obs_Re_CO[mask_name_2]
                    
                            Re_i = (1/obs_ratio_i) * obs_Re_CO_i
                        
                            obs_x.append(Re_i)
                            obs_y.append(obs_Re_CO_i)
                            print('  included NGC 4292 manually with Re: %.2f' %Re_i)
                        if name_i == b'NGC 2697':
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                        
                            obs_ratio_i = obs_ratios[mask_name_2] 
                            obs_Re_CO_i = obs_Re_CO[mask_name_2]
                    
                            Re_i = (1/obs_ratio_i) * obs_Re_CO_i
                        
                            obs_x.append(Re_i)
                            obs_y.append(obs_Re_CO_i)
                            print('  included NGC 4292 manually with log Re: %.2f' %Re_i)
                    
            
            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
            assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
            

            print('Sample length Davis+13 ATLAS3D (detected):   %s'%len(obs_x))
            
            axs.scatter(np.log10(obs_x), np.log10(obs_y), marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+13')
            #axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
    
            
    #-----------------
    # Add detection hist for excl. FR and incl. FR
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(-1, 2, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(-0.3, 1.5)
    axs.set_ylim(-0.5, 2)
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
    axs.set_xlabel(r'log$_{10}$ $r_{\mathrm{50}}$ (%s) [pkpc]'%dict_aperture[aperture])
    if not add_detection_hist:
        axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-0.5, 2)
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
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    if add_observational:
        handles_in.append(handles[-1])
        labels_in.append(labels[-1])
    axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/r50_r50H2/%s_%s_%s_r50_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/r50_r50H2/%s_%s_%s_r50_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# returns r50 HI - r50 H2, detected only
def _etg_r50H1_r50H2(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                     aperture_h1 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   h1_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   add_detection_hist = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()

    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    
    H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
    H1_mass.convert_to_units('Msun')
    H1_mass.convert_to_physical()

    u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
    r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
    u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
    mag_plot = u_mag - r_mag

    central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

    kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
    
    
    #================================
    # Calculated values
    r50_h1 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_atomic_hydrogen'))(data)[soap_indicies_sample]
    r50_h1.convert_to_units('kpc')
    r50_h1.convert_to_physical()
    
    r50_h2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_molecular_hydrogen'))(data)[soap_indicies_sample]
    r50_h2.convert_to_units('kpc')
    r50_h2.convert_to_physical()
    
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h1      = H1_mass > cosmo_quantity(h1_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2      = np.logical_and.reduce([mask_h1, mask_h2])       
    
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50_h1[mask_h2], r50_h2[mask_h2]) 
    print('\nSpearman incl. FR:    r50_h1 - r50_h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50_h1[mask_h2_SRs], r50_h2[mask_h2_SRs])
    print('Spearman excl. FR:    r50_h1 - r50_h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(r50_h1)
        
        Y_VALUE_1 = np.log10(r50_h2)
        
        C_VALUE_1 = np.zeros(len(stellar_mass))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (-1, 2, -1, 2)                      ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
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
        extent = (-1, 2, -1, 2)                      ###
        gridsize = (25,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of all ETGs for excl. FR and incl. FR
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(-0.5, 3, 0.2)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(r50_h1[mask_SRs])
        Y_MEDIAN_1 = r50_h2[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(r50_h1)
        Y_MEDIAN_2 = r50_h2
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    #-----------------
    # Add observations
    if add_observational:
        """
        Pick observations we want to add
        """
        print('No obs available')
            
    #-----------------
    # Add detection hist for excl. FR and incl. FR
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(-1, 2, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
            
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(-0.5, 2)
    axs.set_ylim(-0.5, 2)
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
    axs.set_xlabel(r'log$_{10}$ $r_{\mathrm{50,H_{1}}}$ (%s) [pkpc]'%dict_aperture[aperture])
    if not add_detection_hist:
        axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
        
        ax_hist.minorticks_on()
        ax_hist.set_xlim(-0.01, 1.01)
        ax_hist.set_ylim(-0.5, 2)
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
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles_in = [handles[1], handles[0]]
    labels_in = [labels[1], labels[0]]
    if add_observational:
        handles_in.append(handles[-1])
        labels_in.append(labels[-1])
    axs.legend(handles_in, labels_in, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/r50H1_r50H2/%s_%s_%s_r50H1_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/r50H1_r50H2/%s_%s_%s_r50H1_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#-----------------
# returns r50 - h2 mass
def _etg_r50_h2mass(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   print_typical_galaxy = True,       # median galaxy properties of detected ETG
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    r50 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    r50 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
        
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(r50)
        Y_VALUE_1 = np.log10(H2_mass)
        C_VALUE_1 = kappa_stars
        S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
        cb = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H2_mass)
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)
        
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
        extent = (-1, 2, 6, 11)                     ###
        gridsize = (30,12)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H2':  
        ### Plots hexbin showing median log10 H2_mass
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (-1, 2, 6, 11)                     ###
        gridsize = (30,12)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(-0.5, 3, 0.2)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(r50[mask_SRs])
        Y_MEDIAN_1 = H2_mass[mask_SRs]
        
        X_MEDIAN_2 = np.log10(r50)
        Y_MEDIAN_2 = H2_mass
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
    #-----------------
    # Add observations
    if add_observational:
        """
        Pick observations we want to add
        """
        add_Young2011 = True
        
        if add_Young2011:      # ($z=0.0$)
            # Load the observational data
            with h5py.File('%s/Davis2019_ATLAS3D.hdf5'%obs_dir, 'r') as file:
                obs_names_1 = file['data/Galaxy/values'][:]
                obs_H2_1       = file['data/log_H2/values'][:] * u.Unit(file['data/log_H2/values'].attrs['units'])
                obs_Mstar_1    = file['data/log_Mstar/values'][:] * u.Unit(file['data/log_Mstar/values'].attrs['units'])
                obs_Re         = file['data/Re/values'][:] * u.Unit(file['data/Re/values'].attrs['units'])
                obs_mask_1     = file['data/det_mask/values'][:]
                obs_isvirgo_1  = file['data/Virgo/values'][:]
                obs_iscentral_1  = file['data/BCG/values'][:]
            
            obs_names_1 = np.array(obs_names_1)
            obs_mask_1  = np.array(obs_mask_1, dtype=bool)
            obs_isvirgo_1  = np.array(obs_isvirgo_1, dtype=bool)
            obs_iscentral_1  = np.array(obs_iscentral_1, dtype=bool)
            obs_H2_1.to('Msun')
            obs_Mstar_1.to('Msun')
            obs_Re.to('kpc')
            
            #--------------------------------------------
            
            if 'central' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[obs_iscentral_1]
                obs_H2_1    = obs_H2_1[obs_iscentral_1]
                obs_Re      = obs_Re[obs_iscentral_1]
                obs_mask_1  = obs_mask_1[obs_iscentral_1]
            if 'satellite' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[~obs_iscentral_1]
                obs_H2_1    = obs_H2_1[~obs_iscentral_1]
                obs_Re      = obs_Re[~obs_iscentral_1]
                obs_mask_1  = obs_mask_1[~obs_iscentral_1]
            if 'cluster' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[obs_isvirgo_1]
                obs_H2_1    = obs_H2_1[obs_isvirgo_1]
                obs_Re      = obs_Re[obs_isvirgo_1]
                obs_mask_1  = obs_mask_1[obs_isvirgo_1]
            if 'group' in sample_input['name_of_preset']:
                obs_Mstar_1 = obs_Mstar_1[~obs_isvirgo_1]
                obs_H2_1    = obs_H2_1[~obs_isvirgo_1]
                obs_Re      = obs_Re[~obs_isvirgo_1]
                obs_mask_1  = obs_mask_1[~obs_isvirgo_1]

            print('Sample length Young+11 ATLAS3D (detected):   %s'%len(obs_Mstar_1[obs_mask_1]))
            #print('Sample length Cappellari+11 ATLAS3D (non-det):    %s'%len(obs_Mstar_1[~obs_mask_1]))
            
            obs_x = np.log10(np.array(obs_Re))
            obs_y = np.array(obs_H2_1)
            
            axs.scatter(obs_x[obs_mask_1], obs_y[obs_mask_1], marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Young+11')
            #¢axs.scatter(obs_x[~obs_mask_1], np.log10(obs_y[~obs_mask_1]), marker='^', s=6, alpha=1, zorder=5, facecolors='none', linewidths=0.3,  edgecolors='r', label='Cappellari+11 (non-det.)')
            #axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
    
        
        
    #-----------
    # Axis formatting
    axs.set_xlim(-0.3, 1.5)
    axs.set_ylim(6, 11)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
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
    axs.set_xlabel(r'log$_{10}$ $r_{\mathrm{50}}$ (%s) [pkpc]'%dict_aperture[aperture])
    axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ [M$_{\odot}$]')


	#-----------
    # colorbar
    if scatter_or_hexbin == 'scatter_new':
        fig.colorbar(cb, ax=axs, label='log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$', extend='both')      #, extend='max'
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies')
    if scatter_or_hexbin == 'hexbin_H2':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$]', extend='min')
      
    
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/r50_MH2/%s_%s_%s_r50_MH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/r50_MH2/%s_%s_%s_r50_MH2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# returns h2 mass - r50 H2
def _etg_h2mass_r50H2(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'scatter_new',         # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = True,
                   #---------------
                   print_fdet           = True,
                   print_typical_galaxy = True,       # median galaxy properties of detected ETG
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_detection_hist = False

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    r50_h2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_molecular_hydrogen'))(data)[soap_indicies_sample]
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
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(r50_h2[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    r50_h2 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(r50_h2[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    r50_h2 - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(H2_mass)
        Y_VALUE_1 = np.log10(r50_h2)
        C_VALUE_1 = kappa_stars
        S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
    if scatter_or_hexbin == 'scatter_old':
        ### Plots scatter with o for detections and x for non-detect, with marker size = log10 H2
        cb = axs.scatter(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], c='r', s=S_VALUE_1[mask_h2], marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(X_VALUE_1[~mask_h2], Y_VALUE_1[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter separtarately for kappa < 0.4 and kappa > 0.4
        C_VALUE_1 = np.log10(H2_mass)
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)
        
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
        extent = (6, 11, -1, 2)                     ###
        gridsize = (20,15)
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
    if scatter_or_hexbin == 'hexbin_H2':  
        ### Plots hexbin showing median log10 H2_mass
        
        # Define default colormap for H2
        norm = colors.Normalize(vmin=6.5, vmax=11)      ###
        
        # Create custom colormap with grey at the bottom
        viridis = mpl.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 9))
        newcolors[:1, :] = colors.to_rgba('grey')
        newcmp = ListedColormap(newcolors)
        
        # hexbin for all values
        extent = (6, 11, -1, 2)                     ###
        gridsize = (20,15)
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(6, 11, 0.2)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(H2_mass[mask_SRs])
        Y_MEDIAN_1 = r50_h2[mask_SRs]

        X_MEDIAN_2 = np.log10(H2_mass)
        Y_MEDIAN_2 = r50_h2
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    
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
            obs_H2_duplicate = obs_H2_1
            
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
            for name_i in obs_names_2:
                if name_i in obs_names_1:
                    mask_name_1 = np.where(name_i == obs_names_1)[0]
                    mask_name_2 = np.where(name_i == obs_names_2)[0]
                    obs_x.append(obs_Re_CO[mask_name_2])
                    obs_y.append(obs_H2_1[mask_name_1])
                else:
                    if 'central' not in sample_input['name_of_preset']:
                        print('cannot find %s in 260 ATLAS3D sample'%name_i)
                        if name_i == b'NGC 4550':
                            mask_name_1 = np.where(name_i == obs_names_duplicate)[0]
                            mask_name_2 = np.where(name_i == obs_names_2)[0]
                            obs_x.append(obs_Re_CO[mask_name_2])
                            obs_y.append(obs_H2_duplicate[mask_name_1])
                            print('  included NGC 4550')
                        if name_i == b'NGC 4292':
                            print('  excluded NGC 4292 (missing H2 mass)')
                        if name_i == b'NGC 2697':
                            print('  excluded NGC 2697 (missing H2 mass)')
                    

            obs_x = np.array(obs_x)
            obs_y = np.array(obs_y)
            assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
            

            print('Sample length Davis+13 ATLAS3D (detected):   %s'%len(obs_x))
            
            axs.scatter(obs_y, np.log10(obs_x), marker='^', s=8, alpha=1, zorder=5, c='r', edgecolors='none', label='Davis+13')
            #axs.axvline(np.log10(0.82*(10**11.41)), ls='--', linewidth=1, c='grey')
      
    #-----------
    # Axis formatting
    axs.set_xlim(6, 11)
    axs.set_ylim(-0.5, 2)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
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
    axs.set_xlabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')


	#-----------
    # colorbar
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
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.105, y=0.938, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
    #-----------
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    if scatter_or_hexbin == 'scatter_new':
        handles = [handles[5], handles[4], (handles[0], handles[2]), (handles[1], handles[3]), handles[6]]
        labels = [labels[5], labels[4], labels[0], labels[1], labels[6]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [handles[1], handles[0], handles[2]]
        labels = [labels[1], labels[0], labels[2]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/MH2_r50H2/%s_%s_%s_MH2_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/MH2_r50H2/%s_%s_%s_MH2_r50H2_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#----------------
# Plot stelmass - sigma ratio = density within X / ring of density Y, detected only
def _gas_surface_ratios(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   gas_type = 'h2',                    # [ 'h2' / 'h1' / 'gas' ]
                     surfdens_aperture_1   = '',                         # Radius 1, smaller
                     surfdens_aperture_2   = '',                         # Radius 2, larger
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                     add_median_line = False,
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
        fig = plt.figure(figsize=(10/3, 2.2))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(1, 4.5),
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
    
    
    #================================
    # Select apertures with holes
    if gas_type == 'h2':
        mass_radius_1 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_1, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = mass_radius_2 - mass_radius_1
    elif gas_type == 'h1':
        mass_radius_1 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_2, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = mass_radius_2 - mass_radius_1
    elif gas_type == 'gas':
        mass_radius_1 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_1, 'gas_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_2, 'gas_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = mass_radius_2 - mass_radius_1
    mass_radius_1.convert_to_physical()
    mass_radius_2.convert_to_physical()
    mass_radius_1.convert_to_units('Msun')
    mass_radius_2.convert_to_units('Msun')
    
    
    # Calculate surface density for small radius, and larger ring, ignore systems with less than 107 H2
    surf_density_1 = mass_radius_1 / (np.pi*(cosmo_quantity(float(surfdens_aperture_1), u.kpc, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)**2))
    surf_density_2 = mass_radius_2 / (np.pi*(cosmo_quantity(float(surfdens_aperture_2) - float(surfdens_aperture_1), u.kpc, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)**2))
    
    raw_ratio = np.divide(surf_density_1, surf_density_2)
    surf_density_ratio = raw_ratio
    
    # Cap at maximum of 10**3 and 10**-3
    surf_density_ratio[raw_ratio.value > 1e2]  = cosmo_quantity(1e2-0.0001, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    surf_density_ratio[raw_ratio.value < 1e-2] = cosmo_quantity(1e-2, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    
    
    
    #-----------------
    # Spearman with gas mass
    #res = scipy.stats.spearmanr(sfr50[mask_h2], H2_mass[mask_h2]) 
    #print('\nSpearman incl. FR:    SFR - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    #res = scipy.stats.spearmanr(sfr50[mask_h2_SRs], H2_mass[mask_h2_SRs])
    #print('Spearman excl. FR:    SFR - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(raw_ratio)
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        extent = (9.5, 12.5, -3, 3)                     ###
        gridsize = (20,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='Greens', lw=0, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, mincnt=1, zorder=-3, cmap=newcmp, lw=0.02, alpha=0.7)
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
        extent = (9.5, 12.5, -3, 3)                     ###
        gridsize = (20,15)                              ###          25 by default, then multiply by axis_x_range/axis_y_range
        
        #axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, edgecolors='w', lw=1.2)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        #cb = axs.hexbin(X_VALUE_1, Y_VALUE_1, C=C_VALUE_1, reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05)
        
        
        axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap=newcmp, lw=0.05, alpha=0.3)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(X_VALUE_1[mask_h2], Y_VALUE_1[mask_h2], C=C_VALUE_1[mask_h2], reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap=newcmp, lw=0.05, alpha=0.7)
        
    
    #------------
    # Median line of all ETGs for excl. FR and incl. FR
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = sfr50[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = sfr50
        
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians_log), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        bin_width = 0.25
        hist_bins = np.arange(-4, 4, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        if 'LTG' not in sample_input['name_of_preset']:
            bin_n, _           = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
            bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
            mask_positive_n     = bin_n > 0
            bin_f_detected      = bin_n_detected[mask_positive_n]
        
            # step
            mask_positive_n_extra = list(mask_positive_n)
            if True in mask_positive_n_extra:
                last_true_index = len(mask_positive_n_extra) - 1 - mask_positive_n_extra[::-1].index(True)
            mask_positive_n_extra.insert(last_true_index + 1, True)
            mask_positive_n_extra = np.array(mask_positive_n_extra)
            ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
            # errorbar
            hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
            #ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
        
        #==========================================
        ### ETG (incl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_h2], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2], bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]
        
        # step
        mask_positive_n_extra = list(mask_positive_n)
        if True in mask_positive_n_extra:
            last_true_index = len(mask_positive_n_extra) - 1 - mask_positive_n_extra[::-1].index(True)
        
        mask_positive_n_extra.insert(last_true_index + 1, True)
        mask_positive_n_extra = np.array(mask_positive_n_extra)
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
    #=========================================================
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(-2.03, 2.03)
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
    dict_ylabel = {'h2': 'H_{2}',
                   'h1': 'H_{\mathrm{I}}',
                   'gas': 'gas'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    if not add_detection_hist:
        axs.set_ylabel(r'log$_{10}$ $\Sigma_{\mathrm{%s}}^{\mathrm{%s\:kpc}}/\Sigma_{\mathrm{%s}}^{\mathrm{%s\:kpc}}$'%(dict_ylabel[gas_type], surfdens_aperture_1, dict_ylabel[gas_type], surfdens_aperture_2))
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'log$_{10}$ $\Sigma_{\mathrm{%s}}^{\mathrm{%s\:kpc}}/\Sigma_{\mathrm{%s}}^{\mathrm{%s\:kpc}}$'%(dict_ylabel[gas_type], surfdens_aperture_1, dict_ylabel[gas_type], surfdens_aperture_2))
        
        ax_hist.set_xlim(left=0.9)
        ax_hist.set_ylim(-2.03, 2.03)
        #ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        #ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'Count')
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
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s%s>>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    if add_detection_hist:
        fig_text(x=0.107, y=0.963, ha='left', s=text_title, fontsize=7, ax=ax_hist, delim=('<<', '>>'),
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
        
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/gas_sigma_ratio/%s_%s_%s_Mstar_%s_sigmaratio_%s-%s_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], gas_type, surfdens_aperture_1, surfdens_aperture_2, aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/gas_sigma_ratio/%s_%s_%s_Mstar_%s_sigmaratio_%s-%s_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], gas_type, surfdens_aperture_1, surfdens_aperture_2, aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#----------------
# Returns stelmass - H2 / H1 / gas hole detection rate, binned by M*
def _aperture_fdet_missing_gas(csv_samples = [], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture    = 'exclusive_sphere_50kpc', 
                   hole_gas_type = 'gas',                    # [ 'h2' / 'h1' / 'gas' ]
                     hole_aperture   = '',                         # Radius within which to look for missing gas
                     hole_limit      = 10**1,                    # ^ must have less than this gas within this aperture
                   #---------------
                   print_fdet         = True,
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
    #---------------------------
    # Extract data from samples:
    define_dict_labels = True
    if define_dict_labels:
        dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                        'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$ (centrals)',
                        'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$ (satellites)',
                       'all_ETGs': 'ETGs (excl. FRs)',
                        'all_ETGs_centrals': 'ETGs (excl. FRs, centrals)',
                        'all_ETGs_satellites': 'ETGs (excl. FRs, satellites',
                       'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)',
                        'all_ETGs_plus_redspiral_centrals': 'ETGs (incl. FRs, centrals)',
                        'all_ETGs_plus_redspiral_satellites': 'ETGs (incl. FRs, satellites)'}
        dict_colors = {'all_galaxies': 'k',
                        'all_galaxies_centrals': 'k',
                        'all_galaxies_satellites': 'k',
                       'all_ETGs': 'C0',
                        'all_ETGs_centrals': 'C0',
                        'all_ETGs_satellites': 'C0',
                       'all_ETGs_plus_redspiral': 'C1',
                        'all_ETGs_plus_redspiral_centrals': 'C1',
                        'all_ETGs_plus_redspiral_satellites': 'C1'}
        dict_ls     = {'all_galaxies': '-',
                        'all_galaxies_centrals': '-',
                        'all_galaxies_satellites': '-',
                       'all_ETGs': '--',
                        'all_ETGs_centrals': '--',
                        'all_ETGs_satellites': '--',
                       'all_ETGs_plus_redspiral': '-.',
                        'all_ETGs_plus_redspiral_centrals': '-.',
                        'all_ETGs_plus_redspiral_satellites': '-.'}
    labels_list = []
    for csv_sample_i in csv_samples:
        soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = csv_sample_i)
        print(sample_input['name_of_preset'])
        
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
        box_size = data.metadata.boxsize[0]
    
    
        #-------------------------------
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
                
        u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
        r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
        u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        mag_plot = u_mag - r_mag
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
        
        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
        
        
        #======================
        # Select apertures with holes
        if hole_gas_type == 'h2':
            mass_hole = attrgetter('%s.%s'%(hole_aperture, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        elif hole_gas_type == 'h1':
            mass_hole = attrgetter('%s.%s'%(hole_aperture, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        elif hole_gas_type == 'gas':
            mass_hole = attrgetter('%s.%s'%(hole_aperture, 'gas_mass'))(data)[soap_indicies_sample]
        mass_hole.convert_to_units('Msun')
        

        #==========================================================
        # Useful masks
        #mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        #mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        #mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
        #mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
        #mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
        #mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4
        
        
        
        
        
        
        #=============================
        # Of all galaxies within sample, how many have missing gas (gas mass below hole_limit):
        mask_det_missing_gas = mass_hole < ((hole_limit)*u.Msun)
        
        # Fraction of those with H2 with rings
        f_missing_gas = len(stellar_mass[mask_det_missing_gas])/len(stellar_mass) 
        f_missing_gas_err = binom_conf_interval(k=len(stellar_mass[mask_det_missing_gas]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_missing_gas_err_lower = f_missing_gas - f_missing_gas_err[0]
        f_missing_gas_err_upper = f_missing_gas_err[1] - f_missing_gas
        # print
        if print_fdet:
            print('FRACTION OF ALL SAMPLE WITH MISSING %s:   %s    %s' %(hole_gas_type, title_text_in, hole_aperture))
            print('  >  f_missing_gas:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(f_missing_gas, f_missing_gas_err_lower, f_missing_gas_err_upper, len(stellar_mass[mask_det_missing_gas]), len(stellar_mass)))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
        
        #--------------
        # Of galaxies with H2_mass > h2_detection_limit, how many have holes with < h2_hole_limit within h2_hole_aperture_list
        # Bin data by stellar mass, find fraction within stellar bin and append
        bin_step = 0.2
        hist_bins = np.arange(9.5, 13, bin_step)  # Binning edges
        bin_n, _           = np.histogram(np.log10(stellar_mass), bins=hist_bins)
        # bin_n_detected is the number with missing gas
        bin_n_detected, _  = np.histogram(np.log10(stellar_mass[mask_det_missing_gas]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
    
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        
        #------------------
        # Plot bar of fraction
        #axs.step((hist_bins[1:][mask_positive_n]), , where='pre', color=dict_colors[sample_input['name_of_preset']], linewidth=1.0, label=dict_labels[sample_input['name_of_preset']])
        axs.stairs(np.append(100*bin_f_detected, 0), edges=np.insert(np.append(hist_bins[1:][mask_positive_n], hist_bins[1:][mask_positive_n][-1]+bin_step), 0, hist_bins[1:][mask_positive_n][0]-bin_step, axis=0) , orientation='vertical', baseline=0, fill=False, color=dict_colors[sample_input['name_of_preset']], linewidth=1.0, label=dict_labels[sample_input['name_of_preset']], path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        axs.errorbar(hist_bins_midpoint, 100*bin_f_detected, xerr=None, yerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], ecolor=dict_colors[sample_input['name_of_preset']], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
         
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(0, 75)
    #axs.set_xscale("log")
    #axs.set_yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    axs.tick_params(axis='y', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    dict_ylabel = {'h2': 'H$_{2}$',
                   'h1': 'H$_{\mathrm{I}}$',
                   'gas': 'gas'}
    axs.set_ylabel('Incidence of $M_{\mathrm{%s}}=0$ \namong $M_*$ bin '%(dict_ylabel[hole_gas_type]) + '$\:$[\%]')

    
    #-----------  
    # Annotations
    #axs.text(0.80, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.915, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<<%s>><<..>><<%s>>'%(run_name_title, title_text_in)
    fig_text(x=0.133, y=0.98, ha='left', s=text_title, fontsize=7, ax=axs, delim=('<<', '>>'),
            highlight_textprops=[
                {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
                {"color": "white"},
                {"color": "black"}
            ])
    
    
    
    #-----------
    # Legend
    axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=1.3)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/gas_missing_fdet/%s_%s_%s_fdet_missing_gas_%s_%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], np.log10(hole_limit), hole_aperture, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/gas_missing_fdet/%s_%s_%s_fdet_missing_gas_%s_%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], np.log10(hole_limit), hole_aperture, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#----------------
# Returns stelmass - ellip
def _etg_stelmass_ellip(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                     use_projected = False,
                   h2_detection_limit = 10**7,
                   scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                     add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    
    def _compute_intrinsic_ellipticity_triaxiality():
        # construct inertia tensor
        inertiatensor_raw = attrgetter('%s.%s'%('bound_subhalo', 'stellar_inertia_tensor_noniterative'))(data)[soap_indicies_sample]
        i11 = inertiatensor_raw[:,0]
        i22 = inertiatensor_raw[:,1]
        i33 = inertiatensor_raw[:,2]
        i12 = inertiatensor_raw[:,3]
        i13 = inertiatensor_raw[:,4]
        i23 = inertiatensor_raw[:,5]
        
        inertiatensor = np.empty((len(i11), 3, 3))
        inertiatensor[:, 0, 0] = i11
        inertiatensor[:, 1, 1] = i22
        inertiatensor[:, 2, 2] = i33
        inertiatensor[:, 0, 1] = inertiatensor[:, 1, 0] = i12
        inertiatensor[:, 0, 2] = inertiatensor[:, 2, 0] = i13
        inertiatensor[:, 1, 2] = inertiatensor[:, 2, 1] = i23
        
        i11 = 0
        i22 = 0
        i33 = 0
        i12 = 0
        i13 = 0
        i23 = 0
            
        # Compute eigenvalues (sorted in descending order)
        eigenvalues = np.linalg.eigvalsh(inertiatensor)
        eigenvalues_sorted = np.sort(eigenvalues, axis=1)[:, ::-1]  # Sorted: I1 >= I2 >= I3
        inertiatensor = 0
        
        length_a2 = eigenvalues_sorted[:,0]  # Assign to principal moments, length_a2 == a**2, so length = sqrt length_a2. a = longest axis, b = intermediate, c = shortest
        length_b2 = eigenvalues_sorted[:,1]
        length_c2 = eigenvalues_sorted[:,2]
        #print(length_a2)
        #print(length_b2)
        #print(length_c2)
    
        # Compute Triaxiality Parameter
        triaxiality = np.divide((length_a2 - length_b2), (length_a2 - length_c2)) #if (length_a2 - length_c2) != 0 else np.nan  # Avoid division by zero
        ellipticity = 1 - np.divide(np.sqrt(length_c2), np.sqrt(length_a2))
        
        return cosmo_array(ellipticity, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0), cosmo_array(triaxiality, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)     
    
    def _compute_projected_ellipticity_triaxiality():
        # construct inertia tensor
        inertiatensor_raw = attrgetter('%s.%s'%('bound_subhalo', 'projected_stellar_inertia_tensor_noniterative'))(data)[soap_indicies_sample]
        i11 = inertiatensor_raw[:,0]
        i22 = inertiatensor_raw[:,1]
        i12 = inertiatensor_raw[:,2]
        
        inertiatensor = np.empty((len(i11), 2, 2))
        inertiatensor[:, 0, 0] = i11
        inertiatensor[:, 1, 1] = i22
        inertiatensor[:, 0, 1] = inertiatensor[:, 1, 0] = i12
        
        i11 = 0
        i22 = 0
        i12 = 0
            
        # Compute eigenvalues (sorted in descending order)
        eigenvalues = np.linalg.eigvalsh(inertiatensor)
        eigenvalues_sorted = np.sort(eigenvalues, axis=1)[:, ::-1]  # Sorted: I1 >= I2 >= I3
        inertiatensor = 0
        
        length_a2 = eigenvalues_sorted[:,0]  # Assign to principal moments, length_a2 == a**2, so length = sqrt length_a2. a = longest axis, b = intermediate, c = shortest
        length_b2 = eigenvalues_sorted[:,1]
        #print(length_a2)
        #print(length_b2)
    
        # Compute Triaxiality Parameter
        triaxiality = np.divide((length_a2 - length_b2), (length_a2 - length_c2)) #if (length_a2 - length_c2) != 0 else np.nan  # Avoid division by zero
        ellipticity = 1 - np.divide(np.sqrt(length_c2), np.sqrt(length_a2))
        
        return cosmo_array(ellipticity, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0), cosmo_array(triaxiality, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)     
    

    if use_projected:
        raise Exception('not currently configured projected')
    else:
        ellip, triax            = _compute_intrinsic_ellipticity_triaxiality()
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(ellip[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    ellip - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(ellip[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    ellip - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = ellip
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = ellip[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = ellip
        
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
            #y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
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
                
        axs.plot(np.flip(bin_centers), np.flip(medians), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
    #-----------------
    # Add observations
    if add_observational:
        """
        Pick observations we want to add
        """
        print('no obs available')
        
        
        print('add krajnovic, add det hist')
            
    #-----------------
    # Add detection hist for excl. FR and incl. FR
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(0, 1, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
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
        axs.set_ylabel(r'$\epsilon$')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'$\epsilon$')
        
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
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
        
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_ellip/%s_%s_%s_Mstar_ellip%s_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], ('_proj' if use_projected else ''), aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_ellip/%s_%s_%s_Mstar_ellip%s_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], ('_proj' if use_projected else ''), aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - triax
def _etg_stelmass_triax(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                     use_projected = False,
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    
    def _compute_intrinsic_ellipticity_triaxiality():
        # construct inertia tensor
        inertiatensor_raw = attrgetter('%s.%s'%('bound_subhalo', 'stellar_inertia_tensor_noniterative'))(data)[soap_indicies_sample]
        i11 = inertiatensor_raw[:,0]
        i22 = inertiatensor_raw[:,1]
        i33 = inertiatensor_raw[:,2]
        i12 = inertiatensor_raw[:,3]
        i13 = inertiatensor_raw[:,4]
        i23 = inertiatensor_raw[:,5]
        
        inertiatensor = np.empty((len(i11), 3, 3))
        inertiatensor[:, 0, 0] = i11
        inertiatensor[:, 1, 1] = i22
        inertiatensor[:, 2, 2] = i33
        inertiatensor[:, 0, 1] = inertiatensor[:, 1, 0] = i12
        inertiatensor[:, 0, 2] = inertiatensor[:, 2, 0] = i13
        inertiatensor[:, 1, 2] = inertiatensor[:, 2, 1] = i23
        
        i11 = 0
        i22 = 0
        i33 = 0
        i12 = 0
        i13 = 0
        i23 = 0
            
        # Compute eigenvalues (sorted in descending order)
        eigenvalues = np.linalg.eigvalsh(inertiatensor)
        eigenvalues_sorted = np.sort(eigenvalues, axis=1)[:, ::-1]  # Sorted: I1 >= I2 >= I3
        inertiatensor = 0
        
        length_a2 = eigenvalues_sorted[:,0]  # Assign to principal moments, length_a2 == a**2, so length = sqrt length_a2. a = longest axis, b = intermediate, c = shortest
        length_b2 = eigenvalues_sorted[:,1]
        length_c2 = eigenvalues_sorted[:,2]
        #print(length_a2)
        #print(length_b2)
        #print(length_c2)
    
        # Compute Triaxiality Parameter
        triaxiality = np.divide((length_a2 - length_b2), (length_a2 - length_c2)) #if (length_a2 - length_c2) != 0 else np.nan  # Avoid division by zero
        ellipticity = 1 - np.divide(np.sqrt(length_c2), np.sqrt(length_a2))
        
        return cosmo_array(ellipticity, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0), cosmo_array(triaxiality, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)     
    
    def _compute_projected_ellipticity_triaxiality():
        # construct inertia tensor
        inertiatensor_raw = attrgetter('%s.%s'%('bound_subhalo', 'projected_stellar_inertia_tensor_noniterative'))(data)[soap_indicies_sample]
        i11 = inertiatensor_raw[:,0]
        i22 = inertiatensor_raw[:,1]
        i12 = inertiatensor_raw[:,2]
        
        inertiatensor = np.empty((len(i11), 2, 2))
        inertiatensor[:, 0, 0] = i11
        inertiatensor[:, 1, 1] = i22
        inertiatensor[:, 0, 1] = inertiatensor[:, 1, 0] = i12
        
        i11 = 0
        i22 = 0
        i12 = 0
            
        # Compute eigenvalues (sorted in descending order)
        eigenvalues = np.linalg.eigvalsh(inertiatensor)
        eigenvalues_sorted = np.sort(eigenvalues, axis=1)[:, ::-1]  # Sorted: I1 >= I2 >= I3
        inertiatensor = 0
        
        length_a2 = eigenvalues_sorted[:,0]  # Assign to principal moments, length_a2 == a**2, so length = sqrt length_a2. a = longest axis, b = intermediate, c = shortest
        length_b2 = eigenvalues_sorted[:,1]
        #print(length_a2)
        #print(length_b2)
    
        # Compute Triaxiality Parameter
        triaxiality = np.divide((length_a2 - length_b2), (length_a2 - length_c2)) #if (length_a2 - length_c2) != 0 else np.nan  # Avoid division by zero
        ellipticity = 1 - np.divide(np.sqrt(length_c2), np.sqrt(length_a2))
        
        return cosmo_array(ellipticity, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0), cosmo_array(triaxiality, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)     
    

    if use_projected:
        raise Exception('not currently configured projected')
    else:
        ellip, triax            = _compute_intrinsic_ellipticity_triaxiality()
    
    
    #==========================================================
    # Useful masks
    mask_h2      = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_SRs     = kappa_stars < cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_h2_SRs  = np.logical_and.reduce([mask_h2, mask_SRs])       # detection kappa < 0.4
    mask_h2_FRs  = np.logical_and.reduce([mask_h2, ~mask_SRs])      # detection kappa > 0.4
    mask_X_SRs  = np.logical_and.reduce([~mask_h2, mask_SRs])       # non-detection kappa < 0.4
    mask_X_FRs  = np.logical_and.reduce([~mask_h2, ~mask_SRs])      # non-detection kappa > 0.4        
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(triax[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    triax - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(triax[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    triax - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = triax
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = triax[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = triax
        
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
            #y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
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
                
        axs.plot(np.flip(bin_centers), np.flip(medians), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        hist_bins = np.arange(0, 1, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
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
        axs.set_ylabel(r'Triaxiality')
    if add_detection_hist:
        axs.set_yticklabels([])
        ax_hist.set_ylabel(r'Triaxiality')
        
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
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
        
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_triax/%s_%s_%s_Mstar_triax%s_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], ('_proj' if use_projected else ''), aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_triax/%s_%s_%s_Mstar_triax%s_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], ('_proj' if use_projected else ''), aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - kappaco stars
def _etg_stelmass_kappaco(soap_indicies_sample=[], sample_input=[], title_text_in = '',
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
                   
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
    box_size = data.metadata.boxsize[0]


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
    
    
    #-----------------
    # Spearman with gas mass
    res = scipy.stats.spearmanr(kappa_stars[mask_h2], H2_mass[mask_h2]) 
    print('\nSpearman incl. FR:    kappa_stars - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    res = scipy.stats.spearmanr(kappa_stars[mask_h2_SRs], H2_mass[mask_h2_SRs])
    print('Spearman excl. FR:    kappa_stars - h2 rank: %.3f    p-value:   %.3e' %(res.statistic, res.pvalue))
    
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = kappa_stars
        C_VALUE_1 = np.zeros(len(np.log10(stellar_mass)))
        C_VALUE_1[H2_mass.value > 0] = np.log10(H2_mass[H2_mass.value > 0])
        #S_VALUE_1 = (np.log10(H2_mass)-(np.log10(h2_detection_limit)-1))**2.5
    
    
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
        ### ETG (excl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_1 = np.log10(stellar_mass[mask_SRs])
        Y_MEDIAN_1 = kappa_stars[mask_SRs]
        
        ### ETG (incl. FR) sample
        # Feed median, gas is found under H2_mass
        X_MEDIAN_2 = np.log10(stellar_mass)
        Y_MEDIAN_2 = kappa_stars
        
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
            #y_bin = y_bin[H2_mass[mask_SRs][mask].value > h2_detection_limit]
            
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
                
        axs.plot(np.flip(bin_centers), np.flip(medians), color='C0',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C0', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
        
        
        #------------------------------------------    
        # Compute statistics in each bin
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
        
        axs.plot(np.flip(bin_centers), np.flip(medians), color='C1',   ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(medians_masked), color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(lower_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
        axs.plot(np.flip(bin_centers), np.flip(upper_1sigma_masked), color='C1', linewidth=0.6, ls=(5, (10, 3)), zorder=20, path_effects=[outline2], alpha=0.9)
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
        hist_bins = np.arange(0, 1, bin_width)  # Binning edges
        
        #------------------------------------------
        ### ETG (excl. FR) sample
        bin_n, _           = np.histogram(Y_VALUE_1[mask_SRs], bins=hist_bins)
        bin_n_detected, _  = np.histogram(Y_VALUE_1[mask_h2_SRs], bins=hist_bins)
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
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
        ax_hist.stairs(bin_f_detected, hist_bins[mask_positive_n_extra], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        
    
        
        
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
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs',
                  'all_ETGs_centrals': 'ETGs, centrals',
                  'all_ETGs_satellites': 'ETGs, satellites',
                  'all_ETGs_cluster': 'ETGs, cluster',
                  'all_ETGs_groupfield': 'ETGs, group/field',
                  'all_ETGs_plus_redspiral': "ETGs",
                  'all_ETGs_plus_redspiral_centrals': "ETGs, centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs, satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs, cluster',
                  'all_ETGs_plus_redspiral_cluster_centrals': 'ETGs, cluster centrals',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs, group/field',
                  'all_LTGs': 'LTGs',
                  'all_LTGs_excl_redspiral': 'LTGs'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
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
    
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})

    
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
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5, handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            handles = [handles[1], handles[0]]
            labels = [labels[1], labels[0]]
            if add_observational:
                handles.append(handles[-1])
                labels.append(labels[-1])
            axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.3, markerscale=1.5)
        if add_detection_hist:
            if add_observational:
                ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=True)
    if add_detection_hist and not add_median_line:
        axs.plot([0,1], [-20,-21], color='C0', linewidth=1, label='excl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        axs.plot([0,1], [-20,-21], color='C1', linewidth=1, label='incl. FRs', zorder=20, path_effects=[outline], alpha=0.9)
        handles, labels = axs.get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = [labels[1], labels[0]]
        axs.legend(handles, labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, handlelength=0.8, markerfirst=True)
        
    
            
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_kappaco/%s_%s_%s_Mstar_kappaco_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_kappaco/%s_%s_%s_Mstar_kappaco_%s_H2ap%s_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, scatter_or_hexbin, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()























#===================================================================================
### Load a sample from a given snapshot and a given run
sim_box_size_name = 'L100_m6'
sim_type_name     = 'HYBRID_AGN_m6'    # THERMAL_AGN_m6    HYBRID_AGN_m6
snapshot_name     = '127'


### Load ETGs w/o FRs
#soap_indicies_sample_all_ETGs,                           _, sample_input_all_ETGs                           = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs'%(sim_box_size_name, sim_type_name, snapshot_name))   
#soap_indicies_sample_all_ETGs_centrals,                  _, sample_input_all_ETGs_centrals                  = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_centrals'%(sim_box_size_name, sim_type_name, snapshot_name))  
#satellites
#soap_indicies_sample_all_ETGs_cluster,                   _, sample_input_all_ETGs_cluster                   = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_cluster'%(sim_box_size_name, sim_type_name, snapshot_name))   
#soap_indicies_sample_all_ETGs_groupfield,                _, sample_input_all_ETGs_groupfield                = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_groupfield'%(sim_box_size_name, sim_type_name, snapshot_name))  

### Load LTGs
#soap_indicies_sample_all_LTGs,                           _, sample_input_all_LTGs                           = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_LTGs'%(sim_box_size_name, sim_type_name, snapshot_name))   
#centrals
#satellites


### Load ETGs incl. FRs
soap_indicies_sample_all_ETGs_plus_redspiral,            _, sample_input_all_ETGs_plus_redspiral            = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral'%(sim_box_size_name, sim_type_name, snapshot_name))  
soap_indicies_sample_all_ETGs_plus_redspiral_centrals,   _, sample_input_all_ETGs_plus_redspiral_centrals   = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_centrals'%(sim_box_size_name, sim_type_name, snapshot_name))  
soap_indicies_sample_all_ETGs_plus_redspiral_satellites, _, sample_input_all_ETGs_plus_redspiral_satellites   = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_satellites'%(sim_box_size_name, sim_type_name, snapshot_name))  
soap_indicies_sample_all_ETGs_plus_redspiral_cluster,    _, sample_input_all_ETGs_plus_redspiral_cluster    = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_cluster'%(sim_box_size_name, sim_type_name, snapshot_name)) 
soap_indicies_sample_all_ETGs_plus_redspiral_cluster_centrals,    _, sample_input_all_ETGs_plus_redspiral_cluster_centrals    = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_cluster_centrals'%(sim_box_size_name, sim_type_name, snapshot_name)) 
soap_indicies_sample_all_ETGs_plus_redspiral_groupfield, _, sample_input_all_ETGs_plus_redspiral_groupfield = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_groupfield'%(sim_box_size_name, sim_type_name, snapshot_name)) 

### Load LTGs excl. FRs
soap_indicies_sample_all_LTGs_excl_redspiral,            _, sample_input_all_LTGs_excl_redspiral            = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_LTGs_excl_redspiral'%(sim_box_size_name, sim_type_name, snapshot_name))  
#centrals
#satellites                                                                                                                                                         
#===================================================================================





#===================================
### H1 plots:   H1 mass, fraction, and environment
# Plot stelmass - H1 mass
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_ETGs_plus_redspiral_satellites, soap_indicies_sample_all_ETGs_plus_redspiral_groupfield, soap_indicies_sample_all_ETGs_plus_redspiral_cluster, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_ETGs_plus_redspiral_satellites, sample_input_all_ETGs_plus_redspiral_groupfield, sample_input_all_ETGs_plus_redspiral_cluster, sample_input_all_LTGs_excl_redspiral]):
    _etg_stelmass_h1mass(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_count',         # [ scatter_old / scatter_new / hexbin_count / hexbin_H1 ]
                          add_median_line      = True,
                          print_fdet = True,
                          print_typical_galaxy = True,
                        savefig       = True)"""
#------------
# Plot stelmass - H1 mass fraction (H1 / H1 + M*)
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_ETGs_plus_redspiral_satellites, soap_indicies_sample_all_ETGs_plus_redspiral_groupfield, soap_indicies_sample_all_ETGs_plus_redspiral_cluster, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_ETGs_plus_redspiral_satellites, sample_input_all_ETGs_plus_redspiral_groupfield, sample_input_all_ETGs_plus_redspiral_cluster, sample_input_all_LTGs_excl_redspiral]):
    _etg_stelmass_h1massfraction(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H1',         # [ scatter_old / scatter_new / hexbin_count / hexbin_H1 ]
                          add_median_line      = True,
                          print_fdet = True,
                        savefig       = True)"""
# Plots stelmass - H1 mass and H1 fraction as double plot of above
"""_etg_stelmass_h1mass_double(soap_indicies_sample=soap_indicies_sample_all_ETGs_plus_redspiral, sample_input=sample_input_all_ETGs_plus_redspiral,
                          add_median_line      = True,
                        savefig       = True)"""

     

#==========================================
###     H2 plots: H2 mass, fraction
# Plot stelmass - H2 mass
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_ETGs_plus_redspiral_satellites, soap_indicies_sample_all_ETGs_plus_redspiral_groupfield, soap_indicies_sample_all_ETGs_plus_redspiral_cluster, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_ETGs_plus_redspiral_satellites, sample_input_all_ETGs_plus_redspiral_groupfield, sample_input_all_ETGs_plus_redspiral_cluster, sample_input_all_LTGs_excl_redspiral]):
    _etg_stelmass_h2mass(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_count',         # [ scatter_old / scatter_new / hexbin_count / hexbin_H2 ]
                          add_median_line      = True,
                          print_fdet = True,
                          print_typical_galaxy = True,
                        savefig       = True)"""
#------------
# Plot stelmass - H2 mass fraction (H2 / H2 + M*)
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_ETGs_plus_redspiral_satellites, soap_indicies_sample_all_ETGs_plus_redspiral_groupfield, soap_indicies_sample_all_ETGs_plus_redspiral_cluster, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_ETGs_plus_redspiral_satellites, sample_input_all_ETGs_plus_redspiral_groupfield, sample_input_all_ETGs_plus_redspiral_cluster, sample_input_all_LTGs_excl_redspiral]):
    _etg_stelmass_h2massfraction(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_new / hexbin_count / hexbin_H2 ]
                          add_median_line      = True,
                          print_fdet = True,
                        savefig       = True)"""
# Plots stelmass - H2 mass and H2 fraction as double plot of above
"""_etg_stelmass_h2mass_double(soap_indicies_sample=soap_indicies_sample_all_ETGs_plus_redspiral, sample_input=sample_input_all_ETGs_plus_redspiral,
                          add_median_line      = True,
                        savefig       = True)"""
#------------
# Plot fdet with H2_aperture as an errorplot line plot. DOESNT USE LOADED SAMPLES FROM ABOVE
"""_aperture_fdet(sim_box_size_name_1 = 'L200_m6', sim_type_name_1     = 'THERMAL_AGN_m6',
               sim_box_size_name_2 = 'L100_m6', sim_type_name_2     = 'HYBRID_AGN_m6',
                        aperture_h2_list = ['exclusive_sphere_3kpc', 'exclusive_sphere_10kpc', 'exclusive_sphere_30kpc', 'exclusive_sphere_50kpc'],        # list of apertures to consider
                        h2_detection_limit = 10**7,
                        savefig       = True)"""


#==========================================
###     H2 plots: environment
# Plot M200c mass - H2 mass
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_m200c_h2mass(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_count',         # [ scatter_old / scatter_new / hexbin_count / hexbin_H2 ]
                        add_median_line = True,
                        savefig       = True)"""
#------------
# Plot stelmass - M200c mass
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_m200c(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
    


#==========================================
###      H2 plots: u-r, SFR, sSFR, SFE
# Plot stelmass - u-r, coloured by kappa to highlight the sample of included FRs in L100
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'scatter_kappa',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
# Plot stelmass - u-r, hexbin
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                          add_detection_hist = True,
                          add_median_line = True,
                        savefig       = True)"""
#------------
# Plot stelmass - SFR
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_sfr(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
#------------
# Plot stelmass - sSFR
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_ssfr(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
#------------
# Plot stelmass - t_dep = H2 / SFR = 1 / SFE
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_LTGs_excl_redspiral]):
    _etg_stelmass_tdep(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""



#==========================================
###     ETG H2 plots: H2 morph and extent
# Plot stelmass - r50
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_r50(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
# Plot stelmass - r50 HI, detected only
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_r50H1(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H1',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H1 ]
                        savefig       = True)"""
# Plot stelmass - r50 H2, detected only
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_r50H2(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
#------------
# Plot stelmass - r50 H2 / r50
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_LTGs_excl_redspiral]):
    _etg_stelmass_r50r50H2(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""

#------------
# Plot r50 - r50 HI, detected only
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_r50_r50H1(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H1',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)
# Plot r50 - r50 H2, detected only
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_LTGs_excl_redspiral]):
    _etg_r50_r50H2(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
# Plot r50 HI - r50 H2, detected only
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_r50H1_r50H2(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
#------------
# Plot r50 - H2 mass
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_LTGs_excl_redspiral]):
    _etg_r50_h2mass(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_count',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
#------------
# Plot H2 mass - r50 H2, detected only
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_LTGs_excl_redspiral]):
    _etg_h2mass_r50H2(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_count',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)"""
#------------
# Plot stelmass - sigma ratio = density within X / ring of density Y
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_LTGs_excl_redspiral]):
    for hole_gas_type_i in ['h2', 'h1', 'gas']: 
        _gas_surface_ratios(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                                scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                                gas_type = hole_gas_type_i,                    # [ 'h2' / 'h1' / 'gas' ]
                                  surfdens_aperture_1   = '3',    # Radius 1
                                  surfdens_aperture_2   = '50',    # Radius 2
                                savefig       = True)
    
        _gas_surface_ratios(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                                scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                                gas_type = hole_gas_type_i,                    # [ 'h2' / 'h1' / 'gas' ]
                                  surfdens_aperture_1   = '10',    # Radius 1
                                  surfdens_aperture_2   = '50',    # Radius 2
                                savefig       = True)"""
#------------                            
# Plot fraction with no gas particles in 10kpc and 50kpc w.r.t total sample fed. DOESNT USE LOADED SAMPLES FROM ABOVE
"""_aperture_fdet_missing_gas(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                              hole_aperture   = 'exclusive_sphere_50kpc',    # Radius within which to look for any gas particles
                            savefig       = True)
_aperture_fdet_missing_gas(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                              hole_aperture   = 'exclusive_sphere_50kpc',    # Radius within which to look for any gas particles
                              savefig_txt = 'centrals', 
                            savefig       = True)
_aperture_fdet_missing_gas(csv_samples = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                              hole_aperture   = 'exclusive_sphere_50kpc',    # Radius within which to look for any gas particles
                            savefig       = True)
_aperture_fdet_missing_gas(csv_samples = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies_centrals', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_centrals', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                              hole_aperture   = 'exclusive_sphere_50kpc',    # Radius within which to look for any gas particles
                              savefig_txt = 'centrals', 
                            savefig       = True)"""
                            
                            
                            
#==========================================
###     ETG H2 plots: galaxy morph with H2        
# Plot stelmass - ellip
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_ellip(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        use_projected = False,
                        savefig       = True)
    
    raise Exception('current pause 08yhoihlj')
# Plot stelmass - triax
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_triax(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        use_projected = False,
                        savefig       = True)
    
    raise Exception('current pause 08yhoihlj')
# Plot stelmass - kappaco stars
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_kappaco(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H2',         # [ scatter_old / scatter_kappa / scatter_new / hexbin_count / hexbin_H2 ]
                        savefig       = True)
    
    raise Exception('current pause 08yhoihlj')
                        

                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                            
                            


