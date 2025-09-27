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
# Returns stelmass - H1 mass, coloured by H1 detection, size by H1 mass
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

            with h5py.File('%s/Cappellari2011_masses.hdf5'%obs_dir, 'r') as file:
                obs_names_2 = file['data/Galaxy/values'][:]
                obs_masses  = file['data/Mstar/values'][:] * u.Unit(file['data/Mstar/values'].attrs['units'])
            
            obs_names_1 = np.array(obs_names_1)
            obs_names_2 = np.array(obs_names_2)
            obs_y = obs_y.to('Msun')
            obs_masses = obs_masses.to('Msun')
            
            # Match galaxy names to get mass (log)
            obs_x = []
            for name_i in obs_names_1:
                mask_name = np.argwhere(name_i == obs_names_2).squeeze()
                obs_x.append(obs_masses[mask_name])
                
            obs_x = np.array(obs_x)
            assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
                        
            axs.scatter(obs_x, obs_y, marker='^', s=7, alpha=1, zorder=5, c='r', edgecolors='none', label='Serra+12')
        
        
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
# Returns stelmass - H2 mass fraction ( H1 / H1 + M* ), coloured by H2 detection, size by H1 mass
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
                
            
    
    #==========================================================
    # Plot scatter or hexbin
    
    # Define inputs
    with np.errstate(divide='ignore'):
        X_VALUE_1 = np.log10(stellar_mass)
        Y_VALUE_1 = np.log10(H1_mass_fraction)
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

            with h5py.File('%s/Cappellari2011_masses.hdf5'%obs_dir, 'r') as file:
                obs_names_2 = file['data/Galaxy/values'][:]
                obs_masses  = file['data/Mstar/values'][:] * u.Unit(file['data/Mstar/values'].attrs['units'])
            
            obs_names_1 = np.array(obs_names_1)
            obs_names_2 = np.array(obs_names_2)
            obs_HI = obs_HI.to('Msun')
            obs_masses = obs_masses.to('Msun')
            
            # Match galaxy names to get mass (log)
            obs_x = []
            for name_i in obs_names_1:
                mask_name = np.argwhere(name_i == obs_names_2).squeeze()
                obs_x.append(obs_masses[mask_name])
                
            obs_x = np.array(obs_x)
            obs_y = obs_HI - np.log10((10**obs_HI) + (10**obs_x))
            
            assert len(obs_x) == len(obs_y), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_y))
                        
            axs.scatter(obs_x, obs_y, marker='^', s=7, alpha=1, zorder=5, c='r', edgecolors='none', label='Serra+12')
        
        
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











#===================================================================================
### Load a sample from a given snapshot and a given run
sim_box_size_name = 'L100_m6'
sim_type_name     = 'THERMAL_AGN_m6'
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
# Plot stelmass - H1 mass fraction (H1 / H1 + M*)
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals, soap_indicies_sample_all_ETGs_plus_redspiral_satellites, soap_indicies_sample_all_ETGs_plus_redspiral_groupfield, soap_indicies_sample_all_ETGs_plus_redspiral_cluster, soap_indicies_sample_all_LTGs_excl_redspiral], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals, sample_input_all_ETGs_plus_redspiral_satellites, sample_input_all_ETGs_plus_redspiral_groupfield, sample_input_all_ETGs_plus_redspiral_cluster, sample_input_all_LTGs_excl_redspiral]):
    _etg_stelmass_h1massfraction(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        scatter_or_hexbin       = 'hexbin_H1',         # [ scatter_old / scatter_new / hexbin_count / hexbin_H1 ]
                          add_median_line      = True,
                          print_fdet = True,
                        savefig       = True)
                        
    raise Exception('current break 98yhoi')
                        
                        

























