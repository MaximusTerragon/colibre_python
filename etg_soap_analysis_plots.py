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
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
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
# Returns stelmass - u-r, coloured by H2 detection, size by H2 mass
def _etg_stelmass_u_r(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                   aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                   add_detection_hist = False,
                   h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
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
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]

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

    
    #-----------------
    # Scatter
    vmin = 0
    vmax = 0.8
    
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))

    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], mag_plot[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], mag_plot[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], mag_plot[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], mag_plot[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(0.5, 3.2, 0.1)  # Binning edges
        bin_n, _           = np.histogram(mag_plot, bins=hist_bins)
        bin_n_detected, _  = np.histogram(mag_plot[mask_h2], bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(0.8, 3.3)
    axs.set_xscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_aperture_h2 = {'exclusive_sphere_3kpc': '3 pkpc',
                        'exclusive_sphere_10kpc': '10 pkpc',
                        'exclusive_sphere_30kpc': '30 pkpc', 
                        'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$u^{*} - r^{*}$')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(0.8, 3.3)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    axs.text(0.05, 0.915, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    if add_observational:
        plt.text(10**11.9, 2.65, '↑ red sequence', fontsize=7, color='r', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
        plt.text(10**11.9, 2.4, '↓ blue cloud', fontsize=7, color='b', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')

    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_u_r/%s_%s_%s_young14_u-r_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_u_r/%s_%s_%s_young14_u-r_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()

#-----------------
# Returns soap aperture - detection rate for 3 samples
def _aperture_fdet(csv_samples = [], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                   aperture_h2_list = ['exclusive_sphere_3kpc', 'exclusive_sphere_10kpc', 'exclusive_sphere_30kpc', 'exclusive_sphere_50kpc'],        # list of apertures to consider
                   h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
    #---------------------------
    # Extract data from samples:
    define_dict_labels = True
    if define_dict_labels:
        dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                        'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                        'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                       'all_ETGs': 'ETGs',
                        'all_ETGs_centrals': 'ETGs',
                        'all_ETGs_satellites': 'ETGs',
                       'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)',
                        'all_ETGs_plus_redspiral_centrals': 'ETGs (incl. FRs)',
                        'all_ETGs_plus_redspiral_satellites': 'ETGs (incl. FRs)'}
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
        
        
        # Loop over desired H2 apertures and add
        aperture_x_list     = []
        fdet_list           = []
        fdet_err_lower_list = []
        fdet_err_upper_list = []
        for aperture_h2_i in aperture_h2_list:
            H2_mass = attrgetter('%s.%s'%(aperture_h2_i, 'molecular_hydrogen_mass'))(data)
            H2_mass.convert_to_units('Msun')
            H2_mass.convert_to_physical()
            H2_mass = H2_mass[soap_indicies_sample]
            
            
            #--------------
            # Masking LTGs (ETGs are mask_high_kappa):
            mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
            # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
            mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                  mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
            dict_sample_numbers = {'total': len(stellar_mass),
                                   'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                                 'ETG': len(stellar_mass[~mask_high_kappa])},
                                   'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                                     'ETG': len(stellar_mass[~mask_blue_LTG])}}
            #print('Total sample:  ', dict_sample_numbers['total'])
            #print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
            #print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
            #print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
            #print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
            
            
            #--------------
            # Find detection rate of sample and H2_aperture
            mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
            # Total sample
            f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
            f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
            f_detected_err_lower = f_detected - f_detected_err[0]
            f_detected_err_upper = f_detected_err[1] - f_detected
            # print
            if print_fdet:
                print('TOTAL H2 GAS DETECTION RATE in ETG sample:   %s    %s' %(title_text_in, aperture_h2_i))
                print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
                #answer_fdet = input("\n-----------------\nContinue? y/n? ")
                #if answer_fdet == 'n':
                #    raise Exception('Manual break at fdet')
                
            
            #--------------
            # Append fractions
            x_dict = {'exclusive_sphere_3kpc': 3, 
                      'exclusive_sphere_10kpc': 10,
                      'exclusive_sphere_30kpc': 30, 
                      'exclusive_sphere_50kpc': 50}
            aperture_x_list.append(x_dict[aperture_h2_i])
            fdet_list.append(f_detected)
            fdet_err_lower_list.append(f_detected_err_lower)
            fdet_err_upper_list.append(f_detected_err_upper)


        #------------
        # PLOT EACH LINE FOR EACH SAMPLE:
        # Plot several lines: total_sample, ETG, ETG+FR
        axs.errorbar(np.array(aperture_x_list), 100*np.array(fdet_list), yerr=[100*np.array(fdet_err_lower_list), 100*np.array(fdet_err_upper_list)], xerr=None, color=dict_colors[sample_input['name_of_preset']], ecolor=dict_colors[sample_input['name_of_preset']], ls=dict_ls[sample_input['name_of_preset']], capsize=2, lw=1.0, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7, label=dict_labels[sample_input['name_of_preset']])
        
    
    #-----------
    # Axis formatting
    axs.set_xlim(0, 55)
    axs.set_ylim(0, 100)
    axs.minorticks_on()
    #axs.tick_params(axis='x', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    axs.set_xlabel(r'SOAP aperture radius [pkpc]')
    axs.set_ylabel('Detection rate among sub-sample\n' + r'($\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}>7.0$) [\%]')
    
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    axs.text(0.05, 0.915, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    axs.set_title(r'%s' %(title_text_in), size=7, loc='left', pad=3)
    

    #-----------
    # Legend
    axs.legend(loc='lower right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/aperture_fdet/%s_%s_%s_fdet_h2aperture_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/aperture_fdet/%s_%s_%s_yfdet_h2aperture_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - hole detection rate, binned by M*
def _aperture_fdet_rings(csv_samples = [], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture    = 'exclusive_sphere_50kpc', 
                   aperture_h2 = 'exclusive_sphere_50kpc',        
                   h2_detection_limit = 10**7,
                   h2_hole_aperture   = '',                         # Radius within which to look for H2 hole
                     h2_hole_limit      = 10**6,                    # ^ must have less than this H2 mass within this aperture
                   #---------------
                   print_fdet         = False,
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
    #---------------------------
    # Extract data from samples:
    define_dict_labels = True
    if define_dict_labels:
        dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                        'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                        'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                       'all_ETGs': 'ETGs',
                        'all_ETGs_centrals': 'ETGs',
                        'all_ETGs_satellites': 'ETGs',
                       'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)',
                        'all_ETGs_plus_redspiral_centrals': 'ETGs (incl. FRs)',
                        'all_ETGs_plus_redspiral_satellites': 'ETGs (incl. FRs)'}
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
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        stellar_mass = stellar_mass[soap_indicies_sample]
        
        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
        H2_mass.convert_to_units('Msun')
        H2_mass.convert_to_physical()
        H2_mass = H2_mass[soap_indicies_sample]
        
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
        
        
        #======================
        # Select apertures with holes
        H2_mass_hole = attrgetter('%s.%s'%(h2_hole_aperture, 'molecular_hydrogen_mass'))(data)
        H2_mass_hole.convert_to_units('Msun')
        H2_mass_hole = H2_mass_hole[soap_indicies_sample]
        

        #--------------
        # Masking LTGs (ETGs are mask_high_kappa):
        mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
        # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
        mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
        dict_sample_numbers = {'total': len(stellar_mass),
                               'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                             'ETG': len(stellar_mass[~mask_high_kappa])},
                               'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                                 'ETG': len(stellar_mass[~mask_blue_LTG])}}
        #print('Total sample:  ', dict_sample_numbers['total'])
        #print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
        #print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
        #print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
        #print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
        
        #=============================
        # Of galaxies with H2_mass > h2_detection_limit, how many have holes with < h2_hole_limit within h2_hole_aperture_list
        # Find detection rate of sample and H2_aperture
        mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
        # Find detection rate of sample and H2_aperture, but with holes
        mask_h2_hole = H2_mass_hole < ((h2_hole_limit)*u.Msun)
        mask_h2_det_holes = np.logical_and.reduce([mask_h2, mask_h2_hole]).squeeze()
        
        # Fraction of those with H2 with rings
        f_detected_hole = len(stellar_mass[mask_h2_det_holes])/len(stellar_mass[mask_h2]) 
        f_detected_hole_err = binom_conf_interval(k=len(stellar_mass[mask_h2_det_holes]), n=len(stellar_mass[mask_h2]), confidence_level= 0.68269, interval='jeffreys')
        f_detected_hole_err_lower = f_detected_hole - f_detected_hole_err[0]
        f_detected_hole_err_upper = f_detected_hole_err[1] - f_detected_hole
        # print
        if print_fdet:
            print('FRACTION OF H2-DETECTED WITH H2 HOLES:   %s    %s' %(title_text_in, h2_hole_aperture))
            print('  >  f_det_hole:    %.3f (-%.3f + %.3f)'%(f_detected_hole, f_detected_hole_err_lower, f_detected_hole_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
        
        #--------------
        # Of galaxies with H2_mass > h2_detection_limit, how many have holes with < h2_hole_limit within h2_hole_aperture_list
        # Bin data by stellar mass, find fraction within stellar bin and append
        bin_step = 0.2
        hist_bins = np.arange(9.5, 13, bin_step)  # Binning edges
        bin_n, _           = np.histogram(np.log10(stellar_mass[mask_h2]), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(stellar_mass[mask_h2_det_holes]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
    
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        
        #------------------
        # Plot bar of fraction
        #axs.step((hist_bins[1:][mask_positive_n]), , where='pre', color=dict_colors[sample_input['name_of_preset']], linewidth=1.0, label=dict_labels[sample_input['name_of_preset']])
        axs.stairs(np.append(100*bin_f_detected, 0), edges=np.insert(np.append(hist_bins[1:][mask_positive_n], hist_bins[1:][mask_positive_n][-1]+bin_step), 0, hist_bins[1:][mask_positive_n][0]-bin_step, axis=0) , orientation='vertical', baseline=0, fill=False, color=dict_colors[sample_input['name_of_preset']], linewidth=1.0, label=dict_labels[sample_input['name_of_preset']])
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        axs.errorbar(hist_bins_midpoint, 100*bin_f_detected, xerr=None, yerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], ecolor=dict_colors[sample_input['name_of_preset']], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
         
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12)
    axs.set_ylim(0, 100)
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
    axs.set_ylabel('Incidence of H$_{2}$ holes among\ndetected sub-sample [\%]')

    
    #-----------  
    # Annotations
    axs.text(0.80, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.915, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    axs.set_title(r'%s' %(title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    axs.legend(loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/h2_rings_fdet/%s_%s_%s_fdet_h2holes%s_%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], np.log10(h2_hole_limit), h2_hole_aperture, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/h2_rings_fdet/%s_%s_%s_fdet_h2holes%s_%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], np.log10(h2_hole_limit), h2_hole_aperture, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - SFR, coloured by H2 detection, size by H2 mass
def _etg_stelmass_sfr(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra / Calculated values
    sfr50 = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)
    sfr50.convert_to_units(u.Msun/u.yr)
    sfr50.convert_to_physical()
    sfr50 = sfr50[soap_indicies_sample]
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], sfr50[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], sfr50[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], sfr50[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], sfr50[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(-2, 5, 0.25)  # Binning edges
        bin_n, _           = np.histogram(np.log10(sfr50), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(sfr50[mask_h2]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(10**((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(10**(hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, 10**hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(10**-2, 10**5)
    axs.set_xscale("log")
    axs.set_yscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'SFR [M$_{\odot}$ yr$^{-1}$]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(10**-2, 10**5)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_SFR/%s_%s_%s_Mstar_SFR_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_SFR/%s_%s_%s_Mstar_SFR_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - sSFR, coloured by H2 detection, size by H2 mass
def _etg_stelmass_ssfr(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra / Calculated values
    sfr50 = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)
    sfr50.convert_to_units(u.Msun/u.yr)
    sfr50.convert_to_physical()
    sfr50 = sfr50[soap_indicies_sample]
    
    ssfr50 = np.divide(sfr50, stellar_mass)
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], ssfr50[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], ssfr50[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], ssfr50[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], ssfr50[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(-13, -8, 0.25)  # Binning edges
        bin_n, _           = np.histogram(np.log10(ssfr50), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(ssfr50[mask_h2]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(10**((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(10**(hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, 10**hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(10**-13, 10**-8)
    axs.set_xscale("log")
    axs.set_yscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'sSFR [yr$^{-1}$]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(10**-13, 10**-8)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_sSFR/%s_%s_%s_Mstar_sSFR_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_sSFR/%s_%s_%s_Mstar_sSFR_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - H2 mass, coloured by H2 detection, size by H2 mass
def _etg_stelmass_h2mass(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
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
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Calculated values
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], H2_mass[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], H2_mass[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], H2_mass[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], H2_mass[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(5, 11, 0.2)  # Binning edges
        bin_n, _           = np.histogram(np.log10(H2_mass), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(H2_mass[mask_h2]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(10**((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(10**(hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, 10**hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(10**6, 10**11)
    axs.set_xscale("log")
    axs.set_yscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$M_{\mathrm{H_{2}}}$ [M$_{\odot}$]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(10**6, 10**11)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_MH2/%s_%s_%s_Mstar_MH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_MH2/%s_%s_%s_Mstar_MH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - H2 mass fraction ( H2 / H2 + M* ), coloured by H2 detection, size by H2 mass
def _etg_stelmass_h2massfraction(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
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
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #======================
    # Calculated values
    H2_mass_fraction = np.divide(H2_mass, H2_mass + stellar_mass)
    

    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], H2_mass_fraction[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], H2_mass_fraction[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], H2_mass_fraction[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], H2_mass_fraction[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(-7, 0, 0.2)  # Binning edges
        bin_n, _           = np.histogram(np.log10(H2_mass_fraction), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(H2_mass_fraction[mask_h2]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(10**((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(10**(hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, 10**hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(10**-7, 10**0)
    axs.set_xscale("log")
    axs.set_yscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$f_{\mathrm{H_{2}}}$')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(10**7, 10**0)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_fracH2/%s_%s_%s_Mstar_fracH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_fracH2/%s_%s_%s_Mstar_fracH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - m200c, coloured by H2 detection, size by H2 mass
def _etg_stelmass_m200c(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra / Calculated values
    m200c = data.spherical_overdensity_200_crit.total_mass
    m200c.convert_to_units('Msun')
    m200c.convert_to_physical()
    m200c = m200c[soap_indicies_sample]
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], m200c[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], m200c[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], m200c[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], m200c[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(10, 15, 0.25)  # Binning edges
        bin_n, _           = np.histogram(np.log10(m200c), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(m200c[mask_h2]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(10**((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(10**(hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, 10**hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(10**10, 10**15)
    axs.set_xscale("log")
    axs.set_yscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$M_{\mathrm{200c}}$ [M$_{\odot}$]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(10**10, 10**15)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_M200c/%s_%s_%s_Mstar_M200c_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_M200c/%s_%s_%s_Mstar_M200c_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - r50, coloured by H2 detection, size by H2 mass
def _etg_stelmass_r50(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra / Calculated values
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    r50 = r50[soap_indicies_sample]
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], r50[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], r50[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], r50[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], r50[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(0, 20, 0.25)  # Binning edges
        bin_n, _           = np.histogram(r50, bins=hist_bins)
        bin_n_detected, _  = np.histogram(r50[mask_h2], bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(0, 20)
    axs.set_xscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$r_{\mathrm{50}}$ [pkpc]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(0, 20)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_r50/%s_%s_%s_Mstar_r50_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_r50/%s_%s_%s_Mstar_r50_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - r50_H2, coloured by H2 detection, size by H2 mass
def _etg_stelmass_r50H2(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra / Calculated values
    
    raise Exception('currently not added r50H2 from SOAP')
    
    r50H2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)
    r50H2.convert_to_units('kpc')
    r50H2.convert_to_physical()
    r50H2 = r50H2[soap_indicies_sample]
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], r50H2[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], r50H2[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], r50H2[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], r50H2[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(0, 20, 0.25)  # Binning edges
        bin_n, _           = np.histogram(r50H2, bins=hist_bins)
        bin_n_detected, _  = np.histogram(r50H2[mask_h2], bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(0, 20)
    axs.set_xscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$r_{\mathrm{50,H_{2}}}$ [pkpc]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(0, 20)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_r50H2/%s_%s_%s_Mstar_r50H2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_r50H2/%s_%s_%s_Mstar_r50H2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - ellipticity, coloured by H2 detection, size by H2 mass
def _etg_stelmass_ellip(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra values / Calculated values
    
    # Calculate intrinsic triax and ellipticity:
    print('@@@@ using non-iterative ellipticity, replace when SOAP updates')
    def _compute_intrinsic_ellipticity_triaxiality():
        # construct inertia tensor
        i11, i22, i33, i12, i13, i23 = attrgetter('%s.%s'%(aperture, 'stellar_inertia_tensor_noniterative'))(data)
        
        
        print('raw inertia:')
        print(attrgetter('%s.%s'%(aperture, 'stellar_inertia_tensor_noniterative'))(data))
        print(' ')
        print(' i11:')
        print(i11)
        
        inertiatensor = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])
        
        print(' ')
        print(' inertiatensor:')
        print(inertiatensor)
        
        raise Exception('current break, check above tensor^')
    
        # Compute eigenvalues (sorted in descending order)
        eigenvalues = np.linalg.eigvalsh(inertiatensor)[::-1]  # Sorted: I1 >= I2 >= I3
        length_a2, length_b2, length_c2 = eigenvalues  # Assign to principal moments, length_a2 == a**2, so length = sqrt length_a2. a = longest axis, b = intermediate, c = shortest
        #print(length_a2)
        #print(length_b2)
        #print(length_c2)
    
        # Compute Triaxiality Parameter
        triaxiality = (length_a2 - length_b2) / (length_a2 - length_c2) if (length_a2 - length_c2) != 0 else np.nan  # Avoid division by zero
        ellipticity = 1 - (np.sqrt(length_c2)/np.sqrt(length_a2))
        
        return cosmo_quantity(ellipticity, u.dimensionless, comoving=True, scale_factor=sg.metadata.a, scale_exponent=0), cosmo_quantity(triaxiality, u.dimensionless, comoving=True, scale_factor=sg.metadata.a, scale_exponent=0)     
    
    # remove 'noniterative' from above when L25 box  
    ellip, triax            = _compute_intrinsic_ellipticity_triaxiality()
    
    
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], ellip[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], ellip[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], ellip[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], ellip[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(0, 1.1, 0.05)  # Binning edges
        bin_n, _           = np.histogram(ellip, bins=hist_bins)
        bin_n_detected, _  = np.histogram(ellip[mask_h2], bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(0, 1)
    axs.set_xscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$\epsilon$')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(0, 1)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_ellip/%s_%s_%s_Mstar_ellip_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_ellip/%s_%s_%s_Mstar_ellip_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - triaxiality, coloured by H2 detection, size by H2 mass
def _etg_stelmass_triax(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra values / Calculated values
    
    # Calculate intrinsic triax and ellipticity:
    print('@@@@ using non-iterative ellipticity, replace when SOAP updates')
    def _compute_intrinsic_ellipticity_triaxiality():
        # construct inertia tensor
        i11, i22, i33, i12, i13, i23 = attrgetter('%s.%s'%(aperture, 'stellar_inertia_tensor_noniterative'))(data)
        
        
        print('raw inertia:')
        print(attrgetter('%s.%s'%(aperture, 'stellar_inertia_tensor_noniterative'))(data))
        print(' ')
        print(' i11:')
        print(i11)
        
        inertiatensor = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])
        
        print(' ')
        print(' inertiatensor:')
        print(inertiatensor)
        
        raise Exception('current break, check above tensor^')
    
        # Compute eigenvalues (sorted in descending order)
        eigenvalues = np.linalg.eigvalsh(inertiatensor)[::-1]  # Sorted: I1 >= I2 >= I3
        length_a2, length_b2, length_c2 = eigenvalues  # Assign to principal moments, length_a2 == a**2, so length = sqrt length_a2. a = longest axis, b = intermediate, c = shortest
        #print(length_a2)
        #print(length_b2)
        #print(length_c2)
    
        # Compute Triaxiality Parameter
        triaxiality = (length_a2 - length_b2) / (length_a2 - length_c2) if (length_a2 - length_c2) != 0 else np.nan  # Avoid division by zero
        ellipticity = 1 - (np.sqrt(length_c2)/np.sqrt(length_a2))
        
        return cosmo_quantity(ellipticity, u.dimensionless, comoving=True, scale_factor=sg.metadata.a, scale_exponent=0), cosmo_quantity(triaxiality, u.dimensionless, comoving=True, scale_factor=sg.metadata.a, scale_exponent=0)     
    
    # remove 'noniterative' from above when L25 box  
    ellip, triax            = _compute_intrinsic_ellipticity_triaxiality()
    
    
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], triax[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], triax[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], triax[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], triax[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(0, 1.1, 0.05)  # Binning edges
        bin_n, _           = np.histogram(triax, bins=hist_bins)
        bin_n_detected, _  = np.histogram(triax[mask_h2], bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(0, 1)
    axs.set_xscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'Triaxiality')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(0, 1)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_triax/%s_%s_%s_Mstar_triax_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_triax/%s_%s_%s_Mstar_triax_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - stellar kappa co, coloured by H2 detection, size by H2 mass
def _etg_stelmass_kappaco(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra values / Calculated values
    
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], kappa_stars[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], kappa_stars[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], kappa_stars[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], kappa_stars[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(0, 1.1, 0.05)  # Binning edges
        bin_n, _           = np.histogram(kappa_stars, bins=hist_bins)
        bin_n_detected, _  = np.histogram(kappa_stars[mask_h2], bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(0, 1)
    axs.set_xscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$\kappa_{\mathrm{co}}^{*}$')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(0, 1)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_kappaco/%s_%s_%s_Mstar_kappaco_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_kappaco/%s_%s_%s_Mstar_kappaco_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - veldisp (in 50 kpc, not r50), coloured by H2 detection, size by H2 mass
def _etg_stelmass_veldisp(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra / Calculated values
    
    # Stellar velocity dispersion and dispersion parameter from Davis+19
    def _compute_velocity_dispersion(aperture_veldisp='50'):        # km/s
        # construct inertia tensor
        i11, i22, i33, i12, i13, i23 = attrgetter('%s.%s'%(aperture, 'stellar_inertia_tensor_noniterative'))(data)
        
        stellar_vel_disp_matrix = (attrgetter('exclusive_sphere_%skpc.stellar_velocity_dispersion_matrix'%(aperture_veldisp))(data))
        stellar_vel_disp_matrix.convert_to_units(u.km**2 / u.s**2)
        stellar_vel_disp_matrix.convert_to_physical()
        stellar_vel_disp = np.sqrt((stellar_vel_disp_matrix[0][0] + stellar_vel_disp_matrix[0][1] + stellar_vel_disp_matrix[0][2])/3)
        
        return stellar_vel_disp
    stellar_vel_disp50 = _compute_velocity_dispersion(aperture_veldisp='50')
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], stellar_vel_disp50[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], stellar_vel_disp50[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], stellar_vel_disp50[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], stellar_vel_disp50[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        hist_bins = np.arange(1, 3, 0.1)  # Binning edges
        bin_n, _           = np.histogram(np.log10(stellar_vel_disp50), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(stellar_vel_disp50[mask_h2]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(10**((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(10**(hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, 10**hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(10, 1000)
    axs.set_xscale("log")
    axs.set_yscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$\sigma _{\mathrm{*}}$ [km s$^{-1}$]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(10, 1000)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_veldisp/%s_%s_%s_Mstar_veldisp_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_veldisp/%s_%s_%s_Mstar_veldisp_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - stellar specific angular momentum (in 50 kpc, not r50), coloured by H2 detection, size by H2 mass
def _etg_stelmass_lstar(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   add_kappa_colourbar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   #---------------
                   print_fdet         = False,
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
    if not add_detection_hist:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig = plt.figure(figsize=(10/3, 2.5))
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    #-----------------
    # Add SOAP data
    simulation_run  = sample_input['simulation_run']
    simulation_type = sample_input['simulation_type']
    snapshot_no     = sample_input['snapshot_no']
    simulation_dir  = sample_input['simulation_dir']
    soap_catalogue_file = sample_input['soap_catalogue_file']
    data = sw.load(f'%s'%halo_catalogue_file)

    # Get metadata from file
    z = data.metadata.redshift
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0]


    #-------------------------------
    # Get essential SOAP data for analysis
    stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
    stellar_mass.convert_to_units('Msun')
    stellar_mass.convert_to_physical()
    stellar_mass = stellar_mass[soap_indicies_sample]
    
    H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)
    H2_mass.convert_to_units('Msun')
    H2_mass.convert_to_physical()
    H2_mass = H2_mass[soap_indicies_sample]
    
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
    
    #================================
    # Extra / Calculated values
    
    L_stars = attrgetter('%s.%s'%(aperture, 'angular_momentum_stars'))(data)
    L_stars.convert_to_units(u.Msun * u.km**2 / u.s)
    L_stars.convert_to_physical()
    specific_L_stars = L_stars / stellar_mass
    
    
    #--------------
    # Kappa and colourbar [not used]
    vmin = 0
    vmax = 0.8
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.seismic_r(np.linspace(0, 0.45, 128))
    colors2 = plt.cm.seismic_r(np.linspace(0.55, 1, 128))
    # combine them and build a new colormap
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)
    # Normalise colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mymap)         #cmap=cm.coolwarm)
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa])},
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG])}}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of LTG kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('Number of LTG kappa>0.4 and u-r<2:             %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    
    
    #---------------------
    # Plot 2 scatters: one for H2 detections and one for non-detections
    mask_h2 = H2_mass > ((h2_detection_limit)*u.Msun)
    if add_kappa_colourbar:
        sc_points = axs.scatter(stellar_mass[mask_h2], specific_L_stars[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], specific_L_stars[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(stellar_mass[mask_h2], specific_L_stars[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        axs.scatter(stellar_mass[~mask_h2], specific_L_stars[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        
        print('specific_L_stars:')
        print(specific_L_stars)
        print(min(np.log10(specific_L_stars)))
        print(max(np.log10(specific_L_stars)))
        
        raise Exception('current pause -establish units in above, and also add to ylim for graph and detection graph')
        
        hist_bins = np.arange(1, 3, 0.1)  # Binning edges
        bin_n, _           = np.histogram(np.log10(specific_L_stars), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(specific_L_stars[mask_h2]), bins=hist_bins)
        mask_positive_n     = bin_n > 0
        bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
        
        # returns a % upper and a % lower
        bin_f_detected_err = binom_conf_interval(k=bin_n_detected[mask_positive_n], n=bin_n[mask_positive_n], confidence_level= 0.68269, interval='jeffreys')
        bin_f_detected_err_lower = bin_f_detected - bin_f_detected_err[0]
        bin_f_detected_err_upper = bin_f_detected_err[1] - bin_f_detected
        
        # Total sample
        f_detected = len(stellar_mass[mask_h2])/len(stellar_mass) 
        f_detected_err = binom_conf_interval(k=len(stellar_mass[mask_h2]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower = f_detected - f_detected_err[0]
        f_detected_err_upper = f_detected_err[1] - f_detected
        
        
        # barh
        ax_hist.barh(10**((hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(10**(hist_bins[:-1])[mask_positive_n]), 100*bin_f_detected, height=0.1, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        ax_hist.axvline(f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(100*bin_f_detected, 10**hist_bins_midpoint, xerr=[100*bin_f_detected_err_lower, 100*bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('Gas detection rate bins in ETGs:')
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            print('--------------------------------')
            print('TOTAL GAS DETECTION RATE in ETG sample:')
            print('  >  f_det:    %.3f (-%.3f + %.3f)'%(f_detected, f_detected_err_lower, f_detected_err_upper))
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(10**9.5, 10**12)
    axs.set_ylim(10, 1000)
    axs.set_xscale("log")
    axs.set_yscale("log")
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
    axs.set_xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'$\sigma _{\mathrm{*}}$ [km s$^{-1}$]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 100)
        ax_hist.set_ylim(10, 1000)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r'Det. [\%]')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if add_kappa_colourbar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)"
                  }
    axs.set_title(r'%s%s' %(title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.1, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if add_kappa_colourbar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_lstar/%s_%s_%s_Mstar_lstars_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_lstar/%s_%s_%s_Mstar_lstars_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()




#===================================================================================
# Load a sample from a given snapshot
soap_indicies_sample_1, _, sample_input_1 = _load_soap_sample(sample_dir, csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs')   
soap_indicies_sample_2, _, sample_input_2 = _load_soap_sample(sample_dir, csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral')   
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample5450_all_galaxies_centrals
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample4621_all_galaxies_satellites
                                                                                   
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample1081_all_ETGs_centrals
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample1143_all_ETGs_satellites
                                                                                   
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral 
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample1531_all_ETGs_plus_redspiral_centrals
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample2514_all_ETGs_plus_redspiral_satellites
                                                                                   
#===================================================================================


#===================================
# Plot per Young+11 as just red and black pluses and circles, with hist bars for detection limit
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        aperture_h2 = 'exclusive_sphere_50kpc',
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)"""
# Plot with a lower detection limit of 10**6
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                            aperture_h2 = 'exclusive_sphere_50kpc',
                            h2_detection_limit = 10**6,
                            add_detection_hist = True,
                              print_fdet = True,
                             title_text_in = ' 10**6 thresh',
                             savefig_txt = 'test_lower_h2detection_thresh', 
                            savefig       = True)"""
# Plot with coloured kappa
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        aperture_h2 = 'exclusive_sphere_50kpc',
                        add_kappa_colourbar = True,
                        savefig       = True)"""
                    
# Plot similar to Young+14: M* vs u-r, size by H2 reservoir >107, but vary the H2 apertures 3 -> 10 -> 30 -> 50
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    # Plot as just red and black pluses and circles, with hist bars for detection limit - 10 kpc aperture
    print('\n##### 3 kpc aperture test:   %s' %sample_input_i['name_of_preset'])
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        aperture_h2 = 'exclusive_sphere_3kpc',
                        add_detection_hist = True,
                          print_fdet = True,
                         title_text_in = ' 3pkpc H2',
                        savefig       = True)
    print('\n##### 10 kpc aperture test:   %s' %sample_input_i['name_of_preset'])
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        aperture_h2 = 'exclusive_sphere_10kpc',
                        add_detection_hist = True,
                          print_fdet = True,
                         title_text_in = ' 10pkpc H2',
                        savefig       = True)
    print('\n##### 30 kpc aperture test:   %s' %sample_input_i['name_of_preset'])
    _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        aperture_h2 = 'exclusive_sphere_30kpc',
                        add_detection_hist = True,
                          print_fdet = True,
                         title_text_in = ' 30pkpc H2',
                        savefig       = True)"""

#---------------------
# Plot fdet with H2_aperture as an errorplot line plot. DOESNT USE LOADED SAMPLES FROM ABOVE
"""_aperture_fdet(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                        aperture_h2_list = ['exclusive_sphere_3kpc', 'exclusive_sphere_10kpc', 'exclusive_sphere_30kpc', 'exclusive_sphere_50kpc'],        # list of apertures to consider
                        h2_detection_limit = 10**7,
                          print_fdet = True,
                        savefig       = True)
_aperture_fdet(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample5450_all_galaxies_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample1081_all_ETGs_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample1531_all_ETGs_plus_redspiral_centrals'],
                        aperture_h2_list = ['exclusive_sphere_3kpc', 'exclusive_sphere_10kpc', 'exclusive_sphere_30kpc', 'exclusive_sphere_50kpc'],        # list of apertures to consider
                        h2_detection_limit = 10**7,
                          print_fdet = True,
                         title_text_in = 'centrals',
                         savefig_txt = '_centrals',
                        savefig       = True)  
_aperture_fdet(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample4621_all_galaxies_satellites', 'L100_m6_THERMAL_AGN_m6_127_sample1143_all_ETGs_satellites', 'L100_m6_THERMAL_AGN_m6_127_sample2514_all_ETGs_plus_redspiral_satellites'],
                        aperture_h2_list = ['exclusive_sphere_3kpc', 'exclusive_sphere_10kpc', 'exclusive_sphere_30kpc', 'exclusive_sphere_50kpc'],        # list of apertures to consider
                        h2_detection_limit = 10**7,
                          print_fdet = True,
                         title_text_in = 'satellites',
                         savefig_txt = '_satellite',
                        savefig       = True)  """
                        
#----------------------       
# Plot fraction with H2 holes in 3kpc and 10kpc w.r.t sample detected with H2 > h2_detection_limit     
"""_aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                        h2_detection_limit = 10**7,
                        h2_hole_aperture   = 'exclusive_sphere_3kpc',    # Radius within which to look for H2 hole
                          h2_hole_limit    = 10**6,
                          print_fdet = True,
                         title_text_in = 'Hole aperture: 3 pkpc',
                        savefig       = True)
_aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                        h2_detection_limit = 10**7,
                        h2_hole_aperture   = 'exclusive_sphere_10kpc',    # Radius within which to look for H2 hole
                          h2_hole_limit    = 10**6,
                          print_fdet = True,
                         title_text_in = 'Hole aperture: 10 pkpc',
                        savefig       = True)"""
# Plot fraction with H2 holes in 3kpc and 10kpc w.r.t sample detected with H2 > h2_detection_limit     
"""_aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample5450_all_galaxies_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample1081_all_ETGs_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample1531_all_ETGs_plus_redspiral_centrals'],
                        h2_detection_limit = 10**7,
                        h2_hole_aperture   = 'exclusive_sphere_3kpc',    # Radius within which to look for H2 hole
                          h2_hole_limit    = 10**6,
                          print_fdet = True,
                         title_text_in = 'Hole aperture: 3 pkpc, centrals',
                         savefig_txt = '_centrals',
                        savefig       = True)
_aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample5450_all_galaxies_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample1081_all_ETGs_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample1531_all_ETGs_plus_redspiral_centrals'],
                        h2_detection_limit = 10**7,
                        h2_hole_aperture   = 'exclusive_sphere_10kpc',    # Radius within which to look for H2 hole
                          h2_hole_limit    = 10**6,
                          print_fdet = True,
                         title_text_in = 'Hole aperture: 10 pkpc, centrals',
                         savefig_txt = '_centrals',
                        savefig       = True)"""


#===================================
# Star formation 
# Plot stelmass - SFR
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_sfr(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)


# Plot stelmass - sSFR
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_ssfr(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)


             
#===================================
# Masses
# Plot stelmass - H2 mass
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_h2mass(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)


#===================================
# Plot stelmass - H2 mass fraction (H2 / H2 + M*)
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_h2massfraction(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)


#===================================
# Plot stelmass - M200c mass
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_m200c(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)

                     
#===================================
# Stellar and H2 extents
# Plot stelmass - r50 stars
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_r50(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)
#-------------------- 
# Plot stelmass - r50 H2
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_r50(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)"""


#===================================
# Morphological proxies
# Plot stelmass - ellip stars
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_ellip(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)
#-------------------
# Plot stelmass - triaxiality stars
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_triax(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)

#-------------------
# Plot stelmass - kappa_co stars
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_kappaco(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)

#===================================
# Stellar kinematics 
# Plot stelmass - stellar velocity dispersion. (note we use 50 kpc aperture, observations use r50)
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_veldisp(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)
#-------------------
# Plot stelmass - stellar specific angular momentum. (note we use 50 kpc aperture, observations use r50)
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_1, soap_indicies_sample_2], [sample_input_1, sample_input_2]):
    _etg_stelmass_lstar(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                          print_fdet = True,
                        savefig       = True)



"""
Mstar_SFR
Mstar_sSFR
Mstar_mukin
r50_MH2
r50_r50H2
lstar_MH2
"""






