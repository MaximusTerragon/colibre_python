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
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
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



#   scp -r /Users/c22048063/Documents/COLIBRE/obs_data c22048063@physxlogin.astro.cf.ac.uk:/home/cosmos/c22048063/COLIBRE/
#   scp -r code/*.py c22048063@physxlogin.astro.cf.ac.uk:/home/user/c22048063/Documents/COLIBRE/code
#   scp -r c22048063@physxlogin.astro.cf.ac.uk:/home/cosmos/c22048063/COLIBRE/figures/etg_soap_analysis/Mstar_MH2 /Users/c22048063/Documents/COLIBRE/figures/etg_soap_analysis
#


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================
outline=mpe.withStroke(linewidth=1.6, foreground='black')







# Returns stelmass - H2 / H1 / gas hole detection rate, binned by M*
def _aperture_fdet_rings(csv_samples = [], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture    = 'exclusive_sphere_50kpc', 
                   aperture_h2 = 'exclusive_sphere_50kpc',     
                     h2_detection_limit = 10**7,
                   hole_gas_type = 'h2',                    # [ 'h2' / 'h1' / 'gas' ]
                     hole_aperture   = '',                         # Radius within which to look for H2 hole
                     hole_limit      = 10**6,                    # ^ must have less than this H2 mass within this aperture
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
                       'all_ETGs': 'ETGs (excl. FRs)',
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
        
        
        #======================
        # Select apertures with holes
        if hole_gas_type == 'h2':
            mass_hole = attrgetter('%s.%s'%(hole_aperture, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        elif hole_gas_type == 'h1':
            mass_hole = attrgetter('%s.%s'%(hole_aperture, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        elif hole_gas_type == 'gas':
            mass_hole = attrgetter('%s.%s'%(hole_aperture, 'gas_mass'))(data)[soap_indicies_sample]
        mass_hole.convert_to_units('Msun')
        

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
        mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        # Find detection rate of sample and H2_aperture, but with holes of different gas type
        mask_hole = mass_hole < ((hole_limit)*u.Msun)
        mask_det_holes = np.logical_and.reduce([mask_h2, mask_hole]).squeeze()
        
        # Fraction of those with H2 with rings
        f_detected_hole = len(stellar_mass[mask_det_holes])/len(stellar_mass[mask_h2]) 
        f_detected_hole_err = binom_conf_interval(k=len(stellar_mass[mask_det_holes]), n=len(stellar_mass[mask_h2]), confidence_level= 0.68269, interval='jeffreys')
        f_detected_hole_err_lower = f_detected_hole - f_detected_hole_err[0]
        f_detected_hole_err_upper = f_detected_hole_err[1] - f_detected_hole
        # print
        if print_fdet:
            print('FRACTION OF H2 > %.1e -DETECTED WITH %s HOLES:   %s    %s' %(h2_detection_limit, hole_gas_type, title_text_in, hole_aperture))
            print('  >  f_det_hole:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(f_detected_hole, f_detected_hole_err_lower, f_detected_hole_err_upper, len(stellar_mass[mask_det_holes]), len(stellar_mass[mask_h2]) ))    
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
        
        #--------------
        # Of galaxies with H2_mass > h2_detection_limit, how many have holes with < h2_hole_limit within h2_hole_aperture_list
        # Bin data by stellar mass, find fraction within stellar bin and append
        bin_step = 0.2
        hist_bins = np.arange(9.5, 13, bin_step)  # Binning edges
        bin_n, _           = np.histogram(np.log10(stellar_mass[mask_h2]), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(stellar_mass[mask_det_holes]), bins=hist_bins)
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
    axs.set_xlim(9.5, 12.5)
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
    dict_ylabel = {'h2': 'H$_{2}$',
                   'h1': 'H$_{\mathrm{I}}$',
                   'gas': 'gas'}
    axs.set_ylabel('Incidence of %s holes among\n$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ in bin '%dict_ylabel[hole_gas_type] + '$\:$[\%]')

    
    #-----------  
    # Annotations
    axs.text(0.80, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.915, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    axs.legend(loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/gas_holes_fdet/%s_%s_%s_fdet_%sholes%s_%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], hole_gas_type, np.log10(hole_limit), hole_aperture, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/gas_holes_fdet/%s_%s_%s_fdet_%sholes%s_%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], hole_gas_type, np.log10(hole_limit), hole_aperture, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns ratio of H2 surface density in 10 kpc / 50 kpc, 3 kpc / 50 kpc, and 3 kpc / 10 kpc
def _gas_surface_ratios(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture    = 'exclusive_sphere_50kpc', 
                   aperture_h2 = 'exclusive_sphere_50kpc',     
                     h2_detection_limit = 10**7,
                   gas_type = 'gas',                    # [ 'h2' / 'h1' / 'gas' ]
                     surfdens_aperture_1   = '',                         # Radius 1, smaller
                     surfdens_aperture_2   = '',                         # Radius 2, larger
                   scatter_or_hexbin       = 'scatter',         # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
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
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
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
    
    
    #======================
    # Select apertures with holes
    if gas_type == 'h2':
        mass_radius_1 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_1, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
    elif gas_type == 'h1':
        mass_radius_1 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_2, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
    elif gas_type == 'gas':
        mass_radius_1 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_1, 'gas_mass'))(data)[soap_indicies_sample]
        mass_radius_2 = attrgetter('exclusive_sphere_%skpc.%s'%(surfdens_aperture_2, 'gas_mass'))(data)[soap_indicies_sample]
    mass_radius_1.convert_to_physical()
    mass_radius_2.convert_to_physical()
    mass_radius_1.convert_to_units('Msun')
    mass_radius_2.convert_to_units('Msun')
    

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
    # Calculate surface density 1 and 2, ignore systems with less than H2<10**7 in radius_2
    mask_positive_H2          = mass_radius_2 > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    mask_positive_and_h2_det = np.logical_and.reduce([mask_positive_H2, H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)])
    
    # Surface density ratio
    surf_density_1 = mass_radius_1 / (np.pi*(cosmo_quantity(float(surfdens_aperture_1), u.kpc, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)**2))
    surf_density_2 = mass_radius_2 / (np.pi*(cosmo_quantity(float(surfdens_aperture_2), u.kpc, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)**2))
    surf_density_ratio = surf_density_1/surf_density_2
        
    # What is the median surf density ratio among H2-detected ETGs?, ignore systems with no H2 in radius_2
    print('MEDIAN %s SURFACE DENSITY RATIO %s/%s for sample: %s' %(gas_type, surfdens_aperture_1, surfdens_aperture_2, sample_input['name_of_preset']))
    print('    > median surface density ratio H2 > 0 in %s:                    %.2e' %(surfdens_aperture_2, np.median(surf_density_ratio[mask_positive_H2])))
    print('    > median surface density ratio H2 > 0 in %s and H2 > %.1e:      %.2e' %(surfdens_aperture_2, h2_detection_limit, np.median(surf_density_ratio[mask_positive_and_h2_det])))
    
    # Set limits
    log_maximum_ratio = np.log10((surfdens_aperture_2/surfdens_aperture_1)**2)
    log_minimum_ratio = log_maximum_ratio + np.log10(0.001)         # 0.001 means down to 0.1% of mass in r1
    
    #==============================
    # Plot scatter or hexbin
    if scatter_or_hexbin == 'scatter':
        print('Scatter (old) not available for this plot ')
    if scatter_or_hexbin == 'scatter_new':
        ### Plots scatter as usual, perhaps with different size markers for H2
        
        mask_low_kappa_plot  = np.logical_and.reduce([~mask_high_kappa, mask_positive_H2])
        mask_high_kappa_plot = np.logical_and.reduce([mask_high_kappa, mask_positive_H2])
        
        # scatter for kappa < 0.4
        axs.scatter(np.log10(stellar_mass[mask_low_kappa_plot]), np.log10(surf_density_ratio[mask_low_kappa_plot]), c='C0', s=1.5, marker='o', linewidths=0, edgecolor='none', alpha=0.75, label='$\kappa_{\mathrm{co}}^{*}<0.4$')
        # scatter for kappa > 0.4
        axs.scatter(np.log10(stellar_mass[mask_high_kappa_plot]), np.log10(surf_density_ratio[mask_high_kappa_plot]), c='C1', s=1.5, marker='s', linewidths=0, edgecolor='none', alpha=0.75, label='$\kappa_{\mathrm{co}}^{*}>0.4$')
    if scatter_or_hexbin == 'hexbin_count':  
        ### Plots hexbin showing number of galaxies in bin
        
        # Set lower density ratio of 10**log_minimum_ratio where there isn't enough H2 in radius_2        
        surf_density_ratio[~mask_positive_H2] = cosmo_quantity(10**log_minimum_ratio, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        # Set lower density ratio of 10**log_minimum_ratio for any values below 10**log_minimum_ratio
        surf_density_ratio[surf_density_ratio < cosmo_quantity(10**log_minimum_ratio, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)] = cosmo_quantity(10**-8, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        
        # hexbin for all values
        extent = (9.5,12.5,np.log10(10**log_minimum_ratio), np.max(np.log10(surf_density_ratio)))
        gridsize = (30,15)
        axs.hexbin(np.log10(stellar_mass), np.log10(surf_density_ratio), bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=1, zorder=-4, cmap='plasma', lw=0.01, alpha=0.5)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(np.log10(stellar_mass), np.log10(surf_density_ratio), bins='log', gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', vmin=1, vmax=100, mincnt=5, zorder=-3, cmap='plasma', lw=0.01)
    if scatter_or_hexbin == 'hexbin_H2':  
        ### Plots hexbin showing median log10 H2_mass
        
        # Set lower density ratio of 10**log_minimum_ratio where there isn't enough H2 in radius_2
        surf_density_ratio[~mask_positive_H2] = cosmo_quantity(10**log_minimum_ratio, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        # Set lower density ratio of 10**log_minimum_ratio for any values below 10**log_minimum_ratio
        surf_density_ratio[surf_density_ratio < cosmo_quantity(10**log_minimum_ratio, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)] = cosmo_quantity(10**-8, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        
        # Add a norm
        norm = colors.Normalize(vmin=5, vmax=10)
        
        # hexbin for all values
        extent = (9.5,12.5,np.log10(10**log_minimum_ratio), np.max(np.log10(surf_density_ratio)))
        gridsize = (30,15)
        axs.hexbin(np.log10(stellar_mass), np.log10(surf_density_ratio), C=np.log10(H2_mass), reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=1, zorder=-4, cmap='viridis', edgecolors='w', lw=1.8)
        # normal hexbin with mincnt >= 5: override the grey background of other points first with white background, then plot normal
        cb = axs.hexbin(np.log10(stellar_mass), np.log10(surf_density_ratio), C=np.log10(H2_mass), reduce_C_function=np.median, norm=norm, gridsize=gridsize, extent=extent, xscale='linear', yscale='linear', mincnt=5, zorder=-3, cmap='viridis', lw=0.01)
        
         
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(np.floor(log_minimum_ratio), np.ceil(log_maximum_ratio))
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
    dict_ylabel = {'h2': 'H_{2}',
                   'h1': 'H_{\mathrm{I}}',
                   'gas': 'gas'}
    axs.set_ylabel(r'log$_{10}$ $\Sigma_{\mathrm{%s}}^{\mathrm{%s\:kpc}}/\Sigma_{\mathrm{%s}}^{\mathrm{%s\:kpc}}$'%(dict_ylabel[gas_type], surfdens_aperture_1, dict_ylabel[gas_type], surfdens_aperture_2))
    
    #-----------
    # colorbar for hexbin
    if scatter_or_hexbin == 'hexbin_count':
        fig.colorbar(cb, ax=axs, label='Number of galaxies', extend='max')
    if scatter_or_hexbin == 'hexbin_H2':
        fig.colorbar(cb, ax=axs, label='Median log$_{10}$ $M_{\mathrm{H_2}}$ [M$_{\odot}$]', extend='both')
    
    #-----------  
    # Annotations    
    #axs.text(0.80, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.915, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Add ratio lines
    axs.text(12, log_maximum_ratio + np.log10(0.5) - 0.2, '$50\%$', fontsize=7, color='grey')
    axs.avhline(log_maximum_ratio + np.log10(0.5), ls='-', c='grey', zorder=-10)
    
    axs.text(12, log_maximum_ratio + np.log10(0.1) - 0.2, '$10\%$', fontsize=7, color='grey')
    axs.avhline(log_maximum_ratio + np.log10(0.1), ls='-', c='grey', zorder=-10)
    
    axs.text(12, log_maximum_ratio + np.log10(0.01) - 0.2, '$1\%$', fontsize=7, color='grey')
    axs.avhline(log_maximum_ratio + np.log10(0.01), ls='-', c='grey', zorder=-10)
    
    #-----------
    # Legend
    if scatter_or_hexbin == 'scatter':
        axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', markerscale=2)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/gas_sigma_ratio/%s_%s_%s_%s_%s_sigmaratio%s-%s%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], gas_type, surfdens_aperture_1, surfdens_aperture_2, ('' if scatter_or_hexbin=='scatter' else scatter_or_hexbin), savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/gas_sigma_ratio/%s_%s_%s_%s_%s_sigmaratio%s-%s%s_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], gas_type, surfdens_aperture_1, surfdens_aperture_2, ('' if scatter_or_hexbin=='scatter' else scatter_or_hexbin), savefig_txt_save, file_format))
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
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(r50[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(r50[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(r50[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(r50[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(r50[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(np.log10(0.5), np.log10(30), bin_width)  # Binning edges
        bin_n, _           = np.histogram(np.log10(r50), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(r50[mask_h2]), bins=hist_bins)
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
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
    
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = r50.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, np.log10(medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, np.log10(medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, np.log10(lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, np.log10(upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(np.log10(0.5), np.log10(30))
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
    axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50}}$ [pkpc]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(np.log10(0.5), np.log10(30))
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    ##axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower right', handletextpad=0.4, handlelength=1.0, alignment='right', markerfirst=False)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
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
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    
    raise Exception('currently not added r50H2 from SOAP')
    
    r50H2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50H2.convert_to_units('kpc')
    r50H2.convert_to_physical()
    
    
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(r50H2[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(r50H2[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(r50H2[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(r50H2[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(r50H2[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(np.log10(0.5), np.log10(30), bin_width)  # Binning edges
        bin_n, _           = np.histogram(np.log10(r50H2), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(r50H2[mask_h2]), bins=hist_bins)
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
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
    
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = r50H2.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, np.log10(medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, np.log10(medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, np.log10(lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, np.log10(upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(np.log10(0.5), np.log10(30))
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
    axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(np.log10(0.5), np.log10(30))
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='right', markerfirst=False, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_r50H2/%s_%s_%s_Mstar_r50H2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_r50H2/%s_%s_%s_Mstar_r50H2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns r50 - r50_H2, coloured by H2 detection, size by H2 mass
def _etg_r50_r50H2(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    raise Exception('currently not added r50H2 from SOAP')
    
    r50H2 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50H2.convert_to_units('kpc')
    r50H2.convert_to_physical()
    
    
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(r50[mask_h2]), np.log10(r50H2[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(r50[~mask_h2]), np.log10(r50H2[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(r50[mask_h2]), np.log10(r50H2[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(r50[~mask_h2]), np.log10(r50H2[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(r50[~mask_h2]), np.log10(r50H2[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(np.log10(0.5), np.log10(30), bin_width)  # Binning edges
        bin_n, _           = np.histogram(np.log10(r50H2), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(r50H2[mask_h2]), bins=hist_bins)
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
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
    
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(np.log10(0.5), np.log10(30), 0.2)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(r50) < hist_bins[i + 1])
            y_bin = r50H2.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, np.log10(medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, np.log10(medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, np.log10(lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, np.log10(upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(np.log10(0.5), np.log10(30))
    axs.set_ylim(np.log10(0.5), np.log10(30))
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
    axs.set_xlabel(r'log$_{10}$ $r_{\mathrm{50}$ [pkpc]]')
    axs.set_ylabel(r'log$_{10}$ $r_{\mathrm{50,H_{2}}}$ [pkpc]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(np.log10(0.5), np.log10(30))
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='right', markerfirst=False, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/r50_r50H2/%s_%s_%s_r50_r50H2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/r50_r50H2/%s_%s_%s_r50_r50H2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns r50 - H2 mass, coloured by H2 detection, size by H2 mass
def _etg_r50_h2mass(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
    r50.convert_to_units('kpc')
    r50.convert_to_physical()
    
    
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(r50[mask_h2]), np.log10(H2_mass[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(r50[~mask_h2]), np.log10(H2_mass[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(r50[mask_h2]), np.log10(H2_mass[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(r50[~mask_h2]), np.log10(H2_mass[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(r50[~mask_h2]), np.log10(H2_mass[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.25
        hist_bins = np.arange(5, 11, bin_width)  # Binning edges
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
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, 10**hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(np.log(1), np.log(30+1), 0.05)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(r50) >= hist_bins[i]) & (np.log10(r50) < hist_bins[i + 1])
            y_bin = H2_mass.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[y_bin > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, np.log10(medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, np.log10(medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, np.log10(lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, np.log10(upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(np.log10(0.5), np.log10(30))
    axs.set_ylim(6, 11)
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
    axs.set_xlabel(r'log$_{10}$ $r_{\mathrm{50}}$ [pkpc]')
    axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ [M$_{\odot}$]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(6, 11)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/r50_MH2/%s_%s_%s_r50_MH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/r50_MH2/%s_%s_%s_r50_MH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
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
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    # Extra values / Calculated values
    
    # Calculate intrinsic triax and ellipticity:
    print('@@@@ using non-iterative ellipticity, replace when SOAP updates')
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), ellip[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), ellip[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), ellip[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), ellip[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), ellip[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(0, 1.1, bin_width)  # Binning edges
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
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
    
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = ellip.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, (medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, (medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, (lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, (upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(0, 1)
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
    axs.set_ylabel(r'$\epsilon$')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(0, 1)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
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
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    # Extra values / Calculated values
    
    # Calculate intrinsic triax and ellipticity:
    print('@@@@ using non-iterative ellipticity, replace when SOAP updates')
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), triax[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), triax[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), triax[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), triax[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), triax[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(0, 1.1, bin_width)  # Binning edges
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
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
    
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
        
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = triax.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, (medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, (medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, (lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, (upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(0, 1.4)
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
    axs.set_ylabel(r'Triaxiality')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(0, 1.4)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
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
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), kappa_stars[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), kappa_stars[~mask_h2], c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), kappa_stars[mask_h2], c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), kappa_stars[~mask_h2], c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), kappa_stars[~mask_h2], s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.05
        hist_bins = np.arange(0, 1.1, bin_width)  # Binning edges
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
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh((hist_bins[:-1])[mask_positive_n], bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
    
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = kappa_stars.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, (medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, (medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, (lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, (upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(0, 1)
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
    axs.set_ylabel(r'$\kappa_{\mathrm{co}}^{*}$')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(0, 1)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    ##axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='right', markerfirst=False, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_kappaco/%s_%s_%s_Mstar_kappaco_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_kappaco/%s_%s_%s_Mstar_kappaco_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - veldisp (interpolated between available SOAP, incl 2r50), coloured by H2 detection, size by H2 mass
def _etg_stelmass_veldisp(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                     aperture_sigma = 'r50',     # [ exclusive_sphere_50kpc / r50 ]
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    
    # Stellar velocity dispersion
    def _compute_vel_disp(aperture_veldisp='50kpc'):        # Msun^1/3 / sigma, Msun^1/3 km-1 s
        stellar_vel_disp_matrix = (attrgetter('exclusive_sphere_%s.stellar_velocity_dispersion_matrix'%(aperture_veldisp))(data))[soap_indicies_sample]
        stellar_vel_disp_matrix.convert_to_units(u.km**2 / u.s**2)
        stellar_vel_disp_matrix.convert_to_physical()
        stellar_vel_disp = np.sqrt((stellar_vel_disp_matrix[:,0] + stellar_vel_disp_matrix[:,1] + stellar_vel_disp_matrix[:,2])/3)

        return stellar_vel_disp
        
    if aperture_sigma == 'r50':
        r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
        r50.convert_to_units('kpc')
        r50.convert_to_physical()
        
        stellar_vel_disp50 = _compute_vel_disp(aperture_veldisp='50kpc')
        stellar_vel_disp30 = _compute_vel_disp(aperture_veldisp='30kpc')
        stellar_vel_disp10 = _compute_vel_disp(aperture_veldisp='10kpc')
        stellar_vel_disp3 = _compute_vel_disp(aperture_veldisp='3kpc')
        stellar_vel_disp1 = _compute_vel_disp(aperture_veldisp='1kpc')
        stellar_vel_disp_2r50 = _compute_vel_disp(aperture_veldisp='2xhalf_mass_radius_stars')
        
        stellar_vel_disp_plot = []
        for i,_ in enumerate(r50):
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(stellar_vel_disp_plot[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(stellar_vel_disp_plot[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(stellar_vel_disp_plot[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(stellar_vel_disp_plot[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(stellar_vel_disp_plot[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.075
        hist_bins = np.arange(np.log10(10), np.log10(1000), bin_width)  # Binning edges
        bin_n, _           = np.histogram(np.log10(stellar_vel_disp_plot), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(stellar_vel_disp_plot[mask_h2]), bins=hist_bins)
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
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
    
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = stellar_vel_disp_plot.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, np.log10(medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, np.log10(medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, np.log10(lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, np.log10(upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(np.log10(10), np.log10(1000))
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
    dict_ylabel     = {'exclusive_sphere_50kpc': '$\sigma _{\mathrm{*}}$', 
                       'r50': '$\sigma _{\mathrm{50}}$'}
    axs.set_xlabel(r'log$_{10}$ $M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    axs.set_ylabel(r'log$_{10}$ %s [km s$^{-1}$]'%dict_ylabel[aperture_sigma])
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(np.log10(10), np.log10(1000))
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
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
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = False,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    L_stars = attrgetter('%s.%s'%(aperture, 'angular_momentum_stars'))(data)[soap_indicies_sample]
    L_stars = np.linalg.norm(L_stars)
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(specific_L_stars[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(specific_L_stars[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(specific_L_stars[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(specific_L_stars[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(specific_L_stars[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised
        bin_width = 0.2
        hist_bins = np.arange(19, 23, bin_width)  # Binning edges
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
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(19.5, 23)
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
    axs.set_ylabel(r'log$_{10}$ $l _{\mathrm{*}}$ [km$^{2}$ s$^{-1}$]')    
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(19.5, 23)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='lower left', handletextpad=0.4, alignment='left', markerfirst=True, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_lstar/%s_%s_%s_Mstar_lstars_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_lstar/%s_%s_%s_Mstar_lstars_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - eta kin = Msun^1/3 / sigma, Msun^1/3 km-1 s (interpolated between available SOAP, incl 2r50), coloured by H2 detection, size by H2 mass
def _etg_stelmass_etakin(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                     aperture_sigma = 'r50',                    # [ 'r50' / exclusive_sphere_50kpc ]
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,          # Median of detected
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    
    if aperture_sigma == 'r50':
        r50 = attrgetter('%s.%s'%(aperture, 'half_mass_radius_stars'))(data)[soap_indicies_sample]
        r50.convert_to_units('kpc')
        r50.convert_to_physical()
        
        stellar_vel_disp50 = _compute_vel_disp(aperture_veldisp='50kpc')
        stellar_vel_disp30 = _compute_vel_disp(aperture_veldisp='30kpc')
        stellar_vel_disp10 = _compute_vel_disp(aperture_veldisp='10kpc')
        stellar_vel_disp3 = _compute_vel_disp(aperture_veldisp='3kpc')
        stellar_vel_disp1 = _compute_vel_disp(aperture_veldisp='1kpc')
        stellar_vel_disp_2r50 = _compute_vel_disp(aperture_veldisp='2xhalf_mass_radius_stars')
        
        stellar_vel_disp_plot = []
        for i,_ in enumerate(r50):
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(eta_kin_plot[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(eta_kin_plot[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(stellar_mass[mask_h2]), np.log10(eta_kin_plot[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(eta_kin_plot[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(stellar_mass[~mask_h2]), np.log10(eta_kin_plot[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised        
        bin_width = 0.025
        hist_bins = np.arange(1, 2, bin_width)  # Binning edges
        bin_n, _           = np.histogram(np.log10(eta_kin_plot), bins=hist_bins)
        bin_n_detected, _  = np.histogram(np.log10(eta_kin_plot[mask_h2]), bins=hist_bins)
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
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    # Median line of detected ETGs
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(9.5, 13.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = np.log10(eta_kin_plot.value[mask])
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[H2_mass.value[mask] > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, (medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, (medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, (lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, (upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(9.5, 12.5)
    axs.set_ylim(1.1, 1.9)
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
    axs.set_ylabel(r'log$_{10}$ $\eta _{\mathrm{kin}}$ [M$_{\odot}^{1/3}$ km$^{-1}$ s]')
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(1.1, 1.9)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    #axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    ##axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='right', markerfirst=False, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/Mstar_etakin/%s_%s_%s_Mstar_etakin_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/Mstar_etakin/%s_%s_%s_Mstar_etakin_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stellar specific angular momentum  - H2 mass(in 50 kpc, not r50), coloured by H2 detection, size by H2 mass
def _etg_lstar_h2mass(soap_indicies_sample=[], sample_input=[], title_text_in = '',
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                     aperture_h2 = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   scatter_kappa_coloubar = False,
                     add_detection_hist = False,
                     h2_detection_limit = 10**7,
                   add_median_line = True,
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
        gs  = fig.add_gridspec(1, 2,  width_ratios=(4, 1.1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
                        
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
    L_stars = attrgetter('%s.%s'%(aperture, 'angular_momentum_stars'))(data)[soap_indicies_sample]
    L_stars = np.linalg.norm(L_stars)
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
    mask_h2 = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
    if scatter_kappa_coloubar:
        sc_points = axs.scatter(np.log10(specific_L_stars[mask_h2]), np.log10(H2_mass[mask_h2]), c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(specific_L_stars[~mask_h2]), np.log10(H2_mass[~mask_h2]), c=kappa_stars[~mask_h2], s=4.0, cmap=mymap, norm=norm, marker='P', alpha=0.5, linewidths=0, edgecolor='none')
    else:
        sc_points = axs.scatter(np.log10(specific_L_stars[mask_h2]), np.log10(H2_mass[mask_h2]), c='r', s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, marker='o', alpha=0.5, linewidths=0, edgecolor='none')
        #axs.scatter(np.log10(specific_L_stars[~mask_h2]), np.log10(H2_mass[~mask_h2]), c='k', s=4.0, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        axs.scatter(np.log10(specific_L_stars[~mask_h2]), np.log10(H2_mass[~mask_h2]), s=4.5, marker='o', alpha=0.75, linewidths=0.4, edgecolor='grey', facecolor='none')
    
    #----------------
    # Add H2 detection rate within a bin above he_detection_limit
    if add_detection_hist:
        # we want the fraction within a bin, not normalised     
        bin_width = 0.2   
        hist_bins = np.arange(5, 11, bin_width)  # Binning edges
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
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='w', edgecolor='k', linewidth=0.7) 
        ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', color='r', edgecolor='k', linewidth=0.7, alpha=0.7) 
        
        #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
        
        # errorbar
        hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
        #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
        ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
        
        # print
        if print_fdet:
            print('--------------------------------')
            print('H2 > %.1e detection rate bins in ETGs:   sample: %s' %(h2_detection_limit, sample_input['name_of_preset']))
            print('   bins with galaxies:   %s'%hist_bins[:-1][mask_positive_n])  
            print('   bin counts:           %s'%bin_n[mask_positive_n])
            print('   f_det:                %s'%np.around(bin_f_detected,3))  
            print('     f_deterr lower: %s'%np.around(bin_f_detected_err_lower,3))  
            print('     f_deterr upper: %s'%np.around(bin_f_detected_err_upper,3)) 
            #answer_fdet = input("\n-----------------\nContinue? y/n? ")
            #if answer_fdet == 'n':
            #    raise Exception('Manual break at fdet')
        
    
    #----------------
    # Print H2 detection
    print('--------------------------------')
    print('FRACTION OF TOTAL SAMPLE %s ABOVE H2 MASS:' %(sample_input['name_of_preset']))
    # Total sample
    for h2_detection_limit_i in [10**6, 10**7, 10**8, 10**9, 10**10]:
        mask_h2_i = H2_mass > cosmo_quantity(h2_detection_limit_i, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        f_detected_i = len(stellar_mass[mask_h2_i])/len(stellar_mass) 
        f_detected_err_i = binom_conf_interval(k=len(stellar_mass[mask_h2_i]), n=len(stellar_mass), confidence_level= 0.68269, interval='jeffreys')
        f_detected_err_lower_i = f_detected_i - f_detected_err_i[0]
        f_detected_err_upper_i = f_detected_err_i[1] - f_detected_i
        print('  >  H2 > %.1e:    %.3f (-%.3f + %.3f), \tcount: %s / %s'%(h2_detection_limit_i, f_detected_i, f_detected_err_lower_i, f_detected_err_upper_i, len(stellar_mass[mask_h2_i]), len(stellar_mass)))
    
    #------------
    if add_median_line:
        #-----------------
        # Define binning parameters
        hist_bins = np.arange(19.5, 23.1, 0.25)  # Binning edges
    
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        bins_n = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(specific_L_stars) >= hist_bins[i]) & (np.log10(specific_L_stars) < hist_bins[i + 1])
            y_bin = H2_mass.value[mask]
        
            # Remove <107 H2 mass from sample
            y_bin = y_bin[y_bin > h2_detection_limit]
            
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
        medians_masked = np.ma.masked_where(bins_n < 10, medians)
        lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
        upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
        
        axs.plot(bin_centers, (medians), color='k',   ls=(0, (1, 1)), linewidth=1, zorder=10)
        axs.plot(bin_centers, (medians_masked), color='k', linewidth=1, label='$M_{\mathrm{H_{2}}}>10^{7}$ M$_{\odot}$ median', zorder=20)
        axs.plot(bin_centers, (lower_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        axs.plot(bin_centers, (upper_1sigma_masked), color='k', linewidth=0.7, ls='--', zorder=20)
        #axs.fill_between(bin_centers, lower_1sigma_masked, upper_1sigma_masked, color='k', alpha=0.3)
    
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    axs.set_xlim(19.5, 23)
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
    axs.set_xlabel(r'log$_{10}$ $l _{\mathrm{*}}$ [km$^{2}$ s$^{-1}$]')   
    axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ [M$_{\odot}$]') 
    if add_detection_hist:
        #ax_hist.set_xscale("log")
        ax_hist.minorticks_on()
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(6, 11)
        #ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.set_xticklabels([0, '', 0.5, '', 1])
        ax_hist.set_xlabel(r'$f_{\mathrm{H_{2}}>10^{7}}}$')
        #ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    if scatter_kappa_coloubar:
        fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    ##axs.text(0.77, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    #axs.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_galaxies_centrals': 'All central galaxies',
                  'all_galaxies_satellites': 'All satellite galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_centrals': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), centrals',
                  'all_ETGs_satellites': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), satellites',
                  'all_ETGs_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), cluster',
                  'all_ETGs_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$), group/field',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs)",
                  'all_ETGs_plus_redspiral_centrals': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), centrals",
                  'all_ETGs_plus_redspiral_satellites': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), satellites",
                  'all_ETGs_plus_redspiral_cluster': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), cluster',
                  'all_ETGs_plus_redspiral_groupfield': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. FRs), group/field',
                  'all_LTGs': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$)',
                  'all_LTGs_excl_redspiral': 'LTGs ($\kappa_{\mathrm{co}}^{*}>0.4$, excl. red FRs)'
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    axs.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3)
    
    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            axs.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            axs.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    axs.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, alignment='right', markerfirst=False, handlelength=1.0)
    
    
    #-----------
    # other
    #plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '_' + '%s'%('kappa' if scatter_kappa_coloubar else '') + '%s'%('_fdet' if add_detection_hist else '') + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/lstar_MH2/%s_%s_%s_lstar_MH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/lstar_MH2/%s_%s_%s_lstar_MH2_%s_H2ap%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], aperture_h2, savefig_txt_save, file_format))
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

### Load ETGs incl. FRs
soap_indicies_sample_all_ETGs_plus_redspiral,            _, sample_input_all_ETGs_plus_redspiral            = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral'%(sim_box_size_name, sim_type_name, snapshot_name))  
soap_indicies_sample_all_ETGs_plus_redspiral_centrals,   _, sample_input_all_ETGs_plus_redspiral_centrals   = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_centrals'%(sim_box_size_name, sim_type_name, snapshot_name))  
#satellites
soap_indicies_sample_all_ETGs_plus_redspiral_cluster,    _, sample_input_all_ETGs_plus_redspiral_cluster    = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_cluster'%(sim_box_size_name, sim_type_name, snapshot_name)) 
soap_indicies_sample_all_ETGs_plus_redspiral_groupfield, _, sample_input_all_ETGs_plus_redspiral_groupfield = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_ETGs_plus_redspiral_groupfield'%(sim_box_size_name, sim_type_name, snapshot_name)) 

### Load LTGs
soap_indicies_sample_all_LTGs,                           _, sample_input_all_LTGs                           = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_LTGs'%(sim_box_size_name, sim_type_name, snapshot_name))   
#centrals
#satellites

### Load LTGs excl. FRs
soap_indicies_sample_all_LTGs_excl_redspiral,            _, sample_input_all_LTGs_excl_redspiral            = _load_soap_sample(sample_dir, csv_sample = '%s_%s_%s_sample_all_LTGs_excl_redspiral'%(sim_box_size_name, sim_type_name, snapshot_name))  
#centrals
#satellites                                                                                                                                                         
#===================================================================================


# IN APPROXIMATE ORDER OF PRESENTATION:








#==========================================
###     ETG H2 plots: H2 morph and extent
# Plot stelmass - r50 stars
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_r50(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)"""
#-------------------- 
# Plot stelmass - r50 H2
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_r50H2(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)"""
#-------------------- 
# Plot r50  - r50 H2
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_r50_r50H2(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)"""
#-------------------- 
# Plot r50  - H2 mass
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_r50_h2mass(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = False,
                        add_median_line = True,
                        savefig       = True)"""
#----------------------
# Plot surface desity ratio of H2 / HI / gas in several apertures
"""for hole_gas_type_i in ['h2', 'h1', 'gas']:  
    for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
        _gas_surface_ratios(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                                gas_type = hole_gas_type_i,
                                  surfdens_aperture_1   = '3',    # Radius 1
                                  surfdens_aperture_2   = '50',    # Radius 2
                                  scatter_or_hexbin = 'hexbin_count',    # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                                savefig       = True)
                                
        _gas_surface_ratios(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                                gas_type = hole_gas_type_i,
                                  surfdens_aperture_1   = '3',    # Radius 1
                                  surfdens_aperture_2   = '50',    # Radius 2
                                  scatter_or_hexbin = 'hexbin_H2',    # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                                savefig       = True)
                                
        raise Exception('current pause 6y98yoh')
                                
                                
        _gas_surface_ratios(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                                gas_type = hole_gas_type_i,
                                  surfdens_aperture_1   = '10',    # Radius 1
                                  surfdens_aperture_2   = '50',    # Radius 2
                                  scatter_or_hexbin = 'hexbin_count',    # [ scatter / scatter_new / hexbin_count / hexbin_H2 ]
                                savefig       = True)"""
#----------------------       
# Plot fraction with H2 / HI / gas holes in 3kpc and 10kpc w.r.t sample detected with H2 > h2_detection_limit. DOESNT USE LOADED SAMPLES FROM ABOVE
"""for hole_gas_type_i in ['h2', 'h1', 'gas']:  
    _aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                            h2_detection_limit = 10**7,
                            hole_gas_type = hole_gas_type_i,
                              hole_aperture   = 'exclusive_sphere_3kpc',    # Radius within which to look for H2 hole
                              print_fdet = True,
                              hole_limit    = 10**6,
                             title_text_in = 'Hole aperture: 3 pkpc',
                            savefig       = True)
    _aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                            h2_detection_limit = 10**7,
                            hole_gas_type = hole_gas_type_i,
                              hole_aperture   = 'exclusive_sphere_10kpc',    # Radius within which to look for H2 hole
                              print_fdet = True,
                              hole_limit    = 10**6,
                             title_text_in = 'Hole aperture: 10 pkpc',
                            savefig       = True)
    # Plot fraction with H2 / HI / gas holes in 3kpc and 10kpc w.r.t sample detected with H2 > h2_detection_limit     
    _aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                            h2_detection_limit = 10**7,
                            hole_gas_type = hole_gas_type_i,
                              hole_aperture   = 'exclusive_sphere_3kpc',    # Radius within which to look for H2 hole
                              print_fdet = True,
                              hole_limit    = 10**6,
                             title_text_in = 'Hole aperture: 3 pkpc, centrals',
                             savefig_txt = '_centrals',
                            savefig       = True)
    _aperture_fdet_rings(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_centrals', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                            h2_detection_limit = 10**7,
                            hole_gas_type = hole_gas_type_i,
                              hole_aperture   = 'exclusive_sphere_10kpc',    # Radius within which to look for H2 hole
                              print_fdet = True,
                             title_text_in = 'Hole aperture: 10 pkpc, centrals',
                             savefig_txt = '_centrals',
                            savefig       = True)"""





#===================================
# Morphological proxies
# Plot stelmass - ellip stars
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_ellip(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)"""
#add projected as option to above
#-------------------
# Plot stelmass - triaxiality stars
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_triax(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)
#-------------------
# Plot stelmass - kappa_co stars
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_kappaco(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)"""


#===================================
# Stellar kinematics 
# Plot stelmass - stellar velocity dispersion. (note we use 50 kpc aperture, observations use r50)
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_veldisp(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)"""
#-------------------
# Plot stelmass - stellar specific angular momentum. (note we use 50 kpc aperture, observations use r50)
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_lstar(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)
#-------------------
# Plot stellar specific angular momentum - H2 mass (note we use 50 kpc aperture, observations use r50)
for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_lstar_h2mass(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = False,
                        add_median_line = True,
                        savefig       = True)"""
#-------------------
# Plot stelmass - etakin = Msun^1/3 / sigma, Msun^1/3 km-1 s. (using sigma interpolated to r50 and Msun in 50kpc)
"""for soap_indicies_sample_i, sample_input_i in zip([soap_indicies_sample_all_ETGs_plus_redspiral, soap_indicies_sample_all_ETGs_plus_redspiral_centrals], [sample_input_all_ETGs_plus_redspiral, sample_input_all_ETGs_plus_redspiral_centrals]):
    _etg_stelmass_etakin(soap_indicies_sample=soap_indicies_sample_i, sample_input=sample_input_i,
                        add_detection_hist = True,
                        savefig       = True)"""
                        
                        








