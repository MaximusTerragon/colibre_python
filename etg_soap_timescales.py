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
from astropy.cosmology import z_at_value, FlatLambdaCDM
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



#------------------
"""
ETG snap sample code
	use galaxy tree to see how our sample changes over several snapshots; same galaxies? 
	of the ones at z=0, trackid to see how many are h2 detected at an earlier snapshot
	... make samples at each snapshot, which gives trackid list and soap index
	compare trackid over past snapshots... z=0 to ... z=0.1, z=0.3, etc.
	which left due to H2, which left due to morph/ETG criterion?
"""
# Returns H2 mass function for given set of samples
def _etg_sample_timescales(z0_sample = '',  title_text_in = '',
                            snapshot_list = [],
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          aperture_h2 = 'exclusive_sphere_50kpc', 
                            h2_detection_limit = 10**7,
                          plot_lowStelmass = True,
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
    # Extract z=0 sample and TrackIDs
    soap_indicies_sample, trackid_sample, sample_input = _load_soap_sample(sample_dir, csv_sample = z0_sample)
    trackid_sample = np.array(trackid_sample)
    print('Length of z=0 sample:      ', len(trackid_sample))
        
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
    lookbacktime = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0 = 0.0486).age(z)).value
    run_name = data.metadata.run_name
    box_size = data.metadata.boxsize[0].to(u.Mpc)
    
    #-----------------
    # Loop over snapshot list given. Will be in reverse chronological order
    dict_class = {'z': [z],
                  'lookbacktime': [0],
                    'etg_h2': [len(trackid_sample)],    # = stays h2 etg
                    'etg_non': [0],                     # = stays etg but becomes non-H2 det
                    'ltg_h2': [0],                      # = stays h2, but was ltg
                    'ltg_non': [0],                     # = becomes non-h2 det and was ltg
                    'other': [0],                       # = trackID not found for some reason
                    'min_Mstar': [0]}                   # = goes below 10^9.5 limit
    for snap_i in snapshot_list[1:]:
        
        print('\nLoading snap_i:  ', snap_i)
        soap_catalogue_file_i = os.path.join(colibre_base_path, simulation_dir, "SOAP-HBT/halo_properties_0%s.hdf5"%snap_i,)
        data = sw.load(f'%s'%soap_catalogue_file)
    
        # Get metadata from file
        z = data.metadata.redshift
        lookbacktime = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0 = 0.0486).age(z)).value
        dict_class['z'].append(z)
        dict_class['lookbacktime'].append(lookbacktime)
        
        
        
        
        print('lookbacktime test: ', lookbacktime)
        
        
        
        
        #==========================================================
        # Get trackid, stelmass, molecular hydrogen, magnitude data and kappaco (etg classif.), central_sat
        trackid = attrgetter('input_halos_hbtplus.track_id')(data)
        trackid = np.array(trackid)
        
        
        # Find indicies of where TrackID was found
        soap_indicies_sample = np.nonzero(np.in1d(trackid, trackid_sample))[0]
        print('TrackID found in %i sample:     %i'%(snap_i, len(soap_indicies_sample)))
        
        #-----------------------------
        # Append any Trackid not found
        num_notfound = len(trackid_sample) - len(trackid)
        print('    TrackID not found (other):   ', num_notfound)
        dict_class['other'].append(num_notfound)
        
        
        
        #==========================================================
        # Extract values from TrackIDs found: stelmass, molecular hydrogen, and kappa + magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        
        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        H2_mass.convert_to_units('Msun')
        
        u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])[soap_indicies_sample]
        r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])[soap_indicies_sample]
        u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        mag_plot = u_mag - r_mag

        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]

        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)[soap_indicies_sample]
        
        #==========================================================
        # Useful masks - can be combined to give desired sample
        mask_Mstar      = stellar_mass > cosmo_quantity(10**9.5, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_h2         = H2_mass > cosmo_quantity(h2_detection_limit, u.Msun, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_highkappa  = kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_blue       = mag_plot > cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)
        mask_ltg        = np.logical_and.reduce([mask_blue, mask_highkappa])
        
        mask_etg_h2     = np.logical_and.reduce([mask_Mstar, ~mask_ltg, mask_h2])
        mask_etg_non    = np.logical_and.reduce([mask_Mstar, ~mask_ltg, ~mask_h2])
        mask_ltg_h2     = np.logical_and.reduce([mask_Mstar, mask_ltg, mask_h2])
        mask_ltg_non    = np.logical_and.reduce([mask_Mstar, ~mask_ltg, ~mask_h2])
        
        #----------------------------
        # Find how many no longer reach stelmass criterion - can be optionally excluded in plot with plot_lowStelmass = False
        num_min_Mstar = len(stellar_mass[~mask_Mstar])
        print('    Was low-mass (min_Mstar):   ', num_min_Mstar)
        dict_class['min_Mstar'].append(num_notfound)
        
        
        #----------------------------
        # Of those that continue to meet stelmass criterion, split into etg/ltg and h2/non-det
        num_etg_h2 = len(stellar_mass[mask_etg_h2])
        print('    Stay ETG, stay H2 (etg_h2):   ', num_etg_h2)
        dict_class['etg_h2'].append(num_etg_h2)
        
        num_etg_non = len(stellar_mass[mask_etg_non])
        print('    Stay ETG, H2 non-det (etg_non):   ', num_etg_non)
        dict_class['etg_non'].append(num_etg_non)
        
        num_ltg_h2 = len(stellar_mass[mask_ltg_h2])
        print('    Was LTG, stay H2 (ltg_h2):   ', num_ltg_h2)
        dict_class['ltg_h2'].append(num_ltg_h2)
        
        num_ltg_non = len(stellar_mass[mask_ltg_non])
        print('    Was LTG, H2 non-det (ltg_non):   ', num_ltg_non)
        dict_class['ltg_non'].append(num_ltg_non)
        
        
        #----------------------------
        # Sanity check
        print('\n-------SANITY CHECK-------')
        print('Total TrackID found: %s with %s not found' %(trackid, num_notfound))
        print('   ...of which stayed Mstar: ', len(stellar_mass[mask_Mstar])
        print('      ...of which ETG H2:    ', num_etg_h2)
        print('      ...of which ETG non:   ', num_etg_non)
        print('      ...of which LTG H2:    ', num_ltg_h2)
        print('      ...of which LTG non:   ', num_ltg_non)
    
     
     
    #-----------
    # Axis formatting
    
    
    #redshift on top, lookback on bottom like subhalo_evolution.py
    #np.flip all arrays
    #plot filled... there may be a plot that does this for me
    #plot as fractions, depending on if 'other' is included, and if plot_lowStelmass = False
    #change savefile name
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    
    #-----------
    # Axis formatting
    plt.xlim(6, 11)
    plt.ylim(10**(-6), 10**(-1))
    #plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    axs.minorticks_on()
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc',
                     'inclusive_sphere_10kpc': '10 pkpc',
                     'inclusive_sphere_30kpc': '30 pkpc', 
                     'inclusive_sphere_50kpc': '50 pkpc'}
    plt.xlabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}$ (%s) [M$_{\odot}$]'%(dict_aperture[aperture_h2]))
    plt.ylabel(r'dn/dlog$_{10}$($M_{\mathrm{H_{2}}}$) [cMpc$^{-3}$]')
      
    #-----------  
    # Annotations
    #plt.text(0.8, 0.9, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    
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
    fig_text(x=0.195, y=0.935, ha='left', s=text_title, fontsize=7, ax=axs,
        highlight_textprops=[
            {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
            {"color": "white"},
            {"color": "black"}
        ])
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3)
    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture_h2 + '_' + savefig_txt
        
        plt.savefig("%s/etg_time_analysis/%s_%s_ALL_SAMPLES_sample_H2massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_time_analysis/%s_%s_ALL_SAMPLES_sample_H2massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()





#-------------------
# Takes the ETG (incl. FRs) sample at z=0 and splits them into: 
#       'etg_h2'  = stays h2 etg
#       'etg_non' = stays etg but non
#       'ltg_h2'  = stays h2, but was ltg
#       'ltg_non' = becomes non-h2 det and was ltg
#       'other'   = TrackID was not found (possible merger or just quirk with HBT)
#       [optional] 'low_mstar' = falls below the M* = 10^9.5 of our sample
_etg_sample_timescales(z0_sample = ['L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                           snapshot_list = [127, 119, 114, 110],
                           plot_lowStelmass = True,     # includes category where the etg ends up below stelmass limit
                         showfig       = False,
                         savefig       = True)






















