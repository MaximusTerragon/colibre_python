import swiftsimio as sw
import swiftgalaxy
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import h5py
import csv
import time
import math
from swiftsimio import cosmo_quantity, cosmo_array
from operator import attrgetter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.ticker import PercentFormatter
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
# Returns stelmass - sfr or stelmass - ssfr, coloured by H2 mass, size by H2 mass
def _etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample, sample_input=sample_input, 
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. red spirals)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '--',
                   'all_ETGs_plus_redspiral': '-.'}
    for csv_sample_i in csv_samples:
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
        
        mask_kappa = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
        print('Total sample:  ', len(stellar_mass))
        print('Number of LTG: ', len(stellar_mass[mask_kappa]))
        print('Number of ETG: ', len(stellar_mass[~mask_kappa]))
        
        
        # Plot 2 scatters: one for H2 detections and one for non-detections
        mask_h2 = H2_mass > ((10**6)*u.Msun)
        plt.scatter(stellar_mass[mask_h2], mag_plot[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])**3)/192, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        plt.scatter(stellar_mass[~mask_h2], mag_plot[~mask_h2], c=kappa_stars[~mask_h2], s=1.5, cmap=mymap, norm=norm, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        
        

    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        #plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        print('no obs available yet')
        
    #-----------
    # Axis formatting
    plt.xlim(10**9, 10**12)
    plt.ylim(0.8, 3.3)
    plt.xscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    plt.xlabel(r'$M_{*}$ ($%s$) [M$_{\odot}$]'%dict_aperture[aperture])
    plt.ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    plt.text(0.8, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    if add_observational:
        plt.text(10**11.9, 2.65, '↑ red sequence', fontsize=7, color='r', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
        plt.text(10**11.9, 2.4, '↓ blue cloud', fontsize=7, color='b', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')

    #-----------
    # Legend
    msizes = np.array([5, 6, 7, 8, 9])
    msizes = (msizes**3)/192
    l1, = plt.plot([],[], 'or', markersize=msizes[1])
    l2, = plt.plot([],[], 'or', markersize=msizes[2])
    l3, = plt.plot([],[], 'or', markersize=msizes[3])
    labels = [r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=6.0$', '7.0', '8.0']
    leg = plt.legend([l1, l2, l3], labels, ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left')
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture + '_' + savefig_txt
        
        plt.savefig("%s/etg_soap_analysis/%s_%s_%s_young14_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_soap_analysis/%s_%s_%s_young14_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()



#=====================================
# Load a sample from a given snapshot
soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs')   
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs
                                                                                   # L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral
#=====================================


#--------------
# Plot similar to Young+14
_etg_stelmass_u_r(soap_indicies_sample=soap_indicies_sample, sample_input=sample_input,
                    showfig       = False,
                    savefig       = True)







# M* vs u*-r* (no dust), size by molecular mass to mimic Young+14 (remove 1.36 He). Note Cappellari+13 find M_JAM ~ M* (at 2Re). | Young+11,+14, Davis+19? | 30/50pkpc aperture via SOAP


# M* - M_H2, colour by kappa… hints at stellarmassloss? | Young+14, Davis+19, Davis+22, Michalowski+24, remove He | 30/50pkpc aperture via SOAP
#       What are some typical molecular masses of ETGs at a given mass? | Young+11,+14, Davis+19, Davis+22, Michalowski+24, remove He | 30/50pkpc aperture via SOAP
#       What sort of ETGs (M*,kappa,r50,environ) host the most molecular mass within our sample? | Young+11,+14, Davis+19, remove He | 30/50pkpc aperture via SOAP
#       What would be the approximate molecular mass detection rate when compared to ATLAS3D limit of ~107 for molecular mass (using CO detections… Young+11,+14… inner 2-4 kpc)… check observational aperture... Or MASSIVE (Davis+19) | Young+11,+14, Davis+19, remove He | 1/5/10/30/50pkpc aperture via SOAP
#       How does H2 scale with other morphological properties, such as ellipticity and triaxiality? | 30/50pkpc aperture via SOAP
#       Are these results sensitive to 30 vs 50 kpc aperture? | 30/50pkpc aperture via SOAP

# Molecular gas surface densities vs LTG sample | Young+11, remove He | 30/50pkpc aperture via SOAP, local densities/radial requires snapshots

# Environment: M200,crit vs M_mol, detection rates | Young+11, Davis+11, remove He | 30/50pkpc aperture via SOAP
# Correlogram? https://python-graph-gallery.com/111-custom-correlogram/ 


# r50 vs M_ H2 | Davis+13 ?  | 30/50pkpc aperture via SOAP

# r50 vs r50H2 … how does this compare to the LTG sample? | Davis+13 |  30pkpc aperture via SOAP + snapshots for H2
#       Select only field galaxies/environment? As ETGs in cluster had smaller H2 extent | Davis+13 

# M_ H2 surface density vs radius, stack some galaxies and see what happens  | Davis+13 |  snapshots for H2

# Specific angular momentum of stars vs M_H2 | Young+11, Davis+11, remove He | 30/50pkpc aperture via SOAP for stars

# σ_stars  and bulge fraction vs M_ H2 | Davis+19,  remove He | 30/50pkpc aperture via SOAP for stars

# mu_kin = M^1/3/σ (roughly kinematic bulge fraction at fixed stellar mass) vs stellar mass,  measure detection rate  | Davis+19,  remove He | 30/50pkpc aperture via SOAP for stars



