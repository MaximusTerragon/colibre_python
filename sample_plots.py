import swiftsimio as sw
import swiftgalaxy
import numpy as np
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
import matplotlib.cm as cm
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



#-----------------
# Returns stelmass - sfr or stelmass - ssfr, coloured by kappa stars
def _sample_stelmass_u_r(soap_indicies_sample = [],  sample_input = {},
                   #=====================================
                   # Graph settings
                   aperture = 'exclusive_sphere_50kpc', 
                   #----------
                   add_ur2_line = True,        # Adapts based on imput mass_type, and using references from pipeline
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
    fig = plt.figure(figsize=(10/2, 2.5))
    gs  = fig.add_gridspec(1, 2,  width_ratios=(1, 2.5),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.5)
    # Create the Axes.
    ax_scat = fig.add_subplot(gs[1])
    ax_hist = fig.add_subplot(gs[0])
                        
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
    mag_plot = u_mag - r_mag
    mag_plot = mag-plot[soap_indicies_sample]
    
    central_sat = attrgetter('input_halos.is_central')(data)
    central_sat = central_sat[soap_indicies_sample]
    
    kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
    kappa_stars = kappa_stars[soap_indicies_sample]
    
    
    #===============================
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
        
    # Plot scatter
    ax_scat.scatter(stellar_mass, mag_plot, c=kappa_stars, s=1.5, cmap=mymap, norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)

        
    #-----------------
    # Histogram
    hist_bins = np.arange(0.5, 3.1, 0.05)  # Binning edges
    
    n_LTG, _    = np.histogram(mag_plot[mask_kappa], weights=(np.ones(len(stellar_mass[mask_kappa]))/len(stellar_mass[mask_kappa])), bins=hist_bins)
    n_LTG, _, _ = ax_hist.hist(mag_plot[mask_kappa], weights=(np.ones(len(stellar_mass[mask_kappa]))/len(stellar_mass[mask_kappa])), bins=hist_bins, alpha=0.5, facecolor='b', orientation='horizontal', label='$\kappa_{\mathrm{co}}^{*}>0.4$\n($f_{\mathrm{LTG}}=%.2f$)'%(len(stellar_mass[mask_kappa])/len(stellar_mass)))
        
    n_ETG, _    = np.histogram(mag_plot[~mask_kappa], weights=(np.ones(len(stellar_mass[~mask_kappa]))/len(stellar_mass[~mask_kappa])), bins=hist_bins)
    n_ETG, _, _ = ax_hist.hist(mag_plot[~mask_kappa], weights=(np.ones(len(stellar_mass[~mask_kappa]))/len(stellar_mass[~mask_kappa])), bins=hist_bins, alpha=0.5, facecolor='r', orientation='horizontal', label='$\kappa_{\mathrm{co}}^{*}<0.4$\n($f_{\mathrm{ETG}}=%.2f$)'%(len(stellar_mass[~mask_kappa])/len(stellar_mass)))
        
        
    #-----------------
    # Add observations
    if add_ur2_line:
        ax_scat.axhline(2, lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10))
        ax_hist.axhline(2, lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10))
        
        
    #-----------
    # Axis formatting
    ax_scat.set_xlim(10**(9.5), 10**12)
    ax_hist.set_ylim(0.5, 3)
    ax_scat.set_ylim(0.5, 3)
    ax_scat.set_xscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    ax_scat.minorticks_on()
    ax_hist.minorticks_on()
    ax_scat.tick_params(axis='x', which='minor')
    dict_aperture = {'exclusive_sphere_30kpc': '30', 
                     'exclusive_sphere_50kpc': '50'}
    ax_scat.set_xlabel(r'$M_{*}$ ($%s$ pkpc) [M$_{\odot}$]'%dict_aperture[aperture])
    ax_hist.set_xlabel(r'Normalised freq.')
    ax_scat.set_yticklabels([])
    ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    fig.colorbar(mapper, ax=ax_scat, label='$\kappa_{\mathrm{co}}^{*}$')      #, extend='max'  
      
    #-----------  
    # Annotations
    ax_scat.text(10**11.5, 2.8, '${z=%.2f}$' %z, fontsize=7)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}\leq0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}\leq0.4$ and 'red spirals')"
                  }
    ax_hist.set_title(r'%s' %(title_dict[sample_input['name_of_preset']]), size=7, loc='left', pad=3) 
    if add_ur2_line:
        ax_scat.text(10**11.9, 2.65, "'red spiral' threshold", fontsize=7, color='r')
        
    #-----------
    # Legend
    ax_hist.legend(loc='lower left', ncol=1, frameon=False)
        
        
    #-----------
    # other
    #plt.tight_layout()
    ax_hist.grid(alpha=0.4, zorder=-5, lw=0.7)
    
    if savefig:
        savefig_txt_save = aperture + '_' + savefig_txt
        
        plt.savefig("%s/sample_plots/%s_%s_%s_sample_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/sample_plots/%s_%s_%s_sample_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()





#=====================================
# Load a sample from a given snapshot
soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies')
                                                                                    # L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies
                                                                                    # L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs
                                                                                    # L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral
#=====================================




#-------------------
# Similar to the correa+17 plot.   USE WITH ALL GALAXIES
_sample_stelmass_u_r(soap_indicies_sample=soap_indicies_sample, sample_input=sample_input,
                     showfig       = False,
                     savefig       = True)















