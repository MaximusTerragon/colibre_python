import swiftsimio as sw
import swiftgalaxy
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import h5py
import csv
import time
import math
import astropy
from swiftsimio import cosmo_quantity, cosmo_array
from operator import attrgetter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patheffects as mpe
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from highlight_text import fig_text
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
outline=mpe.withStroke(linewidth=1.6, foreground='black')


#-----------------
# Returns stelmass - sfr or stelmass - ssfr, coloured by kappa stars
def _sample_stelmass_u_r(csv_sample = '',  title_text_in = '',
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
    # Load a sample from a given snapshot
    soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = csv_sample)                                                      

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
    
    # Masking LTGs (ETGs are mask_high_kappa):
    mask_high_kappa    = (kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
    # Masking blue LTGs: (ETGs are ~mask_blue_LTG)
    mask_blue_LTG = np.logical_and.reduce([kappa_stars > cosmo_quantity(0.4, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                              mag_plot < cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)]).squeeze()
    # Mask all red galaxies u-r>2
    mask_red = (mag_plot > cosmo_quantity(2, u.dimensionless, comoving=False, scale_factor= data.metadata.a, scale_exponent=0)).squeeze()
    
    dict_sample_numbers = {'total': len(stellar_mass),
                           'kappa_cut': {'LTG': len(stellar_mass[mask_high_kappa]),
                                         'ETG': len(stellar_mass[~mask_high_kappa]),
                                         'LTG_red': len(stellar_mass[np.logical_and.reduce([mask_red, mask_high_kappa]).squeeze()]),
                                         'ETG_red': len(stellar_mass[np.logical_and.reduce([mask_red, ~mask_high_kappa]).squeeze()])
                                         },
                           'kappa_mag_cut': {'LTG': len(stellar_mass[mask_blue_LTG]),
                                             'ETG': len(stellar_mass[~mask_blue_LTG]),
                                             'LTG_red': len(stellar_mass[np.logical_and.reduce([mask_red, mask_blue_LTG]).squeeze()]),
                                             'ETG_red': len(stellar_mass[np.logical_and.reduce([mask_red, ~mask_blue_LTG]).squeeze()])
                                         }}
    print('Total sample:  ', dict_sample_numbers['total'])
    print('Number of kappa>0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG'], dict_sample_numbers['kappa_cut']['LTG']/dict_sample_numbers['total']))
    print('     of which u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['LTG_red'], dict_sample_numbers['kappa_cut']['LTG_red']/dict_sample_numbers['kappa_cut']['LTG']))
    print('Number of kappa<0.4:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG'], dict_sample_numbers['kappa_cut']['ETG']/dict_sample_numbers['total']))
    print('     of which u-r>2:  %s  (%.3f)' %(dict_sample_numbers['kappa_cut']['ETG_red'], dict_sample_numbers['kappa_cut']['ETG_red']/dict_sample_numbers['kappa_cut']['ETG']))
    print('Number of LTG kappa>0.4 and u-r<2:               %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG'], dict_sample_numbers['kappa_mag_cut']['LTG']/dict_sample_numbers['total']))
    print('                   of which u-r>2:               %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['LTG_red'], dict_sample_numbers['kappa_mag_cut']['LTG_red']/dict_sample_numbers['kappa_mag_cut']['LTG']))
    print('Number of ETG kappa<0.4 or kappa>0.4 + u-r>2:    %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG'], dict_sample_numbers['kappa_mag_cut']['ETG']/dict_sample_numbers['total']))
    print('                              of which u-r>2:    %s  (%.3f)' %(dict_sample_numbers['kappa_mag_cut']['ETG_red'], dict_sample_numbers['kappa_mag_cut']['ETG_red']/dict_sample_numbers['kappa_mag_cut']['ETG']))
    
    # Plot scatter
    ax_scat.scatter(np.log10(stellar_mass), mag_plot, c=kappa_stars, s=1.5, cmap=mymap, norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)

        
    #-----------------
    # Histogram
    hist_bins = np.arange(0.5, 3.1, 0.05)  # Binning edges
    
    n_LTG, _    = np.histogram(mag_plot[mask_high_kappa], weights=(np.ones(len(stellar_mass[mask_high_kappa]))/len(stellar_mass[mask_high_kappa])), bins=hist_bins)
    n_LTG, _, _ = ax_hist.hist(mag_plot[mask_high_kappa], weights=(np.ones(len(stellar_mass[mask_high_kappa]))/len(stellar_mass[mask_high_kappa])), bins=hist_bins, alpha=0.5, facecolor='b', orientation='horizontal', label='$\kappa_{\mathrm{co}}^{*}>0.4$')
        
    n_ETG, _    = np.histogram(mag_plot[~mask_high_kappa], weights=(np.ones(len(stellar_mass[~mask_high_kappa]))/len(stellar_mass[~mask_high_kappa])), bins=hist_bins)
    n_ETG, _, _ = ax_hist.hist(mag_plot[~mask_high_kappa], weights=(np.ones(len(stellar_mass[~mask_high_kappa]))/len(stellar_mass[~mask_high_kappa])), bins=hist_bins, alpha=0.5, facecolor='r', orientation='horizontal', label='$\kappa_{\mathrm{co}}^{*}<0.4$')
        
        
    #-----------------
    # Add observations
    if add_ur2_line:
        ax_scat.axhline(2, lw=1, ls='--', color='k', zorder=10, path_effects=[outline], label='ETG threshold\nfor $\kappa_{\mathrm{co}}^{*}>0.4$')
        ax_hist.axhline(2, lw=1, ls='--', color='k', zorder=10, path_effects=[outline])
        
        # Add Schawinski+14 from Correa+17
        outline_obs=mpe.withStroke(linewidth=1.3, foreground='black')
        ax_scat.plot([9, 15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=0.8, ls='-', color='darkviolet', label='Schawinski+14', zorder=10, path_effects=[outline])
        
        
    #-----------
    # Axis formatting
    ax_scat.set_xlim(9.5, 12.5)
    ax_hist.set_ylim(0.5, 3)
    ax_scat.set_ylim(0.5, 3)
    #ax_scat.set_xscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    ax_scat.minorticks_on()
    ax_hist.minorticks_on()
    ax_scat.tick_params(axis='x', which='minor')
    dict_aperture = {'exclusive_sphere_30kpc': '30', 
                     'exclusive_sphere_50kpc': '50'}
    ax_scat.set_xlabel(r'log$_{10}$ $M_{*}$ ($%s$ pkpc) [M$_{\odot}$]'%dict_aperture[aperture])
    ax_hist.set_xlabel(r'Normalised freq.')
    ax_scat.set_yticklabels([])
    ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
        
    #------------
    # colorbar
    fig.colorbar(mapper, ax=ax_scat, label='$\kappa_{\mathrm{co}}^{*}$', extend='max')      #, extend='max'  
      
    #-----------  
    # Annotations
    ax_scat.text(0.05, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = ax_scat.transAxes)

    ax_scat.text(0.85, 0.94, '↑ red\nsequence', fontsize=7, color='r', rotation=14, rotation_mode='anchor', horizontalalignment='center', verticalalignment='top', transform = ax_scat.transAxes, bbox=dict(facecolor='r', edgecolor='r', boxstyle='round', alpha=0.2))
    ax_scat.text(0.88, 0.78, '↓ blue\ncloud', fontsize=7, color='b', rotation=14, rotation_mode='anchor', horizontalalignment='center', verticalalignment='top', transform = ax_scat.transAxes, bbox=dict(facecolor='b', edgecolor='b', boxstyle='round', alpha=0.2))
    
    title_dict = {'all_galaxies': 'all galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. red FRs)"
                  }
    title_run_dict = {'L100_m6': 'L100m6', 
                      'L200_m6': 'L200m6'}
    title_type_dict = {'THERMAL_AGN_m6': '',
                       'HYBRID_AGN_m6': 'h'}
    title_color_dict = {'L100m6': "#1B9E77", 
                        'L100m6h': "#D95F02", 
                        'L200m6': "#7570B3"}
    run_name_title = '%s%s'%(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']])
    text_title = r'<%s><..><%s%s>'%(run_name_title, title_dict[sample_input['name_of_preset']], title_text_in)
    fig_text(x=0.105, y=0.96, ha='left', s=text_title, fontsize=7, ax=ax_scat,
        highlight_textprops=[
            {"color": title_color_dict[run_name_title], "fontname": 'Courier New', "bbox": {"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'}},
            {"color": "white"},
            {"color": "black"}
        ])      
    #ax_hist.set_title(r'%s%s, %s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_dict[sample_input['name_of_preset']], title_text_in), size=7, loc='left', pad=3) 
    #if add_ur2_line:
        #ax_scat.text(10**11.35, 1.7, "$\kappa_{\mathrm{co}}^{*}>0.4$\nred spirals", fontsize=7, color='k')
        
    #-----------
    # Legend
    ax_hist.legend(loc='lower right', ncol=1, frameon=False)
    ax_scat.legend(loc='lower right', ncol=1, frameon=False, handlelength=1.0, labelcolor='linecolor')
        
        
    #-----------
    # other
    #plt.tight_layout()
    ax_hist.grid(alpha=0.4, lw=0.7, zorder=-20)
    ax_scat.grid(alpha=0.4, lw=0.7, zorder=-20)
    
    if savefig:
        savefig_txt_save = aperture + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/u_r_sample_plot/%s_%s_%s_sample_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/u_r_sample_plot/%s_%s_%s_sample_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#------------------
# Returns stellar mass function for given set of samples
def _sample_stellar_mass_function(csv_samples = [],  title_text_in = '',
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.6], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_centrals': 'ETGs (excl. FRs)',
                   'all_ETGs_satellites': 'ETGs (excl. FRs)',
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
                   'all_ETGs': '-',
                   'all_ETGs_centrals': '-',
                   'all_ETGs_satellites': '-',
                   'all_ETGs_plus_redspiral': '-',
                   'all_ETGs_plus_redspiral_centrals': '-',
                   'all_ETGs_plus_redspiral_satellites': '-'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_galaxies_centrals': 'o',
                   'all_galaxies_satellites': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_centrals': 's',
                   'all_ETGs_satellites': 's',
                   'all_ETGs_plus_redspiral': 'D',
                   'all_ETGs_plus_redspiral_centrals': 'D',
                   'all_ETGs_plus_redspiral_satellites': 'D'}
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
        box_size = data.metadata.boxsize[0].to(u.Mpc)
    
    
        #-------------------------------
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
    
        #---------------
        # Histograms   
        hist_bin_width = 0.2
        lower_mass_limit = 10**9.5
        upper_mass_limit = 10**13
        
        # start with histogram of masses (n, count)
        hist_masses, bin_edges =  np.histogram(np.log10(stellar_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        # turn into a density by dividing by volume (n/cMpc**3, density)
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**-3
        # divide by bin widths (log10 M*) to get dn/dlog10(M) in [cMpc**-3]
        hist_masses = hist_masses/hist_bin_width        
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(np.log10(stellar_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_err = (np.sqrt(hist_n)/(box_size)**3)/hist_bin_width
    
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask_finite = np.isfinite(np.log10(hist_masses))
        hist_masses = hist_masses[hist_mask_finite]
        bin_midpoints   = bin_midpoints[hist_mask_finite]
        hist_err    = hist_err[hist_mask_finite]
        hist_n      = hist_n[hist_mask_finite]
        
        
        #-----------
        # Line formatting
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        ms_i    = dict_ms[sample_input['name_of_preset']]
        
        # Plotting all points as dashed 
        axs.plot(np.flip(bin_midpoints), np.flip(hist_masses), color=linecol, ls=(0, (1, 1)), zorder=-4, path_effects=[outline])
        
        # Plotting only those with hist_n >= 10
        hist_masses_masked = np.ma.masked_where(hist_n < 10, hist_masses)
        hist_err_masked    = np.ma.masked_where(hist_n < 10, hist_err)
        
        #axs.fill_between(bin_midpoints, (10**(np.log10(hist_masses_masked-hist_err_masked))), (10**(np.log10(hist_masses_masked+hist_err_masked))), alpha=0.4, fc=linecol, zorder=-5)
        lines = axs.plot(bin_midpoints, hist_masses_masked, label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol, zorder=-3, path_effects=[outline])
        
    #-----------------
    # Add observations
    axs.plot([],[])    # skip C0
    if add_observational:
        """
        Pick observations we want to add
        """
        add_leja2020        = False     # z=0.2
        # Driver results:
        add_driver2022_old  = False      # z=0       #   VALUES FROM ROB MCGIBBON FILE
        add_driver2022      = True      # z=0       # values from paper, extracted and cosmo adjusted
        add_driver2022_etg  = True      # z=0
        add_driver2022_ltg  = False      # z=0
        # Kelvin results which are weird:       (extracted from graph)
        add_kelvin2014               = True   # z=0, morph types E and S0-Sa
        add_kelvin2014_etg           = False   # z=0, morph types E and S0-Sa
        add_kelvin2014_etg_plus_lbs  = True   # z=0, morph types E and S0-Sa, plus LBS galaxies (little blue spheroids)
        add_kelvin2014_ltg           = False   # z=0, morph types Sab-Scd, Sd-Irr)
        # Kelvin schechter functions results which are weird:
        add_kelvin2014_schechter               = False   # z=0, morph types E and S0-Sa
        add_kelvin2014_etg_schechter           = False   # z=0, morph types E and S0-Sa
        add_kelvin2014_etg_plus_lbs_schechter  = False   # z=0, morph types E and S0-Sa, plus LBS galaxies (little blue spheroids)
        add_kelvin2014_ltg_schechter           = False   # z=0, morph types Sab-Scd, Sd-Irr)
        
        if add_leja2020:    #  ($z=0.2$)
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
                
                obs_x     = file['x/z000.200_values'][:] * u.Unit(file['x/z000.200_values'].attrs['units'])
                obs_y = file['y/z000.200_values'][:] * u.Unit(file['y/z000.200_values'].attrs['units'])
                obs_y_err= file['y/z000.200_scatter'][:] * u.Unit(file['y/z000.200_scatter'].attrs['units'])
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Masses were adjusted using Planck+15 h=0.6777, COLIBRE uses h=0.681. Scale as (h_old/h_new)**2
            h_old = 0.6777
            h_new = 0.681
            obs_x = obs_x * (h_old/h_new)**2
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Leja+20 ($z=0.2$)', ls='-', linewidth=1, alpha=0.9, zorder=-20, c='C2')      
        if add_driver2022_old:      # ($z=0.0$)
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
            
                obs_x     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                obs_y = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                obs_y_err= file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Masses were adjusted using Planck+15 h=0.6777, COLIBRE uses h=0.681. Scale as (h_old/h_new)**2
            h_old = 0.6777
            h_new = 0.681
            obs_x = obs_x * (h_old/h_new)**2
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 (GAMA-DR4)', ls='none', linewidth=1, elinewidth=0.7, marker='^', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C0')              
        if add_driver2022:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""

                obs_x     = file['data/total/x/values'][:] * u.Unit(file['data/total/x/values'].attrs['units'])
                obs_y = file['data/total/y/values'][:] * u.Unit(file['data/total/y/values'].attrs['units'])
                obs_y_err= file['data/total/y/scatter'][:] * u.Unit(file['data/total/y/scatter'].attrs['units'])
            
            
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 (GAMA-DR4)', ls='none', linewidth=1, elinewidth=0.7, marker='s', ms=3, alpha=0.9, zorder=-20, c='C7', markerfacecolor='none', markeredgecolor='C7', markeredgewidth=0.6)
        if add_driver2022_etg:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
            
                obs_x     = file['data/etg/x/values'][:] * u.Unit(file['data/etg/x/values'].attrs['units'])
                obs_y = file['data/etg/y/values'][:] * u.Unit(file['data/etg/y/values'].attrs['units'])
                obs_y_err= file['data/etg/y/scatter'][:] * u.Unit(file['data/etg/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 ETGs', ls='none', linewidth=1, elinewidth=0.7, marker='^', ms=3, markerfacecolor='C4', alpha=0.9, zorder=-20, c='C4')
        if add_driver2022_ltg:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
            
                obs_x     = file['data/ltg/x/values'][:] * u.Unit(file['data/ltg/x/values'].attrs['units'])
                obs_y = file['data/ltg/y/values'][:] * u.Unit(file['data/ltg/y/values'].attrs['units'])
                obs_y_err= file['data/ltg/y/scatter'][:] * u.Unit(file['data/ltg/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 LTGs', ls='none', linewidth=1, elinewidth=0.7, marker='D', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C4')
        
        if add_kelvin2014:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
                
                obs_x     = file['data/total/x/values'][:] * u.Unit(file['data/total/x/values'].attrs['units'])
                obs_y = file['data/total/y/values'][:] * u.Unit(file['data/total/y/values'].attrs['units'])
                obs_y_err= file['data/total/y/scatter'][:] * u.Unit(file['data/total/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 (GAMA-DR1)', ls='none', linewidth=1, elinewidth=0.7, marker='o', ms=3, alpha=0.9, zorder=-20, c='C5', markerfacecolor='none', markeredgecolor='C5', markeredgewidth=0.6)
        if add_kelvin2014_etg:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
            
                obs_x     = file['data/etg/x/values'][:] * u.Unit(file['data/etg/x/values'].attrs['units'])
                obs_y = file['data/etg/y/values'][:] * u.Unit(file['data/etg/y/values'].attrs['units'])
                obs_y_err= file['data/etg/y/scatter'][:] * u.Unit(file['data/etg/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 ETGs', ls='none', linewidth=1, elinewidth=0.7, marker='v', ms=3, markerfacecolor='C6', alpha=0.9, zorder=-20, c='C6')
        if add_kelvin2014_etg_plus_lbs:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
            
                obs_x     = file['data/etg_plus_lbs/x/values'][:] * u.Unit(file['data/etg_plus_lbs/x/values'].attrs['units'])
                obs_y = file['data/etg_plus_lbs/y/values'][:] * u.Unit(file['data/etg_plus_lbs/y/values'].attrs['units'])
                obs_y_err= file['data/etg_plus_lbs/y/scatter'][:] * u.Unit(file['data/etg_plus_lbs/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 ETGs incl. LBS', ls='none', linewidth=1, elinewidth=0.7, marker='v', ms=3, alpha=0.9, zorder=-20, markerfacecolor='C3', c='C3')
        if add_kelvin2014_ltg:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
            
                obs_x     = file['data/ltg/x/values'][:] * u.Unit(file['data/ltg/x/values'].attrs['units'])
                obs_y = file['data/ltg/y/values'][:] * u.Unit(file['data/ltg/y/values'].attrs['units'])
                obs_y_err= file['data/ltg/y/scatter'][:] * u.Unit(file['data/ltg/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 LTGs', ls='none', linewidth=1, elinewidth=0.7, marker='D', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C8')
        
        if add_kelvin2014_schechter:  # ($z=0.0$)
            # Schechter line values, see Schechter 1976
            add_kelvin2014 = {'SchechterFunc_1': {'log_Mstar': 10.64,
                                                     'log_Mstar_err': 0.07,       # 1 standard deviation errors
                                                     'log_Mstar_units': u.Msun,
                                                     'alpha': -0.43,
                                                     'alpha_err': 0.35,
                                                     'alpha_units': u.dimensionless,
                                                     'phi_star': 4.18,             #* (u.Mpc ** -3)
                                                     'phi_star_err': 1.52,         #* (u.Mpc ** -3)
                                                     'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                          'SchechterFunc_2': {'log_Mstar': 10.64,
                                                     'log_Mstar_err': 0.07,       # 1 standard deviation errors
                                                     'log_Mstar_units': u.Msun,
                                                     'alpha': -1.50,
                                                     'alpha_err': 0.22,
                                                     'alpha_units': u.dimensionless,
                                                     'phi_star': 0.74,             #* (u.Mpc ** -3)
                                                     'phi_star_err': 1.13,         #* (u.Mpc ** -3)
                                                     'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                            'h_obs': 0.70}
                                    
            # adjust masses for cosmology
            h_colibre = 0.681
            log_Mstar_1 = np.log10((10**add_kelvin2014['SchechterFunc_1']['log_Mstar']) * (add_kelvin2014['h_obs']/h_colibre)**2)
            log_Mstar_2 = np.log10((10**add_kelvin2014['SchechterFunc_2']['log_Mstar']) * (add_kelvin2014['h_obs']/h_colibre)**2)
            
            # Mass range (linear scale)
            dlogM = 0.1
            M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
            # Schechter function in linear mass units
            def schechter_mass(M, M_star, alpha, phi_star):
                return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
            
            # Evaluate Schechter function
            phi_M_1 = schechter_mass(M, 10**log_Mstar_1, add_kelvin2014['SchechterFunc_1']['alpha'], add_kelvin2014['SchechterFunc_1']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
            phi_M_2 = schechter_mass(M, 10**log_Mstar_2, add_kelvin2014['SchechterFunc_2']['alpha'], add_kelvin2014['SchechterFunc_2']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
            
            # Plot double schechter
            axs.plot(np.log10(M), np.log10(phi_M_1 + phi_M_2), label='Kelvin+14 (GAMA-DR1)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C6')  
        if add_kelvin2014_etg_schechter:  # ($z=0.0$)
            # Schechter line values, see Schechter 1976
            dict_kelvin2014_etg = {'SchechterFunc': {'log_Mstar': 10.60,
                                                     'log_Mstar_err': 0.08,       # 1 standard deviation errors
                                                     'log_Mstar_units': u.Msun,
                                                     'alpha': -0.27,
                                                     'alpha_err': 0.20,
                                                     'alpha_units': u.dimensionless,
                                                     'phi_star': 3.96,             #* (u.Mpc ** -3)
                                                     'phi_star_err': 0.37,         #* (u.Mpc ** -3)
                                                     'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                    'h_obs': 0.70}
                                    
            # adjust masses for cosmology
            h_colibre = 0.681
            log_Mstar = np.log10((10**dict_kelvin2014_etg['SchechterFunc']['log_Mstar']) * (dict_kelvin2014_etg['h_obs']/h_colibre)**2)
            
            # Mass range (linear scale)
            dlogM = 0.1
            M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
            # Schechter function in linear mass units
            def schechter_mass(M, M_star, alpha, phi_star):
                return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
            # Evaluate Schechter function
            phi_M = schechter_mass(M, 10**log_Mstar, dict_kelvin2014_etg['SchechterFunc']['alpha'], dict_kelvin2014_etg['SchechterFunc']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
                
            axs.plot(np.log10(M), np.log10(phi_M), label='Kelvin+14 ETGs (GAMA-DR1)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C2')  
        if add_kelvin2014_etg_plus_lbs_schechter:  # ($z=0.0$)
            # Schechter line values, see Schechter 1976
            add_kelvin2014_etg_plus_lbs = {'SchechterFunc_1': {'log_Mstar': 10.65,
                                                     'log_Mstar_err': 0.08,       # 1 standard deviation errors
                                                     'log_Mstar_units': u.Msun,
                                                     'alpha': -0.37,
                                                     'alpha_err': 0.23,
                                                     'alpha_units': u.dimensionless,
                                                     'phi_star': 3.63,             #* (u.Mpc ** -3)
                                                     'phi_star_err': 1.38,         #* (u.Mpc ** -3)
                                                     'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                          'SchechterFunc_2': {'log_Mstar': 10.65,
                                                     'log_Mstar_err': 0.08,       # 1 standard deviation errors
                                                     'log_Mstar_units': u.Msun,
                                                     'alpha': -2.13,
                                                     'alpha_err': 1.23,
                                                     'alpha_units': u.dimensionless,
                                                     'phi_star': 0.01,             #* (u.Mpc ** -3)
                                                     'phi_star_err': 0.05,         #* (u.Mpc ** -3)
                                                     'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                            'h_obs': 0.70}
                                    
            # adjust masses for cosmology
            h_colibre = 0.681
            log_Mstar_1 = np.log10((10**add_kelvin2014_etg_plus_lbs['SchechterFunc_1']['log_Mstar']) * (add_kelvin2014_etg_plus_lbs['h_obs']/h_colibre)**2)
            log_Mstar_2 = np.log10((10**add_kelvin2014_etg_plus_lbs['SchechterFunc_2']['log_Mstar']) * (add_kelvin2014_etg_plus_lbs['h_obs']/h_colibre)**2)
            
            # Mass range (linear scale)
            dlogM = 0.1
            M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
            # Schechter function in linear mass units
            def schechter_mass(M, M_star, alpha, phi_star):
                return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
            
            # Evaluate Schechter function
            phi_M_1 = schechter_mass(M, 10**log_Mstar_1, add_kelvin2014_etg_plus_lbs['SchechterFunc_1']['alpha'], add_kelvin2014_etg_plus_lbs['SchechterFunc_1']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
            phi_M_2 = schechter_mass(M, 10**log_Mstar_2, add_kelvin2014_etg_plus_lbs['SchechterFunc_2']['alpha'], add_kelvin2014_etg_plus_lbs['SchechterFunc_2']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
            
            # Plot double schechter
            axs.plot(np.log10(M), np.log10(phi_M_1 + phi_M_2), label='Kelvin+14 ETGs incl. LBS\n(GAMA-DR1)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C5')  
        if add_kelvin2014_ltg_schechter:
            # Schechter line values, see Schechter 1976
            dict_kelvin2014_ltg = {'SchechterFunc': {'log_Mstar': 10.70,
                                                     'log_Mstar_err': 0.07,       # 1 standard deviation errors
                                                     'log_Mstar_units': u.Msun,
                                                     'alpha': -1.37,
                                                     'alpha_err': 0.04,
                                                     'alpha_units': u.dimensionless,
                                                     'phi_star': 0.98,             #* (u.Mpc ** -3)
                                                     'phi_star_err': 0.14,         #* (u.Mpc ** -3)
                                                     'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                    'h_obs': 0.70}
                                    
            # adjust masses for cosmology
            h_colibre = 0.681
            log_Mstar = np.log10((10**dict_kelvin2014_ltg['SchechterFunc']['log_Mstar']) * (dict_kelvin2014_ltg['h_obs']/h_colibre)**2)
            
            # Mass range (linear scale)
            dlogM = 0.1
            M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
            # Schechter function in linear mass units
            def schechter_mass(M, M_star, alpha, phi_star):
                return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
            # Evaluate Schechter function
            phi_M = schechter_mass(M, 10**log_Mstar, dict_kelvin2014_ltg['SchechterFunc']['alpha'], dict_kelvin2014_ltg['SchechterFunc']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
                
            axs.plot(np.log10(M), np.log10(phi_M), label='Kelvin+14 (LTGs)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C4')
            
            
            
    #-----------
    # Axis formatting
    plt.xlim(9.5, 12.5)
    plt.ylim(10**(-5.5), 10**(-1.5))
    axs.minorticks_on()
    #plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))

    dict_aperture = {'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    plt.xlabel(r'log$_{10}$ $M_*$ (%s) [M$_{\odot}$]'%(dict_aperture[aperture]))
    plt.ylabel(r'dn/dlog$_{10}$($M_*$) [cMpc$^{-3}$]')
      
    #-----------  
    # Annotations
    plt.text(0.80, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    
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
    #axs.set_title(r'%s%s%s' %(title_run_dict[sample_input['simulation_run']], title_type_dict[sample_input['simulation_type']], title_text_in), size=7, loc='left', pad=3, bbox={"edgecolor": title_color_dict[run_name_title], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})
    
    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/stellar_mass_func/%s_%s_ALL_SAMPLES_sample_stellarmassfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/stellar_mass_func/%s_%s_ALL_SAMPLES_sample_stellarmassfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns 3-panel stellar mass function for given set of samples
# Returns stellar mass function for given set of samples
def _sample_stellar_mass_function_3x1(csv_samples1 = [], csv_samples2 = [], csv_samples3 = [], title_text_in = '',
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[7.5, 2.6], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_plus_redspiral': 'ETGs (incl. FRs)'}
    dict_colors = {'all_galaxies': 'k',
                   'all_ETGs': 'C0',
                   'all_ETGs_plus_redspiral': 'C1'}
    dict_ls     = {'all_galaxies': '-',
                   'all_ETGs': '-',
                   'all_ETGs_plus_redspiral': '-'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_plus_redspiral': 'D'}
    sim_name_title_list = []
    for ax_i, csv_samples in zip((ax1, ax2, ax3), (csv_samples1, csv_samples2, csv_samples3)):
        for i, csv_sample_i in enumerate(csv_samples):
            
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
            # Get stelmass, molecular hydrogen, and magnitude data
            stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
            stellar_mass.convert_to_units('Msun')
    
            central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
    
            #---------------
            # Histograms   
            hist_bin_width = 0.2
            lower_mass_limit = 10**9.5
            upper_mass_limit = 10**13
        
            # start with histogram of masses (n, count)
            hist_masses, bin_edges =  np.histogram(np.log10(stellar_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
            # turn into a density by dividing by volume (n/cMpc**3, density)
            hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**-3
            # divide by bin widths (log10 M*) to get dn/dlog10(M) in [cMpc**-3]
            hist_masses = hist_masses/hist_bin_width        
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(np.log10(stellar_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
            hist_err = (np.sqrt(hist_n)/(box_size)**3)/hist_bin_width
    
            # Masking out nans
            with np.errstate(divide='ignore', invalid='ignore'):
                hist_mask_finite = np.isfinite(np.log10(hist_masses))
            hist_masses = hist_masses[hist_mask_finite]
            bin_midpoints   = bin_midpoints[hist_mask_finite]
            hist_err    = hist_err[hist_mask_finite]
            hist_n      = hist_n[hist_mask_finite]
        
        
            #-----------
            # Line formatting
            label_i = dict_labels[sample_input['name_of_preset']]
            linecol = dict_colors[sample_input['name_of_preset']]
            ls_i    = dict_ls[sample_input['name_of_preset']]
            ms_i    = dict_ms[sample_input['name_of_preset']]
        
            # Plotting all points as dashed 
            ax_i.plot(np.flip(bin_midpoints), np.flip(hist_masses), color=linecol, ls=(0, (1, 1)), zorder=-4, path_effects=[outline])
        
            # Plotting only those with hist_n >= 10
            hist_masses_masked = np.ma.masked_where(hist_n < 10, hist_masses)
            hist_err_masked    = np.ma.masked_where(hist_n < 10, hist_err)
        
            #ax_i.fill_between(bin_midpoints, (10**(np.log10(hist_masses_masked-hist_err_masked))), (10**(np.log10(hist_masses_masked+hist_err_masked))), alpha=0.4, fc=linecol, zorder=-5)
            lines = ax_i.plot(bin_midpoints, hist_masses_masked, label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol, zorder=-3, path_effects=[outline])
        
        #-----------------
        # Add observations
        ax_i.plot([],[])    # skip C0
        if add_observational:
            """
            Pick observations we want to add
            """
            add_leja2020        = False     # z=0.2
            # Driver results:
            add_driver2022_old  = False      # z=0       #   VALUES FROM ROB MCGIBBON FILE
            add_driver2022      = True      # z=0       # values from paper, extracted and cosmo adjusted
            add_driver2022_etg  = True      # z=0
            add_driver2022_ltg  = False      # z=0
            # Kelvin results which are weird:       (extracted from graph)
            add_kelvin2014               = True   # z=0, morph types E and S0-Sa
            add_kelvin2014_etg           = False   # z=0, morph types E and S0-Sa
            add_kelvin2014_etg_plus_lbs  = True   # z=0, morph types E and S0-Sa, plus LBS galaxies (little blue spheroids)
            add_kelvin2014_ltg           = False   # z=0, morph types Sab-Scd, Sd-Irr)
            # Kelvin schechter functions results which are weird:
            add_kelvin2014_schechter               = False   # z=0, morph types E and S0-Sa
            add_kelvin2014_etg_schechter           = False   # z=0, morph types E and S0-Sa
            add_kelvin2014_etg_plus_lbs_schechter  = False   # z=0, morph types E and S0-Sa, plus LBS galaxies (little blue spheroids)
            add_kelvin2014_ltg_schechter           = False   # z=0, morph types Sab-Scd, Sd-Irr)
        
            if add_leja2020:    #  ($z=0.2$)
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
                
                    obs_x     = file['x/z000.200_values'][:] * u.Unit(file['x/z000.200_values'].attrs['units'])
                    obs_y = file['y/z000.200_values'][:] * u.Unit(file['y/z000.200_values'].attrs['units'])
                    obs_y_err= file['y/z000.200_scatter'][:] * u.Unit(file['y/z000.200_scatter'].attrs['units'])
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Masses were adjusted using Planck+15 h=0.6777, COLIBRE uses h=0.681. Scale as (h_old/h_new)**2
                h_old = 0.6777
                h_new = 0.681
                obs_x = obs_x * (h_old/h_new)**2
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Leja+20 ($z=0.2$)', ls='-', linewidth=1, alpha=0.9, zorder=-20, c='C2')      
            if add_driver2022_old:      # ($z=0.0$)
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
            
                    obs_x     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                    obs_y = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                    obs_y_err= file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
                
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Masses were adjusted using Planck+15 h=0.6777, COLIBRE uses h=0.681. Scale as (h_old/h_new)**2
                h_old = 0.6777
                h_new = 0.681
                obs_x = obs_x * (h_old/h_new)**2
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 (GAMA-DR4)', ls='none', linewidth=1, elinewidth=0.7, marker='^', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C0')              
            if add_driver2022:      # ($z=0.0$)
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5'%obs_dir, 'r') as file:
                
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""

                    obs_x     = file['data/total/x/values'][:] * u.Unit(file['data/total/x/values'].attrs['units'])
                    obs_y = file['data/total/y/values'][:] * u.Unit(file['data/total/y/values'].attrs['units'])
                    obs_y_err= file['data/total/y/scatter'][:] * u.Unit(file['data/total/y/scatter'].attrs['units'])
            
            
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 (GAMA-DR4)', ls='none', linewidth=1, elinewidth=0.7, marker='s', ms=3, alpha=0.9, zorder=-20, c='C7', markerfacecolor='none', markeredgecolor='C7', markeredgewidth=0.6)
            if add_driver2022_etg:      # ($z=0.0$)
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5'%obs_dir, 'r') as file:
                
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
            
                    obs_x     = file['data/etg/x/values'][:] * u.Unit(file['data/etg/x/values'].attrs['units'])
                    obs_y = file['data/etg/y/values'][:] * u.Unit(file['data/etg/y/values'].attrs['units'])
                    obs_y_err= file['data/etg/y/scatter'][:] * u.Unit(file['data/etg/y/scatter'].attrs['units'])
                
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 ETGs', ls='none', linewidth=1, elinewidth=0.7, marker='^', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C4')
            if add_driver2022_ltg:      # ($z=0.0$)
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5'%obs_dir, 'r') as file:
                
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
            
                    obs_x     = file['data/ltg/x/values'][:] * u.Unit(file['data/ltg/x/values'].attrs['units'])
                    obs_y = file['data/ltg/y/values'][:] * u.Unit(file['data/ltg/y/values'].attrs['units'])
                    obs_y_err= file['data/ltg/y/scatter'][:] * u.Unit(file['data/ltg/y/scatter'].attrs['units'])
                
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Driver+22 LTGs', ls='none', linewidth=1, elinewidth=0.7, marker='D', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C4')
        
            if add_kelvin2014:      # ($z=0.0$)
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
                
                    obs_x     = file['data/total/x/values'][:] * u.Unit(file['data/total/x/values'].attrs['units'])
                    obs_y = file['data/total/y/values'][:] * u.Unit(file['data/total/y/values'].attrs['units'])
                    obs_y_err= file['data/total/y/scatter'][:] * u.Unit(file['data/total/y/scatter'].attrs['units'])
                
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 (GAMA-DR1)', ls='none', linewidth=1, elinewidth=0.7, marker='o', ms=3, alpha=0.9, zorder=-20, c='C5', markerfacecolor='none', markeredgecolor='C5', markeredgewidth=0.6)
            if add_kelvin2014_etg:      # ($z=0.0$)
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
            
                    obs_x     = file['data/etg/x/values'][:] * u.Unit(file['data/etg/x/values'].attrs['units'])
                    obs_y = file['data/etg/y/values'][:] * u.Unit(file['data/etg/y/values'].attrs['units'])
                    obs_y_err= file['data/etg/y/scatter'][:] * u.Unit(file['data/etg/y/scatter'].attrs['units'])
                
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 ETGs', ls='none', linewidth=1, elinewidth=0.7, marker='v', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C6')
            if add_kelvin2014_etg_plus_lbs:      # ($z=0.0$)
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
            
                    obs_x     = file['data/etg_plus_lbs/x/values'][:] * u.Unit(file['data/etg_plus_lbs/x/values'].attrs['units'])
                    obs_y = file['data/etg_plus_lbs/y/values'][:] * u.Unit(file['data/etg_plus_lbs/y/values'].attrs['units'])
                    obs_y_err= file['data/etg_plus_lbs/y/scatter'][:] * u.Unit(file['data/etg_plus_lbs/y/scatter'].attrs['units'])
                
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 ETGs incl. LBS', ls='none', linewidth=1, elinewidth=0.7, marker='v', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C3')
            if add_kelvin2014_ltg:      # ($z=0.0$)
                # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
                with h5py.File('%s/GalaxyStellarMassFunction/Kelvin2014.hdf5'%obs_dir, 'r') as file:
                
                    """# A description of the file and data is in the metadata
                    print(f'File keys: {file.keys()}')
                    for k, v in file['metadata'].attrs.items():
                        print(f'{k}: {v}')
                    # Data
                    print(file['x'].keys())
                    print(file['y'].keys())
                    print(' ')"""
            
                    obs_x     = file['data/ltg/x/values'][:] * u.Unit(file['data/ltg/x/values'].attrs['units'])
                    obs_y = file['data/ltg/y/values'][:] * u.Unit(file['data/ltg/y/values'].attrs['units'])
                    obs_y_err= file['data/ltg/y/scatter'][:] * u.Unit(file['data/ltg/y/scatter'].attrs['units'])
                
                obs_x = obs_x.to('Msun')
                obs_y = obs_y.to('Mpc**(-3)')
                obs_y_err = obs_y_err.to('Mpc**(-3)')
            
                # Convert to log
                #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
                #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
                #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
            
                ax_i.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Kelvin+14 LTGs', ls='none', linewidth=1, elinewidth=0.7, marker='D', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='C8')
        
            if add_kelvin2014_schechter:  # ($z=0.0$)
                # Schechter line values, see Schechter 1976
                add_kelvin2014 = {'SchechterFunc_1': {'log_Mstar': 10.64,
                                                         'log_Mstar_err': 0.07,       # 1 standard deviation errors
                                                         'log_Mstar_units': u.Msun,
                                                         'alpha': -0.43,
                                                         'alpha_err': 0.35,
                                                         'alpha_units': u.dimensionless,
                                                         'phi_star': 4.18,             #* (u.Mpc ** -3)
                                                         'phi_star_err': 1.52,         #* (u.Mpc ** -3)
                                                         'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                              'SchechterFunc_2': {'log_Mstar': 10.64,
                                                         'log_Mstar_err': 0.07,       # 1 standard deviation errors
                                                         'log_Mstar_units': u.Msun,
                                                         'alpha': -1.50,
                                                         'alpha_err': 0.22,
                                                         'alpha_units': u.dimensionless,
                                                         'phi_star': 0.74,             #* (u.Mpc ** -3)
                                                         'phi_star_err': 1.13,         #* (u.Mpc ** -3)
                                                         'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                                'h_obs': 0.70}
                                    
                # adjust masses for cosmology
                h_colibre = 0.681
                log_Mstar_1 = np.log10((10**add_kelvin2014['SchechterFunc_1']['log_Mstar']) * (add_kelvin2014['h_obs']/h_colibre)**2)
                log_Mstar_2 = np.log10((10**add_kelvin2014['SchechterFunc_2']['log_Mstar']) * (add_kelvin2014['h_obs']/h_colibre)**2)
            
                # Mass range (linear scale)
                dlogM = 0.1
                M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
                # Schechter function in linear mass units
                def schechter_mass(M, M_star, alpha, phi_star):
                    return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
            
                # Evaluate Schechter function
                phi_M_1 = schechter_mass(M, 10**log_Mstar_1, add_kelvin2014['SchechterFunc_1']['alpha'], add_kelvin2014['SchechterFunc_1']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
                phi_M_2 = schechter_mass(M, 10**log_Mstar_2, add_kelvin2014['SchechterFunc_2']['alpha'], add_kelvin2014['SchechterFunc_2']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
            
                # Plot double schechter
                ax_i.plot(np.log10(M), np.log10(phi_M_1 + phi_M_2), label='Kelvin+14 (GAMA-DR1)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C6')  
            if add_kelvin2014_etg_schechter:  # ($z=0.0$)
                # Schechter line values, see Schechter 1976
                dict_kelvin2014_etg = {'SchechterFunc': {'log_Mstar': 10.60,
                                                         'log_Mstar_err': 0.08,       # 1 standard deviation errors
                                                         'log_Mstar_units': u.Msun,
                                                         'alpha': -0.27,
                                                         'alpha_err': 0.20,
                                                         'alpha_units': u.dimensionless,
                                                         'phi_star': 3.96,             #* (u.Mpc ** -3)
                                                         'phi_star_err': 0.37,         #* (u.Mpc ** -3)
                                                         'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                        'h_obs': 0.70}
                                    
                # adjust masses for cosmology
                h_colibre = 0.681
                log_Mstar = np.log10((10**dict_kelvin2014_etg['SchechterFunc']['log_Mstar']) * (dict_kelvin2014_etg['h_obs']/h_colibre)**2)
            
                # Mass range (linear scale)
                dlogM = 0.1
                M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
                # Schechter function in linear mass units
                def schechter_mass(M, M_star, alpha, phi_star):
                    return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
                # Evaluate Schechter function
                phi_M = schechter_mass(M, 10**log_Mstar, dict_kelvin2014_etg['SchechterFunc']['alpha'], dict_kelvin2014_etg['SchechterFunc']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
                
                ax_i.plot(np.log10(M), np.log10(phi_M), label='Kelvin+14 ETGs (GAMA-DR1)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C2')  
            if add_kelvin2014_etg_plus_lbs_schechter:  # ($z=0.0$)
                # Schechter line values, see Schechter 1976
                add_kelvin2014_etg_plus_lbs = {'SchechterFunc_1': {'log_Mstar': 10.65,
                                                         'log_Mstar_err': 0.08,       # 1 standard deviation errors
                                                         'log_Mstar_units': u.Msun,
                                                         'alpha': -0.37,
                                                         'alpha_err': 0.23,
                                                         'alpha_units': u.dimensionless,
                                                         'phi_star': 3.63,             #* (u.Mpc ** -3)
                                                         'phi_star_err': 1.38,         #* (u.Mpc ** -3)
                                                         'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                              'SchechterFunc_2': {'log_Mstar': 10.65,
                                                         'log_Mstar_err': 0.08,       # 1 standard deviation errors
                                                         'log_Mstar_units': u.Msun,
                                                         'alpha': -2.13,
                                                         'alpha_err': 1.23,
                                                         'alpha_units': u.dimensionless,
                                                         'phi_star': 0.01,             #* (u.Mpc ** -3)
                                                         'phi_star_err': 0.05,         #* (u.Mpc ** -3)
                                                         'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                                'h_obs': 0.70}
                                    
                # adjust masses for cosmology
                h_colibre = 0.681
                log_Mstar_1 = np.log10((10**add_kelvin2014_etg_plus_lbs['SchechterFunc_1']['log_Mstar']) * (add_kelvin2014_etg_plus_lbs['h_obs']/h_colibre)**2)
                log_Mstar_2 = np.log10((10**add_kelvin2014_etg_plus_lbs['SchechterFunc_2']['log_Mstar']) * (add_kelvin2014_etg_plus_lbs['h_obs']/h_colibre)**2)
            
                # Mass range (linear scale)
                dlogM = 0.1
                M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
                # Schechter function in linear mass units
                def schechter_mass(M, M_star, alpha, phi_star):
                    return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
            
                # Evaluate Schechter function
                phi_M_1 = schechter_mass(M, 10**log_Mstar_1, add_kelvin2014_etg_plus_lbs['SchechterFunc_1']['alpha'], add_kelvin2014_etg_plus_lbs['SchechterFunc_1']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
                phi_M_2 = schechter_mass(M, 10**log_Mstar_2, add_kelvin2014_etg_plus_lbs['SchechterFunc_2']['alpha'], add_kelvin2014_etg_plus_lbs['SchechterFunc_2']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
            
                # Plot double schechter
                ax_i.plot(np.log10(M), np.log10(phi_M_1 + phi_M_2), label='Kelvin+14 ETGs incl. LBS\n(GAMA-DR1)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C5')  
            if add_kelvin2014_ltg_schechter:
                # Schechter line values, see Schechter 1976
                dict_kelvin2014_ltg = {'SchechterFunc': {'log_Mstar': 10.70,
                                                         'log_Mstar_err': 0.07,       # 1 standard deviation errors
                                                         'log_Mstar_units': u.Msun,
                                                         'alpha': -1.37,
                                                         'alpha_err': 0.04,
                                                         'alpha_units': u.dimensionless,
                                                         'phi_star': 0.98,             #* (u.Mpc ** -3)
                                                         'phi_star_err': 0.14,         #* (u.Mpc ** -3)
                                                         'phi_star_units': u.Mpc**-3},   # ALSO REMOVE DEX units [dex **-1  Mpc ** -3]
                                        'h_obs': 0.70}
                                    
                # adjust masses for cosmology
                h_colibre = 0.681
                log_Mstar = np.log10((10**dict_kelvin2014_ltg['SchechterFunc']['log_Mstar']) * (dict_kelvin2014_ltg['h_obs']/h_colibre)**2)
            
                # Mass range (linear scale)
                dlogM = 0.1
                M = 10**np.arange(9, 13, dlogM)            #np.logspace(9, 13, 500)  # from 10^9 to 10^13 Msun
                # Schechter function in linear mass units
                def schechter_mass(M, M_star, alpha, phi_star):
                    return np.log(10) * phi_star * (M / M_star)**(alpha + 1) * np.exp(-M / M_star)
                # Evaluate Schechter function
                phi_M = schechter_mass(M, 10**log_Mstar, dict_kelvin2014_ltg['SchechterFunc']['alpha'], dict_kelvin2014_ltg['SchechterFunc']['phi_star']) * dlogM * 0.01     # multiply by bin width, and remove dex**-1 dependence
                
                ax_i.plot(np.log10(M), np.log10(phi_M), label='Kelvin+14 (LTGs)', ls='-', linewidth=1, alpha=0.9, zorder=-30, c='C4')
            
        
         
            
    #-----------
    # Axis formatting
    for ax_i in (ax1, ax2, ax3):
        ax_i.set_xlim(9.5, 12.5)
        ax_i.set_ylim(10**(-5.5), 10**(-1.5))
        ax_i.minorticks_on()
        #plt.xscale("log")
        ax_i.set_yscale("log")
        #plt.yticks(np.arange(-5, -1.4, 0.5))
        #plt.xticks(np.arange(9.5, 12.5, 0.5))
        
        dict_aperture = {'exclusive_sphere_30kpc': '30 pkpc', 
                         'exclusive_sphere_50kpc': '50 pkpc'}
        ax_i.set_xlabel(r'log$_{10}$ $M_*$ (%s) [M$_{\odot}$]'%(dict_aperture[aperture]))
        ax_i.set_ylabel(r'dn/dlog$_{10}$($M_*$) [cMpc$^{-3}$]')

    # axs formatting tweaks
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])

    #-----------  
    # Annotations
    ax1.text(0.77, 0.9, '${z=%.2f}$' %z, fontsize=7, transform = ax1.transAxes)
    ax2.text(0.77, 0.9, '${z=%.2f}$' %z, fontsize=7, transform = ax2.transAxes)
    ax3.text(0.77, 0.9, '${z=%.2f}$' %z, fontsize=7, transform = ax3.transAxes)
    
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
    ax1.set_title(r'L100m6%s' %(title_text_in), size=7, x=0.1, y=1.02, pad=3, c=title_color_dict['L100m6'], bbox={"edgecolor": title_color_dict['L100m6'], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})
    ax2.set_title(r'L200m6%s' %(title_text_in), size=7, x=0.1, y=1.02, pad=3, c=title_color_dict['L200m6'], bbox={"edgecolor": title_color_dict['L200m6'], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})
    ax3.set_title(r'L100m6h%s' %(title_text_in), size=7, x=0.12, y=1.02, pad=3, c=title_color_dict['L100m6h'], bbox={"edgecolor": title_color_dict['L100m6h'], "facecolor": "none", "linewidth": 1, "pad": 0.3, "boxstyle": 'round'})
    
    
    #-----------
    # Legend
    ax1.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/stellar_mass_func/LXXX_both_ALL_SAMPLES_sample_stellarmassfunc_%s%s.%s" %(fig_dir, sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/stellar_mass_func/LXXX_both_ALL_SAMPLES_sample_stellarmassfunc_%s%s.%s" %(fig_dir, sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#------------------
# Returns H2 mass function for given set of samples
def _sample_H2_mass_function(csv_samples = [],  title_text_in = '',
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          aperture_h2 = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
                   'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_centrals': 'ETGs (excl. FRs)',
                   'all_ETGs_satellites': 'ETGs (excl. FRs)',
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
                   'all_ETGs': '-',
                   'all_ETGs_centrals': '-',
                   'all_ETGs_satellites': '-',
                   'all_ETGs_plus_redspiral': '-',
                   'all_ETGs_plus_redspiral_centrals': '-',
                   'all_ETGs_plus_redspiral_satellites': '-'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_galaxies_centrals': 'o',
                   'all_galaxies_satellites': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_centrals': 's',
                   'all_ETGs_satellites': 's',
                   'all_ETGs_plus_redspiral': 'D',
                   'all_ETGs_plus_redspiral_centrals': 'D',
                   'all_ETGs_plus_redspiral_satellites': 'D'}
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
        box_size = data.metadata.boxsize[0].to(u.Mpc)
    
    
        #-------------------------------
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        
        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        H2_mass.convert_to_units('Msun')
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
    
        #---------------
        # Histograms   
        hist_bin_width = 0.2
        lower_mass_limit = 10**4
        upper_mass_limit = 10**11
        
    
        hist_masses, bin_edges =  np.histogram(np.log10(H2_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**3
        hist_masses = hist_masses/hist_bin_width        # density
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(np.log10(H2_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_err = (np.sqrt(hist_n)/(box_size)**3)/hist_bin_width
    
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask_finite = np.isfinite(np.log10(hist_masses))
        hist_masses = hist_masses[hist_mask_finite]
        bin_midpoints   = bin_midpoints[hist_mask_finite]
        hist_err    = hist_err[hist_mask_finite]
        hist_n      = hist_n[hist_mask_finite]              # hist of bin counts
        
        #-----------
        # Line formatting
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        ms_i    = dict_ms[sample_input['name_of_preset']]
        
        # Plotting all points as dashed 
        axs.plot(np.flip(bin_midpoints), np.flip(hist_masses), color=linecol, ls=(0, (1, 1)), zorder=-4, path_effects=[outline])
        
        # Plotting only those with hist_n >= 10
        hist_masses_masked = np.ma.masked_where(hist_n < 10, hist_masses)
        hist_err_masked    = np.ma.masked_where(hist_n < 10, hist_err)
        #axs.fill_between(bin_midpoints, (10**(np.log10(hist_masses_masked-hist_err_masked))), (10**(np.log10(hist_masses_masked+hist_err_masked))), alpha=0.4, fc=linecol, zorder=-5)
        lines = axs.plot(bin_midpoints, hist_masses_masked, label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol, zorder=-3, path_effects=[outline])
        
        """# Find bins with more than 10 entry
        hist_mask_n = hist_n >= 10
        
        # Make fake arrays to add to plotted entries to manually get math.nan values, thus a break
        fa_good = np.zeros(len(bin_midpoints))
        fa_bad = np.zeros(len(bin_midpoints))
        fa_good[~hist_mask_n] = math.nan
        fa_bad[hist_mask_n] = math.nan
        
        #-----------
        # Plotting error for n >= 10, and single dots for others
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        ms_i    = dict_ms[sample_input['name_of_preset']]
        
        axs.fill_between(bin_midpoints+fa_good, (10**(np.log10(hist_masses-hist_err)))+fa_good, (10**(np.log10(hist_masses+hist_err)))+fa_good, alpha=0.4, fc=linecol, zorder=-5)
        lines = axs.plot(bin_midpoints+fa_good, hist_masses+fa_good, label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol, zorder=-3)
        axs.plot(bin_midpoints+fa_bad, hist_masses+fa_bad, marker=ms_i, color=lines[0].get_color(), ls='--', zorder=-4)"""
        
    #-----------------
    # Add observations
    axs.plot([],[])    # skip C0
    if add_observational:
        """
        Pick observations we want to add
        """
        add_andreani2020    = False
        add_fletcher2021    = False     # most robust where sample bias has been taken into account
        add_lagos2015       = False
        #add_guo2023 = True
        
        add_lagos2014_uncorr       = True
        add_lagos2014_1Vcorr       = True
        
        if add_andreani2020:
            logm = np.arange(6.5, 10.2, 0.25)
            phi      = np.array([-2.36, -2.14, -2.02, -1.96, -1.93, -1.94, -1.98, -2.04, -2.12, -2.24, -2.40, -2.64, -3.78, -5.2, -6.00])
            #phi_err  = np.array([])
            
            # removing 1.36 multiplier for He correction --> now is pure H2
            phi = np.log10((10**phi)/1.36)
            
            #phi_err_lower = 10**phi - (10**(phi-phi_err))
            #phi_err_upper = (10**(phi+phi_err)) - 10**phi
            
            axs.plot(logm, phi, ls='none', linewidth=1, marker='o', ms=2, fillstyle='none', alpha=0.9, zorder=-20, label='Andreani+20 (HRS) ($z=0.0$)')
            #axs.errorbar(10**logm, 10**phi, yerr=np.array([phi_err_lower, phi_err_upper]), label='Andreani+20 (HRS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=2, fillstyle='none', alpha=0.9, zorder=-20)                
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
            
                obs_x     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                obs_y = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                obs_y_err= file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
            obs_x = obs_x.to('Msun')/1.36     # divide to account for x1.36 He correction in observations
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Fletcher+21 (xCOLD GASS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=2, fillstyle='none', alpha=0.9, zorder=-20)           
        if add_lagos2015:
            print('lagos not available')
        
        if add_lagos2014_1Vcorr:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyH2MassFunction/Lagos2014_H2.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
                
                obs_x     = file['data/massfunc/1Vcorr/x/values'][:] * u.Unit(file['data/massfunc/1Vcorr/x/values'].attrs['units'])
                obs_y = file['data/massfunc/1Vcorr/y/values'][:] * u.Unit(file['data/massfunc/1Vcorr/y/values'].attrs['units'])
                obs_y_err= file['data/massfunc/1Vcorr/y/scatter'][:] * u.Unit(file['data/massfunc/1Vcorr/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Lagos+14 $1/V_{\mathrm{max}}$ corr', ls='none', elinewidth=0.7, marker='^', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='r', capsize=1.6)
        if add_lagos2014_uncorr:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyH2MassFunction/Lagos2014_H2.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
                
                obs_x     = file['data/massfunc/uncorr/x/values'][:] * u.Unit(file['data/massfunc/uncorr/x/values'].attrs['units'])
                obs_y = file['data/massfunc/uncorr/y/values'][:] * u.Unit(file['data/massfunc/uncorr/y/values'].attrs['units'])
                obs_y_err= file['data/massfunc/uncorr/y/scatter'][:] * u.Unit(file['data/massfunc/uncorr/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Lagos+14 uncorr', ls='none', elinewidth=0.7, marker='^', ms=3.9, markeredgewidth=0.6, markerfacecolor='none', alpha=0.9, zorder=-20, markeredgecolor='r', c='r', capsize=1.6)
            
    
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
        
        plt.savefig("%s/etg_sample_plots/H2_mass_func/%s_%s_ALL_SAMPLES_sample_H2massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/H2_mass_func/%s_%s_ALL_SAMPLES_sample_H2massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#------------------
# Returns H2 mass fraction = H2/H2+M* function for given set of samples
def _sample_H2_mass_frac_function(csv_samples = [],  title_text_in = '',
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          aperture_h2 = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
                   'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_centrals': 'ETGs (excl. FRs)',
                   'all_ETGs_satellites': 'ETGs (excl. FRs)',
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
                   'all_ETGs': '-',
                   'all_ETGs_centrals': '-',
                   'all_ETGs_satellites': '-',
                   'all_ETGs_plus_redspiral': '-',
                   'all_ETGs_plus_redspiral_centrals': '-',
                   'all_ETGs_plus_redspiral_satellites': '-'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_galaxies_centrals': 'o',
                   'all_galaxies_satellites': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_centrals': 's',
                   'all_ETGs_satellites': 's',
                   'all_ETGs_plus_redspiral': 'D',
                   'all_ETGs_plus_redspiral_centrals': 'D',
                   'all_ETGs_plus_redspiral_satellites': 'D'}
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
        box_size = data.metadata.boxsize[0].to(u.Mpc)
    
    
        #-------------------------------
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        
        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        H2_mass.convert_to_units('Msun')
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
        #======================
        # Calculated values
        #H2_mass_fraction = np.divide(H2_mass, H2_mass + stellar_mass)
        H2_mass_fraction = np.divide(H2_mass, stellar_mass)
    
        #---------------
        # Histograms   
        hist_bin_width = 0.2
        lower_mass_limit = 1e-6
        upper_mass_limit = 10
        
    
        hist_masses, bin_edges =  np.histogram(np.log10(H2_mass_fraction), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**3
        hist_masses = hist_masses/hist_bin_width        # density
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(np.log10(H2_mass_fraction), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_err = (np.sqrt(hist_n)/(box_size)**3)/hist_bin_width
    
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask_finite = np.isfinite(np.log10(hist_masses))
        hist_masses = hist_masses[hist_mask_finite]
        bin_midpoints   = bin_midpoints[hist_mask_finite]
        hist_err    = hist_err[hist_mask_finite]
        hist_n      = hist_n[hist_mask_finite]
        
        
        #-----------
        # Line formatting
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        ms_i    = dict_ms[sample_input['name_of_preset']]
        
        # Plotting all points as dashed 
        axs.plot(np.flip(bin_midpoints), np.flip(hist_masses), color=linecol, ls=(0, (1, 1)), zorder=-4, path_effects=[outline])
        
        # Plotting only those with hist_n >= 10
        hist_masses_masked = np.ma.masked_where(hist_n < 10, hist_masses)
        hist_err_masked    = np.ma.masked_where(hist_n < 10, hist_err)
        #axs.fill_between(bin_midpoints, (10**(np.log10(hist_masses_masked-hist_err_masked))), (10**(np.log10(hist_masses_masked+hist_err_masked))), alpha=0.4, fc=linecol, zorder=-5)
        lines = axs.plot(bin_midpoints, hist_masses_masked, label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol, zorder=-3, path_effects=[outline])
        
    #-----------------
    # Add observations
    axs.plot([],[])    # skip C0
    if add_observational:
        """
        Pick observations we want to add
        """
        add_lagos2014       = True          # Outputs H2 / M*
        
        if add_lagos2014:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyH2MassFunction/Lagos2014_H2.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
                
                obs_x     = file['data/massfracfunc/x/values'][:] * u.Unit(file['data/massfracfunc/x/values'].attrs['units'])
                obs_y = file['data/massfracfunc/y/values'][:] * u.Unit(file['data/massfracfunc/y/values'].attrs['units'])
                obs_y_err= file['data/massfracfunc/y/scatter'][:] * u.Unit(file['data/massfracfunc/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('dimensionless')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Scale to correct fgas
            #obs_x = obs_x * (1/(1+obs_x))
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Lagos+14', ls='none', elinewidth=0.7, marker='^', ms=3.9, markeredgewidth=0.6, markerfacecolor='none', alpha=0.9, zorder=-20, markeredgecolor='r', c='r', capsize=1.6)
        
    #-----------
    # Axis formatting
    plt.xlim(-6, 0)
    plt.ylim(10**(-6), 10**(-1))
    #plt.xscale("log")
    plt.yscale("log")
    axs.minorticks_on()
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))

    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc',
                     'inclusive_sphere_10kpc': '10 pkpc',
                     'inclusive_sphere_30kpc': '30 pkpc', 
                     'inclusive_sphere_50kpc': '50 pkpc'}
    #plt.xlabel(r'log$_{10}$ $f_{\mathrm{H_{2}}}$ (%s)'%(dict_aperture[aperture_h2]))
    #plt.ylabel(r'dn/dlog$_{10}$($f_{\mathrm{H_{2}}}$) [cMpc$^{-3}$]')
    plt.xlabel(r'log$_{10}$ $M_{\mathrm{H_{2}}}/M_*$ (%s)'%(dict_aperture[aperture_h2]))
    plt.ylabel(r'dn/dlog$_{10}$($M_{\mathrm{H_{2}}}/M_*$) [cMpc$^{-3}$]')
      
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
    axs.legend(loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture_h2 + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/H2_mass_func/%s_%s_ALL_SAMPLES_sample_H2massfracfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/H2_mass_func/%s_%s_ALL_SAMPLES_sample_H2massfracfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#------------------
# Returns H1 mass function for given set of samples
def _sample_H1_mass_function(csv_samples = [],  title_text_in = '',
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          aperture_h1 = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
                        
    #---------------------------
    # Extract data from samples:
    dict_labels = {'all_galaxies': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_centrals': 'ETGs (excl. FRs)',
                   'all_ETGs_satellites': 'ETGs (excl. FRs)',
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
                   'all_ETGs': '-',
                   'all_ETGs_centrals': '-',
                   'all_ETGs_satellites': '-',
                   'all_ETGs_plus_redspiral': '-',
                   'all_ETGs_plus_redspiral_centrals': '-',
                   'all_ETGs_plus_redspiral_satellites': '-'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_galaxies_centrals': 'o',
                   'all_galaxies_satellites': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_centrals': 's',
                   'all_ETGs_satellites': 's',
                   'all_ETGs_plus_redspiral': 'D',
                   'all_ETGs_plus_redspiral_centrals': 'D',
                   'all_ETGs_plus_redspiral_satellites': 'D'}
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
        box_size = data.metadata.boxsize[0].to(u.Mpc)
    
    
        #-------------------------------
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        
        H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        H1_mass.convert_to_units('Msun')
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
    
        #---------------
        # Histograms   
        hist_bin_width = 0.2
        lower_mass_limit = 10**4
        upper_mass_limit = 10**11
        
    
        hist_masses, bin_edges =  np.histogram(np.log10(H1_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**3
        hist_masses = hist_masses/hist_bin_width        # density
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(np.log10(H1_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_err = (np.sqrt(hist_n)/(box_size)**3)/hist_bin_width
    
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask_finite = np.isfinite(np.log10(hist_masses))
        hist_masses = hist_masses[hist_mask_finite]
        bin_midpoints   = bin_midpoints[hist_mask_finite]
        hist_err    = hist_err[hist_mask_finite]
        hist_n      = hist_n[hist_mask_finite]
        
        
        #-----------
        # Line formatting
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        ms_i    = dict_ms[sample_input['name_of_preset']]
        
        # Plotting all points as dashed 
        axs.plot(np.flip(bin_midpoints), np.flip(hist_masses), color=linecol, ls=(0, (1, 1)), zorder=-4, path_effects=[outline])
        
        # Plotting only those with hist_n >= 10
        hist_masses_masked = np.ma.masked_where(hist_n < 10, hist_masses)
        hist_err_masked    = np.ma.masked_where(hist_n < 10, hist_err)
        #axs.fill_between(bin_midpoints, (10**(np.log10(hist_masses_masked-hist_err_masked))), (10**(np.log10(hist_masses_masked+hist_err_masked))), alpha=0.4, fc=linecol, zorder=-5)
        lines = axs.plot(bin_midpoints, hist_masses_masked, label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol, zorder=-3, path_effects=[outline])
        
    #-----------------
    # Add observations
    axs.plot([],[])    # skip C0
    if add_observational:
        """
        Pick observations we want to add
        """
        add_andreani2020    = False
        add_fletcher2021    = False     # most robust where sample bias has been taken into account
        add_lagos2015       = False
        #add_guo2023 = True
        add_lagos2014_uncorr       = True
        add_lagos2014_1Vcorr       = True
        
        if add_andreani2020:
            logm = np.arange(6.5, 10.2, 0.25)
            phi      = np.array([-2.36, -2.14, -2.02, -1.96, -1.93, -1.94, -1.98, -2.04, -2.12, -2.24, -2.40, -2.64, -3.78, -5.2, -6.00])
            #phi_err  = np.array([])
            
            # removing 1.36 multiplier for He correction --> now is pure H2
            phi = np.log10((10**phi)/1.36)
            
            #phi_err_lower = 10**phi - (10**(phi-phi_err))
            #phi_err_upper = (10**(phi+phi_err)) - 10**phi
            
            axs.plot(logm, phi, ls='none', linewidth=1, marker='o', ms=2, fillstyle='none', alpha=0.9, zorder=-20, label='Andreani+20 (HRS) ($z=0.0$)')
            #axs.errorbar(10**logm, 10**phi, yerr=np.array([phi_err_lower, phi_err_upper]), label='Andreani+20 (HRS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=2, fillstyle='none', alpha=0.9, zorder=-20)                
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
            
                obs_x     = file['x/values'][:] * u.Unit(file['x/values'].attrs['units'])
                obs_y = file['y/values'][:] * u.Unit(file['y/values'].attrs['units'])
                obs_y_err= file['y/scatter'][:] * u.Unit(file['y/scatter'].attrs['units'])
            obs_x = obs_x.to('Msun')/1.36     # divide to account for x1.36 He correction in observations
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Fletcher+21 (xCOLD GASS) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=2, fillstyle='none', alpha=0.9, zorder=-20)           
        if add_lagos2015:
            print('lagos not available')
        
        if add_lagos2014_1Vcorr:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyHIMassFunction/Lagos2014_H1.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
                
                obs_x     = file['data/massfunc/1Vcorr/x/values'][:] * u.Unit(file['data/massfunc/1Vcorr/x/values'].attrs['units'])
                obs_y = file['data/massfunc/1Vcorr/y/values'][:] * u.Unit(file['data/massfunc/1Vcorr/y/values'].attrs['units'])
                obs_y_err= file['data/massfunc/1Vcorr/y/scatter'][:] * u.Unit(file['data/massfunc/1Vcorr/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Lagos+14 $1/V_{\mathrm{max}}$ corr', ls='none', elinewidth=0.7, marker='^', ms=3, fillstyle='none', alpha=0.9, zorder=-20, c='r', capsize=1.6)
        if add_lagos2014_uncorr:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyHIMassFunction/Lagos2014_H1.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
                
                obs_x     = file['data/massfunc/uncorr/x/values'][:] * u.Unit(file['data/massfunc/uncorr/x/values'].attrs['units'])
                obs_y = file['data/massfunc/uncorr/y/values'][:] * u.Unit(file['data/massfunc/uncorr/y/values'].attrs['units'])
                obs_y_err= file['data/massfunc/uncorr/y/scatter'][:] * u.Unit(file['data/massfunc/uncorr/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('Msun')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Lagos+14 uncorr', ls='none', elinewidth=0.7, marker='^', ms=3.9, markeredgewidth=0.6, markerfacecolor='none', alpha=0.9, zorder=-20, markeredgecolor='r', c='r', capsize=1.6)
        
        
    #-----------
    # Axis formatting
    plt.xlim(6, 11)
    plt.ylim(10**(-7), 10**(-1))
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
    plt.xlabel(r'log$_{10}$ $M_{\mathrm{H_{I}}}$ (%s) [M$_{\odot}$]'%(dict_aperture[aperture_h1]))
    plt.ylabel(r'dn/dlog$_{10}$($M_{\mathrm{H_{I}}}$) [cMpc$^{-3}$]')
      
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
    axs.legend(loc='lower center', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture_h1 + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/H1_mass_func/%s_%s_ALL_SAMPLES_sample_H1massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/H1_mass_func/%s_%s_ALL_SAMPLES_sample_H1massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#------------------
# Returns H1 mass fraction = H1/H1+M* function for given set of samples
def _sample_H1_mass_frac_function(csv_samples = [],  title_text_in = '',
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          aperture_h1 = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
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
                   'all_galaxies_centrals': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_galaxies_satellites': r'Total $M_{*}>10^{9.5}$ M$_\odot$',
                   'all_ETGs': 'ETGs (excl. FRs)',
                   'all_ETGs_centrals': 'ETGs (excl. FRs)',
                   'all_ETGs_satellites': 'ETGs (excl. FRs)',
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
                   'all_ETGs': '-',
                   'all_ETGs_centrals': '-',
                   'all_ETGs_satellites': '-',
                   'all_ETGs_plus_redspiral': '-',
                   'all_ETGs_plus_redspiral_centrals': '-',
                   'all_ETGs_plus_redspiral_satellites': '-'}
    dict_ms     = {'all_galaxies': 'o',
                   'all_galaxies_centrals': 'o',
                   'all_galaxies_satellites': 'o',
                   'all_ETGs': 's',
                   'all_ETGs_centrals': 's',
                   'all_ETGs_satellites': 's',
                   'all_ETGs_plus_redspiral': 'D',
                   'all_ETGs_plus_redspiral_centrals': 'D',
                   'all_ETGs_plus_redspiral_satellites': 'D'}
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
        box_size = data.metadata.boxsize[0].to(u.Mpc)
    
    
        #-------------------------------
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        
        H1_mass = attrgetter('%s.%s'%(aperture_h1, 'atomic_hydrogen_mass'))(data)[soap_indicies_sample]
        H1_mass.convert_to_units('Msun')
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
        #======================
        # Calculated values
        #H1_mass_fraction = np.divide(H1_mass, H1_mass + stellar_mass)
        H1_mass_fraction = np.divide(H1_mass, stellar_mass)
    
        #---------------
        # Histograms   
        hist_bin_width = 0.2
        lower_mass_limit = 1e-6
        upper_mass_limit = 10
        
    
        hist_masses, bin_edges =  np.histogram(np.log10(H1_mass_fraction), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**3
        hist_masses = hist_masses/hist_bin_width        # density
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(np.log10(H1_mass_fraction), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_err = (np.sqrt(hist_n)/(box_size)**3)/hist_bin_width
    
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask_finite = np.isfinite(np.log10(hist_masses))
        hist_masses = hist_masses[hist_mask_finite]
        bin_midpoints   = bin_midpoints[hist_mask_finite]
        hist_err    = hist_err[hist_mask_finite]
        hist_n      = hist_n[hist_mask_finite]
        
        
        #-----------
        # Line formatting
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        ms_i    = dict_ms[sample_input['name_of_preset']]
        
        # Plotting all points as dashed 
        axs.plot(np.flip(bin_midpoints), np.flip(hist_masses), color=linecol, ls=(0, (1, 1)), zorder=-4, path_effects=[outline])
        
        # Plotting only those with hist_n >= 10
        hist_masses_masked = np.ma.masked_where(hist_n < 10, hist_masses)
        hist_err_masked    = np.ma.masked_where(hist_n < 10, hist_err)
        #axs.fill_between(bin_midpoints, (10**(np.log10(hist_masses_masked-hist_err_masked))), (10**(np.log10(hist_masses_masked+hist_err_masked))), alpha=0.4, fc=linecol, zorder=-5)
        lines = axs.plot(bin_midpoints, hist_masses_masked, label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol, zorder=-3, path_effects=[outline])
        
    #-----------------
    # Add observations
    axs.plot([],[])    # skip C0
    if add_observational:
        """
        Pick observations we want to add
        """
        add_lagos2014       = True          # Outputs H2 / M*
        
        if add_lagos2014:      # ($z=0.0$)
            # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
            with h5py.File('%s/GalaxyHIMassFunction/Lagos2014_H1.hdf5'%obs_dir, 'r') as file:
                
                """# A description of the file and data is in the metadata
                print(f'File keys: {file.keys()}')
                for k, v in file['metadata'].attrs.items():
                    print(f'{k}: {v}')
                # Data
                print(file['x'].keys())
                print(file['y'].keys())
                print(' ')"""
                
                obs_x     = file['data/massfracfunc/x/values'][:] * u.Unit(file['data/massfracfunc/x/values'].attrs['units'])
                obs_y = file['data/massfracfunc/y/values'][:] * u.Unit(file['data/massfracfunc/y/values'].attrs['units'])
                obs_y_err= file['data/massfracfunc/y/scatter'][:] * u.Unit(file['data/massfracfunc/y/scatter'].attrs['units'])
                
            obs_x = obs_x.to('dimensionless')
            obs_y = obs_y.to('Mpc**(-3)')
            obs_y_err = obs_y_err.to('Mpc**(-3)')
            
            # Scale to correct fgas
            #obs_x = obs_x * (1/(1+obs_x))
            
            # Convert to log
            #err_minus_log = np.log10(obs_y) - np.log10(obs_y - obs_y_err[0])
            #err_plus_log  = np.log10(obs_y + obs_y_err[1]) - np.log10(obs_y)
            #log_obs_y_err = np.vstack([err_minus_log, err_plus_log])
                        
            axs.errorbar(np.log10(obs_x), obs_y, yerr=obs_y_err, label='Lagos+14', ls='none', elinewidth=0.7, marker='^', ms=3.9, markeredgewidth=0.6, markerfacecolor='none', alpha=0.9, zorder=-20, markeredgecolor='r', c='r', capsize=1.6)
        
    #-----------
    # Axis formatting
    plt.xlim(-5, 0)
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
    plt.xlabel(r'log$_{10}$ $M_{\mathrm{H_{I}}}/M_*$ (%s)'%(dict_aperture[aperture_h1]))
    plt.ylabel(r'dn/dlog$_{10}$($M_{\mathrm{H_{I}}}/M_*$) [cMpc$^{-3}$]')
      
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
    axs.legend(loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture_h1 + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/H1_mass_func/%s_%s_ALL_SAMPLES_sample_H1massfracfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/H1_mass_func/%s_%s_ALL_SAMPLES_sample_H1massfracfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()



#=====================================

#-------------------
# Similar to the correa+17 plot.   USE WITH ALL GALAXIES
"""_sample_stelmass_u_r(csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies',
                     showfig       = False,
                     savefig       = True)
_sample_stelmass_u_r(csv_sample = 'L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies',
                     showfig       = False,
                     savefig       = True)
_sample_stelmass_u_r(csv_sample = 'L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies',
                     showfig       = False,
                     savefig       = True)"""


#-------------------
# Stellar mass function of samples
"""_sample_stellar_mass_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     showfig       = False,
                     savefig       = True)
_sample_stellar_mass_function(csv_samples = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     showfig       = False,
                     savefig       = True)
_sample_stellar_mass_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     showfig       = False,
                     savefig       = True)
_sample_stellar_mass_function_3x1(csv_samples1 = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                                  csv_samples2 = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                                  csv_samples3 = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     showfig       = False,
                     savefig       = True)"""



#-------------------
# H1 mass function of samples
_sample_H1_mass_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)
"""_sample_H1_mass_function(csv_samples = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)
_sample_H1_mass_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)"""
# Central and satellite
"""_sample_H1_mass_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                        title_text_in = 'centrals',
                        savefig_txt = 'centrals', 
                     savefig       = True)
_sample_H1_mass_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_satellites'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                        title_text_in = 'satellites',
                        savefig_txt = 'satellites', 
                     savefig       = True)"""
# H1 mass fraction = H1 / H1 + M* function of samples
"""_sample_H1_mass_frac_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)
_sample_H1_mass_frac_function(csv_samples = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)
_sample_H1_mass_frac_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)"""
# Central and satellite
"""_sample_H1_mass_frac_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                        title_text_in = 'centrals',
                        savefig_txt = 'centrals', 
                     savefig       = True)
_sample_H1_mass_frac_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_satellites'],
                     aperture_h1 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                        title_text_in = 'satellites',
                        savefig_txt = 'satellites', 
                     savefig       = True)"""

#-------------------
# H2 mass function of samples
"""_sample_H2_mass_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                         savefig       = True)"""
"""_sample_H2_mass_function(csv_samples = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                         savefig       = True)
_sample_H2_mass_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                         savefig       = True)"""
# Central and satellite
"""_sample_H2_mass_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                        title_text_in = 'centrals',
                        savefig_txt = 'centrals', 
                         savefig       = True)
_sample_H2_mass_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_satellites'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                        title_text_in = 'satellites',
                        savefig_txt = 'satellites', 
                         savefig       = True) """             
                         
# H2 mass fraction = H2 / H2 + M* function of samples
"""_sample_H2_mass_frac_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                         savefig       = True)
_sample_H2_mass_frac_function(csv_samples = ['L100_m6_HYBRID_AGN_m6_127_sample_all_galaxies', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs', 'L100_m6_HYBRID_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                         savefig       = True)
_sample_H2_mass_frac_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                         savefig       = True)"""
# Central and satellite
"""_sample_H2_mass_frac_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_centrals', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_centrals'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                        title_text_in = 'centrals',
                        savefig_txt = 'centrals', 
                         savefig       = True)
_sample_H2_mass_frac_function(csv_samples = ['L200_m6_THERMAL_AGN_m6_127_sample_all_galaxies_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_satellites', 'L200_m6_THERMAL_AGN_m6_127_sample_all_ETGs_plus_redspiral_satellites'],
                         aperture_h2 = 'exclusive_sphere_50kpc',
                         showfig       = False,
                        title_text_in = 'satellites',
                        savefig_txt = 'satellites', 
                         savefig       = True)"""







