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



#-----------------
# Returns stelmass - sfr or stelmass - ssfr, coloured by kappa stars
def _sample_stelmass_u_r(csv_sample = '',
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
    ax_scat.scatter(stellar_mass, mag_plot, c=kappa_stars, s=1.5, cmap=mymap, norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)

        
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
        ax_scat.axhline(2, lw=1, ls='--', color='k', zorder=10)
        ax_hist.axhline(2, lw=1, ls='--', color='k', zorder=10)
        
        
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
    ax_scat.text(0.78, 0.92, '${z=%.2f}$' %z, fontsize=7, transform = ax_scat.transAxes)
    title_dict = {'all_galaxies': 'All galaxies',
                  'all_ETGs': 'ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$)',
                  'all_ETGs_plus_redspiral': "ETGs ($\kappa_{\mathrm{co}}^{*}<0.4$ incl. red FRs)"
                  }
    ax_hist.set_title(r'%s' %(title_dict[sample_input['name_of_preset']]), size=7, loc='left', pad=3) 
    #if add_ur2_line:
        #ax_scat.text(10**11.35, 1.7, "$\kappa_{\mathrm{co}}^{*}>0.4$\nred spirals", fontsize=7, color='k')
        
    #-----------
    # Legend
    ax_hist.legend(loc='lower right', ncol=1, frameon=False)
        
        
    #-----------
    # other
    #plt.tight_layout()
    ax_hist.grid(alpha=0.4, zorder=-5, lw=0.7)
    
    if savefig:
        savefig_txt_save = aperture + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/%s_%s_%s_sample_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/%s_%s_%s_sample_u-r_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], sample_input['name_of_preset'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#------------------
# Returns stellar mass function for given set of samples
def _sample_stellar_mass_function(csv_samples = [],
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.3], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
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
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
    
        #---------------
        # Histograms   
        hist_bin_width = 0.1
        lower_mass_limit = 10**9.5
        upper_mass_limit = 10**13
        
    
        hist_masses, bin_edges =  np.histogram(np.log10(stellar_mass), bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
        hist_masses = hist_masses[:]/(box_size)**3      # in units of /cMpc**3
        hist_masses = hist_masses/hist_bin_width        # density
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
        
        # Find bins with more than 1 entry
        hist_mask_n = hist_n > 1
    
    
        #-----------
        # Plotting error for n >= 1, and single dots for others
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
            
        axs.fill_between(10**bin_midpoints[hist_mask_n], 10**(np.log10(hist_masses[hist_mask_n]-hist_err[hist_mask_n])), 10**(np.log10(hist_masses[hist_mask_n]+hist_err[hist_mask_n])), alpha=0.4, fc=linecol)
        lines = axs.plot(10**bin_midpoints[hist_mask_n], 10**np.log10(hist_masses[hist_mask_n]), label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol)
        axs.plot(10**bin_midpoints[~hist_mask_n], 10**np.log10(hist_masses[~hist_mask_n]), 'o', color=lines[0].get_color(), ms=2)
        
    #-----------------
    # Add observations
    axs.plot([],[])    # skip C0
    if add_observational:
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
            
            axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Leja+20 ($z=0.2$)', ls='-', linewidth=1, alpha=0.8, zorder=-20, c='C2')      
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
            
            axs.errorbar(obs_mass, obs_size, yerr=obs_size_scatter, label='Driver+22 (GAMA-DR4) ($z=0.0$)', ls='none', linewidth=1, marker='o', ms=1.2, alpha=0.8, zorder=-20, c='C3')              
        
    #-----------
    # Axis formatting
    plt.xlim(10**9.5, 10**12)
    plt.ylim(10**(-5.5), 10**(-1.5))
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))

    dict_aperture = {'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    plt.xlabel(r'$M_*$ (%s) [M$_{\odot}$]'%(dict_aperture[aperture]))
    plt.ylabel(r'dn/dlog$_{10}$($M$) [cMpc$^{-3}$]')
      
    #-----------  
    # Annotations
    plt.text(0.80, 0.90, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/%s_%s_ALL_SAMPLES_sample_stellarmassfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/%s_%s_ALL_SAMPLES_sample_stellarmassfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#------------------
# Returns H2 mass function for given set of samples
def _sample_H2_mass_function(csv_samples = [],
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          aperture_h2 = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.3], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
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
        hist_n      = hist_n[hist_mask_finite]
        
        # Find bins with more than 1 entry
        hist_mask_n = hist_n > 1
    
    
        #-----------
        # Plotting error for n >= 1, and single dots for others
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        
        axs.fill_between(10**bin_midpoints[hist_mask_n], 10**(np.log10(hist_masses[hist_mask_n]-hist_err[hist_mask_n])), 10**(np.log10(hist_masses[hist_mask_n]+hist_err[hist_mask_n])), alpha=0.4, fc=linecol)
        lines = axs.plot(10**bin_midpoints[hist_mask_n], 10**np.log10(hist_masses[hist_mask_n]), label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol)
        axs.plot(10**bin_midpoints[~hist_mask_n], 10**np.log10(hist_masses[~hist_mask_n]), 'o', color=lines[0].get_color(), ms=2)
        
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
        
        if add_andreani2020:
            logm = np.arange(6.5, 10.2, 0.25)
            phi      = np.array([-2.36, -2.14, -2.02, -1.96, -1.93, -1.94, -1.98, -2.04, -2.12, -2.24, -2.40, -2.64, -3.78, -5.2, -6.00])
            #phi_err  = np.array([])
            
            # removing 1.36 multiplier for He correction --> now is pure H2
            phi = np.log10((10**phi)/1.36)
            
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
    
    #-----------
    # Axis formatting
    plt.xlim(10**6, 10**11)
    plt.ylim(10**(-6), 10**(-1))
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))

    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    plt.xlabel(r'$M_{\mathrm{H_{2}}}$ (%s) [M$_{\odot}$]'%(dict_aperture[aperture_h2]))
    plt.ylabel(r'dn/dlog$_{10}$($M$) [cMpc$^{-3}$]')
      
    #-----------  
    # Annotations
    plt.text(0.8, 0.9, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    
    #-----------
    # Legend
    axs.legend(loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture_h2 + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/%s_%s_ALL_SAMPLES_sample_H2massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/%s_%s_ALL_SAMPLES_sample_H2massfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#------------------
# Returns H2 mass fraction = H2/H2+M* function for given set of samples
def _sample_H2_mass_frac_function(csv_samples = [],
                          #=====================================
                          aperture = 'exclusive_sphere_50kpc', 
                          aperture_h2 = 'exclusive_sphere_50kpc', 
                          #----------
                          add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
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
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.3], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
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
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)[soap_indicies_sample]
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        
        H2_mass = attrgetter('%s.%s'%(aperture_h2, 'molecular_hydrogen_mass'))(data)[soap_indicies_sample]
        H2_mass.convert_to_units('Msun')
    
        central_sat = attrgetter('input_halos.is_central')(data)[soap_indicies_sample]
    
        #======================
        # Calculated values
        H2_mass_fraction = np.divide(H2_mass, H2_mass + stellar_mass)
    
        #---------------
        # Histograms   
        hist_bin_width = 0.2
        lower_mass_limit = 1e-6
        upper_mass_limit = 1
        
    
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
        
        # Find bins with more than 1 entry
        hist_mask_n = hist_n > 1
    
    
        #-----------
        # Plotting error for n >= 1, and single dots for others
        label_i = dict_labels[sample_input['name_of_preset']]
        linecol = dict_colors[sample_input['name_of_preset']]
        ls_i    = dict_ls[sample_input['name_of_preset']]
        
        axs.fill_between(10**bin_midpoints[hist_mask_n], 10**(np.log10(hist_masses[hist_mask_n]-hist_err[hist_mask_n])), 10**(np.log10(hist_masses[hist_mask_n]+hist_err[hist_mask_n])), alpha=0.4, fc=linecol)
        lines = axs.plot(10**bin_midpoints[hist_mask_n], 10**np.log10(hist_masses[hist_mask_n]), label='%s'%label_i, ls=ls_i, linewidth=1, c=linecol)
        axs.plot(10**bin_midpoints[~hist_mask_n], 10**np.log10(hist_masses[~hist_mask_n]), 'o', color=lines[0].get_color(), ms=2)
        
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
        
        if add_andreani2020:
            logm = np.arange(6.5, 10.2, 0.25)
            phi      = np.array([-2.36, -2.14, -2.02, -1.96, -1.93, -1.94, -1.98, -2.04, -2.12, -2.24, -2.40, -2.64, -3.78, -5.2, -6.00])
            #phi_err  = np.array([])
            
            # removing 1.36 multiplier for He correction --> now is pure H2
            phi = np.log10((10**phi)/1.36)
            
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
    
    #-----------
    # Axis formatting
    plt.xlim(10**-6, 10**0)
    plt.ylim(10**(-6), 10**(-1))
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))

    dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                     'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    plt.xlabel(r'$f_{\mathrm{H_{2}}}$ (%s)'%(dict_aperture[aperture_h2]))
    plt.ylabel(r'dn/dlog$_{10}$($f_{\mathrm{H_{2}}}$) [cMpc$^{-3}$]')
      
    #-----------  
    # Annotations
    plt.text(0.8, 0.9, '${z=%.2f}$' %z, fontsize=7, transform = axs.transAxes)
    
    #-----------
    # Legend
    axs.legend(loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.3)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = aperture_h2 + '_' + savefig_txt
        
        plt.savefig("%s/etg_sample_plots/%s_%s_ALL_SAMPLES_sample_H2massfracfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)         
        print("\n  SAVED: %s/etg_sample_plots/%s_%s_ALL_SAMPLES_sample_H2massfracfunc_%s%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['snapshot_no'], savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()




#=====================================
# L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies
# L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs
# L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral

#-------------------
# Similar to the correa+17 plot.   USE WITH ALL GALAXIES
"""_sample_stelmass_u_r(csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies',
                     showfig       = False,
                     savefig       = True)"""


#-------------------
# Stellar mass function of samples
"""_sample_stellar_mass_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                     showfig       = False,
                     savefig       = True)"""
#-------------------
# H2 mass function of samples
"""_sample_H2_mass_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                     aperture_h2 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)
_sample_H2_mass_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                     aperture_h2 = 'exclusive_sphere_10kpc',
                     showfig       = False,
                     savefig       = True)"""


#-------------------
# H2 mass fraction function of samples
_sample_H2_mass_frac_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                     aperture_h2 = 'exclusive_sphere_50kpc',
                     showfig       = False,
                     savefig       = True)
_sample_H2_mass_frac_function(csv_samples = ['L100_m6_THERMAL_AGN_m6_127_sample10071_all_galaxies', 'L100_m6_THERMAL_AGN_m6_127_sample2224_all_ETGs', 'L100_m6_THERMAL_AGN_m6_127_sample4045_all_ETGs_plus_redspiral'],
                     aperture_h2 = 'exclusive_sphere_10kpc',
                     showfig       = False,
                     savefig       = True)







