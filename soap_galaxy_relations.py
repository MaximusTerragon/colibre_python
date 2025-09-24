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
# Returns stelmass - gasmass scatter, and as mass fraction, coloured by kappa stars
def _stelmass_gasmass(simulation_run = ['L100_m6', 'L0025N0752'], 
                   simulation_type = ['THERMAL_AGN_m6', 'THERMAL_AGN_m5'],
                    snapshot_no   = [127, 123],        # available for L100_m6: 127, 119, 114, 102, 092
                   #=====================================
                   # Graph settings
                   mass_type = 'gas_mass',    # [ gas_mass / star_forming_gas_mass / gas_mass_in_cold_dense_gas
                                              #   molecular_hydrogen_mass / atomic_hydrogen_mass / 
                                              #   molecular_and_atomic_hydrogen_mass ]
                   centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                   aperture = 'exclusive_sphere_50kpc', 
                   #----------
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   plot_fractions    = False,         # Converts to mass fraction instead of mass
                   format_z          = False,        # For variations in z graphs
                   plot_bins_medians = True,         # Plots medians
                     plot_bins_percentiles = True,   # Plots 1 sigma percentiles
                   hist_bin_width = 0.25,
                     lower_mass_limit = 10**8.8,
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
                        
    #-----------------
    # Add SOAP data
    if len(snapshot_no) > len(simulation_run):
        simulation_run_list = simulation_run * len(snapshot_no)
        simulation_type_list = simulation_type * len(snapshot_no)
        snapshot_no_list = snapshot_no
    else:
        simulation_run_list = simulation_run
        simulation_type_list = simulation_type
        snapshot_no_list = snapshot_no
        
    
    run_name_list = []
    redshift_list = []
    for sim_run_i, sim_type_i, snap_i in zip(simulation_run_list, simulation_type_list, snapshot_no_list):
        # Load data
        simulation_dir = sim_run_i + '/' + sim_type_i
        soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
    
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0]
        #lookbacktime
        if run_name not in run_name_list:
            run_name_list.append(run_name)
        redshift_list.append(z)
        print('\n', run_name, snap_i)
    
    
        # Get stelmass and gas mass data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
        central_sat = attrgetter('input_halos.is_central')(data)
        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)
            
        if mass_type == 'molecular_and_atomic_hydrogen_mass':
            HI_mass = attrgetter('%s.%s'%(aperture, 'atomic_hydrogen_mass'))(data)
            H2_mass = attrgetter('%s.%s'%(aperture, 'molecular_hydrogen_mass'))(data)
            gas_mass = HI_mass + H2_mass
            HI_mass = 0
            H2_mass = 0
        else:
            gas_mass = attrgetter('%s.%s'%(aperture, mass_type))(data)
        
        # convert in place
        stellar_mass.convert_to_units('Msun')
        gas_mass.convert_to_units('Msun')
        
        
        #print('max stelmass: %.2e' %np.max(stellar_mass))

        # Mask for central/satellite and no DM-only galaxies
        if centrals_or_satellites == 'centrals':
            candidates = np.argwhere(np.logical_and(central_sat == cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            gas_mass = gas_mass[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking only centrals: ', len(stellar_mass))
        elif centrals_or_satellites == 'satellites':
            candidates = np.argwhere(np.logical_and(central_sat == cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            gas_mass = gas_mass[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking only satellites: ', len(stellar_mass))
        else:
            candidates = np.argwhere(stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
            stellar_mass = stellar_mass[candidates]
            gas_mass = gas_mass[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking all galaxies: ', len(stellar_mass))

        if plot_fractions:
            gas_mass = gas_mass/(gas_mass + stellar_mass)

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
        mapper = cm.ScalarMappable(norm=norm, cmap='mymap')         #cmap=cm.coolwarm)
        
        plt.scatter(stellar_mass, gas_mass, c=kappa_stars, s=1.5, cmap='mymap', norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)
        
        
        #-----------------
        # Hist medians and percentiles
        
        # Define binning parameters
        hist_bins = np.arange(np.log10(lower_mass_limit), np.log10(10e15) + hist_bin_width, hist_bin_width)  # Binning edges
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = gas_mass.value[mask]
            
            # Remove zero values before computing statistics
            y_bin = y_bin[y_bin > 0]
            
            if len(y_bin) >= 10:  # Ensure the bin contains data
                medians.append(np.median(y_bin))
                lower_1sigma.append(np.percentile(y_bin, 16))  # -1σ (16th percentile)
                upper_1sigma.append(np.percentile(y_bin, 84))  # +1σ (84th percentile)
                bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
        # Convert bin centers back to linear space for plotting
        bin_centers = 10**np.array(bin_centers)
        
        plt.plot(bin_centers, medians, color='k', label='Median', linewidth=1)
        plt.fill_between(bin_centers, lower_1sigma, upper_1sigma, color='k', alpha=0.3, label='1-Sigma')


    #-----------------
    # Add observations
    if add_observational:
        print('no obs available') 
        
        
    #-----------
    # Axis formatting
    plt.xlim(10**9, 10**12)
    if plot_fractions:
        plt.ylim(10**-4, 1)
    else:
        plt.ylim(10**5, 10**11)
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    
    dict_aperture = {'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_ylabel = {'mass': {'gas_mass': r'$M_{\mathrm{gas}}$ [M$_{\odot}$]',
                            'star_forming_gas_mass': r'$M_{\mathrm{gas,SF}}$ [M$_{\odot}$]',
                            'molecular_hydrogen_mass': r'$M_{\mathrm{H_{2}}}$ [M$_{\odot}$]',
                            'atomic_hydrogen_mass': r'$M_{\mathrm{HI}}$ [M$_{\odot}$]',
                            'molecular_gas_mass': r'$M_{\mathrm{mol}}$ (incl. He) [M$_{\odot}$]',
                            'molecular_and_atomic_hydrogen_mass': r'$M_{\mathrm{HI+H_{2}}}$ [M$_{\odot}$]',
                            'molecular_and_atomic_gas_mass': r'$M_{\mathrm{HI+H_{mol}}}$ [M$_{\odot}$]',
                            'gas_mass_in_cold_dense_gas': r'$M_{\mathrm{cold,dense}}$ [M$_{\odot}$]' 
                            },
                   'fraction': {'gas_mass': r'$M_{\mathrm{gas}}/(M_{*}+M_{\mathrm{gas}})$',
                                'star_forming_gas_mass': r'$M_{\mathrm{gas,SF}}/(M_{\mathrm{gas,SF}}+M_{*})$',
                                'molecular_hydrogen_mass': r'$M_{\mathrm{H_{2}}}/(M_{\mathrm{H_{2}}}+M_{*})$',
                                'atomic_hydrogen_mass': r'$M_{\mathrm{HI}}/(M_{\mathrm{HI}}+M_{*})$',
                                'molecular_gas_mass': '$M_{\mathrm{mol}}/(M_{\mathrm{mol}}+M_{*})$',
                                'molecular_and_atomic_hydrogen_mass': r'$M_{\mathrm{HI+H_{2}}}/(M_{\mathrm{HI+H_{2}}}+M_{*})$',
                                'molecular_and_atomic_gas_mass': r'$M_{\mathrm{HI+H_{mol}}}/(M_{\mathrm{HI+H_{mol}}}+M_{*})$',
                                'gas_mass_in_cold_dense_gas': r'$M_{\mathrm{cold,dense}}/(M_{\mathrm{cold,dense}}+M_{*})$'
                                }}
    plt.xlabel(r'$M_*$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    plt.ylabel(r'%s'%dict_ylabel['%s'%('fraction' if plot_fractions else 'mass')][mass_type])
    
    
    #------------
    # colorbar
    fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'
    
      
    #-----------  
    # Annotations
    if not format_z:
        if plot_fractions:
            plt.text(10**11.2, 0.5, '${z=%.2f}$' %z, fontsize=7)
        else:
            plt.text(10**11.2, 10**10.3, '${z=%.2f}$' %z, fontsize=7)
    if centrals_or_satellites != 'both':
        axs.set_title(r'%s' %(centrals_or_satellites), size=7, loc='left', pad=3)

    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '%s'%('redshift_' if format_z else '') + aperture + '_' + centrals_or_satellites + '_' + savefig_txt
        add_snap_directory = ('snap_%s/'%snapshot_no[0] if len(snapshot_no)==1 else '')
        
        plt.savefig("%s/soap_galaxy_relations/%s%s%s_%s_stelmass-%s%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, ('_fraction' if plot_fractions else ''), savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600) 
        print("\n  SAVED: %s/soap_galaxy_relations/%s%s%s_%s_stelmass-%s%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, ('_fraction' if plot_fractions else ''), savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - gasmass scatter, and as mass fraction, coloured by kappa stars
def _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6', 'L0025N0752'], 
                   simulation_type = ['THERMAL_AGN_m6', 'THERMAL_AGN_m5'],
                    snapshot_no   = [127, 123],        # available for L100_m6: 127, 119, 114, 102, 092
                    aperture_list = [30, 50], 
                   #=====================================
                   # Graph settings
                   mass_type = 'gas_mass',    # [ gas_mass / star_forming_gas_mass / gas_mass_in_cold_dense_gas
                                              #   molecular_hydrogen_mass / atomic_hydrogen_mass / 
                                              #   molecular_and_atomic_hydrogen_mass ]
                   centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                   #----------
                   plot_fractions    = False,         # Converts to mass fraction instead of mass
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
                    
    add_observational = False        # Adapts based on imput mass_type, and using references from pipeline
    format_z          = False        # For variations in z graphs
    plot_bins_medians = True         # Plots medians
    plot_bins_percentiles = True   # Plots 1 sigma percentiles
    hist_bin_width = 0.25
    lower_mass_limit = 10**8.8

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
                        
    #-----------------
    # Add SOAP data
    if len(snapshot_no) > len(simulation_run):
        simulation_run_list = simulation_run * len(snapshot_no)
        simulation_type_list = simulation_type * len(snapshot_no)
        snapshot_no_list = snapshot_no
    else:
        simulation_run_list = simulation_run
        simulation_type_list = simulation_type
        snapshot_no_list = snapshot_no
        
    
    run_name_list = []
    redshift_list = []
    for sim_run_i, sim_type_i, snap_i in zip(simulation_run_list, simulation_type_list, snapshot_no_list):
        # Load data
        simulation_dir = sim_run_i + '/' + sim_type_i
        soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
    
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0]
        #lookbacktime
        if run_name not in run_name_list:
            run_name_list.append(run_name)
        redshift_list.append(z)
        print('\n', run_name, snap_i)
    
    
        # Get stelmass and gas mass data
        dict_plot = {}
        for aperture in aperture_list:
            dict_plot.update({'%s'%aperture : {'stellar_mass': 0, 'gas_mass': 0}})
            
            stellar_mass = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, 'stellar_mass'))(data)
            central_sat = attrgetter('input_halos.is_central')(data)
            kappa_stars = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, 'kappa_corot_stars'))(data)
            
            
            if mass_type == 'molecular_and_atomic_hydrogen_mass':
                HI_mass = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, 'atomic_hydrogen_mass'))(data)
                H2_mass = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, 'molecular_hydrogen_mass'))(data)
                gas_mass = HI_mass + H2_mass
                HI_mass = 0
                H2_mass = 0
            else:
                gas_mass = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, mass_type))(data)
        
            # convert in place
            stellar_mass.convert_to_units('Msun')
            gas_mass.convert_to_units('Msun')

            # Mask for central/satellite and no DM-only galaxies
            if centrals_or_satellites == 'centrals':
                candidates = np.argwhere(np.logical_and(central_sat == cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                        stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
                stellar_mass = stellar_mass[candidates]
                gas_mass = gas_mass[candidates]
                kappa_stars = kappa_stars[candidates]
                print('Masking only centrals: ', len(stellar_mass))
            elif centrals_or_satellites == 'satellites':
                candidates = np.argwhere(np.logical_and(central_sat == cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                        stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
                stellar_mass = stellar_mass[candidates]
                gas_mass = gas_mass[candidates]
                kappa_stars = kappa_stars[candidates]
                print('Masking only satellites: ', len(stellar_mass))
            else:
                candidates = np.argwhere(stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
                stellar_mass = stellar_mass[candidates]
                gas_mass = gas_mass[candidates]
                kappa_stars = kappa_stars[candidates]
                print('Masking all galaxies: ', len(stellar_mass))

            if plot_fractions:
                gas_mass = gas_mass/(gas_mass + stellar_mass)
                
            dict_plot['%s'%aperture]['gas_mass'] = gas_mass
            dict_plot['%s'%aperture]['stellar_mass'] = stellar_mass

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
        mapper = cm.ScalarMappable(norm=norm, cmap='mymap')         #cmap=cm.coolwarm)
        
        #plt.scatter(stellar_mass, gas_mass, c=kappa_stars, s=1.5, cmap='mymap', norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)
        
        
        #-----------------
        # Hist medians and percentiles
        
        # Define binning parameters
        hist_bins = np.arange(np.log10(lower_mass_limit), np.log10(10e15) + hist_bin_width, hist_bin_width)  # Binning edges
        
        for aperture in aperture_list:
            # Compute statistics in each bin
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            for i in range(len(hist_bins) - 1):
                mask = (np.log10(dict_plot['%s'%aperture]['stellar_mass']) >= hist_bins[i]) & (np.log10(dict_plot['%s'%aperture]['stellar_mass']) < hist_bins[i + 1])
                y_bin = dict_plot['%s'%aperture]['gas_mass'].value[mask]
            
                # Remove zero values before computing statistics
                y_bin = y_bin[y_bin > 0]
            
                if len(y_bin) >= 10:  # Ensure the bin contains data
                    medians.append(np.median(y_bin))
                    lower_1sigma.append(np.percentile(y_bin, 16))  # -1σ (16th percentile)
                    upper_1sigma.append(np.percentile(y_bin, 84))  # +1σ (84th percentile)
                    bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
            # Convert bin centers back to linear space for plotting
            bin_centers = 10**np.array(bin_centers)
        
            line, = plt.plot(bin_centers, medians, linewidth=1, label='%s kpc'%aperture)
            plt.fill_between(bin_centers, lower_1sigma, upper_1sigma, color=line.get_color(), alpha=0.3)
        


    #-----------------
    # Add observations
    if add_observational:
        print('no obs available') 
        
        
    #-----------
    # Axis formatting
    plt.xlim(10**9, 10**12)
    if plot_fractions:
        plt.ylim(10**-4, 1)
    else:
        plt.ylim(10**5, 10**11)
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    
    dict_ylabel = {'mass': {'gas_mass': r'$M_{\mathrm{gas}}$ [M$_{\odot}$]',
                            'star_forming_gas_mass': r'$M_{\mathrm{gas,SF}}$ [M$_{\odot}$]',
                            'molecular_hydrogen_mass': r'$M_{\mathrm{H_{2}}}$ [M$_{\odot}$]',
                            'atomic_hydrogen_mass': r'$M_{\mathrm{HI}}$ [M$_{\odot}$]',
                            'molecular_gas_mass': r'$M_{\mathrm{mol}}$ (incl. He) [M$_{\odot}$]',
                            'molecular_and_atomic_hydrogen_mass': r'$M_{\mathrm{HI+H_{2}}}$ [M$_{\odot}$]',
                            'molecular_and_atomic_gas_mass': r'$M_{\mathrm{HI+H_{mol}}}$ [M$_{\odot}$]',
                            'gas_mass_in_cold_dense_gas': r'$M_{\mathrm{cold,dense}}$ [M$_{\odot}$]' 
                            },
                   'fraction': {'gas_mass': r'$M_{\mathrm{gas}}/(M_{*}+M_{\mathrm{gas}})$',
                                'star_forming_gas_mass': r'$M_{\mathrm{gas,SF}}/(M_{\mathrm{gas,SF}}+M_{*})$',
                                'molecular_hydrogen_mass': r'$M_{\mathrm{H_{2}}}/(M_{\mathrm{H_{2}}}+M_{*})$',
                                'atomic_hydrogen_mass': r'$M_{\mathrm{HI}}/(M_{\mathrm{HI}}+M_{*})$',
                                'molecular_gas_mass': '$M_{\mathrm{mol}}/(M_{\mathrm{mol}}+M_{*})$',
                                'molecular_and_atomic_hydrogen_mass': r'$M_{\mathrm{HI+H_{2}}}/(M_{\mathrm{HI+H_{2}}}+M_{*})$',
                                'molecular_and_atomic_gas_mass': r'$M_{\mathrm{HI+H_{mol}}}/(M_{\mathrm{HI+H_{mol}}}+M_{*})$',
                                'gas_mass_in_cold_dense_gas': r'$M_{\mathrm{cold,dense}}/(M_{\mathrm{cold,dense}}+M_{*})$'
                                }}
    aperture_annotate = ''
    for aperture in aperture_list:
        aperture_annotate = aperture_annotate + '%s_'%aperture
    plt.xlabel(r'$M_*$ (%s) [M$_{\odot}$]'%aperture_annotate)
    plt.ylabel(r'%s'%dict_ylabel['%s'%('fraction' if plot_fractions else 'mass')][mass_type])
    
    
    #------------
    # colorbar
    #fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'
    
      
    #-----------  
    # Annotations
    if not format_z:
        if plot_fractions:
            plt.text(10**11.2, 0.5, '${z=%.2f}$' %z, fontsize=7)
        else:
            plt.text(10**11.2, 10**10.3, '${z=%.2f}$' %z, fontsize=7)
    if centrals_or_satellites != 'both':
        axs.set_title(r'%s' %(centrals_or_satellites), size=7, loc='left', pad=3)

    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '%s'%('redshift_' if format_z else '') + aperture_annotate + '_' + centrals_or_satellites + '_' + savefig_txt
        add_snap_directory = ('snap_%s/'%snapshot_no[0] if len(snapshot_no)==1 else '')
        
        plt.savefig("%s/soap_galaxy_relations/%s%s%s_%s_aperturecheck_stelmass-%s%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, ('_fraction' if plot_fractions else ''), savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600) 
        print("\n  SAVED: %s/soap_galaxy_relations/%s%s%s_%s_aperturecheck_stelmass-%s%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, ('_fraction' if plot_fractions else ''), savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
# Returns stelmass - gasmass scatter but at different radii for gas, but at fixed stellar aperture of 50 kpc
def _stelmass_gasmass_compare_radii(simulation_run = ['L100_m6', 'L0025N0752'], 
                   simulation_type = ['THERMAL_AGN_m6', 'THERMAL_AGN_m5'],
                    snapshot_no   = [127, 123],        # available for L100_m6: 127, 119, 114, 102, 092
                    aperture_list = [1, 3, 10, 30, 50], 
                   #=====================================
                   # Graph settings
                   mass_type = 'gas_mass',    # [ gas_mass / star_forming_gas_mass / gas_mass_in_cold_dense_gas
                                              #   molecular_hydrogen_mass / atomic_hydrogen_mass / 
                                              #   molecular_and_atomic_hydrogen_mass ]
                   centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                   #=====================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'png',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
    

    plot_fractions    = False         # Converts to mass fraction instead of mass
    add_observational = False        # Adapts based on imput mass_type, and using references from pipeline
    format_z          = False        # For variations in z graphs
    plot_bins_medians = True         # Plots medians
    plot_bins_percentiles = True   # Plots 1 sigma percentiles
    hist_bin_width = 0.25
    lower_mass_limit = 10**8.8

    #---------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
                        
    #-----------------
    # Add SOAP data
    if len(snapshot_no) > len(simulation_run):
        simulation_run_list = simulation_run * len(snapshot_no)
        simulation_type_list = simulation_type * len(snapshot_no)
        snapshot_no_list = snapshot_no
    else:
        simulation_run_list = simulation_run
        simulation_type_list = simulation_type
        snapshot_no_list = snapshot_no
        
    
    run_name_list = []
    redshift_list = []
    for sim_run_i, sim_type_i, snap_i in zip(simulation_run_list, simulation_type_list, snapshot_no_list):
        # Load data
        simulation_dir = sim_run_i + '/' + sim_type_i
        soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
    
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0]
        #lookbacktime
        if run_name not in run_name_list:
            run_name_list.append(run_name)
        redshift_list.append(z)
        print('\n', run_name, snap_i)
    
    
        # Get stelmass and gas mass data
        dict_plot = {}
        for aperture in aperture_list:
            dict_plot.update({'%s'%aperture : {'stellar_mass': 0, 'gas_mass': 0}})
            
            stellar_mass = attrgetter('%s.%s'%('exclusive_sphere_50kpc', 'stellar_mass'))(data)
            stellar_mass_at_r = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, 'stellar_mass'))(data)
            central_sat = attrgetter('input_halos.is_central')(data)
            kappa_stars = attrgetter('%s.%s'%('exclusive_sphere_50kpc', 'kappa_corot_stars'))(data)
            
            
            if mass_type == 'molecular_and_atomic_hydrogen_mass':
                HI_mass = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, 'atomic_hydrogen_mass'))(data)
                H2_mass = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, 'molecular_hydrogen_mass'))(data)
                gas_mass = HI_mass + H2_mass
                HI_mass = 0
                H2_mass = 0
            else:
                gas_mass = attrgetter('%s.%s'%('exclusive_sphere_%skpc'%aperture, mass_type))(data)
        
            # convert in place
            stellar_mass.convert_to_units('Msun')
            gas_mass.convert_to_units('Msun')

            # Mask for central/satellite and no DM-only galaxies
            if centrals_or_satellites == 'centrals':
                candidates = np.argwhere(np.logical_and(central_sat == cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                        stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
                stellar_mass = stellar_mass[candidates]
                stellar_mass_at_r = stellar_mass[candidates]
                gas_mass = gas_mass[candidates]
                kappa_stars = kappa_stars[candidates]
                print('Masking only centrals: ', len(stellar_mass))
            elif centrals_or_satellites == 'satellites':
                candidates = np.argwhere(np.logical_and(central_sat == cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=data.metadata.a, scale_exponent=0), 
                                                        stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
                stellar_mass = stellar_mass[candidates]
                stellar_mass_at_r = stellar_mass[candidates]
                gas_mass = gas_mass[candidates]
                kappa_stars = kappa_stars[candidates]
                print('Masking only satellites: ', len(stellar_mass))
            else:
                candidates = np.argwhere(stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
                stellar_mass = stellar_mass[candidates]
                stellar_mass_at_r = stellar_mass[candidates]
                gas_mass = gas_mass[candidates]
                kappa_stars = kappa_stars[candidates]
                print('Masking all galaxies: ', len(stellar_mass))

            if plot_fractions:
                gas_mass = gas_mass/(gas_mass + stellar_mass_at_r)
                raise Exception('tweak the fraction to include total gas here')
                
            dict_plot['%s'%aperture]['gas_mass'] = gas_mass
            dict_plot['%s'%aperture]['stellar_mass'] = stellar_mass

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
        mapper = cm.ScalarMappable(norm=norm, cmap='mymap')         #cmap=cm.coolwarm)
        
        #plt.scatter(stellar_mass, gas_mass, c=kappa_stars, s=1.5, cmap='mymap', norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)
        
        
        #-----------------
        # Hist medians and percentiles
        
        # Define binning parameters
        hist_bins = np.arange(np.log10(lower_mass_limit), np.log10(10e15) + hist_bin_width, hist_bin_width)  # Binning edges
        cmap = plt.get_cmap('Spectral')
        for iii, aperture in enumerate(aperture_list):
            # Compute statistics in each bin
            medians = []
            lower_1sigma = []
            upper_1sigma = []
            bin_centers = []
            for i in range(len(hist_bins) - 1):
                mask = (np.log10(dict_plot['%s'%aperture]['stellar_mass']) >= hist_bins[i]) & (np.log10(dict_plot['%s'%aperture]['stellar_mass']) < hist_bins[i + 1])
                y_bin = dict_plot['%s'%aperture]['gas_mass'].value[mask]
            
                # Remove zero values before computing statistics
                y_bin = y_bin[y_bin > 0]
            
                if len(y_bin) >= 10:  # Ensure the bin contains data
                    medians.append(np.median(y_bin))
                    lower_1sigma.append(np.percentile(y_bin, 16))  # -1σ (16th percentile)
                    upper_1sigma.append(np.percentile(y_bin, 84))  # +1σ (84th percentile)
                    bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        
            # Convert bin centers back to linear space for plotting
            bin_centers = 10**np.array(bin_centers)
        
            linecol = cmap((iii+1) / (len(aperture_list)))
            line, = plt.plot(bin_centers, medians, linewidth=1, color=linecol, label='%s kpc'%aperture)
            plt.fill_between(bin_centers, lower_1sigma, upper_1sigma, color=line.get_color(), alpha=0.3)
        


    #-----------------
    # Add observations
    if add_observational:
        print('no obs available') 
        
        
    #-----------
    # Axis formatting
    plt.xlim(10**9, 10**12)
    if plot_fractions:
        plt.ylim(10**-4, 1)
    else:
        plt.ylim(10**3, 10**11)
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    
    dict_ylabel = {'mass': {'gas_mass': r'$M_{\mathrm{gas}}$ [M$_{\odot}$]',
                            'star_forming_gas_mass': r'$M_{\mathrm{gas,SF}}$ [M$_{\odot}$]',
                            'molecular_hydrogen_mass': r'$M_{\mathrm{H_{2}}}$ [M$_{\odot}$]',
                            'atomic_hydrogen_mass': r'$M_{\mathrm{HI}}$ [M$_{\odot}$]',
                            'molecular_gas_mass': r'$M_{\mathrm{mol}}$ (incl. He) [M$_{\odot}$]',
                            'molecular_and_atomic_hydrogen_mass': r'$M_{\mathrm{HI+H_{2}}}$ [M$_{\odot}$]',
                            'molecular_and_atomic_gas_mass': r'$M_{\mathrm{HI+H_{mol}}}$ [M$_{\odot}$]',
                            'gas_mass_in_cold_dense_gas': r'$M_{\mathrm{cold,dense}}$ [M$_{\odot}$]' 
                            },
                   'fraction': {'gas_mass': r'$M_{\mathrm{gas}}/(M_{*}+M_{\mathrm{gas}})$',
                                'star_forming_gas_mass': r'$M_{\mathrm{gas,SF}}/(M_{\mathrm{gas,SF}}+M_{*})$',
                                'molecular_hydrogen_mass': r'$M_{\mathrm{H_{2}}}/(M_{\mathrm{H_{2}}}+M_{*})$',
                                'atomic_hydrogen_mass': r'$M_{\mathrm{HI}}/(M_{\mathrm{HI}}+M_{*})$',
                                'molecular_gas_mass': '$M_{\mathrm{mol}}/(M_{\mathrm{mol}}+M_{*})$',
                                'molecular_and_atomic_hydrogen_mass': r'$M_{\mathrm{HI+H_{2}}}/(M_{\mathrm{HI+H_{2}}}+M_{*})$',
                                'molecular_and_atomic_gas_mass': r'$M_{\mathrm{HI+H_{mol}}}/(M_{\mathrm{HI+H_{mol}}}+M_{*})$',
                                'gas_mass_in_cold_dense_gas': r'$M_{\mathrm{cold,dense}}/(M_{\mathrm{cold,dense}}+M_{*})$'
                                }}
    plt.xlabel(r'$M_*$ (50 kpc) [M$_{\odot}$]')
    plt.ylabel(r'%s'%dict_ylabel['%s'%('fraction' if plot_fractions else 'mass')][mass_type])
    
    
    #------------
    # colorbar
    #fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'
    
      
    #-----------  
    # Annotations
    if not format_z:
        if plot_fractions:
            plt.text(10**11.2, 0.5, '${z=%.2f}$' %z, fontsize=7)
        else:
            plt.text(10**11.2, 10**10.3, '${z=%.2f}$' %z, fontsize=7)
    if centrals_or_satellites != 'both':
        axs.set_title(r'%s' %(centrals_or_satellites), size=7, loc='left', pad=3)

    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '%s'%('redshift_' if format_z else '') + '_' + centrals_or_satellites + '_' + savefig_txt
        add_snap_directory = ('snap_%s/'%snapshot_no[0] if len(snapshot_no)==1 else '')
        
        plt.savefig("%s/soap_galaxy_relations/%s%s%s_%s_radiuscheck_stelmass-%s%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, '_'.join(str(x) for x in aperture_list), savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600) 
        print("\n  SAVED: %s/soap_galaxy_relations/%s%s%s_%s_radiuscheck_stelmass-%s%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), mass_type, '_'.join(str(x) for x in aperture_list), savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - sfr or stelmass - ssfr, coloured by kappa stars
def _stelmass_sfr(simulation_run = ['L100_m6'], 
                   simulation_type = ['THERMAL_AGN_m6'],
                    snapshot_no   = [127],        # available for L100_m6: 127, 119, 114, 102, 092
                   #=====================================
                   # Graph settings
                   sfr_or_ssfr      = 'sfr',        # sfr or ssfr
                   centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                   aperture = 'exclusive_sphere_50kpc', 
                   #----------
                   add_observational = False,        # Adapts based on imput mass_type, and using references from pipeline
                   format_z          = False,        # For variations in z graphs
                   hist_bin_width = 0.25,
                     lower_mass_limit = 10**8.8,
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
                        
    #-----------------
    # Add SOAP data
    if len(snapshot_no) > len(simulation_run):
        simulation_run_list = simulation_run * len(snapshot_no)
        simulation_type_list = simulation_type * len(snapshot_no)
        snapshot_no_list = snapshot_no
    else:
        simulation_run_list = simulation_run
        simulation_type_list = simulation_type
        snapshot_no_list = snapshot_no
    
    
    run_name_list = []
    redshift_list = []
    for sim_run_i, sim_type_i, snap_i in zip(simulation_run_list, simulation_type_list, snapshot_no_list):
        # Load data
        simulation_dir = sim_run_i + '/' + sim_type_i
        soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
    
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0]
        #lookbacktime
        if run_name not in run_name_list:
            run_name_list.append(run_name)
        redshift_list.append(z)
        print('\n', run_name, snap_i)
    
    
        # Get stelmass and gas mass data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
        sfr          = attrgetter('%s.%s'%(aperture, 'star_formation_rate'))(data)
        
        stellar_mass.convert_to_units('Msun')
        sfr.convert_to_units(u.Msun/u.yr)
        sfr.convert_to_physical()
        
        # Get other data
        central_sat = attrgetter('input_halos.is_central')(data)
        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)

        # Mask for central/satellite and no DM-only galaxies
        if centrals_or_satellites == 'centrals':
            candidates = np.argwhere(np.logical_and(central_sat == 1, 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            sfr = sfr[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking only centrals: ', len(stellar_mass))
        elif centrals_or_satellites == 'satellites':
            candidates = np.argwhere(np.logical_and(central_sat == 0, 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            sfr = sfr[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking only satellites: ', len(stellar_mass))
        else:
            candidates = np.argwhere(stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
            stellar_mass = stellar_mass[candidates]
            sfr = sfr[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking all galaxies: ', len(stellar_mass))

        if sfr_or_ssfr == 'sfr':
            sfr_plot = sfr
        elif sfr_or_ssfr == 'ssfr':
            sfr_plot = sfr/stellar_mass
    
    
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
        mapper = cm.ScalarMappable(norm=norm, cmap='mymap')         #cmap=cm.coolwarm)
        
        plt.scatter(stellar_mass, sfr_plot, c=kappa_stars, s=1.5, cmap='mymap', norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)
        
        #-----------------
        # Define binning parameters

        hist_bins = np.arange(np.log10(lower_mass_limit), np.log10(10e15) + hist_bin_width, hist_bin_width)  # Binning edges
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        print('a')
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = sfr_plot.value[mask]
            
            # Remove inactive galaxies from ssfr values before computing statistics
            if sfr_or_ssfr == 'ssfr':
                y_bin = y_bin[y_bin > 1e-11]
            else:
                y_bin = y_bin[y_bin > 0]
            
            if len(y_bin) >= 10:  # Ensure the bin contains data
                medians.append(np.median(y_bin))
                lower_1sigma.append(np.percentile(y_bin, 16))  # -1σ (16th percentile)
                upper_1sigma.append(np.percentile(y_bin, 84))  # +1σ (84th percentile)
                bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        

        # Convert bin centers back to linear space for plotting
        bin_centers = 10**np.array(bin_centers)
        
        plt.plot(bin_centers, medians, color='k', label='Median', linewidth=1)
        plt.fill_between(bin_centers, lower_1sigma, upper_1sigma, color='k', alpha=0.3, label='1-Sigma')
        
    

    #-----------------
    # Add observations
    if add_observational:
        print('no obs available') 
        
        
    #-----------
    # Axis formatting
    plt.xlim(10**9, 10**12)
    if sfr_or_ssfr == 'sfr':
        plt.ylim(10**-4, 10**2)
    elif sfr_or_ssfr == 'ssfr':
        plt.ylim(10**-15, 10**-8)
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    dict_aperture = {'exclusive_sphere_30kpc': '30 pkpc', 
                     'exclusive_sphere_50kpc': '50 pkpc'}
    dict_ylabel = {'sfr': 'SFR (%s) [M$_{\odot}$ yr$^{-1}$]'%dict_aperture[aperture],
                   'ssfr': 'sSFR (%s) [yr$^{-1}$]'%dict_aperture[aperture]}
    plt.xlabel(r'$M_{*}$ (%s) [M$_{\odot}$]'%dict_aperture[aperture])
    plt.ylabel(r'%s'%(dict_ylabel[sfr_or_ssfr]))
    
    
    #------------
    # colorbar
    fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    if not format_z:
        if sfr_or_ssfr == 'sfr':
            plt.text(10**11.2, 10**1.5, '${z=%.2f}$' %z, fontsize=7)
        if sfr_or_ssfr == 'ssfr':
            plt.text(10**11.2, 10**-8.5, '${z=%.2f}$' %z, fontsize=7)
    if centrals_or_satellites != 'both':
        axs.set_title(r'%s' %(centrals_or_satellites), size=7, loc='left', pad=3)

    
    #-----------
    # Legend
    axs.legend(loc='lower left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '%s'%('redshift_' if format_z else '') + aperture + '_' + centrals_or_satellites + '_' + savefig_txt
        add_snap_directory = ('snap_%s/'%snapshot_no[0] if len(snapshot_no)==1 else '')
        
        plt.savefig("%s/soap_galaxy_relations/%s%s%s_%s_stelmass-%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), sfr_or_ssfr, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600) 
        print("\n  SAVED: %s/soap_galaxy_relations/%s%s%s_%s_stelmass-%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), sfr_or_ssfr, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()


#-----------------
# Returns stelmass - sfr or stelmass - ssfr, coloured by kappa stars
def _stelmass_u_r_correa2017(simulation_run = ['L100_m6'], 
                   simulation_type = ['THERMAL_AGN_m6'],
                    snapshot_no   = [127],        # available for L100_m6: 127, 119, 114, 102, 092
                   #=====================================
                   # Graph settings
                   magnitudes      = 'u-r',        # [ u-r / u-g ]
                   centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                   aperture = 'exclusive_sphere_50kpc', 
                   #----------
                   add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                   format_z          = False,        # For variations in z graphs
                   lower_mass_limit  = 10**9.5,
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
    if len(snapshot_no) > len(simulation_run):
        simulation_run_list = simulation_run * len(snapshot_no)
        simulation_type_list = simulation_type * len(snapshot_no)
        snapshot_no_list = snapshot_no
    else:
        simulation_run_list = simulation_run
        simulation_type_list = simulation_type
        snapshot_no_list = snapshot_no
    
    
    run_name_list = []
    redshift_list = []
    for sim_run_i, sim_type_i, snap_i in zip(simulation_run_list, simulation_type_list, snapshot_no_list):
        # Load data
        simulation_dir = sim_run_i + '/' + sim_type_i
        soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
    
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0]
        #lookbacktime
        if run_name not in run_name_list:
            run_name_list.append(run_name)
        redshift_list.append(z)
        print('\n', run_name, snap_i)
    
    
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
                
        if magnitudes == 'u-r':
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - r_mag
            u_mag = 0
            r_mag = 0
        if magnitudes == 'u-g':
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            g_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,1])
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            g_mag = cosmo_array(g_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            mag_plot = u_mag - g_mag
            u_mag = 0
            g_mag = 0
        
        # Get other data
        central_sat = attrgetter('input_halos.is_central')(data)
        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)

        # Mask for central/satellite and no DM-only galaxies
        if centrals_or_satellites == 'centrals':
            candidates = np.argwhere(np.logical_and(central_sat == 1, 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            mag_plot = mag_plot[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking only centrals: ', len(stellar_mass))
        elif centrals_or_satellites == 'satellites':
            candidates = np.argwhere(np.logical_and(central_sat == 0, 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            mag_plot = mag_plot[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking only satellites: ', len(stellar_mass))
        else:
            candidates = np.argwhere(stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
            stellar_mass = stellar_mass[candidates]
            mag_plot = mag_plot[candidates]
            kappa_stars = kappa_stars[candidates]
            print('Masking all galaxies: ', len(stellar_mass))
    
    
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
        
        # Plot scatter
        ax_scat.scatter(stellar_mass, mag_plot, c=kappa_stars, s=1.5, cmap=mymap, norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)
        #ax_scat.scatter(stellar_mass[~mask_kappa], mag_plot[~mask_kappa], c=kappa_stars[~mask_kappa], s=1.5, cmap='seismic_r', norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)
        
        #-----------------
        # Histogram
        hist_bins = np.arange(0.5, 3.1, 0.05)  # Binning edges
        
        n_LTG, _    = np.histogram(mag_plot[mask_kappa], weights=(np.ones(len(stellar_mass[mask_kappa]))/len(stellar_mass[mask_kappa])), bins=hist_bins)
        n_LTG, _, _ = ax_hist.hist(mag_plot[mask_kappa], weights=(np.ones(len(stellar_mass[mask_kappa]))/len(stellar_mass[mask_kappa])), bins=hist_bins, alpha=0.5, facecolor='b', orientation='horizontal', label='$\kappa_{\mathrm{co}}^{*}>0.4$\n($f_{\mathrm{LTG}}=%.2f$)'%(len(stellar_mass[mask_kappa])/len(stellar_mass)))
        
        n_ETG, _    = np.histogram(mag_plot[~mask_kappa], weights=(np.ones(len(stellar_mass[~mask_kappa]))/len(stellar_mass[~mask_kappa])), bins=hist_bins)
        n_ETG, _, _ = ax_hist.hist(mag_plot[~mask_kappa], weights=(np.ones(len(stellar_mass[~mask_kappa]))/len(stellar_mass[~mask_kappa])), bins=hist_bins, alpha=0.5, facecolor='r', orientation='horizontal', label='$\kappa_{\mathrm{co}}^{*}<0.4$\n($f_{\mathrm{ETG}}=%.2f$)'%(len(stellar_mass[~mask_kappa])/len(stellar_mass)))
        
        
    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        ax_scat.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        
        
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
    
    ylabel_dict = {'u-r': '$u^{*} - r^{*}$', 
                   'u-g': '$u^{*} - g^{*}$'}
    ax_scat.set_yticklabels([])
    ax_hist.set_ylabel(r'%s'%ylabel_dict[magnitudes])
        
    #------------
    # colorbar
    fig.colorbar(mapper, ax=ax_scat, label='$\kappa_{\mathrm{co}}^{*}$')      #, extend='max'  
      
    #-----------  
    # Annotations
    if not format_z:
        ax_scat.text(10**11.5, 2.8, '${z=%.2f}$' %z, fontsize=7)
    if centrals_or_satellites != 'both':
        ax_hist.set_title(r'%s' %(centrals_or_satellites), size=7, loc='left', pad=3)
    if add_observational:
        ax_scat.text(10**11.9, 2.65, '↑ red sequence', fontsize=7, color='r', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
        ax_scat.text(10**11.9, 2.4, '↓ blue cloud', fontsize=7, color='b', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')

    
    #-----------
    # Legend
    ax_hist.legend(loc='lower left', ncol=1, frameon=False)
        
    #-----------
    # other
    #plt.tight_layout()
    ax_hist.grid(alpha=0.4, zorder=-5, lw=0.7)
    
    if savefig:
        savefig_txt_save = '%s'%('redshift_' if format_z else '') + aperture + '_' + centrals_or_satellites + '_' + savefig_txt
        
        add_snap_directory = ('snap_%s/'%snapshot_no[0] if len(snapshot_no)==1 else '')
        
        plt.savefig("%s/soap_galaxy_relations/%s%s%s_%s_Correa_stelmass-%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), magnitudes, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600) 
        print("\n  SAVED: %s/soap_galaxy_relations/%s%s%s_%s_Correa_stelmass-%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), magnitudes, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()
#-----------------
# Returns stelmass - sfr or stelmass - ssfr, coloured by kappa stars, size by H2 mass
def _stelmass_u_r(simulation_run = ['L100_m6'], 
                   simulation_type = ['THERMAL_AGN_m6'],
                    snapshot_no   = [127],        # available for L100_m6: 127, 119, 114, 102, 092
                   #=====================================
                   # Graph settings
                   magnitudes      = 'u-r',        # [ u-r / u-g ]
                   centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                   aperture = 'exclusive_sphere_50kpc', 
                   #----------
                   add_observational = True,        # Adapts based on imput mass_type, and using references from pipeline
                   format_z          = False,        # For variations in z graphs
                   hist_bin_width = 0.25,
                     lower_mass_limit = 10**8.8,
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
                        
    #-----------------
    # Add SOAP data
    if len(snapshot_no) > len(simulation_run):
        simulation_run_list = simulation_run * len(snapshot_no)
        simulation_type_list = simulation_type * len(snapshot_no)
        snapshot_no_list = snapshot_no
    else:
        simulation_run_list = simulation_run
        simulation_type_list = simulation_type
        snapshot_no_list = snapshot_no
    
    
    run_name_list = []
    redshift_list = []
    for sim_run_i, sim_type_i, snap_i in zip(simulation_run_list, simulation_type_list, snapshot_no_list):
        # Load data
        simulation_dir = sim_run_i + '/' + sim_type_i
        soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
    
        # Get metadata from file
        z = data.metadata.redshift
        run_name = data.metadata.run_name
        box_size = data.metadata.boxsize[0]
        #lookbacktime
        if run_name not in run_name_list:
            run_name_list.append(run_name)
        redshift_list.append(z)
        print('\n', run_name, snap_i)
    
    
        # Get stelmass, molecular hydrogen, and magnitude data
        stellar_mass = attrgetter('%s.%s'%(aperture, 'stellar_mass'))(data)
        stellar_mass.convert_to_units('Msun')
        stellar_mass.convert_to_physical()
        
        # Raw H2 without He adjustment
        H2_mass = attrgetter('%s.%s'%(aperture, 'molecular_hydrogen_mass'))(data)
        H2_mass.convert_to_units('Msun')
        
        if magnitudes == 'u-r':
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,2])
            mag_plot = u_mag - r_mag
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            u_mag = 0
            r_mag = 0
        if magnitudes == 'u-g':
            u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,0])
            g_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture, 'stellar_luminosity'))(data))[:,1])
            mag_plot = u_mag - g_mag
            u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            g_mag = cosmo_array(g_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
            u_mag = 0
            g_mag = 0
        
        # Get other data
        central_sat = attrgetter('input_halos.is_central')(data)
        kappa_stars = attrgetter('%s.%s'%(aperture, 'kappa_corot_stars'))(data)

        # Mask for central/satellite and no DM-only galaxies
        if centrals_or_satellites == 'centrals':
            candidates = np.argwhere(np.logical_and(central_sat == 1, 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            mag_plot = mag_plot[candidates]
            kappa_stars = kappa_stars[candidates]
            H2_mass = H2_mass[candidates]
            print('Masking only centrals: ', len(stellar_mass))
        elif centrals_or_satellites == 'satellites':
            candidates = np.argwhere(np.logical_and(central_sat == 0, 
                                                    stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0))).squeeze()
            stellar_mass = stellar_mass[candidates]
            mag_plot = mag_plot[candidates]
            kappa_stars = kappa_stars[candidates]
            H2_mass = H2_mass[candidates]
            print('Masking only satellites: ', len(stellar_mass))
        else:
            candidates = np.argwhere(stellar_mass > cosmo_quantity(lower_mass_limit, u.Msun, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)).squeeze()
            stellar_mass = stellar_mass[candidates]
            mag_plot = mag_plot[candidates]
            kappa_stars = kappa_stars[candidates]
            H2_mass = H2_mass[candidates]
            print('Masking all galaxies: ', len(stellar_mass))
    
    
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
        plt.scatter(stellar_mass[mask_h2], mag_plot[mask_h2], c=kappa_stars[mask_h2], s=(np.log10(H2_mass[mask_h2])-(np.log10(h2_detection_limit)-1))**2.5, cmap=mymap, norm=norm, marker='o', alpha=0.75, linewidths=0, edgecolor='none')
        plt.scatter(stellar_mass[~mask_h2], mag_plot[~mask_h2], c=kappa_stars[~mask_h2], s=1.5, cmap=mymap, norm=norm, marker='P', alpha=0.75, linewidths=0, edgecolor='none')
        
        
        # Plot scatter
        #ax_scat.scatter(stellar_mass, mag_plot, c=kappa_stars, s=1.5, cmap=mymap, norm=norm, marker='o', linewidths=0, edgecolor='none', alpha=0.75)
        
        
        #-----------------
        # Define binning parameters

        """hist_bins = np.arange(np.log10(lower_mass_limit), np.log10(10e15) + hist_bin_width, hist_bin_width)  # Binning edges
        
        # Compute statistics in each bin
        medians = []
        lower_1sigma = []
        upper_1sigma = []
        bin_centers = []
        print('a')
        for i in range(len(hist_bins) - 1):
            mask = (np.log10(stellar_mass) >= hist_bins[i]) & (np.log10(stellar_mass) < hist_bins[i + 1])
            y_bin = sfr_plot.value[mask]
            
            if len(y_bin) >= 10:  # Ensure the bin contains data
                medians.append(np.median(y_bin))
                lower_1sigma.append(np.percentile(y_bin, 16))  # -1σ (16th percentile)
                upper_1sigma.append(np.percentile(y_bin, 84))  # +1σ (84th percentile)
                bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        

        # Convert bin centers back to linear space for plotting
        bin_centers = 10**np.array(bin_centers)
        
        plt.plot(bin_centers, medians, color='k', label='Median', linewidth=1)
        plt.fill_between(bin_centers, lower_1sigma, upper_1sigma, color='k', alpha=0.3, label='1-Sigma')
        """
        

    #-----------------
    # Add observations
    if add_observational:
        # Schawinski+14, used in Correa+17
        # with equation (u∗ −r∗) = 0.25 log10(M∗/ M) − 0.495
        plt.plot([10**9, 10**15], [(0.25*np.log10(10**9))-0.495, (0.25*np.log10(10**15))-0.495], lw=1.3, ls='--', color='k', label='Schawinski+14', zorder=10)
        
        
    #-----------
    # Axis formatting
    plt.xlim(10**9, 10**12)
    plt.ylim(0.8, 3.3)
    plt.xscale("log")
    #plt.yticks(np.arange(-5, -1.4, 0.5))
    #plt.xticks(np.arange(9.5, 12.5, 0.5))
    axs.minorticks_on()
    axs.tick_params(axis='x', which='minor')
    dict_aperture = {'exclusive_sphere_30kpc': '30 \mathrm{pkpc}', 
                     'exclusive_sphere_50kpc': '50 \mathrm{pkpc}'}
    plt.xlabel(r'$M_{*}$ ($%s$) [M$_{\odot}$]'%dict_aperture[aperture])
    ylabel_dict = {'u-r': '$u^{*} - r^{*}$', 
                   'u-g': '$u^{*} - g^{*}$'}
    plt.ylabel(r'%s'%ylabel_dict[magnitudes])
        
    #------------
    # colorbar
    fig.colorbar(mapper, ax=axs, label='$\kappa_{\mathrm{co}}^{*}$', extend='both')      #, extend='max'  
      
    #-----------  
    # Annotations
    if not format_z:
        plt.text(10**11.2, 3.1, '${z=%.2f}$' %z, fontsize=7)
    if centrals_or_satellites != 'both':
        axs.set_title(r'%s' %(centrals_or_satellites), size=7, loc='left', pad=3)
    if add_observational:
        plt.text(10**11.9, 2.65, '↑ red sequence', fontsize=7, color='r', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
        plt.text(10**11.9, 2.4, '↓ blue cloud', fontsize=7, color='b', rotation=14, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')


    
    #-----------
    # Legend
    for i, h2_i in enumerate([7, 8, 9, 10]):
        s = (h2_i-(np.log10(h2_detection_limit)-1))**2.5
        if i == 0:
            plt.scatter([10**9], [0], c='r', s=s, label=r'$\mathrm{log} \: M_{\mathrm{H_{2}}}/\mathrm{M}_{\odot}=%.1f$'%h2_i, linewidths=0, edgecolor='none')
        else:
            plt.scatter([10**9], [0], c='r', s=s, label='%.1f'%h2_i, linewidths=0, edgecolor='none')
    plt.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper left')
    
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        savefig_txt_save = '%s'%('redshift_' if format_z else '') + aperture + '_' + centrals_or_satellites + '_' + savefig_txt
        
        add_snap_directory = ('snap_%s/'%snapshot_no[0] if len(snapshot_no)==1 else '')
        
        plt.savefig("%s/soap_galaxy_relations/%s%s%s_%s_stelmass-%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), magnitudes, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600) 
        print("\n  SAVED: %s/soap_galaxy_relations/%s%s%s_%s_stelmass-%s_%s.%s" %(fig_dir, add_snap_directory, ('redshift/' if format_z else ''), '_'.join(run_name_list), '_'.join(str(x) for x in snapshot_no), magnitudes, savefig_txt_save, file_format))
    if showfig:
        plt.show()
    plt.close()



#============================================
# Plot at a few different z
# Add observational results if possible
# Add Lagos EAGLE results if possible




for snap_i in [127]:
    #--------------------
    # Plots stellarmass against cold dense gas (T<10**4.5, n>0.1)
    """_stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
    # Plots stellarmass against cold dense gas fractions (gas / stars+gas)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],      
                    mass_type = 'gas_mass_in_cold_dense_gas',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)"""
                
                
    #-------------------- 
    # Plots stellarmass against atomic+H2 gas (excl. He)
    """_stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
    # Plots stellarmass against atomic+molecular H fractions (HI+H2 (He adjusted)/stellar + HI+H2 (He adjusted))
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],      
                    mass_type = 'molecular_and_atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)"""



    #-------------------- 
    # Plots stellarmass against atomic gas
    """_stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
    # Plots stellarmass against atomic gas fractions (H1/stellar+HI)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)"""


    #-----------------
    # Plots stellarmass against H2 gas (ex. He)
    """_stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
    # Plots stellarmass against molecular gas fractions (H2/stellar+H2)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                         plot_fractions = True,
                    savefig = True)"""

    #=======================
    # Comparing apertures 50 vs 30, both stelmass and gas mass
    # Atomic gas aperture check 30 vs 50
    """_stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = False,
                    savefig = True)
    _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = False,
                    savefig = True)
    _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = True,
                    savefig = True)
    # H2 gas aperture check 30 vs 50
    _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = False,
                    savefig = True)
    _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = False,
                    savefig = True)
    _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = True,
                    savefig = True)
    _stelmass_gasmass_compare_aperture(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                        aperture_list = [30, 50], 
                         plot_fractions = True,
                    savefig = True)"""
                
    #==============================
    # Comparing radius out to which we extract gas masses, keeping stellar mass at 50 kpc
    # Atomic gas
    """_stelmass_gasmass_compare_radii(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                        aperture_list = [1, 3, 10, 30, 50], 
                    savefig = True)
    _stelmass_gasmass_compare_radii(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                        aperture_list = [1, 3, 10, 30, 50], 
                    savefig = True)
    _stelmass_gasmass_compare_radii(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'atomic_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                        aperture_list = [1, 3, 10, 30, 50], 
                    savefig = True)"""

    # Hydrogen gas
    """_stelmass_gasmass_compare_radii(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                        aperture_list = [1, 3, 10, 30, 50], 
                    savefig = True)
    _stelmass_gasmass_compare_radii(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                        aperture_list = [1, 3, 10, 30, 50], 
                    savefig = True)
    _stelmass_gasmass_compare_radii(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    mass_type = 'molecular_hydrogen_mass',    # [ molecular_and_atomic_hydrogen_mass / molecular_hydrogen_mass / atomic_hydrogen_mass ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                        aperture_list = [1, 3, 10, 30, 50], 
                    savefig = True)"""


    #====================================
    # Plots stellarmass against SFR, coloured by kappa stars
    """_stelmass_sfr(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    sfr_or_ssfr      = 'sfr',
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_sfr(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    sfr_or_ssfr      = 'sfr',
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_sfr(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    sfr_or_ssfr      = 'sfr',
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)

    # Plots stellarmass against sSFR, coloured by kappa stars
    _stelmass_sfr(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    sfr_or_ssfr      = 'ssfr',
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_sfr(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    sfr_or_ssfr      = 'ssfr',
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_sfr(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    sfr_or_ssfr      = 'ssfr',
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)"""


    #====================================
    # Plots stellarmass against u-r, coloured by kappa with a histogram, analogous to Correa+17 but with 50kpc aperture instead of 30kpc
    """_stelmass_u_r_correa2017(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],  
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_u_r_correa2017(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],  
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_u_r_correa2017(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],  
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)"""
                    
    _stelmass_u_r_correa2017(simulation_run = ['L100_m6'], simulation_type = ['HYBRID_AGN_m6'], snapshot_no     = [snap_i],  
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_u_r_correa2017(simulation_run = ['L100_m6'], simulation_type = ['HYBRID_AGN_m6'], snapshot_no     = [snap_i],  
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_u_r_correa2017(simulation_run = ['L100_m6'], simulation_type = ['HYBRID_AGN_m6'], snapshot_no     = [snap_i],  
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)
                    
    
    #====================================
    # Plots stellarmass against u-r, with point size by molecular hydrogen mass
    """_stelmass_u_r(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    magnitudes = 'u-r',    # [ u-r / u-g ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_u_r(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    magnitudes = 'u-r',    # [ u-r / u-g ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _stelmass_u_r(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    magnitudes = 'u-r',    # [ u-r / u-g ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)"""
    # Plots r-band magnitude M_r against u-r, with point size by molecular hydrogen mass
    """_Mr_u_r(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    magnitudes = 'u-r',    # [ u-r / u-g ]
                     centrals_or_satellites = 'both',      # [ both / centrals / satellites ]
                    savefig = True)
    _Mr_u_r(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    magnitudes = 'u-r',    # [ u-r / u-g ]
                     centrals_or_satellites = 'centrals',      # [ both / centrals / satellites ]
                    savefig = True)
    _Mr_u_r(simulation_run = ['L100_m6'], simulation_type = ['THERMAL_AGN_m6'], snapshot_no     = [snap_i],        
                    magnitudes = 'u-r',    # [ u-r / u-g ]
                     centrals_or_satellites = 'satellites',      # [ both / centrals / satellites ]
                    savefig = True)"""










