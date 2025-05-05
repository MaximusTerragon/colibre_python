import swiftsimio as sw
import swiftgalaxy
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import csv
import time
import math
from operator import attrgetter
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from graphformat import set_rc_params
from generate_schechter import _leja_schechter
from read_dataset_directories_colibre import _assign_directories


# Full example available on RobMcGibbon's COLIBRE_Introduction

#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================



def _plot_galaxy_evolution(simulation_run = ['L100_m6'], 
                   simulation_type  = ['THERMAL_AGN_m6'],
                   snapshot_list    = [92, 102, 114, 119, 127],        # available for L100_m6: 127, 119, 114, 102, 092
                   #-----------------------
                   track_id         = 12345, 
                     print_galaxy   = True,
                   #===========================================================
                   # Graph settings
                   aperture_general = 'exclusive_sphere_30kpc', 
                   aperture_halo    = 'exclusive_sphere_30kpc', 
                   aperture_angle   = 'exclusive_sphere_10kpc', 
                   #----------
                   # Merger settings
                     min_merger_ratio   = 0.1,
                   #-------------------------------------------
                   # BH settings
                     use_closest_to_COP   = False,                   # Uses closest to COP (new method) or most massive in 1 hmr (_old)
                   #-------------------------------------------
                   # Masses   [ Msun ]
                     plot_halomass        = False,
                     plot_stelmass        = True,
                     plot_gasmass         = True,
                       plot_sfmass        = False,
                       plot_nsfmass       = False,
                       plot_colddense     = True,       # t < 10**4.5, n > 0.1 cm**-3
                       plot_HImass        = True,
                       plot_H2mass        = True,
                       plot_molecularmass = False,      # He corrected
                   # Angles   [ deg ]
                     plot_angles          = False,
                     #plot_inclination     = False,
                   # Inflow (gas) [ Msun/yr ]
                     plot_sfr             = True,
                   # sSFR     [ /yr ]
                     plot_ssfr            = True,
                   # Radius   [ pkpc]
                     plot_radius_stars    = False,
                     plot_radius_gas      = False,
                   # Morphology stars/gas [ ]
                     plot_kappa_stars     = True,
                     plot_kappa_gas       = False,
                     plot_ellip           = True,
                     plot_triax           = True,
                   # Metallicity  [ Z ]
                     plot_Z_stars         = False,
                     plot_Z_gas           = False,
                   # Colour magnitude (GAMA bands, no dust per Trayford et al. 2015)
                     plot_u_r             = True,
                     plot_g_r             = False,
                   #-------------------------------------------
                   # Merger settings
                     min_merger_ratio   = 0.1,
                   #-------------------------------------------
                   # Plot redshift at top?
                     redshift_axis       = True,      # Whether to add the redshift axis
                   #==========================================================
                   showfig       = False,
                   savefig       = True,
                     file_format = 'pdf',
                     savefig_txt = '', 
                   #--------------------------
                   print_progress = False,
                     debug = False):
    
    
    # Directories
    simulation_dir = simulation_run + '/' + simulation_type
    soap_dir       = colibre_base_path + simulation_dir + "/SOAP/"
    
    #------------------
    # Dictionary containing all properties wanted
    dict_plot = {
        "soap_indicies": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
          "snapshot": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
          "redshift": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
          "lookbacktime": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Gyr')
        "plot_halomass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
        "plot_stelmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
        "plot_gasmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_sfmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_nsfmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_colddense": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_HImass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_H2mass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_molecularmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_HIH2mass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
        "plot_angles": np.zeros(snapshot_list.shape[0]) * unyt.Unit('deg'),
          "plot_inclination": np.zeros(snapshot_list.shape[0]) * unyt.Unit('deg'),
        "plot_sfr": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun/yr'),
          "plot_ssfr": np.zeros(snapshot_list.shape[0]) * unyt.Unit('/yr'),
        "plot_radius_stars": np.zeros(snapshot_list.shape[0]) * unyt.Unit('kpc'),
          "plot_radius_gas": np.zeros(snapshot_list.shape[0]) * unyt.Unit('kpc'),
        "plot_kappa_stars": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_kappa_gas": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_ellip": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_triax": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "plot_Z_stars": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_Z_gas": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "plot_u_r": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_g_r": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
    }   
    for i, snap_i in enumerate(snapshot_list):
        #print('Loading snapshot', snap_i)
        # Load data
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
        
        # Identify the location of the halo in this SOAP catalogue
        soap_idx = np.argmax(data.input_halos_hbtplus.track_id[:].value == track_id)
        lookbacktime = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(data.metadata.redshift)).value
        
        # Extract desired values
        dict_plot['soap_indicies'][i] = soap_idx
        dict_plot['snapshot'][i]      = snap_i
        dict_plot['redshift'][i]      = data.metadata.redshift
        dict_plot['lookbacktime'][i]  = lookbacktime
        
        print('redshift: ', data.metadata.redshift)
        print('lookbacktime: ', lookbacktime, '\n')
        
        dict_plot['plot_halomass'][i] = (attrgetter('%s.%s'%(aperture_halo, 'quantity'))(data))[soap_idx]
        dict_plot['plot_stelmass'][i] = (attrgetter('%s.%s'%(aperture_general, 'stellar_mass'))(data))[soap_idx]
        dict_plot['plot_gasmass'][i]  = (attrgetter('%s.%s'%(aperture_general, 'gas_mass'))(data))[soap_idx]
        dict_plot['plot_sfmass'][i]   = (attrgetter('%s.%s'%(aperture_general, 'star_forming_gas_mass'))(data))[soap_idx]
        dict_plot['plot_nsfmass'][i]  = (attrgetter('%s.%s'%(aperture_general, 'gas_mass'))(data))[soap_idx] - (attrgetter('%s.%s'%(aperture_general, 'star_forming_gas_mass'))(data))[soap_idx]
        dict_plot['plot_colddense'][i] = (attrgetter('%s.%s'%(aperture_general, 'gas_mass_in_cold_dense_gas'))(data))[soap_idx]
        dict_plot['plot_HImass'][i]   = (attrgetter('%s.%s'%(aperture_general, 'atomic_hydrogen_mass'))(data))[soap_idx]
        dict_plot['plot_H2mass'][i]   = (attrgetter('%s.%s'%(aperture_general, 'molecular_hydrogen_mass'))(data))[soap_idx]
        dict_plot['plot_molecularmass'][i]   = (attrgetter('%s.%s'%(aperture_general, 'molecular_hydrogen_mass'))(data))[soap_idx] * (1 + ((attrgetter('%s.%s'%(aperture_general, 'helium_mass'))(data))[soap_idx]/(attrgetter('%s.%s'%(aperture_general, 'hydrogen_mass'))(data))[soap_idx]))
        
        
        
        
        
        fix 'quantity' below:
        
        
        
        
        
        
        
        
        dict_plot['plot_angle'][i] = (attrgetter('%s.%s'%(aperture_angle, 'quantity'))(data))[soap_idx]
        dict_plot['plot_inclination'][i] = 0
        
        dict_plot['plot_sfr'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        dict_plot['plot_ssfr'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        
        dict_plot['plot_radius_stars'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        dict_plot['plot_radius_gas'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        
        dict_plot['plot_kappa_stars'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        dict_plot['plot_kappa_gas'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        dict_plot['plot_ellip'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        dict_plot['plot_triax'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        
        dict_plot['plot_Z_stars'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        dict_plot['plot_Z_gas'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        
        
        dict_plot['plot_u_r'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx] x
        dict_plot['plot_g_r'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx] x
        
        
        
        
    
    
    #------------------------
    # Default graphs: [ angles, massrate, mass, ssfr, radius, kappa, Z, edd, lbol ]
    plot_height_ratios = []
    plot_names         = []
    if plot_halomass or plot_stelmass or plot_gasmass or plot_sfmass or plot_nsfmass or plot_colddense or plot_HImass or plot_H2mass or plot_molecularmass:
        plot_height_ratios.append(2)
        plot_names.append('mass')
    if plot_angles:
        plot_height_ratios.append(3)
        plot_names.append('angles')
    if plot_sfr:
        plot_height_ratios.append(2)
        plot_names.append('massrate')
    if plot_ssfr:
        plot_height_ratios.append(2)
        plot_names.append('ssfr')
    if plot_radius_stars or plot_radius_gas:
        plot_height_ratios.append(2)
        plot_names.append('radius')
    if plot_kappa_stars or plot_kappa_gas or plot_ellip or plot_triax:
        plot_height_ratios.append(2)
        plot_names.append('morphology')
    if plot_Z_stars or plot_Z_gas:
        plot_height_ratios.append(2)
        plot_names.append('Z')
    if plot_u_r or plot_g_r:
        plot_height_ratios.append(2)
        plot_names.append('Z')
    
    #------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(nrows=len(plot_height_ratios), ncols=1, gridspec_kw={'height_ratios': plot_height_ratios}, figsize=[3.5, 0.52*np.sum(np.array(plot_height_ratios))], sharex=True, sharey=False)
    
    #------------------------
    # Create each graph separately
    # Create redshift axis:
    redshiftticks = [0, 0.1, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20]
    ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
    for i, plot_names_i in enumerate(plot_names):
        
        ### Formatting
        ax_top = axs[i].twiny()
        if redshift_axis:
            ax_top.set_xticks(ageticks)
        
        axs[i].set_xlim(8, 0)
        ax_top.set_xlim(8, 0)
    
    
        
    
#========================================================================

#trackid_list_sample = [12345, 123456, 1234567]

# OPTIONAL - Load a sample from a given snapshot
_, trackid_list_sample, sample_input = _load_soap_sample(sample_dir, csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample_20_galaxy_visual')
savefig_txt_in = ''                                                                 # L100_m6_THERMAL_AGN_m6_127_sample_20_galaxy_visual
                                                                                    # L0025N0752_THERMAL_AGN_m5_123_sample_5_example_sample
# Plot evolution of a given trackid, valid for any snapshot
for TrackID_i in trackid_list_sample:
    _plot_galaxy_evolution(track_id=TrackID_i,
                                savefig = True)
    
    


# format the following:



simulation_dir = "/cosma8/data/dp004/colibre/Runs/L0025N0376/Fiducial_test"
snap_nr = 123   # z=0
soap_filename = f'{simulation_dir}/SOAP/halo_properties_{snap_nr:04}.hdf5'
soap = sw.load(soap_filename)


# Idenfity the TrackID of the galaxy with the largest stellar mass
stellar_mass = soap.exclusive_sphere_50kpc.stellar_mass
soap_idx = np.argmax(stellar_mass)
track_id = soap.input_halos_hbtplus.track_id[soap_idx]
# Grab the FOF ID while we're at it
fof_id = soap.input_halos_hbtplus.host_fofid[soap_idx].value
print(f'Most massive galaxy has TrackId={track_id.value}, FOF ID={fof_id}')


snapshots = np.arange(43, 124, 10)
scale_factors = np.zeros(snapshots.shape[0])
dm_mass = np.zeros(snapshots.shape[0]) * unyt.Unit('Msun')
gas_mass = np.zeros(snapshots.shape[0]) * unyt.Unit('Msun')
stellar_mass = np.zeros(snapshots.shape[0]) * unyt.Unit('Msun')
for i, snap_nr in enumerate(snapshots):
    print('Loading snapshot', snap_nr)
    soap_filename = f'{simulation_dir}/SOAP/halo_properties_{snap_nr:04}.hdf5'
    soap = sw.load(soap_filename)
    scale_factors[i] = soap.metadata.a

    # Identify the location of the halo in this SOAP catalogue
    soap_idx = np.argmax(soap.input_halos_hbtplus.track_id[:].value == track_id)
    dm_mass[i] = soap.spherical_overdensity_200_crit.dark_matter_mass[soap_idx]
    gas_mass[i] = soap.spherical_overdensity_200_crit.gas_mass[soap_idx]
    stellar_mass[i] = soap.exclusive_sphere_50kpc.stellar_mass[soap_idx]
    
    
plt.plot(scale_factors, dm_mass, label='Dark matter', color='k')
plt.plot(scale_factors, gas_mass, label='Gas', color='tab:green')
plt.plot(scale_factors, stellar_mass, label='Stars', color='tab:orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Scale factor')
plt.ylabel('Mass [Msun]')












