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


""" Unyts guide:

This applies to both halo_catalogue and particle data (.stars):

.in_units(u.km/u.s)         -> converts, makes a copy, keeps units as unyt array
.in_units(u.dimensionless)  -> converts, makes a copy, keeps units as unyt array
.to_physical()              -> converts to physical, makes a copy, keeps units as unyt array
.to(u.km/u.s)               -> same as .in_units(u.km/u.s)
    
.convert_to_units('kpc')    -> call function to convert array without copy, keeps units as unyt array
  or .convert_to_units(u.kpc)       but this function is bugged as it keeps the old units, despite converting
                                    values
.convert_to_physical()      -> call function to convert to physical
    
.value                      -> makes a copy, converts to np array (removes units)
.to_value(u.km/u.s)         -> converts, makes a copy, converts to np array (removes units)
    
7*u.Msun                    -> applies a unit to a value

"""
#------------------------------
# Make merger tree; extracts TrackID, snapshot, DescendentTrackID, HostFOFID, NestedParentTrackID, SOAP index, stelmass, gasmass, HI mass, H2 mass from all available SOAP files for ease of use
def _make_merger_tree(simulation_run   = 'L100_m6', 
                      simulation_type  = 'THERMAL_AGN_m6',
                      snapshot_list    = [92, 96, 100, 102, 104, 108, 112, 114, 116, 119, 120, 124, 127],
                      #-------------------------------------------
                      aperture_general = 'exclusive_sphere_50kpc', 
                      aperture_gas     = 'exclusive_sphere_50kpc',
                      #-------------------------------------------
                      csv_file = True,
                        csv_name = '',
                     #--------------------------
                     print_progress = False,
                       debug = False):
                      
    # Directories
    simulation_dir = simulation_run + '/' + simulation_type
    soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
    
    #================================================ 
    # Creating dictionary to collect all galaxies, labelled after TrackID
    galaxy_tree        = {}
    
    for i, snap_i in enumerate(snapshot_list):
        #print('Loading snapshot', snap_i)
        # Load data
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
        run_name = data.metadata.run_name
                
        # Loop over all trackid
        for trackid_i in tqdm(data.input_halos_hbtplus.track_id[:].value):
            # Extract all unique TrackID to galaxy_tree header
            if str(trackid_i) not in galaxy_tree.keys():
                galaxy_tree.update['%s'%trackid_i] = {'snapshot': ,
                                                      'track_id': ,
                                                      'descendent_track_id': ,
                                                      'hist_fof_id': ,
                                                      'nested_parent_track_id': ,
                                                      'soap_index': ,
                                                      'stellar_mass': ,
                                                      'gas_mass': ,
                                                      'atomic_hydrogen_mass': ,
                                                      'molecular_hydrogen_mass': ,
                                                      
                                                  
                                                  
                                                  }
                
                
            else:
        
        
        # Append existing TrackID data
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ((attrgetter('%s.%s'%(aperture_general, 'stellar_mass'))(data))[soap_idx]).to(u.Msun)
        
        
        
        
        
        
        # Identify the location of the halo in this SOAP catalogue
        soap_idx = np.argmax(data.input_halos_hbtplus.track_id[:].value == track_id)
    
    
    
    #================================================ 
    if csv_file: 
        # Converting numpy arrays to lists. When reading, may need to simply convert list back to np.array() (easy)
        class NumpyEncoder(json.JSONEncoder):
            ''' Special json encoder for numpy types '''
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
                  
        
        # Combining all dictionaries
        csv_dict = {'merger_tree': merger_tree}
        
        tree_input = {'simulation_run': simulation_run,
                      'simulation_type': simulation_type,
                      'snapshot_list': csv_sample2}
                      
        csv_dict.update({'tree_input': tree_input})
        
        
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%s_%s_merger_tree_%s.csv' %(output_dir, simulation_run, simulation_type, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s_%s_merger_tree_%s.csv' %(output_dir, simulation_run, simulation_type, csv_name))
    

#------------------------------
# Plots evolution of a given galaxy TrackID
def _plot_galaxy_evolution(simulation_run = 'L100_m6', 
                   simulation_type  = 'THERMAL_AGN_m6',
                   snapshot_list    =      [92, 96, 100, 102, 104, 108, 112, 114, 116, 119, 120, 124, 127],        # available for L100_m6: [92, 96, 100, 102, 104, 108, 112, 114, 116, 119, 120, 124, 127]
                   snapshot_list_mergers = [88, 92, 96, 100, 102, 104, 108, 112, 114, 116, 119, 120, 124],         # [88, 92, 96, 100, 102, 104, 108, 112, 114, 116, 119, 120, 124]
                   #-----------------------
                   track_id         = 12345, 
                     print_galaxy   = True,
                   #===========================================================
                   # Graph settings
                   aperture_general = 'exclusive_sphere_50kpc', 
                   aperture_gas     = 'exclusive_sphere_50kpc', 
                   aperture_halo    = 'spherical_overdensity_200_crit', 
                   aperture_angle   = 'exclusive_sphere_10kpc', 
                   #-------------------------------------------
                   # BH settings
                     use_closest_to_COP   = False,                   # Uses closest to COP (new method) or most massive in 1 hmr (_old)
                   #-------------------------------------------
                   # Masses   [ Msun ]
                     plot_m200c           = True,
                       plot_stelmass      = True,
                       plot_gasmass       = True,
                       plot_sfmass        = False,
                       plot_nsfmass       = False,
                       plot_colddense     = False,       # t < 10**4.5, n > 0.1 cm**-3
                       plot_HImass        = True,
                       plot_H2mass        = True,
                   # Angles   [ deg ]
                     plot_angles          = False,
                   # Inflow (gas) [ Msun/yr ]
                     plot_sfr             = True,
                   # sSFR     [ /yr ]
                     plot_ssfr            = True,
                   # Radius   [ pkpc]
                     plot_radius_stars    = True,
                     plot_radius_gas      = False,
                   # Morphology stars/gas [ ]
                     plot_kappa_stars     = True,
                     plot_kappa_gas       = False,
                     plot_ellip           = True,
                     plot_triax           = True,
                   # Velocity dispersion
                     plot_veldisp         = True,
                   # Metallicity  [ Z ]
                     #plot_Z_stars         = False,
                     #plot_Z_gas           = False,
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
    soap_dir       = colibre_base_path + simulation_dir + "/SOAP-HBT/"
    
    #------------------
    # Dictionary containing all properties wanted
    dict_plot = {
        "soap_indicies": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
          "snapshot": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
          "redshift": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
          "lookbacktime": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Gyr')
        "plot_m200c": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_stelmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_gasmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_sfmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_nsfmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_cdmass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),      # colddense
          "plot_HImass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
          "plot_H2mass": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun'),
        "plot_angles": np.zeros(snapshot_list.shape[0]) * unyt.Unit('deg'),
        "plot_sfr": np.zeros(snapshot_list.shape[0]) * unyt.Unit('Msun/yr'),
          "plot_ssfr": np.zeros(snapshot_list.shape[0]) * unyt.Unit('/yr'),
        "plot_radius_stars": np.zeros(snapshot_list.shape[0]) * unyt.Unit('kpc'),
          "plot_radius_gas": np.zeros(snapshot_list.shape[0]) * unyt.Unit('kpc'),
        "plot_kappa_stars": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_kappa_gas": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_ellip": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_triax": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "plot_veldisp": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "plot_Z_stars": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_Z_gas": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "plot_u_r": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
          "plot_g_r": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "merger_ratio_stars": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "merger_ratio_gas": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "merger_ratio_HI": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless'),
        "merger_ratio_H2": np.zeros(snapshot_list.shape[0]) * unyt.Unit('dimensionless')
    }   
    for i, snap_i in enumerate(snapshot_list):
        #print('Loading snapshot', snap_i)
        # Load data
        snap_i = "{:03d}".format(snap_i)
        data = sw.load(f'{soap_dir}halo_properties_0{snap_i}.hdf5')
        run_name = data.metadata.run_name
                
        
        # Identify the location of the halo in this SOAP catalogue
        soap_idx = np.argmax(data.input_halos_hbtplus.track_id[:].value == track_id)
        lookbacktime = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0 = 0.0486).age(data.metadata.redshift)).value
        
        #------------------
        # Extract desired SOAP values
        dict_plot['soap_indicies'][i] = soap_idx
        dict_plot['snapshot'][i]      = snap_i
        dict_plot['redshift'][i]      = data.metadata.redshift
        dict_plot['lookbacktime'][i]  = lookbacktime
        
        print('redshift: ', data.metadata.redshift)
        print('lookbacktime: ', lookbacktime, '\n')
        
        dict_plot['plot_m200c'][i] = ((attrgetter('%s.%s'%(aperture_halo, 'total_mass'))(data))[soap_idx]).to(u.Msun)
        dict_plot['plot_stelmass'][i] = ((attrgetter('%s.%s'%(aperture_general, 'stellar_mass'))(data))[soap_idx]).to(u.Msun)
        dict_plot['plot_gasmass'][i]  = ((attrgetter('%s.%s'%(aperture_gas, 'gas_mass'))(data))[soap_idx]).to(u.Msun)
        dict_plot['plot_sfmass'][i]   = ((attrgetter('%s.%s'%(aperture_gas, 'star_forming_gas_mass'))(data))[soap_idx]).to(u.Msun)
        dict_plot['plot_nsfmass'][i]  = ((attrgetter('%s.%s'%(aperture_gas, 'gas_mass'))(data))[soap_idx] - (attrgetter('%s.%s'%(aperture_general, 'star_forming_gas_mass'))(data))[soap_idx]).to(u.Msun)
        dict_plot['plot_colddense'][i]   = ((attrgetter('%s.%s'%(aperture_gas, 'gas_mass_in_cold_dense_gas'))(data))[soap_idx]).to(u.Msun)
        dict_plot['plot_HImass'][i]   = ((attrgetter('%s.%s'%(aperture_gas, 'atomic_hydrogen_mass'))(data))[soap_idx]).to(u.Msun)
        dict_plot['plot_H2mass'][i]   = ((attrgetter('%s.%s'%(aperture_gas, 'molecular_hydrogen_mass'))(data))[soap_idx]).to(u.Msun)
        
        def _misalignment_angle(L1, L2):
            # Find the misalignment angle
            angle = np.rad2deg(np.arccos(np.clip(np.dot(L1/np.linalg.norm(L1), L2/np.linalg.norm(L2)), -1.0, 1.0)))     # [deg]
            return angle
        dict_plot['plot_angle'][i] = _misalignment_angle((attrgetter('%s.%s'%(aperture_angle, 'angular_momentum_gas'))(data))[soap_idx], (attrgetter('%s.%s'%(aperture_angle, 'angular_momentum_stars'))(data))[soap_idx])
        
        dict_plot['plot_sfr'][i] = ((attrgetter('%s.%s'%(aperture_general, 'star_formation_rate'))(data))[soap_idx]).to(u.Msun/u.yr)
        dict_plot['plot_ssfr'][i] = ((attrgetter('%s.%s'%(aperture_general, 'star_formation_rate'))(data))[soap_idx]).to(u.Msun/u.yr) / ((attrgetter('%s.%s'%(aperture_general, 'stellar_mass'))(data))[soap_idx]).to(u.Msun)
        
        dict_plot['plot_radius_stars'][i] = (((attrgetter('%s.%s'%(aperture_general, 'half_mass_radius_stars'))(data))[soap_idx]).to(u.kpc)).convert_to_physical()
        dict_plot['plot_radius_gas'][i] = (((attrgetter('%s.%s'%(aperture_general, 'half_mass_radius_gas'))(data))[soap_idx].to(u.kpc)).convert_to_physical()
        
        def _compute_intrinsic_ellipticity_triaxiality():
            # construct inertia tensor
            inertiatensor_raw = attrgetter('%s.%s'%('bound_subhalo', 'stellar_inertia_tensor_noniterative'))(data)[soap_idx]
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
        dict_plot['plot_ellip'][i] = ellip
        dict_plot['plot_triax'][i] = triax
        dict_plot['plot_kappa_stars'][i] = (attrgetter('%s.%s'%(aperture_general, 'kappa_corot_stars'))(data))[soap_idx]
        dict_plot['plot_kappa_gas'][i] = (attrgetter('%s.%s'%(aperture_general, 'kappa_corot_gas'))(data))[soap_idx]
        
        def _compute_velocity_dispersion(aperture_veldisp='50'):        # km/s
            stellar_vel_disp_matrix = (attrgetter('exclusive_sphere_%skpc.stellar_velocity_dispersion_matrix'%(aperture_veldisp))(data))[soap_idx]
            stellar_vel_disp_matrix.convert_to_units(u.km**2 / u.s**2)
            stellar_vel_disp_matrix.convert_to_physical()
            stellar_vel_disp = np.sqrt((stellar_vel_disp_matrix[:,0] + stellar_vel_disp_matrix[:,1] + stellar_vel_disp_matrix[:,2])/3)
        
            return stellar_vel_disp
        stellar_vel_disp50 = _compute_velocity_dispersion(aperture_veldisp='50')
        dict_plot['plot_veldisp'][i] = stellar_vel_disp50
        
        #dict_plot['plot_Z_stars'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        #dict_plot['plot_Z_gas'][i] = (attrgetter('%s.%s'%(aperture_general, 'quantity'))(data))[soap_idx]
        
        u_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture_general, 'stellar_luminosity'))(data))[:,0])[soap_idx]
        g_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture_general, 'stellar_luminosity'))(data))[:,1])[soap_idx]
        r_mag = -2.5*np.log10((attrgetter('%s.%s'%(aperture_general, 'stellar_luminosity'))(data))[:,2])[soap_idx]
        u_mag = cosmo_array(u_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        g_mag = cosmo_array(g_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        r_mag = cosmo_array(r_mag, u.dimensionless, comoving=True, scale_factor=data.metadata.a, scale_exponent=0)
        dict_plot['plot_u_r'][i] = u_mag - r_mag
        dict_plot['plot_g_r'][i] = g_mag - r_mag
        
    #--------------
    # Find locations of mergers, use descendant_track_id to link to track_id
    for i, snap_i in enumerate(snapshot_list):
        # Identify the location of the halo in this SOAP catalogue. Do not include this in descendent_track_id
        soap_idx = np.argmax(data.input_halos_hbtplus.track_id[:].value == track_id)
        
        
        
        
        merger_ratio_stars ... max stelmass in past 2 Gyr
        merger_ratio_gas ... max gasmass in past 2 Gyr
        merger_ratio_HI
        merger_ratio_H2
    
    #------------------------
    # Default graphs: [ angles, massrate, mass, ssfr, radius, kappa, Z, edd, lbol ]
    plot_height_ratios = []
    plot_names         = []
    if plot_m200c or plot_stelmass or plot_gasmass or plot_sfmass or plot_nsfmass or plot_colddense or plot_HImass or plot_H2mass:
        plot_height_ratios.append(3)
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
    if plot_veldisp:
        plot_height_ratios.append(2)
        plot_names.append('velocity')
    if plot_kappa_stars or plot_kappa_gas or plot_ellip or plot_triax:
        plot_height_ratios.append(2)
        plot_names.append('morphology')
    #if plot_Z_stars or plot_Z_gas:
    #    plot_height_ratios.append(2)
    #    plot_names.append('Z')
    if plot_u_r or plot_g_r:
        plot_height_ratios.append(2)
        plot_names.append('mag')
    
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
        
        dict_aperture = {'exclusive_sphere_10kpc': '10 pkpc',
                         'exclusive_sphere_30kpc': '30 pkpc', 
                         'exclusive_sphere_50kpc': '50 pkpc'}
    
        # Plot 0
        if plot_names_i == 'mass':
            # Plot line
            if plot_m200c:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_m200c'], alpha=1.0, lw=0.7, c='p', label='$M_{\mathrm{200c}}$')
            if plot_stelmass:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_stelmass'], alpha=1.0, lw=0.7, c='r', label='$M_{\mathrm{*}}$')
                
            if plot_gasmass:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_gasmass'], alpha=1.0, lw=0.7, c='g', ls='--', label='$M_{\mathrm{gas}}$')
            if plot_sfmass:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_sfmass'], alpha=1.0, lw=0.7, c='orange', ls='--', label='$M_{\mathrm{SF}}$')
            if plot_nsfmass:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_nsfmass'], alpha=1.0, lw=0.7, c='brown', ls='dashdot', label='$M_{\mathrm{NSF}}$')
                
            if plot_colddense:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_colddense'], alpha=1.0, lw=0.7, c='b', ls='--', label='$M_{\mathrm{cold,dense}}$')
                
            if plot_HImass:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_HImass'], alpha=1.0, lw=0.7, c='b', ls='dashdot', label='$M_{\mathrm{HI}}$')
            if plot_H2mass:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_H2mass'], alpha=1.0, lw=0.7, c='c', ls='--', label='$M_{\mathrm{H_{\mathrm{2}}}}$')
            
            
            #---------------------
            ### Formatting
            if redshift_axis:
                ax_top.set_xlabel('Redshift')
                ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
            if not redshift_axis:
                ax_top.set_xticklabels([])
                
            #axs[i].set_ylabel('$\mathrm{log_{10}}(\mathrm{M}/\mathrm{M}_{\odot})$')
            axs[i].set_ylabel('$\mathrm{log_{10}}M$ $[\mathrm{M}_{\odot}]$')
            axs[i].set_ylim(bottom=8.5, top=11.5)
            #axs[i].set_ylim(bottom=5.5, top=11.5)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            
            
            #---------------------
            ### Annotation
            axs[i].set_title('TrackID $=%s$)' %(str(track_id)), size=7, loc='left', pad=3)
            
            
            #---------------------
            ### Legend
            axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers
        
        # Plot 1
        if plot_names_i == 'angles':
            
            # Plot line
            axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_angles'], alpha=1.0, lw=0.7, c='k', label='stars-gas')
            
            
            #---------------------
            ### Formatting
            axs[i].set_ylim(0, 180)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major')
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            axs[i].set_yticks(np.arange(0, 181, 30))
            axs[i].set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
                        
            #---------------------
            ### Legend
            axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers
        
        # Plot 2
        if plot_names_i == 'massrate':
            # Plot line
            axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_sfr'], alpha=1.0, lw=0.7, c='r', label='SFR')
            
            
            #---------------------
            ### Formatting
            axs[i].set_ylabel('$\dot{M}$ $[\mathrm{M}_{\odot} \mathrm{yr}^{-1}]$')
            #axs[i].set_ylim(0, 30)
            axs[i].set_ylim(bottom=0)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                        
            #---------------------
            ### Legend
            axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers
        
        # Plot 3
        if plot_names_i == 'ssfr':
            # Plot line
            axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_ssfr'], alpha=1.0, lw=0.7, c='r')
            
            
            #---------------------
            ### Formatting
            axs[i].set_ylabel('$\mathrm{sSFR}$ $[\mathrm{yr}^{-1}]$')
            axs[i].set_ylim(-12, -7)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            
                        
            #---------------------
            ### Legend
            #axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers
        
        # Plot 4
        if plot_names_i == 'radius':
            # Plot line
            if plot_radius_stars:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_radius_stars'], alpha=1.0, lw=0.7, c='r', label='$r_{\mathrm{50}}$')
            if plot_radius_gas:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_radius_stars'], alpha=1.0, lw=0.7, c='g', ls='--', label='$r_{\mathrm{50,gas}}$')
            
            
            #---------------------
            ### Formatting
            axs[i].set_ylabel('$\mathrm{sSFR}$ $[\mathrm{yr}^{-1}]$')
            axs[i].set_ylim(-12, -7)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            
                        
            #---------------------
            ### Formatting
            axs[i].set_ylabel('Radius [pkpc]')
            axs[i].set_ylim(bottom=0)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            
            #---------------------
            ### Legend
            axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers
        
        # Plot 4.5
        if plot_names_i == 'morphology':
            # Plot line
            if plot_kappa_stars:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_kappa_stars'], alpha=1.0, lw=0.7, c='r', label='$\kappa_{\mathrm{co}}^{\mathrm{*}}$')
            if plot_kappa_gas:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_kappa_gas'], alpha=1.0, lw=0.7, c='g', ls='dashdot', label='$\kappa_{\mathrm{co}}^{\mathrm{gas}}$')
            if plot_ellip:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_ellip'], alpha=1.0, lw=0.7, c='g', ls='--', label='$\epsilon$')
            if plot_triax:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_triax'], alpha=1.0, lw=0.7, c='g', ls=':', label='$T$')
            
            
            #---------------------
            ### Formatting
            axs[i].set_ylabel('Morphology')
            axs[i].set_ylim(0, 1)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            
            
            #---------------------
            ### Legend
            axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers
        
        # Plot 5
        if plot_names_i == 'velocity':
            # Plot line
            if plot_veldisp:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_veldisp'], alpha=1.0, lw=0.7, c='r')
            
            #---------------------
            ### Formatting
            axs[i].set_ylabel('$\sigma_{*}$ [km s$^{-1}$]')
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            
            #---------------------
            ### Legend
            #axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers
        
        # Plot 6
        if plot_names_i == 'mag':
            # Plot line
            if plot_u_r:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_u_r'], alpha=1.0, lw=0.7, c='r', label='$u^{*} - r^{*}$')
            if plot_g_r:
                axs[i].plot(dict_plot['Lookbacktime'], dict_plot['plot_g_r'], alpha=1.0, lw=0.7, c='g', ls='--', label='$g^{*} - r^{*}$')
            
            #---------------------
            ### Formatting
            axs[i].set_ylabel('magnitude')
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
            ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
            
            #---------------------
            ### Legend
            axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
            
            #---------------------
            ### Plot mergers

            
        #---------------------
        axs[i].minorticks_on()
        if redshift_axis:
            axs[i].tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='major')
            axs[i].tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='minor')
        if not redshift_axis:
            axs[i].tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            axs[i].tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        if i == len(plot_names)-1:
            axs[i].set_xlabel('Lookback time [Gyr]')
            
        
    #------------------------
    # Other
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    
    #=====================================
    ### Print summary
    metadata_plot = {'Title': 'TrackID: %s'%track_id}
    
    #--------------
    ### Savefig
    if savefig:
        savefig_txt_save = ('' if savefig_txt_in == None else savefig_txt_in) + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/evolutions/%s/%s_evolution_TrackID_%i_%s.%s" %(fig_dir, save_folder_visual, run_name, track_id, savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/evolutions/%s/%s_evolution_TrackID_%i_%s.%s" %(fig_dir, save_folder_visual, run_name, track_id, savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()



#========================================================================
# Manual sample or load input:

"""trackid_list_sample = [12345, 123456, 1234567]
sample_input = {'name_of_preset': 'test_galaxies',
                'virtual_snapshot_file': '%s'%('/home/cosmos/c22048063/COLIBRE/Runs/L100_m6/THERMAL_AGN_m6/SOAP-HBT/colibre_with_SOAP_membership_0127.hdf5' if answer == '2' else '/cosma8/data/dp004/colibre/Runs/L100_m6/THERMAL_AGN_m6/SOAP-HBT/colibre_with_SOAP_membership_0127.hdf5'),
                'soap_catalogue_file':   '%s'%('/home/cosmos/c22048063/COLIBRE/Runs/L100_m6/THERMAL_AGN_m6/SOAP-HBT/halo_properties_0127.hdf5' if answer == '2' else '/cosma8/data/dp004/colibre/Runs/L100_m6/THERMAL_AGN_m6/SOAP-HBT/halo_properties_0127.hdf5')
            }
savefig_txt_in = ''
save_folder_visual = sample_input['name_of_preset']"""
#------------------------------------------------------------------------
_, trackid_list_sample, sample_input = _load_soap_sample(sample_dir, csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample20_random_h2_ETGs_z0')
                                                                                    # L100_m6_THERMAL_AGN_m6_127_sample20_random_ETGs_z0
                                                                                    # L100_m6_THERMAL_AGN_m6_127_sample20_random_h2_ETGs_z0
savefig_txt_in = ''
save_folder_visual = sample_input['name_of_preset']
#========================================================================




#-----------------------------
# Plot evolution of a given trackid, valid for any snapshot
for TrackID_i in trackid_list_sample:
    _plot_galaxy_evolution(track_id=TrackID_i,
                                savefig = True)
    
    
    raise Exception('current pause in evolution loop')
    
    













