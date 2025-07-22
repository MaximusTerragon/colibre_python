import os
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import swiftsimio as sw
import scipy
import csv
import json
import matplotlib.pyplot as plt
from operator import attrgetter
from tqdm import tqdm
from packaging.version import Version
from swiftsimio import SWIFTDataset, cosmo_quantity, cosmo_array
from read_dataset_directories_colibre import _assign_directories
assert Version(sw.__version__) >= Version("9.0.2")

np.random.seed(0)

#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================


#--------------------------------
# Selects a galaxy (or set of galaxies) that meet criteria, returns soap_catalogue_file, virtual_snapshot_file, soap_indicies
def _create_soap_sample(simulation_run = '',
                         simulation_type = '',
                         snapshot_no     = 123,      # snapshot 45 = 045
                         print_sample    = True,
                       #=================================================
                       # Name of sample --> uses presets see code
                       name_of_preset = 'example_sample',   # example_sample
                                                            # galaxy_visual_test, gas_rich_ETGs_z0p1, gas_rich_ETGs_z0 
                                                            # all_galaxies, 
                                                            # all_ETGs, all_ETGs_plus_redspiral
                                                            # all_LTGs, all_LTGs_excl_redspiral
                                                            # above plus _centrals / _satellites
                                                            # test_galaxies
                       #=================================================
                       csv_file = False,                       # Will write sample to csv file in sapmle_dir
                          csv_name = '',
                       #----------------------
                       debug = False):
    
    #--------------------
    # Load data
    simulation_dir = simulation_run + '/' + simulation_type
    soap_dir       = colibre_base_path + simulation_dir + "/SOAP/"
    
    # Load SOAP and SOAP+particle files
    soap_catalogue_file = os.path.join(colibre_base_path, simulation_dir, "SOAP/halo_properties_0%s.hdf5"%snapshot_no,)
    virtual_snapshot_file = os.path.join(colibre_base_path, simulation_dir, "SOAP/colibre_with_SOAP_membership_0%s.hdf5"%snapshot_no)
    
    # We can load the entire SOAP catalogue using swiftsimio to browse for an interesting galaxy to look at. 
    swiftdata = SWIFTDataset(soap_catalogue_file)
    
    if print_sample:
        print('\n=================')
        print('%s\t%s' %(simulation_run, simulation_type))
        print('Snapshot:  %s' %snapshot_no)
        print('Redshift:  %.2f' %swiftdata.metadata.redshift)
        print('Sample preset:    ->  %s' %name_of_preset)
    
    
    #==========================================
    # Extact trackids
    trackID = swiftdata.input_halos_hbtplus.track_id
    
    #==========================================
    # List presets, will kick out soap_indicies matching criteria
    
    if name_of_preset == 'example_sample':
        # Used parameters
        min_halomass = 1e11
        max_halomass = 2e11                              
        select_first = 5
        selection_criteria = {'min_halomass': min_stelmass, 'max_halomass': max_stelmass, 'select_first': select_first}
        
        # Example
        m200c = swiftdata.spherical_overdensity_200_crit.total_mass
        m200c.convert_to_units('Msun')       # Specify units
        m200c.convert_to_physical()          # Specify physical/comoving
        soap_indicies = np.argwhere(np.logical_and(m200c > cosmo_quantity(min_halomass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                   m200c < cosmo_quantity(max_halomass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0))).squeeze()
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set  
        soap_indicies = soap_indicies[:select_first]
        if print_sample:
            print('  Selected first %s in sample.' %(select_first))
            mask_sort = np.argsort(trackID[soap_indicies])
            print('soap_indicies:\n', soap_indicies[mask_sort])
            print('trackID:\n', ((trackID[soap_indicies])[mask_sort]).to_value())
    elif name_of_preset == 'galaxy_visual_test':     # uses 30 kpc
        # Used parameters
        min_stelmass     = 1e10
        max_stelmass     = 1e15
        only_centrals    = True
        select_random = 20
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'only_centrals': only_centrals, 'select_random': select_random}
        
        # Create additional criteria
        if only_centrals:
            central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        else:
            central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # satellite + central
        
        # Select candidates that meet mass sample
        stelmass30  = swiftdata.exclusive_sphere_30kpc.stellar_mass
        stelmass30.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass30 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass30 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat >= central_sat_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
        soap_indicies = np.random.choice(soap_indicies, select_random, replace=False)
        if print_sample:
            print('  Selected %s random sample: %s' %(select_random, len(soap_indicies)))
            mask_sort = np.argsort(trackID[soap_indicies])
            print('soap_indicies:\n', soap_indicies[mask_sort])
            print('trackID:\n', ((trackID[soap_indicies])[mask_sort]).to_value())
    elif name_of_preset == 'gas_rich_ETGs_z0p1':
        # Used parameters
        min_stelmass     = 1e10
        max_stelmass     = 1e15
        only_centrals    = False
        kappa_co_ETG     = 0.4          # will select less than
        min_h2mass       = 1e9
        select_random    = 40
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'only_centrals': only_centrals, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria
        if only_centrals:
            central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        else:
            central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # satellite + central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        h2mass50  = swiftdata.exclusive_sphere_50kpc.molecular_hydrogen_mass
        h2mass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           h2mass50 > cosmo_quantity(min_h2mass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat >= central_sat_condition, 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
        soap_indicies = np.random.choice(soap_indicies, select_random, replace=False)
        if print_sample:
            print('  Selected %s random sample: %s' %(select_random, len(soap_indicies)))
            mask_sort = np.argsort(trackID[soap_indicies])
            print('soap_indicies:\n', soap_indicies[mask_sort])
            print('trackID:\n', ((trackID[soap_indicies])[mask_sort]).to_value())
    elif name_of_preset == 'gas_rich_ETGs_z0':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        only_centrals    = False
        kappa_co_ETG     = 0.4          # will select less than
        min_h2mass       = 1e9
        select_random    = 25
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'only_centrals': only_centrals, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria
        if only_centrals:
            central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        else:
            central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # satellite + central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        h2mass50  = swiftdata.exclusive_sphere_50kpc.molecular_hydrogen_mass
        h2mass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           h2mass50 > cosmo_quantity(min_h2mass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat >= central_sat_condition, 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
        soap_indicies = np.random.choice(soap_indicies, select_random, replace=False)
        if print_sample:
            print('  Selected %s random sample: %s' %(select_random, len(soap_indicies)))
            mask_sort = np.argsort(trackID[soap_indicies])
            print('soap_indicies:\n', soap_indicies[mask_sort])
            print('trackID:\n', ((trackID[soap_indicies])[mask_sort]).to_value())
    elif name_of_preset == 'all_galaxies':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass}
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
    elif name_of_preset == 'all_galaxies_centrals':
        # Used parameters
        min_stelmass      = 10**(9.5)
        max_stelmass      = 1e15
        central_satellite = 'centrals'
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
    elif name_of_preset == 'all_galaxies_satellites':
        # Used parameters
        min_stelmass      = 10**(9.5)
        max_stelmass      = 1e15
        central_satellite = 'satellites'
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
    elif name_of_preset == 'all_ETGs':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        kappa_co_ETG     = 0.4          # will select less than
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
    elif name_of_preset == 'all_ETGs_centrals':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'centrals'
        kappa_co_ETG     = 0.4          # will select less than
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition, 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
    elif name_of_preset == 'all_ETGs_satellites':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'satellites'
        kappa_co_ETG     = 0.4          # will select less than
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition, 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
    elif name_of_preset == 'all_ETGs_plus_redspiral':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        kappa_co_ETG     = 0.4
        u_r_min          = 2    # will include kappa above but for which u-r is above this
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'kappa_co_ETG': kappa_co_ETG, 'u_r_min': u_r_min}
        
        # Create additional criteria
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        u_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,0])
        r_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,2])
        u_r_mag = cosmo_array(np.zeros(stelmass50.shape), u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        u_r_mag[stelmass50 > 0.0] = u_mag50[stelmass50 > 0.0] - r_mag50[stelmass50 > 0.0]
        
        # Select regular sample as before
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select additional: red FRs for above kappa
        u_r_condition = cosmo_quantity(u_r_min, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        soap_indicies_extra = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 kappa_co > kappa_condition, 
                                                                 u_r_mag > u_r_condition])).squeeze() 
        soap_indicies = np.concatenate([soap_indicies, soap_indicies_extra])
        if print_sample:
            print('  >0.4 kappa sample:   %s' %len(soap_indicies_extra))
        # Select sub-set
    elif name_of_preset == 'all_ETGs_plus_redspiral_centrals':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'centrals'
        kappa_co_ETG     = 0.4
        u_r_min          = 2    # will include kappa above but for which u-r is above this
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG, 'u_r_min': u_r_min}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        u_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,0])
        r_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,2])
        u_r_mag = cosmo_array(np.zeros(stelmass50.shape), u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        u_r_mag[stelmass50 > 0.0] = u_mag50[stelmass50 > 0.0] - r_mag50[stelmass50 > 0.0]
        
        # Select regular sample as before
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition, 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select additional: red FRs for above kappa
        u_r_condition = cosmo_quantity(u_r_min, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        soap_indicies_extra = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 central_sat == central_sat_condition, 
                                                                 kappa_co > kappa_condition, 
                                                                 u_r_mag > u_r_condition])).squeeze() 
        soap_indicies = np.concatenate([soap_indicies, soap_indicies_extra])
        if print_sample:
            print('  >0.4 kappa sample:   %s' %len(soap_indicies_extra))
        # Select sub-set
    elif name_of_preset == 'all_ETGs_plus_redspiral_satellites':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'satellites'
        kappa_co_ETG     = 0.4
        u_r_min          = 2    # will include kappa above but for which u-r is above this
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG, 'u_r_min': u_r_min}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        u_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,0])
        r_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,2])
        u_r_mag = cosmo_array(np.zeros(stelmass50.shape), u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        u_r_mag[stelmass50 > 0.0] = u_mag50[stelmass50 > 0.0] - r_mag50[stelmass50 > 0.0]
        
        # Select regular sample as before
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition, 
                                                           kappa_co < kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select additional: red FRs for above kappa
        u_r_condition = cosmo_quantity(u_r_min, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        soap_indicies_extra = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 central_sat == central_sat_condition, 
                                                                 kappa_co > kappa_condition, 
                                                                 u_r_mag > u_r_condition])).squeeze() 
        soap_indicies = np.concatenate([soap_indicies, soap_indicies_extra])
        if print_sample:
            print('  >0.4 kappa sample:   %s' %len(soap_indicies_extra))
        # Select sub-set
    elif name_of_preset == 'all_LTGs':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        kappa_co_ETG     = 0.4          # will select less than
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           kappa_co > kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
    elif name_of_preset == 'all_LTGs_centrals':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'centrals'
        kappa_co_ETG     = 0.4          # will select less than
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition, 
                                                           kappa_co > kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
    elif name_of_preset == 'all_LTGs_satellites':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'satellites'
        kappa_co_ETG     = 0.4          # will select less than
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                           central_sat == central_sat_condition, 
                                                           kappa_co > kappa_condition])).squeeze() 
        if print_sample:
            print('Initial sample size:   %s' %len(soap_indicies))
        
        # Select sub-set
    elif name_of_preset == 'all_LTGs_excl_redspiral':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        kappa_co_ETG     = 0.4
        u_r_min          = 2    # will include kappa above but for which u-r is above this
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'kappa_co_ETG': kappa_co_ETG, 'u_r_min': u_r_min}
        
        # Create additional criteria
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        u_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,0])
        r_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,2])
        u_r_mag = cosmo_array(np.zeros(stelmass50.shape), u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        u_r_mag[stelmass50 > 0.0] = u_mag50[stelmass50 > 0.0] - r_mag50[stelmass50 > 0.0]
        
        # Select LTGs which are blue
        u_r_condition = cosmo_quantity(u_r_min, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 kappa_co > kappa_condition, 
                                                                 u_r_mag < u_r_condition])).squeeze()
        if print_sample:
            print('Initial sample size (without red spiral):   %s' %len(soap_indicies))
    elif name_of_preset == 'all_LTGs_excl_redspiral_centrals':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'centrals'
        kappa_co_ETG     = 0.4
        u_r_min          = 2    # will include kappa above but for which u-r is above this
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG, 'u_r_min': u_r_min}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        u_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,0])
        r_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,2])
        u_r_mag = cosmo_array(np.zeros(stelmass50.shape), u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        u_r_mag[stelmass50 > 0.0] = u_mag50[stelmass50 > 0.0] - r_mag50[stelmass50 > 0.0]
        
        # Select LTGs which are blue
        u_r_condition = cosmo_quantity(u_r_min, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 central_sat == central_sat_condition, 
                                                                 kappa_co > kappa_condition, 
                                                                 u_r_mag < u_r_condition])).squeeze()
        if print_sample:
            print('Initial sample size (without red spiral):   %s' %len(soap_indicies))
    elif name_of_preset == 'all_LTGs_excl_redspiral_satellites':
        # Used parameters
        min_stelmass     = 10**(9.5)
        max_stelmass     = 1e15
        central_satellite = 'satellites'
        kappa_co_ETG     = 0.4
        u_r_min          = 2    # will include kappa above but for which u-r is above this
        selection_criteria = {'min_stelmass': min_stelmass, 'max_stelmass': max_stelmass, 'central_satellite': central_satellite, 'kappa_co_ETG': kappa_co_ETG, 'u_r_min': u_r_min}
        
        # Create additional criteria. 1 = is central, 0 = is satellite
        central_sat_condition = cosmo_quantity(0, u.dimensionless, comoving=False, scale_factor=swiftdata.metadata.a, scale_exponent=0)    # central
        kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        stelmass50.convert_to_units('Msun')
        central_sat = swiftdata.input_halos.is_central
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        u_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,0])
        r_mag50 = -2.5*np.log10((attrgetter('exclusive_sphere_50kpc.stellar_luminosity')(sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')))[:,2])
        u_r_mag = cosmo_array(np.zeros(stelmass50.shape), u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        u_r_mag[stelmass50 > 0.0] = u_mag50[stelmass50 > 0.0] - r_mag50[stelmass50 > 0.0]
        
        # Select LTGs which are blue
        u_r_condition = cosmo_quantity(u_r_min, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                                 central_sat == central_sat_condition, 
                                                                 kappa_co > kappa_condition, 
                                                                 u_r_mag < u_r_condition])).squeeze()
        if print_sample:
            print('Initial sample size (without red spiral):   %s' %len(soap_indicies))
        # Select sub-set
    elif name_of_preset == 'test_galaxies':
        soap_indicies = np.array([482830, 7088094])
        selection_criteria = {}
    else:
        raise Exception('name of preset not recognised')
    
    trackid_list = trackID[soap_indicies]
    if print_sample:
        print('  Final sample size:   %s   <--' %len(soap_indicies))
    
    
    
    
    # He fraction plot
    """Hmass50  = swiftdata.exclusive_sphere_50kpc.hydrogen_mass.to(u.Msun)
    Hemass50  = swiftdata.exclusive_sphere_50kpc.helium_mass.to(u.Msun)
    He_fraction = u.unyt_array(np.zeros(Hmass50.shape), units=u.dimensionless)
    He_fraction[Hmass50 > 0] = Hemass50[Hmass50 > 0] / Hmass50[Hmass50 > 0]
    
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.8], sharex=True, sharey=False)
    
    axs.scatter(np.log10(stelmass50[soap_indicies]), He_fraction[soap_indicies], s=0.5, edgecolor='none', alpha=0.3)
    axs.set_xlabel('Mstar (50 kpc)')
    axs.set_ylabel('He/H fraction')
    
    plt.savefig("%s/test_figs/He_mass_fraction.png" %(fig_dir), format='png', bbox_inches='tight', dpi=600)    
    print("\n  SAVED: %s/test_figs/He_mass_fraction.png" %(fig_dir))
    """
        
        
    #---------------------
    # Create input dict
    sample_input = {'simulation_run': simulation_run,
                    'simulation_type': simulation_type,
                    'simulation_dir': simulation_dir,
                    'soap_catalogue_file': soap_catalogue_file,
                    'virtual_snapshot_file': virtual_snapshot_file,
                    'snapshot_no': snapshot_no,
                    'redshift': swiftdata.metadata.redshift,
                    'selection_criteria': selection_criteria,
                    'name_of_preset': name_of_preset,
                    'csv_name': csv_name}
    
    
    #=====================================
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
        csv_dict = {'soap_indicies': soap_indicies,     # indicies for a given snapshot
                    'trackid_list': trackid_list,       # to identify same galaxy 
                    'snapshot_no': snapshot_no,
                    'redshift': swiftdata.metadata.redshift,
                    'sample_input':  sample_input}
                    
        json.dump(csv_dict, open('%s/%s_%s_%s_sample%s_%s%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), len(soap_indicies), name_of_preset, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s_%s_%s_sample%s_%s%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), len(soap_indicies), name_of_preset, csv_name))
        
    return soap_indicies, sample_input
    
    
#=======================================
# Create sample from example (change to halo limit, not stelmass)
"""_create_soap_sample(simulation_run = 'L0025N0752', simulation_type = 'THERMAL_AGN_m5_obsolete', 
                    snapshot_no = 123,
                      name_of_preset = 'L025_example_sample',
                    csv_file = True)"""

#----------------
# Select 20 random central galaxies
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'galaxy_visual_test',
                    csv_file = True)"""
# Select 40 random gas-rich ETGs (h2 > 1e9), both central or satellite, within our <0.4 kappa sample
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 119,
                      name_of_preset = 'gas_rich_ETGs_z0p1',
                    csv_file = True)"""
# Select 25 random gas-rich ETGs (h2 > 1e9), both central or satellite, within our <0.4 kappa sample
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'gas_rich_ETGs_z0',
                    csv_file = True)"""
# Select a specific few galaxies
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'test_galaxies',
                    csv_file = True)"""


#=======================================
# Create samples of massive >109.5 galaxies:
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_galaxies',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_galaxies_centrals',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_galaxies_satellites',
                    csv_file = True)"""

#=====================
# Create samples of massive >109.5 galaxies that are ETGs (kappa < 0.4):
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_ETGs',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_ETGs_centrals',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_ETGs_satellites',
                    csv_file = True)

# Create samples of massive >109.5 galaxies that are ETGs (kappa < 0.4), and include disky candidates with kappa > 0.4 and u-r > 2.0:
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_ETGs_plus_redspiral',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_ETGs_plus_redspiral_centrals',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_ETGs_plus_redspiral_satellites',
                    csv_file = True)"""
                    
                    
#=====================
# Create samples of massive >109.5 galaxies that are LTGs (kappa > 0.4):
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_LTGs',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_LTGs_centrals',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_LTGs_satellites',
                    csv_file = True)

# Create samples of massive >109.5 galaxies that are LTGs (kappa > 0.4), and exclude disky candidates with kappa > 0.4 and u-r > 2.0:
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_LTGs_excl_redspiral',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_LTGs_excl_redspiral_centrals',
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      name_of_preset = 'all_LTGs_excl_redspiral_satellites',
                    csv_file = True)             
                    
                    
                    
                    
                    
                    
                    