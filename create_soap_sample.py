import os
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import swiftsimio as sw
import scipy
import csv
import json
import matplotlib.pyplot as plt
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
                       # Selection method
                       create_example_sample = False,   # uses halo
                         select_random  = False,        # or number
                         select_first   = False,        # or number, selects first N objects in sample
                       #--------------------------
                       # Selection criteria
                       min_stelmass     = 1e10,
                       max_stelmass     = 1e15,
                       only_centrals    = False,
                       use_kappa        = False,
                         include_colddensefraction = False,     # ctrl+f 'colddensefraction_condition'
                       #=================================================
                       csv_file = False,                       # Will write sample to csv file in sapmle_dir
                          csv_name = '',
                       #----------------------
                       debug = False):
    
    #--------------------
    # Load data
    simulation_dir = simulation_run + '/' + simulation_type
    
    # Load SOAP and SOAP+particle files
    soap_catalogue_file = os.path.join(colibre_base_path, simulation_dir, "SOAP/halo_properties_0%s.hdf5"%snapshot_no,)
    virtual_snapshot_file = os.path.join(colibre_base_path, simulation_dir, "SOAP/colibre_with_SOAP_membership_0%s.hdf5"%snapshot_no)
    
    # We can load the entire SOAP catalogue using swiftsimio to browse for an interesting galaxy to look at. 
    swiftdata = SWIFTDataset(soap_catalogue_file)
    
    
    #==========================================
    # Extact trackids
    trackID = swiftdata.input_halos_hbtplus.track_id
    
    #------
    kappa_co_ETG          = 0.4          # will select less than
    colddense_limit       = 0.15    # will include kappa above but for which colddensefraction is below this
    #------
    
    if create_example_sample:
        # Example
        m200c = swiftdata.spherical_overdensity_200_crit.total_mass
        m200c.convert_to_units('Msun')       # Specify units
        m200c.convert_to_physical()          # Specify physical/comoving
        soap_indicies = np.argwhere(np.logical_and(m200c > cosmo_quantity(1e11, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), 
                                                   m200c < cosmo_quantity(2e11, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0))).squeeze()
    else:
        # Select candidates that meet mass sample
        stelmass50  = swiftdata.exclusive_sphere_50kpc.stellar_mass
        
        central_sat = swiftdata.input_halos.is_central
        
        kappa_co    = swiftdata.exclusive_sphere_50kpc.kappa_corot_stars
        colddense50 = swiftdata.exclusive_sphere_30kpc.gas_mass_in_cold_dense_gas
        colddensefraction = cosmo_array(np.zeros(stelmass50.shape), u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        colddensefraction[stelmass50 > 0.0] = colddense50[stelmass50 > 0.0] / stelmass50[stelmass50 > 0.0]
        
        
        # Centrals (1) or satellites + centrals (1 + 0)
        if only_centrals:
            central_sat_condition = 1 #cosmo_quantity(1, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)   # central
        else:
            central_sat_condition = 0 #cosmo_quantity(0, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)   # satellite + central
        
        # Kappa
        if use_kappa:
            kappa_condition = cosmo_quantity(kappa_co_ETG, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
        else:
            kappa_condition = cosmo_quantity(1, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
            
        soap_indicies = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), central_sat >= central_sat_condition, kappa_co < kappa_condition])).squeeze() 
        
        
        # The warning is complaining that 1e11 * u.Msun doesn't include information about whether and how the quantity depends on the scale factor (while m200c does). The mass doesn't depend on the scale factor (and we're at a=1 anyway) so we can safely ignore it.
        if print_sample:
            print('\n=================')
            print('%s\t%s' %(simulation_run, simulation_type))
            print('Snapshot:  %s' %snapshot_no)
            print('Redshift:  %.2f' %swiftdata.metadata.redshift)
            print('Initial sample size:   %s' %len(soap_indicies))
        
        if include_colddensefraction:
            # Cold dense fraction for above kappa
            colddense_condition = cosmo_quantity(colddense_limit, u.dimensionless, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0)
            soap_indicies_extra = np.argwhere(np.logical_and.reduce([stelmass50 > cosmo_quantity(min_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), stelmass50 < cosmo_quantity(max_stelmass, u.Msun, comoving=True, scale_factor=swiftdata.metadata.a, scale_exponent=0), central_sat >= central_sat_condition, kappa_co > kappa_condition, colddensefraction < colddense_condition])).squeeze() 
            soap_indicies = np.concatenate([soap_indicies, soap_indicies_extra])
            if print_sample:
                print('  >0.4 kappa sample:   %s' %len(soap_indicies_extra))
        
    trackid_list = trackID[soap_indicies]     
        
    # Print first 10 just to check
    #print(swiftdata.exclusive_sphere_50kpc.stellar_mass[soap_indicies][:10])
    
    if print_sample:
        print('  Final sample size:   %s   <--' %len(soap_indicies))
    if select_random:
        soap_indicies = np.random.choice(soap_indicies, select_random, replace=False)
        if print_sample:
            print('  Selected %s random sample: %s' %(select_random, len(soap_indicies)))
        print(soap_indicies)
    elif select_first:
        soap_indicies = soap_indicies[:select_first]
        if print_sample:
            print('  Selected first %s in sample.' %(select_first))
        print(soap_indicies)
        
    
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
                        'min_stelmass': min_stelmass,
                        'max_stelmass': max_stelmass,
                        'only_centrals': only_centrals,
                        'use_kappa': use_kappa,
                          'kappa_co_ETG': kappa_co_ETG,
                        'include_colddensefraction': include_colddensefraction,
                          'colddense_limit': colddense_limit,
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
                    
        json.dump(csv_dict, open('%s/%s_%s_%ssample_%s_%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), len(soap_indicies), csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s_%s_%ssample_%s_%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), len(soap_indicies), csv_name))
        
    return soap_indicies, sample_input
    
    
#=======================================
# Create sample from example (change to halo limit, not stelmass)
"""_create_soap_sample(simulation_run = 'L0025N0752', simulation_type = 'THERMAL_AGN_m5_obsolete', 
                    snapshot_no = 123,
                      create_example_sample = True,
                      select_first = 5,
                    csv_file = True,
                    csv_name = 'example_sample')"""

#----------------
# Select 20 random central galaxies
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                    select_random = 20,
                      min_stelmass     = 10**10,
                      only_centrals    = True,
                    csv_file = True,
                      csv_name = 'galaxy_visual')"""



#=======================================
# Create samples of massive >1010 galaxies:
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      min_stelmass     = 10**10,
                      only_centrals    = False,
                    csv_file = True,
                      csv_name = 'all_galaxies')

# Create samples of massive >1010 galaxies that are ETGs (kappa < 0.4):
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      min_stelmass     = 10**10,
                      only_centrals    = False,
                      use_kappa        = True,      # set to 0.4, using 50 kpc
                    csv_file = True,
                      csv_name = 'all_ETGs')

# Create samples of massive >1010 galaxies that are ETGs (kappa < 0.4), and include disky candidates with kappa > 0.4 and colddensegasfraction < 0.1:
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      min_stelmass     = 10**10,
                      only_centrals    = False,
                      use_kappa        = True,      # set to 0.4, using 50 kpc
                      include_colddensefraction = True,         # includes galaxies above kappa 0.4 if they have low cold gas fraction
                    csv_file = True,
                      csv_name = 'all_ETGs_plus_lowgasfrac')
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    