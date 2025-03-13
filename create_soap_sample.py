import os
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import swiftsimio as sw
import scipy
import csv
import json
from tqdm import tqdm
from packaging.version import Version
from swiftsimio import SWIFTDataset
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
                       # Selection criteria
                       min_stelmass     = 10**10,
                       max_stelmass     = 10**15,
                       only_centrals    = False,
                         select_random  = False,        # or number
                         select_first   = False,        # or number, selects first N objects in sample
                       create_example_sample = False,   # uses halo
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
    central_sat = swiftdata.input_halos.is_central
    
    if not create_example_sample:
        # Select candidates that meet mass sample
        stelmass30  = swiftdata.exclusive_sphere_30kpc.stellar_mass
        stelmass30.convert_to_units('Msun')       # Specify units
        stelmass30.convert_to_physical()          # Specify physical/comoving
        
        if only_centrals:
            soap_indicies = np.argwhere(np.logical_and(central_sat == 1, np.logical_and(stelmass30 > min_stelmass * u.Msun, stelmass30 < max_stelmass * u.Msun))).squeeze()
        else:
            soap_indicies = np.argwhere(np.logical_and(stelmass30 > min_stelmass * u.Msun, stelmass30 < max_stelmass * u.Msun)).squeeze()

        # The warning is complaining that 1e11 * u.Msun doesn't include information about whether and how the quantity depends on the scale factor (while m200c does). The mass doesn't depend on the scale factor (and we're at a=1 anyway) so we can safely ignore it.
        
        trackid_list = trackID[soap_indicies]
    else:
        # Example
        m200c = swiftdata.spherical_overdensity_200_crit.total_mass
        m200c.convert_to_units('Msun')       # Specify units
        m200c.convert_to_physical()          # Specify physical/comoving
        soap_indicies = np.argwhere(np.logical_and(m200c > 1e11 * u.Msun, m200c < 2e11 * u.Msun)).squeeze()
        trackid_list = trackID[soap_indicies]
    
    
    
    # Print first 10 just to check
    #print(swiftdata.exclusive_sphere_30kpc.stellar_mass[soap_indicies][:10])
    
    if print_sample:
        print('\n=================')
        print('%s\t%s' %(simulation_run, simulation_type))
        print('Snapshot:  %s' %snapshot_no)
        print('Redshift:  %.2f' %swiftdata.metadata.redshift)
        print('Created sample size:   %s' %len(soap_indicies))
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
    print('')
        
    
   
        
    #---------------------
    # Create input dict
    sample_input = {'simulation_run': simulation_run,
                    'simulation_type': simulation_type,
                    'simulation_dir': simulation_dir,
                    'soap_catalogue_file': soap_catalogue_file,
                    'virtual_snapshot_file': virtual_snapshot_file,
                    'snapshot_no': snapshot_no,
                    'redshift': swiftdata.metadata.redshift}
    
    
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
                    
        json.dump(csv_dict, open('%s/%s_%s_%s_%ssample_%s_%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), ('centrals_' if only_centrals else ''), len(soap_indicies), csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s_%s_%s_%ssample_%s_%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), ('centrals_' if only_centrals else ''), len(soap_indicies), csv_name))
        

    return soap_indicies, sample_input
    
    
#=======================================
# Create samples of massive >1010 galaxies:
"""_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                    only_centrals = False,
                    csv_file = True)
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                    only_centrals = True,
                    csv_file = True)"""

#----------------
# Create sample from example (change to halo limit, not stelmass)
"""_create_soap_sample(simulation_run = 'L0025N0752', simulation_type = 'THERMAL_AGN_m5_obsolete', 
                    snapshot_no = 123,
                      create_example_sample = True,
                      select_first = 5,
                    csv_file = True,
                    csv_name = 'example_sample')"""

#----------------
# Select 20 random central galaxies
_create_soap_sample(simulation_run = 'L100_m6', simulation_type = 'THERMAL_AGN_m6', 
                    snapshot_no = 127,
                      select_random = 20,
                      min_stelmass     = 10**10,
                      only_centrals = True,
                    csv_file = True,
                      csv_name = 'galaxy_visual')
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    