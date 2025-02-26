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


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir = _assign_directories(answer)
#====================================


#--------------------------------
# Selects a galaxy (or set of galaxies) that meet criteria, returns soap_catalogue_file, virtual_snapshot_file, soap_indicies
def _create_soap_sample(simulation_run = '',
                         simulation_type = '',
                         snapshot_no     = 123,      # snapshot 45 = 045
                         print_sample    = True,
                       #=================================================
                       # Selection criteria
                       min_stelmass     = 10**9.5,
                       max_stelmass     = 10**15,
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
    # Select candidates that meet mass sample
    stelmass30  = swiftdata.exclusive_sphere_30kpc.stellar_mass
    stelmass30.convert_to_units('Msun')       # Specify units
    stelmass30.convert_to_physical()          # Specify physical/comoving
    
    # The warning is complaining that 1e11 * u.Msun doesn't include information about whether and how the quantity depends on the scale factor (while m200c does). The mass doesn't depend on the scale factor (and we're at a=1 anyway) so we can safely ignore it.
    soap_indicies = np.argwhere(np.logical_and(stelmass30 > min_stelmass * u.Msun, stelmass30 < max_stelmass * u.Msun)).squeeze()
    
    
    """
    m200c = swiftdata.spherical_overdensity_200_crit.total_mass
    m200c.convert_to_units('Msun')       # Specify units
    m200c.convert_to_physical()          # Specify physical/comoving
    soap_indicies = np.argwhere(np.logical_and(m200c > 1e11 * u.Msun, m200c < 2e11 * u.Msun)).squeeze()
    """
    
    
    # Print first 10 just to check
    #print(swiftdata.exclusive_sphere_30kpc.stellar_mass[soap_indicies][:10])
    
    if print_sample:
        print('\n=================')
        print('%s\t%s' %(simulation_run, simulation_type))
        print('Snapshot:  %s' %snapshot_no)
        print('Redshift:  %.2f' %swiftdata.metadata.redshift)
        print('Created sample size:   %s' %len(soap_indicies))
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
                    'snapshot_no': snapshot_no,
                    'redshift': swiftdata.metadata.redshift,
                    'sample_input':  sample_input}
                    
        json.dump(csv_dict, open('%s/%s_%s_%s_sample_%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s_%s_%s_sample_%s.csv' %(sample_dir, simulation_run, simulation_type, str(snapshot_no), csv_name))
        

    return soap_indicies, sample_input
    
    
#=======================================
_create_soap_sample(simulation_run = 'L0025N0752', simulation_type = 'THERMAL_AGN_m5', 
                    snapshot_no = 123,
                    csv_file = True,
                    csv_name = 'example_sample')
                    
                    
# simulation_run = 'L100_m6'    THERMAL_AGN_m6  127
# simulation_run = 'L0025N0752' THERMAL_AGN_m5  123
                    
                    
                    
                    
                    
                    
                    
                    
                    