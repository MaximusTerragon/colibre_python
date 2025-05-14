import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import csv
import json


#-------------------------------
# Loads a SOAP sample from csv file
def _load_soap_sample(sample_dir, csv_sample, print_sample=True):
    # Loading sample
    dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    soap_indicies = np.array(dict_new['soap_indicies'])
    trackid_list  = np.array(dict_new['trackid_list'])
    sample_input  = dict_new['sample_input']
    dict_new = 0
    
    if print_sample:
        print('\n===Loaded Sample===')
        print('%s\t%s' %(sample_input['simulation_run'], sample_input['simulation_type']))
        print('Snapshot:  %s' %sample_input['snapshot_no'])
        print('Redshift:  %.2f' %sample_input['redshift'])
        print('Loaded sample name:   %s' %sample_input['name_of_preset'])
        print('Loaded sample size:   %s' %len(soap_indicies))
        print('')
    
    return soap_indicies, trackid_list, sample_input