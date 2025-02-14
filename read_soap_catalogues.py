import swiftsimio as sw
import swiftgalaxy
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import csv
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from graphformat import set_rc_params
from read_dataset_directories_colibre import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir = _assign_directories(answer)
#====================================


# Specify simulation and snapshot
simulation_dir = "L0025N0752/THERMAL_AGN_m5"
snapshot_no    = 123                            # snapshot 45 = 045
soap_dir       = colibre_base_path + simulation_dir + "/SOAP/"
#soap_dir = ’/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/’


# Load data
data = sw.load(f'{soap_dir}halo_properties_0{snapshot_no}.hdf5')

# Load a specific dataset
m200 = data.spherical_overdensity_200_crit.total_mass


# Specify units
m200.convert_to_units('Msun')
# Specify physical/comoving
m200.convert_to_physical()

# Get metadata from file
z = data.metadata.redshift

print('Redshift: ', z)
print(m200)