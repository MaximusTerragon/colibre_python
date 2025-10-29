import swiftgalaxy
import swiftsimio as sw
from packaging.version import Version
from swiftgalaxy import SWIFTGalaxy, SOAP, MaskCollection
from swiftgalaxy.iterator import SWIFTGalaxies

assert Version(sw.__version__) >= Version("10.0.0")
assert Version(swiftgalaxy.__version__) >= Version("2.0.0")

# swiftgalaxy version: 2.2.1


def _myvis_bug(sg):
    # Load soap_index and 
    soap_index  = sg.halo_catalogue.soap_index
    track_id    = sg.halo_catalogue.input_halos_hbtplus.track_id.squeeze()
    
    print('\nsoap_index:', soap_index)
    print('track_id:', track_id)


#------------------------------------------------------------------------
# Load a sample from a given snapshot
soap_indicies_sample = [510524, 2718163, 7214567, 5982028, 9120823, 4272959, 6326572, 6042135, 10173515, 10428481, 957890, 3928295, 4609304, 8639667, 1702242, 3124949, 3649099, 7704088, 10173512, 6012492]
virtual_snapshot_file = '/home/cosmos/c22048063/COLIBRE/Runs/L100_m6/THERMAL_AGN_m6/SOAP-HBT/colibre_with_SOAP_membership_0127.hdf5'
soap_catalogue_file = '/home/cosmos/c22048063/COLIBRE/Runs/L100_m6/THERMAL_AGN_m6/SOAP-HBT/halo_properties_0127.hdf5'


sgs = SWIFTGalaxies(virtual_snapshot_file, SOAP(soap_catalogue_file, soap_index=soap_indicies_sample, ), auto_recentre=False,
                    preload={"gas.coordinates", "gas.masses", "gas.velocities", "gas.element_mass_fractions.hydrogen", "gas.species_fractions.HI", "gas.species_fractions.H2", "gas.smoothing_lengths", "stars.coordinates", "stars.masses", "stars.smoothing_lengths", },)       
output_figures = sgs.map(_myvis_bug)