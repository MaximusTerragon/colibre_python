from swiftsimio import load
import unyt
from read_dataset_directories_colibre import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir = _assign_directories(answer)
#====================================


simulation_dir = "L0025N0752/THERMAL_AGN_m5"
snapshot_no    = 123                  # snapshot 45 = 045


# Of course, replace this path with your own snapshot should you be using
# custom data.
data = load(colibre_base_path + simulation_dir + "SOAP/colibre_with_SOAP_membership_0%s.hdf5"%snapshot_no)


"""
data.gas, which contains all information about gas particles in the simulation.
data.dark_matter, likewise containing information about the dark matter particles in the simulation.
data.metadata, an instance of swiftsimio.metadata.objects.SWIFTSnapshotMetadata
data.units, an instance of swiftsimio.metadata.objects.SWIFTUnits
"""


boxsize = data.metadata.boxsize

print(boxsize)

boxsize.convert_to_units("kpc")

print(boxsize)

new_units = unyt.cm * unyt.Mpc / unyt.kpc
new_units.simplify()

boxsize.convert_to_units(new_units)





print('List of gas properties: ')
print(data.metadata.gas_properties.field_names)
print(data.metadata.stars_properties.field_names)





