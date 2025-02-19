import os
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import swiftgalaxy
import swiftsimio as sw
import scipy
import csv
import time
from tqdm import tqdm
from packaging.version import Version
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP
from swiftsimio.visualisation.projection import project_gas, project_pixel_grid
from swiftsimio.visualisation import (generate_smoothing_lengths,)  # if not found try `from sw.visualisation import generate_smoothing_lengths`
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from graphformat import set_rc_params
from read_dataset_directories_colibre import _assign_directories
assert Version(sw.__version__) >= Version("9.0.2")
assert Version(swiftgalaxy.__version__) >= Version("1.2.0")


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir = _assign_directories(answer)
#====================================




# Initial file paths and snapnum
#colibre_base_path = "/cosma8/data/dp004/colibre/Runs/"
simulation_dir = "L0025N0752/THERMAL_AGN_m5"
snapshot_no    = 123                            # snapshot 45 = 045


# SOAP records which particles belong to each individual halo in a set of "membership" files, usually found alongside the halo catalogue in a subdirectory, in this case membership_0123/membership_0123.X.hdf5 (where X is replaced by integers, each corresponding to one part of the "raw" snapshot). swiftgalaxy expects to find the information contained in these files directly in the (single, monolithic) simulation snapshot file.
soap_catalogue_file = os.path.join(colibre_base_path, simulation_dir, "SOAP/halo_properties_0%s.hdf5"%snapshot_no,)
virtual_snapshot_file = os.path.join(colibre_base_path, simulation_dir, "SOAP/colibre_with_SOAP_membership_0%s.hdf5"%snapshot_no)


#====================================
# We can load the entire SOAP catalogue using swiftsimio to browse for an interesting galaxy to look at. To avoid having too many particles let's pick something with an m200c of about 1e11 Msun. We'll just grab the first eligible galaxy in the list:
sd = SWIFTDataset(soap_catalogue_file)
m200c = sd.spherical_overdensity_200_crit.total_mass

candidates = np.argwhere(np.logical_and(m200c > 1e11 * u.Msun, m200c < 2e11 * u.Msun)).squeeze()    # The warning is complaining that 1e11 * u.Msun doesn't include information about whether and how the quantity depends on the scale factor (while m200c does). The mass doesn't depend on the scale factor (and we're at a=1 anyway) so we can safely ignore it.
chosen_halo_index = candidates[0]
print('Chosen halo index: ', chosen_halo_index)


#====================================
# Now we can create a SWIFTGalaxy to experiment with. The halo_index selects one row from the halo catalogue. The SWIFTGalaxy object will contain all of the integrated properties of this object from the halo catalogue and also let us access its particles from the snapshot file.
galaxy_data = SWIFTGalaxy(virtual_snapshot_file, SOAP(soap_catalogue_file, soap_index=chosen_halo_index,),)
print('SOAP index: ', galaxy_data.halo_catalogue.soap_index)


# SOAP catalogue data of the galaxy can be accessed through
#Lstars = galaxy_data.halo_catalogue.exclusive_sphere_10kpc.angular_momentum_stars.squeeze()


# We can define a function that uses some of swiftsimio's visualisation tools to make some quick images of this galaxy:
def myvis(galaxy_data = None, plot_annotate = None, savefig_txt_in = None,       # SWIFTGalaxy object containing all the integrated properties + access the particles from the snapshot file
            #=====================================
            # fig settings
              disc_radius=15 * u.kpc,           # gas/stars disc radius 
              halo_radius=200 * u.kpc,          # DM halo disc radius
              fignum=1,
            #=====================================
            showfig       = False,
            savefig       = False,
              file_format = 'pdf',
              savefig_txt = '', 
            #--------------------------
            print_progress = False,
            debug = False):
                       
    # Before visualising the galaxy, we need to initialise some smoothing
    # lengths for the dark matter particles (this bit of code is taken
    # from the `swiftsimio` visualisation documentation).
    if not hasattr(galaxy_data.dark_matter, "smoothing_lengths"):
        galaxy_data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
            (galaxy_data.dark_matter.coordinates + galaxy_data.centre) % galaxy_data.metadata.boxsize,
            galaxy_data.metadata.boxsize,
            kernel_gamma=1.8,
            neighbours=57,
            speedup_fac=2,
            dimension=3,
        )
    disc_region = [-disc_radius, disc_radius, -disc_radius, disc_radius]
    halo_region = [-halo_radius, halo_radius, -halo_radius, halo_radius]
    gas_map = project_gas(
        galaxy_data,
        resolution=256,
        project="masses",
        parallel=True,
        region=disc_region,
    )
    dm_map = project_pixel_grid(
        data=galaxy_data.dark_matter,
        boxsize=galaxy_data.metadata.boxsize,
        resolution=256,
        project="masses",
        parallel=True,
        region=halo_region,
    )
    star_map = project_pixel_grid(
        data=galaxy_data.stars,
        boxsize=galaxy_data.metadata.boxsize,
        resolution=256,
        project="masses",
        parallel=True,
        region=disc_region,
    )
    
    #--------------
    ### Figure initialising
    fig = plt.figure(fignum, figsize=(10, 3))
    sp1, sp2, sp3 = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
    
    #--------------
    ### Plot imshow
    sp1.imshow(colors.LogNorm()(gas_map.value), cmap="viridis", extent=disc_region)
    sp2.imshow(colors.LogNorm()(dm_map), cmap="inferno", extent=halo_region,)
    sp3.imshow(colors.LogNorm()(star_map), cmap="magma", extent=disc_region,)
    
    
    #--------------
    ### general formatting
    sp1.set_xlabel(f"x' [{disc_radius.units}]")
    sp1.set_ylabel(f"y' [{disc_radius.units}]")
    
    sp2.set_xlabel(f"x' [{halo_radius.units}]")
    sp2.set_ylabel(f"y' [{halo_radius.units}]")
    
    sp3.set_xlabel(f"x' [{disc_radius.units}]")
    sp3.set_ylabel(f"y' [{disc_radius.units}]")
    
    
    #--------------
    ### Annotation
    sp1.text(0.9, 0.9, "gas", color="white", ha="right", va="top", transform=sp1.transAxes)
    
    sp2.plot([-disc_radius, -disc_radius, disc_radius, disc_radius, -disc_radius],
             [-disc_radius, disc_radius, disc_radius, -disc_radius, -disc_radius],
             "-k",)
    sp2.text(0.9, 0.9, "DM", ha="right", va="top", color="white", transform=sp2.transAxes)
    sp3.text(0.9, 0.9, "stars", ha="right", va="top", transform=sp3.transAxes)
    
    
    #--------------
    ### Title
    sp2.set_title(f"soap_index={galaxy_data.halo_catalogue.soap_index}")
    
    
    #--------------
    ### other
    fig.subplots_adjust(wspace=0.4)
    
    
    #--------------
    ### Savefig
    if savefig:
        savefig_txt_save = ('' if savefig_txt_in == None else savefig_txt_in) + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/test_figs/L0025_testvis_%s_%s.%s" %(fig_dir, galaxy_data.halo_catalogue.soap_index, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/test_figs/L0025_testvis_%s_%s.%s" %(fig_dir, galaxy_data.halo_catalogue.soap_index, savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
    
    #return fig
    
    
    
#=====================================================
myvis(galaxy_data=galaxy_data, 
        showfig=False,
        savefig=True)


















