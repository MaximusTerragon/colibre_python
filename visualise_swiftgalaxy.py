import os
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import swiftgalaxy
import swiftsimio as sw
import scipy
import csv
import time
import json
from tqdm import tqdm
from packaging.version import Version
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP
from swiftgalaxy.iterator import SWIFTGalaxies
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


#-------------------------------
# Loads a SOAP sample from csv file
def _load_soap_sample(csv_sample, print_sample=True):
    # Loading sample
    dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    soap_indicies = np.array(dict_new['soap_indicies'])
    sample_input  = dict_new['sample_input']
    dict_new = 0
    
    if print_sample:
        print('\n===Loaded Sample===')
        print('%s\t%s' %(sample_input['simulation_run'], sample_input['simulation_type']))
        print('Snapshot:  %s' %sample_input['snapshot_no'])
        print('Redshift:  %.2f' %sample_input['redshift'])
        print('Created sample size:   %s' %len(soap_indicies))
        print('')
    
    return soap_indicies, sample_input


#--------------------------------
# We can define a function that uses some of swiftsimio's visualisation tools to make some quick images of a given galaxy
def _visualize_galaxy(sg, plot_annotate = None, savefig_txt_in = None,      # SWIFTGalaxy object containing all the integrated properties + access the particles from the snapshot file
                      #=====================================
                      # fig settings
                        disc_radius=15 * u.kpc,           # gas/stars disc radius 
                        halo_radius=200 * u.kpc,          # DM halo disc radius
                      #=====================================
                      showfig       = False,
                      savefig       = True,
                        file_format = 'pdf',
                        savefig_txt = '', 
                      #--------------------------
                      print_progress = False,
                        debug = False):
    
    
    #--------------------
    # Load soap_index
    soap_index  = sg.halo_catalogue.soap_index.squeeze()
    track_id    = sg.halo_catalogue.input_halos_hbtplus.track_id.squeeze()
    redshift    = sg.metadata.redshift.squeeze()
    run_name    = sg.metadata.run_name
    lookbacktime = sg.metadata.time.squeeze()
        
    #--------------------
    # Before visualising the galaxy, we need to initialise some smoothing lengths for the dark matter particles (this bit of code is taken from the `swiftsimio` visualisation documentation).
    if not hasattr(sg.dark_matter, "smoothing_lengths"):
        sg.dark_matter.smoothing_lengths = generate_smoothing_lengths(
            (sg.dark_matter.coordinates + sg.centre) % sg.metadata.boxsize,
            sg.metadata.boxsize,
            kernel_gamma=1.8,
            neighbours=57,
            speedup_fac=2,
            dimension=3,
        )
    disc_region = [-disc_radius, disc_radius, -disc_radius, disc_radius]
    halo_region = [-halo_radius, halo_radius, -halo_radius, halo_radius]
    gas_map = project_gas(
        sg,
        resolution=256,
        project="masses",
        parallel=True,
        region=disc_region,
    )
    dm_map = project_pixel_grid(
        data=sg.dark_matter,
        boxsize=sg.metadata.boxsize,
        resolution=256,
        project="masses",
        parallel=True,
        region=halo_region,
    )
    star_map = project_pixel_grid(
        data=sg.stars,
        boxsize=sg.metadata.boxsize,
        resolution=256,
        project="masses",
        parallel=True,
        region=disc_region,
    )
    
    #--------------
    ### Figure initialising
    fig = plt.figure(figsize=(10, 3))
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
    sp2.set_title(f"soap_index={soap_index}, track_id={track_id}")
    
    
    #--------------
    ### other
    fig.subplots_adjust(wspace=0.4)
    
    
    #--------------
    ### Savefig
    if savefig:
        savefig_txt_save = ('' if savefig_txt_in == None else savefig_txt_in) + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/test_figs/%s_testvis_%s_%s.%s" %(fig_dir, run_name, track_id.squeeze(), savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/test_figs/%s_testvis_%s_%s.%s" %(fig_dir, run_name, track_id.squeeze(), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()




#====================================
# Load a sample from a given snapshot
soap_indicies, sample_input = _load_soap_sample(csv_sample = 'L0025N0752_THERMAL_AGN_m5_123_sample_')
savefig_txt_in = ''

# Lets select the first 5 and create images:
soap_indicies_sample = soap_indicies[:5]
#soap_indicies_sample = soap_indicies
print(f"Iterating over limited sample with galaxies {soap_indicies_sample}")

# Now we can create a SWIFTGalaxy to experiment with. The halo_index selects one row from the halo catalogue. The SWIFTGalaxy object will contain all of the integrated properties of this object from the halo catalogue and also let us access its particles from the snapshot file.
sgs = SWIFTGalaxies(sample_input['virtual_snapshot_file'], SOAP(sample_input['soap_catalogue_file'], soap_index=soap_indicies_sample, ),
                    preload={"gas.coordinates", "gas.masses", "gas.smoothing_lengths", "dark_matter.coordinates", "dark_matter.masses", "stars.coordinates", "stars.masses", "stars.smoothing_lengths", },)
           
# Use map to apply `myvis` to each `sg` in `sgs` and give us the results in order.         
output_figures = sgs.map(_visualize_galaxy)





    















