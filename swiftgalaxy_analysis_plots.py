import swiftsimio as sw
import swiftgalaxy
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import scipy
import h5py
import csv
import time
import math
import pandas as pd
from swiftsimio import cosmo_quantity, cosmo_array
from swiftsimio import SWIFTDataset, cosmo_quantity, cosmo_array
from swiftgalaxy import SWIFTGalaxy, SOAP, MaskCollection
from swiftgalaxy.iterator import SWIFTGalaxies
from swiftsimio.visualisation.projection import project_gas, project_pixel_grid
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation import (generate_smoothing_lengths,)  # if not found try `from sw.visualisation import generate_smoothing_lengths`
from swiftsimio.objects import cosmo_factor, a
from operator import attrgetter
from tqdm import tqdm
import matplotlib.pyplot as plt
from graphformat import set_rc_params
from read_dataset_directories_colibre import _assign_directories
from load_soap_sample import _load_soap_sample
from packaging.version import Version
assert Version(sw.__version__) >= Version("9.0.2")
assert Version(swiftgalaxy.__version__) >= Version("1.2.0")


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================


""" Unyts guide:

This applies to both halo_catalogue and particle data (.stars):

.in_units(u.km/u.s)         -> converts, makes a copy, keeps units as unyt array
.in_units(u.dimensionless)  -> converts, makes a copy, keeps units as unyt array
.to_physical()              -> converts to physical, makes a copy, keeps units as unyt array
.to(u.km/u.s)               -> same as .in_units(u.km/u.s)
    
.convert_to_units('kpc')    -> call function to convert array without copy, keeps units as unyt array
  or .convert_to_units(u.kpc)       but this function is bugged as it keeps the old units, despite converting
                                    values
.convert_to_physical()      -> call function to convert to physical
    
.value                      -> makes a copy, converts to np array (removes units)
.to_value(u.km/u.s)         -> converts, makes a copy, converts to np array (removes units)
    
7*u.Msun                    -> applies a unit to a value

"""
#--------------------------------
# Plot distribution of H2 fraction among particles, weighted by particle count, mass fraction count, and cumulative, in 50 kpc... are they mostly near unity or mostly low fraction?
def _plot_h2_massfraction_particles(sg, plot_annotate = None, savefig_txt_in = None,      # SWIFTGalaxy object containing all the integrated properties + access the particles from the snapshot file
                      print_galaxy = True,
                      #=====================================
                      # fig settings
                        analysis_radius= 50,           # pkpc, radius to mask
                        h2_rich_thresh = 0.5,          # mass fraction that is considered 'H2 rich' 
                      #=====================================
                      showfig       = False,
                      savefig       = True,
                        file_format = 'pdf',
                        savefig_txt = '', 
                      #--------------------------
                      print_progress = False,
                        debug = False):
    
    
    #--------------------
    # Load soap_index and 
    soap_index  = sg.halo_catalogue.soap_index
    track_id    = sg.halo_catalogue.input_halos_hbtplus.track_id.squeeze()
    #descendent_id = sg.halo_catalogue.input_halos_hbtplus.descendant_track_id.squeeze()
    redshift    = sg.metadata.redshift.squeeze()
    run_name    = sg.metadata.run_name
    lookbacktime = sg.metadata.time.squeeze()
    
    #------------------------------
    #print('REMOVE MANUAL RECENTER AND auto_recenter in function !!!!!!!')
    #print('Using manual recenter due to velocity bug')
    # Manual recentre because bugged velocity in L00250752 simulation
    # REMOVE FROM L100 RUN
    sg.recentre(sg.halo_catalogue.centre)               # Defaults to The centre of the subhalo as given by the halo finder. For HBTplus this is equal to the position of the most bound particle in the subhalo.
    vcentre = sg.halo_catalogue.velocity_centre         # Defaults to CentreOfMassVelocity of bound subhalo
    vcentre.cosmo_factor = cosmo_factor(a**0, sg.metadata.scale_factor)
    sg.recentre_velocity(vcentre)
    
    
    #------------------------------
    # Bulk properties within aperture of 50 pkpc
    stelmass50      = sg.halo_catalogue.exclusive_sphere_50kpc.stellar_mass.to(u.Msun)
    m200c           = sg.halo_catalogue.spherical_overdensity_200_crit.total_mass.to(u.Msun)
    H2mass50          = sg.halo_catalogue.exclusive_sphere_50kpc.molecular_hydrogen_mass.to(u.Msun)
    H2mass10          = sg.halo_catalogue.exclusive_sphere_10kpc.molecular_hydrogen_mass.to(u.Msun)
    r50stars        = (sg.halo_catalogue.exclusive_sphere_50kpc.half_mass_radius_stars.to_physical()).to(u.kpc)
    r50gas          = (sg.halo_catalogue.exclusive_sphere_50kpc.half_mass_radius_gas.to_physical()).to(u.kpc)
    kappastars        = sg.halo_catalogue.exclusive_sphere_50kpc.kappa_corot_stars
    is_central       = sg.halo_catalogue.input_halos.is_central
    #---------------
    if print_galaxy:
        print('\nSOAP index:                 %i   at    z = %.3f   in   %s' %(soap_index, redshift, run_name))
        print('   trackID:                     %i' %track_id)
        #print('   descendantID:         %i' %descendent_id)
        print('   central or satellite:  ->  %s' %('central' if is_central==cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=sg.metadata.a, scale_exponent=0) else 'satellite'))
        print('   stelmass50:            ->  %.3e      Msun' %stelmass50.squeeze())
        print('   m200 crit              ->  %.3e      Msun' %m200c.squeeze())
        print('      H2 (50kpc):         ->    %.3e    Msun' %H2mass50.squeeze())
        print('      H2 (10kpc)          ->    %.3e    Msun' %H2mass10.squeeze())
        print('   r50 stars:             ->  %.2f      pkpc' %r50stars.squeeze())
        print('    r50 gas:                    %.2f    pkpc' %r50gas.squeeze())
        print('   kappa stars 50:       ->  %.3f'           %kappastars.squeeze())
        
    
    #========================================================
    # histogram of H2 particle mass fractions
    
    # Mask particles within analysis_radius
    def _trim_particles(particle_masses, particle_coord, trim_rad= 50):
        trim_rad = cosmo_quantity(trim_rad, u.kpc, comoving=False, scale_factor=sg.metadata.a, scale_exponent=1)
        
        # Masses and coords
        particle_masses = particle_masses.to(u.Msun)
        particle_masses.convert_to_physical()
        particle_coord  = particle_coord.to(u.kpc)
        particle_coord.convert_to_physical()
        
        # Mask all within trim_rad
        mask = np.argwhere(np.linalg.norm(particle_coord, axis=1) <= trim_rad).squeeze()
        
        return mask
    trim_mask = _trim_particles(sg.gas.masses, sg.gas.coordinates, trim_rad = analysis_radius)
    
    # create new sg instance for h2 mass
    sg.gas.mass_gas_h2  = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * (sg.gas.species_fractions.H2 * 2)
    h2_mass_fractions   = sg.gas.mass_gas_h2/sg.gas.masses
    h2_mass_particle_50 = np.sum(sg.gas.mass_gas_h2[trim_mask])
    h2_mass_rich_50     = np.sum(sg.gas.mass_gas_h2[trim_mask][h2_mass_fractions[trim_mask] > cosmo_quantity(h2_rich_thresh, u.dimensionless, comoving=False, scale_factor=sg.metadata.a, scale_exponent=0)])
    
    print('---------------')
    print('SOAP H2 mass:                    %.3e' %H2mass50.squeeze())
    print('Particle mask mass:              %.3e' %h2_mass_particle_50)
    print('H2-rich particle mask mass:      %.3e' %h2_mass_rich_50)
    
    # sort by H2 fraction, assuming it has >0 H2
    #mask_sort = np.argsort((h2_mass_fractions[h2_mass_fractions > 0]).value)
    
    
    #---------------------------
    # Graph initialising and base formatting
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=[10/2, 6.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    #--------------
    # histogram in log
    bin_step = 0.1
    hist_bins = np.arange(-3, 0, bin_step)
    
    # histogram showing distribution of mass fractions among H2-containing gas particles, weighted by number of particles
    ax1.hist(np.log10(h2_mass_fractions[h2_mass_fractions>0]), bins=hist_bins, weights=np.ones(len(h2_mass_fractions[h2_mass_fractions>0]))/len(h2_mass_fractions[h2_mass_fractions>0]))

    
    # histogram showing distribution of mass fractions among H2-containing gas particles, weighted by contribution to total H2 mass of galaxy
    ax2.hist(np.log10(h2_mass_fractions[h2_mass_fractions>0]), bins=hist_bins, weights=np.divide(np.array(sg.gas.mass_gas_h2[h2_mass_fractions>0]), h2_mass_particle_50).value)
    
    # Plot contribution to H2 mass in 50 pkpc (cumulative version of above)
    ax3.hist(np.log10(h2_mass_fractions[h2_mass_fractions>0]), bins=hist_bins, weights=np.divide(np.array(sg.gas.mass_gas_h2[h2_mass_fractions>0]), h2_mass_particle_50).value, histtype="step", cumulative=True)
    
    
    #----------
    # General formatting
    ax1.set_xlim(-3, 0)
    ax3.set_xlabel('log10 H2 mass fraction in gas particles 50 pkpc')
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_ylabel('fraction of total\nnumber of particles in 50 pkpc')
    ax2.set_ylabel('mass-weighted fraction\nin 50 pkpc')
    ax3.set_ylabel('cumulative mass-weighted\nfraction in 50 pkpc')
    
    
    #--------------
    ### Title
    ax1.set_title(f"soap_index = %i, track_id = %i, redshift = %.3f,  %s \n M200c = %.2e Msun, M*_50 = %.2e Msun\nMH2_10 = %.2e Msun, MH2_50 = %.2e Msun\nfrac in h2-rich = %.3f \nr50 = %.2f pkpc, kappaco = %.2f" %(soap_index, (track_id.squeeze()).to_value(), redshift, ('central' if is_central==cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=sg.metadata.a, scale_exponent=0) else 'satellite'), m200c.squeeze(), stelmass50.squeeze(),  H2mass10.squeeze(), H2mass50.squeeze(), h2_mass_rich_50/h2_mass_particle_50, r50stars.squeeze(), kappastars.squeeze()), fontsize=10, loc='left')
    
    #--------------
    ### other
    fig.subplots_adjust(wspace=0, hspace=0)
    
    #--------------
    ### Savefig
    if savefig:
        savefig_txt_save = ('' if savefig_txt_in == None else savefig_txt_in) + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/etg_swiftgalaxy_analysis/h2_particle_massfractions/%s/%s_h2particlemassfractions_%i_%s.%s" %(fig_dir, save_folder_visual, run_name, track_id, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/etg_swiftgalaxy_analysis/h2_particle_massfractions/%s/%s_h2particlemassfractions_%i_%s.%s" %(fig_dir, save_folder_visual, run_name, track_id, savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()

    return soap_index
 

#--------------------------------
# Analyse each galaxy and extract H2 mass from SOAP, from particle data, and from counting only H2-rich particles
def _analyse_rich_particles_mass_change(sg, plot_annotate = None, savefig_txt_in = None,      # SWIFTGalaxy object containing all the integrated properties + access the particles from the snapshot file
                      print_galaxy = True,
                      #=====================================
                      # fig settings
                        analysis_radius  = 50,           # pkpc, radius to mask
                        h2_rich_thresh = 0.5,          # mass fraction that is considered 'H2 rich' 
                        h2_semirich_thresh = 0.1,          # mass fraction that is considered 'H2 rich' 
                      #=====================================
                      print_progress = False,
                        debug = False):
    
    
    #--------------------
    # Load soap_index and 
    soap_index  = sg.halo_catalogue.soap_index
    track_id    = sg.halo_catalogue.input_halos_hbtplus.track_id.squeeze()
    #descendent_id = sg.halo_catalogue.input_halos_hbtplus.descendant_track_id.squeeze()
    redshift    = sg.metadata.redshift.squeeze()
    run_name    = sg.metadata.run_name
    lookbacktime = sg.metadata.time.squeeze()
    
    
    #------------------------------
    #print('REMOVE MANUAL RECENTER AND auto_recenter in function !!!!!!!')
    #print('Using manual recenter due to velocity bug')
    # Manual recentre because bugged velocity in L00250752 simulation
    # REMOVE FROM L100 RUN
    sg.recentre(sg.halo_catalogue.centre)               # Defaults to The centre of the subhalo as given by the halo finder. For HBTplus this is equal to the position of the most bound particle in the subhalo.
    vcentre = sg.halo_catalogue.velocity_centre         # Defaults to CentreOfMassVelocity of bound subhalo
    vcentre.cosmo_factor = cosmo_factor(a**0, sg.metadata.scale_factor)
    sg.recentre_velocity(vcentre)
    
    
    #------------------------------
    # Bulk properties within aperture of 50 pkpc
    stelmass50      = (sg.halo_catalogue.exclusive_sphere_50kpc.stellar_mass.to(u.Msun)).squeeze()
    m200c           = (sg.halo_catalogue.spherical_overdensity_200_crit.total_mass.to(u.Msun)).squeeze()
    H2mass50          = (sg.halo_catalogue.exclusive_sphere_50kpc.molecular_hydrogen_mass.to(u.Msun)).squeeze()
    H2mass10          = (sg.halo_catalogue.exclusive_sphere_10kpc.molecular_hydrogen_mass.to(u.Msun)).squeeze()
    r50stars        = ((sg.halo_catalogue.exclusive_sphere_50kpc.half_mass_radius_stars.to_physical()).to(u.kpc)).squeeze()
    r50gas          = ((sg.halo_catalogue.exclusive_sphere_50kpc.half_mass_radius_gas.to_physical()).to(u.kpc)).squeeze()
    kappastars        = (sg.halo_catalogue.exclusive_sphere_50kpc.kappa_corot_stars).squeeze()
    is_central       = sg.halo_catalogue.input_halos.is_central
    #---------------
    if print_galaxy:
        print('\nSOAP index:                 %i   at    z = %.3f   in   %s' %(soap_index, redshift, run_name))
        print('   trackID:                     %i' %track_id)
        #print('   descendantID:         %i' %descendent_id)
        print('   central or satellite:  ->  %s' %('central' if is_central==cosmo_quantity(1, u.dimensionless, comoving=False, scale_factor=sg.metadata.a, scale_exponent=0) else 'satellite'))
        print('   stelmass50:            ->  %.3e      Msun' %stelmass50)
        print('   m200 crit              ->  %.3e      Msun' %m200c)
        print('      H2 (50kpc):         ->    %.3e    Msun' %H2mass50)
        print('      H2 (10kpc)          ->    %.3e    Msun' %H2mass10)
        print('   r50 stars:             ->  %.2f      pkpc' %r50stars)
        print('    r50 gas:                    %.2f    pkpc' %r50gas)
        print('   kappa stars 50:       ->  %.3f'           %kappastars)
        
    
    #========================================================
    # histogram of H2 particle mass fractions
    
    # Mask particles within analysis_radius
    def _trim_particles(particle_masses, particle_coord, trim_rad= 50):
        trim_rad = cosmo_quantity(trim_rad, u.kpc, comoving=False, scale_factor=sg.metadata.a, scale_exponent=1)
        
        # Masses and coords
        particle_masses = particle_masses.to(u.Msun)
        particle_masses.convert_to_physical()
        particle_coord  = particle_coord.to(u.kpc)
        particle_coord.convert_to_physical()
        
        # Mask all within trim_rad
        mask = np.argwhere(np.linalg.norm(particle_coord, axis=1) <= trim_rad).squeeze()
        
        return mask
    trim_mask = _trim_particles(sg.gas.masses, sg.gas.coordinates, trim_rad = analysis_radius)
    
    # create new sg instance for h2 mass
    sg.gas.mass_gas_h2  = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * (sg.gas.species_fractions.H2 * 2)
    h2_mass_fractions = sg.gas.mass_gas_h2/sg.gas.masses
    h2_mass_particle_50 = np.sum(sg.gas.mass_gas_h2[trim_mask])
    h2_mass_rich_50     = np.sum(sg.gas.mass_gas_h2[trim_mask][h2_mass_fractions[trim_mask] > cosmo_quantity(h2_rich_thresh, u.dimensionless, comoving=False, scale_factor=sg.metadata.a, scale_exponent=0)])
    h2_mass_semirich_50 = np.sum(sg.gas.mass_gas_h2[trim_mask][h2_mass_fractions[trim_mask] > cosmo_quantity(h2_semirich_thresh, u.dimensionless, comoving=False, scale_factor=sg.metadata.a, scale_exponent=0)])
    
    print('---------------')
    print('SOAP H2 mass:                    %.3e' %H2mass50.squeeze())
    print('Particle mask mass:              %.3e' %h2_mass_particle_50)
    print('H2-rich particle mask mass:      %.3e' %h2_mass_rich_50)
    print('H2-semirich particle mask mass:  %.3e' %h2_mass_semirich_50)
    
    #------------------------
    dict_return = {'run_name': run_name,
                   'soap_index': soap_index,
                   'soap_H2mass50': H2mass50,
                   'h2_mass_particle_50': h2_mass_particle_50,
                   'h2_mass_rich_50': h2_mass_rich_50,
                   'h2_mass_semirich_50': h2_mass_semirich_50}
    
    return dict_return
# Plot to show how much mass of H2 changes if we only use H2-rich gas particles above a certain mass fraction 
def _plot_rich_particles_mass_change(output_data = 0, sample_input = 0, soap_indicies_sample = 0,
                                    #=====================================
                                    showfig       = False,
                                    savefig       = True,
                                      file_format = 'pdf',
                                      savefig_txt = '', 
                                    #--------------------------
                                    print_progress = False,
                                      debug = False):    
    
    #---------------------------
    # Graph initialising and base formatting
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[10, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    # Extract
    soap_index_list = []
    soap_H2mass50_list   = []
    h2_mass_particle_50_list = []
    h2_mass_rich_50_list = []
    h2_mass_semirich_50_list = []
    for galaxy_i in output_data:
        soap_index_list.append(galaxy_i['soap_index'])
        soap_H2mass50_list.append(galaxy_i['soap_H2mass50'])
        h2_mass_particle_50_list.append(galaxy_i['h2_mass_particle_50'])
        h2_mass_rich_50_list.append(galaxy_i['h2_mass_rich_50'])
        h2_mass_semirich_50_list.append(galaxy_i['h2_mass_semirich_50'])
        
    df = pd.DataFrame(data={'soap_index_list': soap_index_list, 'soap_H2mass50_list': soap_H2mass50_list, 'h2_mass_particle_50_list': h2_mass_particle_50_list, 'h2_mass_rich_50_list': h2_mass_rich_50_list, 'h2_mass_semirich_50_list': h2_mass_semirich_50_list})        
    
    print('\n-------------------------------')
    print('Median particle / SOAP difference:   %.3f'%np.median(df['h2_mass_particle_50_list']/df['soap_H2mass50_list']))
    print('   Max particle / SOAP difference:   %.3f'%np.max(df['h2_mass_particle_50_list']/df['soap_H2mass50_list']))
    print('   Min particle / SOAP difference:   %.3f'%np.min(df['h2_mass_particle_50_list']/df['soap_H2mass50_list']))
    print(' ')
    print('Median H2-rich / particle difference:   %.3f'%np.median(df['h2_mass_rich_50_list']/df['h2_mass_particle_50_list']))
    print('   Max H2-rich / particle difference:   %.3f'%np.max(df['h2_mass_rich_50_list']/df['h2_mass_particle_50_list']))
    print('   Min H2-rich / particle difference:   %.3f'%np.min(df['h2_mass_rich_50_list']/df['h2_mass_particle_50_list']))
    print(' ')
    print('Median H2-semirich / particle difference:   %.3f'%np.median(df['h2_mass_semirich_50_list']/df['h2_mass_particle_50_list']))
    print('   Max H2-semirich / particle difference:   %.3f'%np.max(df['h2_mass_semirich_50_list']/df['h2_mass_particle_50_list']))
    print('   Min H2-semirich / particle difference:   %.3f'%np.min(df['h2_mass_semirich_50_list']/df['h2_mass_particle_50_list']))
    print(' ')
    
    
    #--------------
    # scatter1 in log log: particle H2 vs soap H2
    ax1.scatter(df['h2_mass_particle_50_list'], df['soap_H2mass50_list'], s=3, marker='o')
    
    # scatter2 in log log: particle H2 vs gas rich particle H2
    ax2.scatter(df['h2_mass_particle_50_list'], df['h2_mass_rich_50_list'], s=3, marker='o')
    
    # scatter3 in log log: particle H2 vs gas semirich particle H2
    ax3.scatter(df['h2_mass_particle_50_list'], df['h2_mass_semirich_50_list'], s=3, marker='o')
    
    #----------
    # General formatting
    for axs in [ax1, ax2, ax3]:
        axs.plot([10**5, 10**12], [10**5, 10**12], ls='--', lw=0.8)
        
        axs.set_xlim(10**5, 10**11)
        axs.set_ylim(10**5, 10**11)
        axs.set_xscale("log")
        axs.set_yscale("log")
        
        axs.set_xlabel('H2 mass particle data')
    ax1.set_ylabel('H2 mass SOAP')
    ax2.set_ylabel('H2 mass h2-rich particles\n(H2 mass frac $>$ 0.5)')
    ax3.set_ylabel('H2 mass h2-semirich particles\n(H2 mass frac $>$ 0.1)')
    
    
    #--------------
    ### other
    
    
    #--------------
    ### Savefig
    if savefig:
        savefig_txt_save = ('' if savefig_txt_in == None else savefig_txt_in) + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/etg_swiftgalaxy_analysis/h2_soap_particle_rich_particle/%s_%s_h2masschange_soap_particle_rich_%s_%i_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset'], len(soap_indicies_sample), savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/etg_swiftgalaxy_analysis/h2_soap_particle_rich_particle/%s_%s_h2masschange_soap_particle_richs_%s_%i_%s.%s" %(fig_dir, sample_input['simulation_run'], sample_input['simulation_type'], sample_input['name_of_preset'], len(soap_indicies_sample), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
                            
    


#========================================================================
# Manual sample or load input:
"""soap_indicies_sample = [1921622, 10192609] 
sample_input = {'name_of_preset': 'gas_rich_ETGs_z0',
                'simulation_run': 'L100N1504',
                'simulation_type': 'Thermal_non_equilibrium',
                'virtual_snapshot_file': '%s'%('/home/cosmos/c22048063/COLIBRE/Runs/L100_m6/THERMAL_AGN_m6/SOAP/colibre_with_SOAP_membership_0127.hdf5' if answer == '2' else '/cosma8/data/dp004/colibre/Runs/L100_m6/THERMAL_AGN_m6/SOAP/colibre_with_SOAP_membership_0127.hdf5'),
                'soap_catalogue_file':   '%s'%('/home/cosmos/c22048063/COLIBRE/Runs/L100_m6/THERMAL_AGN_m6/SOAP/halo_properties_0127.hdf5' if answer == '2' else '/cosma8/data/dp004/colibre/Runs/L100_m6/THERMAL_AGN_m6/SOAP/halo_properties_0127.hdf5')
            }
savefig_txt_in = ''
save_folder_visual = sample_input['name_of_preset']"""
#------------------------------------------------------------------------
# Load a sample from a given snapshot
soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = 'L100_m6_THERMAL_AGN_m6_127_sample20_random_h2_ETGs_z0')
                                                                                    # L100_m6_THERMAL_AGN_m6_127_sample20_random_ETGs_z0
                                                                                    # L100_m6_THERMAL_AGN_m6_127_sample20_random_h2_ETGs_z0
savefig_txt_in = ''
save_folder_visual = sample_input['name_of_preset']
#========================================================================






#---------------------
# Plot distribution of H2 fraction among particles, weighted by particle count, mass fraction count, and cumulative, in 50 kpc
"""sgs = SWIFTGalaxies(sample_input['virtual_snapshot_file'], SOAP(sample_input['soap_catalogue_file'], soap_index=soap_indicies_sample, ), auto_recentre=False,
                    preload={"gas.coordinates", "gas.masses", "gas.element_mass_fractions.hydrogen", "gas.species_fractions.H2", },)
output_data = sgs.map(_plot_h2_massfraction_particles)      # output_data = NOT THE SAME ORDER as input SOAP indicies
print(soap_indicies_sample)
print(output_data)"""


#---------------------
# Plot how H2 mass changes depending on if SOAP 50, particle data 50, or H2-rich particle data is used
sgs = SWIFTGalaxies(sample_input['virtual_snapshot_file'], SOAP(sample_input['soap_catalogue_file'], soap_index=soap_indicies_sample, ), auto_recentre=False,
                    preload={"gas.coordinates", "gas.masses", "gas.element_mass_fractions.hydrogen", "gas.species_fractions.H2", },)       
output_data = sgs.map(_analyse_rich_particles_mass_change) # output_data = NOT THE SAME ORDER as input SOAP indicies
_plot_rich_particles_mass_change(output_data=output_data, sample_input=sample_input, soap_indicies_sample=soap_indicies_sample,
                                    showfig       = False,
                                    savefig       = True)



