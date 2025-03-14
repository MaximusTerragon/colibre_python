import os
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import swiftgalaxy
import swiftsimio as sw
import scipy
import csv
import time
import json
import math
from tqdm import tqdm
from packaging.version import Version
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP, MaskCollection
from swiftgalaxy.iterator import SWIFTGalaxies
from swiftsimio.visualisation.projection import project_gas, project_pixel_grid
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation import (generate_smoothing_lengths,)  # if not found try `from sw.visualisation import generate_smoothing_lengths`
from swiftsimio.objects import cosmo_factor, a
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from graphformat import set_rc_params
from load_soap_sample import _load_soap_sample
from read_dataset_directories_colibre import _assign_directories
assert Version(sw.__version__) >= Version("9.0.2")
assert Version(swiftgalaxy.__version__) >= Version("1.2.0")

np.random.seed(0)

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
# We can define a function that uses some of swiftsimio's visualisation tools to make some quick images of a given galaxy. Requires a SOAP index
# Visualises stars, gas out to 15 kpc, and DM out to 200 kpc
def _visualize_galaxy_example(sg, plot_annotate = None, savefig_txt_in = None,      # SWIFTGalaxy object containing all the integrated properties + access the particles from the snapshot file
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
    
    #------------------------------
    # Manual recentre because bugged velocity in L00250752 simulation
    # REMOVE FROM L100 RUN
    sg.recentre(sg.halo_catalogue.centre)
    vcentre = sg.halo_catalogue.velocity_centre
    vcentre.cosmo_factor = cosmo_factor(a**0, sg.metadata.scale_factor)
    sg.recentre_velocity(vcentre)
        
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

        plt.savefig("%s/test_figs/%s_testvis_%s_%s.%s" %(fig_dir, run_name, track_id, savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/test_figs/%s_testvis_%s_%s.%s" %(fig_dir, run_name, track_id, savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#--------------------------------
# Visualises stars, gas, SF gas, HII, HI, and H2 out to 30 kpc
def _visualize_galaxy_gas(sg, plot_annotate = None, savefig_txt_in = None,      # SWIFTGalaxy object containing all the integrated properties + access the particles from the snapshot file
                      print_galaxy = True,
                      #=====================================
                      # fig settings
                        disc_radius= 15 * u.kpc,           # gas/stars disc radius 
                        orientation = 'both',               # [ 'face' / 'edge' / 'none' / 'both' ] Orientates to face within 10 kpc
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
    soap_index  = sg.halo_catalogue.soap_index.squeeze()
    track_id    = sg.halo_catalogue.input_halos_hbtplus.track_id.squeeze()
    #descendent_id = sg.halo_catalogue.input_halos_hbtplus.descendant_track_id.squeeze()
    redshift    = sg.metadata.redshift.squeeze()
    run_name    = sg.metadata.run_name
    lookbacktime = sg.metadata.time.squeeze()
    
    
    #------------------------------
    print('REMOVE MANUAL RECENTER AND auto_recenter in function !!!!!!!')
    # Manual recentre because bugged velocity in L00250752 simulation
    # REMOVE FROM L100 RUN
    sg.recentre(sg.halo_catalogue.centre)
    vcentre = sg.halo_catalogue.velocity_centre
    vcentre.cosmo_factor = cosmo_factor(a**0, sg.metadata.scale_factor)
    sg.recentre_velocity(vcentre)
    
    
    # Bulk properties within aperture of 30 pkpc
    stelmass30      = sg.halo_catalogue.exclusive_sphere_30kpc.stellar_mass.to(u.Msun)
    m200c           = sg.halo_catalogue.spherical_overdensity_200_crit.total_mass.to(u.Msun)
    gasmass30       = sg.halo_catalogue.exclusive_sphere_30kpc.gas_mass.to(u.Msun)
    gascoldmass30     = sg.halo_catalogue.exclusive_sphere_30kpc.gas_mass_in_cold_dense_gas.to(u.Msun)
    gassfmass30       = sg.halo_catalogue.exclusive_sphere_30kpc.star_forming_gas_mass.to(u.Msun)
    HImass30          = sg.halo_catalogue.exclusive_sphere_30kpc.atomic_hydrogen_mass.to(u.Msun)
    H2mass30          = sg.halo_catalogue.exclusive_sphere_30kpc.molecular_hydrogen_mass.to(u.Msun)
    Molmass30         = (sg.halo_catalogue.exclusive_sphere_30kpc.molecular_hydrogen_mass.to(u.Msun) * (1 + (sg.halo_catalogue.exclusive_sphere_30kpc.helium_mass.to(u.Msun)/sg.halo_catalogue.exclusive_sphere_30kpc.hydrogen_mass.to(u.Msun))))
    r50stars        = (sg.halo_catalogue.exclusive_sphere_30kpc.half_mass_radius_stars.to_physical()).to(u.kpc)
    r50gas          = (sg.halo_catalogue.exclusive_sphere_30kpc.half_mass_radius_gas.to_physical()).to(u.kpc)
    kappastars        = sg.halo_catalogue.exclusive_sphere_30kpc.kappa_corot_stars
    kappagas          = sg.halo_catalogue.exclusive_sphere_30kpc.kappa_corot_gas
    sfr30           = sg.halo_catalogue.exclusive_sphere_30kpc.star_formation_rate.to(u.Msun/u.yr)
    ssfr30          = sfr30/(sg.halo_catalogue.exclusive_sphere_30kpc.stellar_mass.to(u.Msun))
    u_mag30           = -2.5*np.log10(sg.halo_catalogue.exclusive_sphere_30kpc.stellar_luminosity.squeeze()[0])
    g_mag30           = -2.5*np.log10(sg.halo_catalogue.exclusive_sphere_30kpc.stellar_luminosity.squeeze()[1])
    r_mag30           = -2.5*np.log10(sg.halo_catalogue.exclusive_sphere_30kpc.stellar_luminosity.squeeze()[2])
    is_central      = sg.halo_catalogue.input_halos.is_central
    
    
    #---------------
    # Calculate triax and ellipticity:
    ϵ = 0 -> Perfectly spherical galaxy
    ϵ > 0 -> Elliptical galaxy
    ϵ = 1 -> Highly flattened/disc galaxy or elongated galaxy
    
    T ≈ 0, (a ≈ b > c) -> Oblate (disk-like) galaxy 
    0 < T < 1, (a > b > c) -> Triaxial (ellipsoidal)
    T ≈ 1, (a > b ≈ c) -> Prolate (cigar-shaped)
    
    
    i11, i22, i33, i12, i13, i23 = sg.halo_catalogue.bound_subhalo.stellar_inertia_tensor.squeeze()
    inertiatensor = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])
    
    # Compute eigenvalues (sorted in descending order)
    eigenvalues = np.linalg.eigvalsh(inertiatensor)[::-1]  # Sorted: I1 >= I2 >= I3
    length_a2, length_b2, length_c2 = eigenvalues  # Assign to principal moments, length_a2 == a**2, so length = sqrt length_a2. a = longest axis, b = intermediate, c = shortest
    #print(length_a2)
    #print(length_b2)
    #print(length_c2)
    # Compute Triaxiality Parameter
    triaxiality = (length_a2 - length_b2) / (length_a2 - length_c2) if (length_a2 - length_c2) != 0 else np.nan  # Avoid division by zero
    ellipticity = 1 - (np.sqrt(length_c2)/np.sqrt(length_a2))
    #print(triaxiality)
    #print(ellipticity)
    
    #triaxiality = math.nan
    #ellipticity = math.nan
    
    #---------------
    if print_galaxy:
        print('\nSOAP index:                 %i   at    z = %.3f   in   %s' %(soap_index, redshift, run_name))
        print('   trackID:                     %i' %track_id)
        #print('   descendantID:         %i' %descendent_id)
        print('   central or satellite:  ->  %s' %('central' if is_central==1 else 'satellite'))
        print('   stelmass30:            ->  %.3e      Msun' %stelmass30.squeeze())
        print('   m200 crit              ->  %.3e      Msun' %m200c.squeeze())
        print('   gasmass30:                 %.3e      Msun' %gasmass30.squeeze())
        print('      SF:                       %.3e    Msun' %gassfmass30.squeeze())
        print('      cold/dense:               %.3e    Msun' %gascoldmass30.squeeze())
        print('      HI:                 ->    %.3e    Msun' %HImass30.squeeze())
        print('      H2:                 ->    %.3e    Msun' %H2mass30.squeeze())
        print('      Mol (H2 adjusted):  ->    %.3e    Msun' %Molmass30.squeeze())
        print('   r50 stars:             ->  %.2f      pkpc' %r50stars.squeeze())
        print('    r50 gas:                    %.2f    pkpc' %r50gas.squeeze())
        print('   kappa stars 30:        ->  %.3f'           %kappastars.squeeze())
        print('    kappa gas 30:               %.3f'         %kappagas.squeeze())
        print('    ellipticity stars SH:   ->  %.3f'         %ellipticity)
        print('    triaxiality stars SH:   ->  %.3f'         %triaxiality)
        print('   SFR 30:                    %.3e      Msun/yr' %sfr30.squeeze())
        print('    sSFR 30:              ->   %.3e     /yr' %ssfr30.squeeze())
        print('   u - r (no dust):           %.3f      mag' %(u_mag30-r_mag30))
        
    metadata_plot = {'Title': 'stelmass30: %.1e\nm200crit: %.1e\ngasmass30: %.1e\ngassf30: %.1e\ngascolddensemass30: %.1e\nHImass30: %.1e\nH2mass30: %.1e\nMolmass30: %.1e\nr50stars: %.2f\nr50gas: %.2f\nellipstars: %.2f\ntriaxstars: %.2f\nkappastars: %.2f\nkappagas: %.2f\nsfr30: %.2e\nssfr30: %.2e\nu-r (nodust): %.2f'%(stelmass30.squeeze(), m200c.squeeze(), gasmass30.squeeze(), gassfmass30.squeeze(), gascoldmass30.squeeze(), HImass30.squeeze(), H2mass30.squeeze(), Molmass30.squeeze(), r50stars.squeeze(), r50gas.squeeze(), ellipticity, triaxiality, kappastars.squeeze(), kappagas.squeeze(), sfr30.squeeze(), ssfr30.squeeze(), (u_mag30-r_mag30)),
                     'Author': 'SOAP index: %i\nredshift: %.2f\nTrackID: %i\ncen/sat: %s'%(soap_index, redshift, track_id, 'central' if is_central==1 else 'satellite'),
                     'Subject': run_name,
                     'Producer': ''}
    
    #-----------------------
    # Rotation matrix to align vec1 to vec2 (axis)
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
        
    
    if orientation != 'both':
        if orientation == 'face':
            Lstars = sg.halo_catalogue.exclusive_sphere_10kpc.angular_momentum_stars.squeeze()
            zhat = (Lstars / np.sqrt(np.sum(Lstars**2))).to_value(u.dimensionless)
            rotmat = rotation_matrix_from_vector(zhat, axis='z')
            rotcent = u.unyt_array([0, 0, 0]) * u.Mpc
        elif orientation == 'edge':
            Lstars = sg.halo_catalogue.exclusive_sphere_10kpc.angular_momentum_stars.squeeze()
            zhat = (Lstars / np.sqrt(np.sum(Lstars**2))).to_value(u.dimensionless)
            rotmat = rotation_matrix_from_vector(zhat, axis='x')
            rotcent = u.unyt_array([0, 0, 0]) * u.Mpc
        else:
            rotmat = np.identity(3)
            rotcent = u.unyt_array([0, 0, 0]) * u.Mpc

    
        #---------------------------------------
        # Creating masks for galaxy properties
        mask_gas_sf         = sg.gas.star_formation_rates > (0 * u.Msun/u.yr)
        #mask_gas_HII    = np.logical_and(sg.gas.species_fractions.HII > 0, sg.gas.element_mass_fractions.hydrogen > 0)
        #mask_gas_HI     = np.logical_and(sg.gas.species_fractions.HI > 0, sg.gas.element_mass_fractions.hydrogen > 0) 
        #mask_gas_H2     = np.logical_and(sg.gas.species_fractions.H2 > 0, sg.gas.element_mass_fractions.hydrogen > 0)
        #sg.gas.mass_gas_sf  = sg.gas.masses[mask_gas_sf]
    
        # Creating new objects for datasets of HII, HI, and H2
        sg.gas.mass_gas_hii = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * sg.gas.species_fractions.HII
        sg.gas.mass_gas_hi  = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * sg.gas.species_fractions.HI
        sg.gas.mass_gas_h2  = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * (sg.gas.species_fractions.H2 * 2)
    
        # Creating new objects for mass-weighted temperature, metallicity, and sfr
        sg.gas.mass_weighted_temps  = sg.gas.masses * sg.gas.temperatures
        sg.gas.mass_weighted_Z      = sg.gas.masses * sg.gas.metal_mass_fractions
        #sg.gas.mass_weighted_sfr    = sg.gas.masses * sg.gas.star_formation_rates.to_physical()
        sg.gas.mass_weighted_vel    = sg.gas.masses * (sg.gas.velocities[:,2].in_units(u.km/u.s))    # bugged with scalefactor in L25, see Kyle messages
        sg.gas.mass_weighted_vel.convert_to_physical()
    
    
      
        #--------------------
        disc_region = [-disc_radius, disc_radius, -disc_radius, disc_radius]
    
        # This creates a grid that has units msun / Mpc^2, and can be transformed like any other unyt quantity
        star_map    = project_pixel_grid(data=sg.stars, boxsize=sg.metadata.boxsize, resolution=612, project="masses", parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        gas_map     = project_gas(sg, resolution=612, project="masses",  parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        gassf_map   = project_gas(sg, resolution=612, project="masses", mask=mask_gas_sf, parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        mass_weighted_temp_map = project_gas(sg, resolution=612, project="mass_weighted_temps",  parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        mass_weighted_Z_map    = project_gas(sg, resolution=612, project="mass_weighted_Z",  parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        #mass_weighted_sfr      = project_gas(sg, resolution=612, project="mass_weighted_sfr",  parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        mass_weighted_vel      = project_gas(sg, resolution=612, project="mass_weighted_vel",  parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        gasHII_map  = project_gas(sg, resolution=612, project="mass_gas_hii", parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        gasHI_map   = project_gas(sg, resolution=612, project="mass_gas_hi", parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
        gasH2_map   = project_gas(sg, resolution=612, project="mass_gas_h2", parallel=True, region=disc_region, rotation_matrix=rotmat, rotation_center=rotcent, )
    
        temp_map = (mass_weighted_temp_map / gas_map)
        Z_map    = (mass_weighted_Z_map / gas_map)
        #sfr_map  = (mass_weighted_sfr / gas_map).to(u.Msun/u.yr)
        vel_map  = (mass_weighted_vel / gas_map).to_physical()
    
        #--------------
        ### Figure initialising
        fig = plt.figure(figsize=(20, 20))
        sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9 = [fig.add_subplot(3, 3, i, facecolor='k') for i in range(1, 10)]
    
        #--------------
        ### Plot imshow
        sp1.imshow(colors.LogNorm()(star_map), cmap="magma", extent=disc_region,)
        sp2.imshow(colors.LogNorm()(gas_map.value), cmap="viridis", extent=disc_region)
        sp3.imshow(colors.LogNorm()(gassf_map.value), cmap="viridis", extent=disc_region)
    
        #sp4.imshow(colors.LogNorm()(sfr_map.value), cmap="inferno", extent=disc_region)
        sp4.imshow(vel_map.value, cmap="coolwarm", extent=disc_region)
        sp5.imshow(colors.LogNorm()(temp_map.value), cmap="Spectral_r", extent=disc_region)
        sp6.imshow(colors.LogNorm()(Z_map.value), cmap="managua_r", extent=disc_region)
    
        sp7.imshow(colors.LogNorm()(gasHI_map.value), cmap="Greens", extent=disc_region)
        sp8.imshow(colors.LogNorm()(gasH2_map.value), cmap="Blues", extent=disc_region)
        sp9.imshow(colors.LogNorm()(gasHII_map.value), cmap="Reds", extent=disc_region)
    
    
        #--------------
        ### general formatting
        for sp_i in [sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9]:
            sp_i.set_xlabel(f"x' [{disc_radius.units}]")
            sp_i.set_ylabel(f"y' [{disc_radius.units}]")
    
    
        #--------------
        ### Annotation
        sp1.text(0.9, 0.9, r"$\Sigma_{\mathrm{*}}$", color="white", ha="right", va="top", transform=sp1.transAxes)
        sp2.text(0.9, 0.9, r"$\Sigma_{\mathrm{gas}}$", color="white", ha="right", va="top", transform=sp2.transAxes)
        sp3.text(0.9, 0.9, r"$\Sigma_{\mathrm{gas,SF}}$", color="white", ha="right", va="top", transform=sp3.transAxes)

        #sp4.text(0.9, 0.9, "$\Sigma_{\mathrm{SFR}}$", color="white", ha="right", va="top", transform=sp4.transAxes)
        sp4.text(0.9, 0.9, "Velocity", color="white", ha="right", va="top", transform=sp4.transAxes)
        sp5.text(0.9, 0.9, "Temperature", color="white", ha="right", va="top", transform=sp5.transAxes)
        sp6.text(0.9, 0.9, "Metallicity", color="white", ha="right", va="top", transform=sp6.transAxes)
    
        sp7.text(0.9, 0.9, r"$\Sigma_{\mathrm{HI}}$", color="k", ha="right", va="top", transform=sp7.transAxes)
        sp8.text(0.9, 0.9, r"$\Sigma_{\mathrm{H2}}$", color="k", ha="right", va="top", transform=sp8.transAxes)
        sp9.text(0.9, 0.9, r"$\Sigma_{\mathrm{HII}}$", color="k", ha="right", va="top", transform=sp9.transAxes)
    if orientation == 'both':
        
        # Rotation matrix
        Lstars = sg.halo_catalogue.exclusive_sphere_10kpc.angular_momentum_stars.squeeze()
        zhat = (Lstars / np.sqrt(np.sum(Lstars**2))).to_value(u.dimensionless)
        rotmat_face = rotation_matrix_from_vector(zhat, axis='z')
        rotmat_edge = rotation_matrix_from_vector(zhat, axis='x')
        rotcent = u.unyt_array([0, 0, 0]) * u.Mpc
        
    
        #---------------------------------------
        # Creating masks for galaxy properties
        mask_gas_sf         = sg.gas.star_formation_rates > (0 * u.Msun/u.yr)
    
        # Creating new objects for datasets of HII, HI, and H2
        sg.gas.mass_gas_hii = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * sg.gas.species_fractions.HII
        sg.gas.mass_gas_hi  = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * sg.gas.species_fractions.HI
        sg.gas.mass_gas_h2  = sg.gas.masses * sg.gas.element_mass_fractions.hydrogen * (sg.gas.species_fractions.H2 * 2)
    
        # Creating new objects for mass-weighted temperature, metallicity, and sfr
        sg.gas.mass_weighted_temps  = sg.gas.masses * sg.gas.temperatures
        sg.gas.mass_weighted_Z      = sg.gas.masses * sg.gas.metal_mass_fractions
        #sg.gas.mass_weighted_sfr    = sg.gas.masses * sg.gas.star_formation_rates.to_physical()
        
        velocities_xyz = sg.gas.velocities.to(u.km/u.s)
        velocities_xyz.convert_to_physical() 
        velocities_edge = (Rotation.from_matrix(rotmat_edge)).apply(velocities_xyz)
        sg.gas.mass_weighted_vel    = sg.gas.masses * (velocities_edge[:,2])    # bugged with scalefactor in L25, see Kyle messages
        #sg.gas.mass_weighted_vel.convert_to_physical() 
        velocities_xyz = 0
        velocities_edge = 0
        
        
        #--------------------
        disc_region = [-disc_radius, disc_radius, -disc_radius, disc_radius]
    
        # This creates a grid that has units msun / Mpc^2, and can be transformed like any other unyt quantity
        star_map_face    = project_pixel_grid(data=sg.stars, boxsize=sg.metadata.boxsize, resolution=612, project="masses", parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        star_map_edge    = project_pixel_grid(data=sg.stars, boxsize=sg.metadata.boxsize, resolution=612, project="masses", parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        gas_map_face     = project_gas(sg, resolution=612, project="masses",  parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        gas_map_edge     = project_gas(sg, resolution=612, project="masses",  parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        gassf_map_face   = project_gas(sg, resolution=612, project="masses", mask=mask_gas_sf, parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        gassf_map_edge   = project_gas(sg, resolution=612, project="masses", mask=mask_gas_sf, parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        mass_weighted_vel_map_face = project_gas(sg, resolution=612, project="mass_weighted_vel",  parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        mass_weighted_vel_map_edge = project_gas(sg, resolution=612, project="mass_weighted_vel",  parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        mass_weighted_temp_map_face = project_gas(sg, resolution=612, project="mass_weighted_temps",  parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        mass_weighted_temp_map_edge = project_gas(sg, resolution=612, project="mass_weighted_temps",  parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        mass_weighted_Z_map_face    = project_gas(sg, resolution=612, project="mass_weighted_Z",  parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        mass_weighted_Z_map_edge    = project_gas(sg, resolution=612, project="mass_weighted_Z",  parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        gasHII_map_face  = project_gas(sg, resolution=612, project="mass_gas_hii", parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        gasHII_map_edge  = project_gas(sg, resolution=612, project="mass_gas_hii", parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        gasHI_map_face   = project_gas(sg, resolution=612, project="mass_gas_hi", parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        gasHI_map_edge   = project_gas(sg, resolution=612, project="mass_gas_hi", parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        gasH2_map_face   = project_gas(sg, resolution=612, project="mass_gas_h2", parallel=True, region=disc_region, rotation_matrix=rotmat_face, rotation_center=rotcent, )
        gasH2_map_edge   = project_gas(sg, resolution=612, project="mass_gas_h2", parallel=True, region=disc_region, rotation_matrix=rotmat_edge, rotation_center=rotcent, )
        
        temp_map_face = (mass_weighted_temp_map_face / gas_map_face)
        temp_map_edge = (mass_weighted_temp_map_edge / gas_map_edge)
        vel_map_face  = (mass_weighted_vel_map_face / gas_map_face).to_physical()
        vel_map_edge  = (mass_weighted_vel_map_edge / gas_map_edge).to_physical()
        Z_map_face    = (mass_weighted_Z_map_face / gas_map_face)
        Z_map_edge    = (mass_weighted_Z_map_edge / gas_map_edge)
        
        
        #--------------
        ### Figure initialising
        fig = plt.figure(figsize=(36, 8))
        sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9, sp10, sp11, sp12, sp13, sp14, sp15, sp16, sp17, sp18 = [fig.add_subplot(2, 9, i, facecolor='k') for i in range(1, 19)]
    
        #--------------
        ### Plot imshow
        sp1.imshow(colors.LogNorm()(star_map_face), cmap="magma", extent=disc_region,)
        sp2.imshow(colors.LogNorm()(gas_map_face.value), cmap="viridis", extent=disc_region)
        sp3.imshow(colors.LogNorm()(gassf_map_face.value), cmap="viridis", extent=disc_region)
        sp4.imshow(vel_map_face.value, cmap="coolwarm", extent=disc_region)
        sp5.imshow(colors.LogNorm()(temp_map_face.value), cmap="Spectral_r", extent=disc_region)
        sp6.imshow(colors.LogNorm()(Z_map_face.value), cmap="managua_r", extent=disc_region)
        sp7.imshow(colors.LogNorm()(gasHI_map_face.value), cmap="Greens", extent=disc_region)
        sp8.imshow(colors.LogNorm()(gasH2_map_face.value), cmap="Blues", extent=disc_region)
        sp9.imshow(colors.LogNorm()(gasHII_map_face.value), cmap="Reds", extent=disc_region)
        
        sp10.imshow(colors.LogNorm()(star_map_edge), cmap="magma", extent=disc_region,)
        sp11.imshow(colors.LogNorm()(gas_map_edge.value), cmap="viridis", extent=disc_region)
        sp12.imshow(colors.LogNorm()(gassf_map_edge.value), cmap="viridis", extent=disc_region)
        sp13.imshow(vel_map_edge.value, cmap="coolwarm", extent=disc_region)
        sp14.imshow(colors.LogNorm()(temp_map_edge.value), cmap="Spectral_r", extent=disc_region)
        sp15.imshow(colors.LogNorm()(Z_map_edge.value), cmap="managua_r", extent=disc_region)
        sp16.imshow(colors.LogNorm()(gasHI_map_edge.value), cmap="Greens", extent=disc_region)
        sp17.imshow(colors.LogNorm()(gasH2_map_edge.value), cmap="Blues", extent=disc_region)
        sp18.imshow(colors.LogNorm()(gasHII_map_edge.value), cmap="Reds", extent=disc_region)
    
    
        #--------------
        ### general formatting
        sp1.set_ylabel(f"y' [{disc_radius.units}]")
        sp10.set_ylabel(f"y' [{disc_radius.units}]")
        for sp_i in [sp10, sp11, sp12, sp13, sp14, sp15, sp16, sp17, sp18]:
            sp_i.set_xlabel(f"x' [{disc_radius.units}]")
    
    
        #--------------
        ### Annotation
        sp1.text(0.9, 0.9, r"$\Sigma_{\mathrm{*}}$", color="white", ha="right", va="top", transform=sp1.transAxes)
        sp2.text(0.9, 0.9, r"$\Sigma_{\mathrm{gas}}$", color="white", ha="right", va="top", transform=sp2.transAxes)
        sp3.text(0.9, 0.9, r"$\Sigma_{\mathrm{gas,SF}}$", color="white", ha="right", va="top", transform=sp3.transAxes)
        sp4.text(0.9, 0.9, "Velocity", color="white", ha="right", va="top", transform=sp4.transAxes)
        sp5.text(0.9, 0.9, "Temperature", color="white", ha="right", va="top", transform=sp5.transAxes)
        sp6.text(0.9, 0.9, "Metallicity", color="white", ha="right", va="top", transform=sp6.transAxes)
        sp7.text(0.9, 0.9, r"$\Sigma_{\mathrm{HI}}$", color="k", ha="right", va="top", transform=sp7.transAxes)
        sp8.text(0.9, 0.9, r"$\Sigma_{\mathrm{H2}}$", color="k", ha="right", va="top", transform=sp8.transAxes)
        sp9.text(0.9, 0.9, r"$\Sigma_{\mathrm{HII}}$", color="k", ha="right", va="top", transform=sp9.transAxes)
        sp10.text(0.9, 0.9, r"$\Sigma_{\mathrm{*}}$", color="white", ha="right", va="top", transform=sp10.transAxes)
        sp11.text(0.9, 0.9, r"$\Sigma_{\mathrm{gas}}$", color="white", ha="right", va="top", transform=sp11.transAxes)
        sp12.text(0.9, 0.9, r"$\Sigma_{\mathrm{gas,SF}}$", color="white", ha="right", va="top", transform=sp12.transAxes)
        sp13.text(0.9, 0.9, "Velocity", color="white", ha="right", va="top", transform=sp13.transAxes)
        sp14.text(0.9, 0.9, "Temperature", color="white", ha="right", va="top", transform=sp14.transAxes)
        sp15.text(0.9, 0.9, "Metallicity", color="white", ha="right", va="top", transform=sp15.transAxes)
        sp16.text(0.9, 0.9, r"$\Sigma_{\mathrm{HI}}$", color="k", ha="right", va="top", transform=sp16.transAxes)
        sp17.text(0.9, 0.9, r"$\Sigma_{\mathrm{H2}}$", color="k", ha="right", va="top", transform=sp17.transAxes)
        sp18.text(0.9, 0.9, r"$\Sigma_{\mathrm{HII}}$", color="k", ha="right", va="top", transform=sp18.transAxes)
        
    
    
    
    #--------------
    ### Title
    sp2.set_title(f"soap_index={soap_index}, track_id={track_id}")
    
    
    #--------------
    ### other
    fig.subplots_adjust(wspace=0, hspace=0)
    
    
    #--------------
    ### Savefig
    if savefig:
        savefig_txt_save = ('' if savefig_txt_in == None else savefig_txt_in) + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/galaxy_visuals/%s_gasvis_%s_%i_%s.%s" %(fig_dir, run_name, orientation, track_id, savefig_txt_save, file_format), format=file_format, metadata=metadata_plot, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/galaxy_visuals/%s_gasvis_%s_%i_%s.%s" %(fig_dir, run_name, orientation, track_id, savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()




#========================================================================
# Load a sample from a given snapshot
soap_indicies_sample, _, sample_input = _load_soap_sample(sample_dir, csv_sample = 'L0025N0752_THERMAL_AGN_m5_123_sample_5_example_sample')
savefig_txt_in = ''                                                                 # L100_m6_THERMAL_AGN_m6_127_centrals_sample_20_galaxy_visual
                                                                                    # L0025N0752_THERMAL_AGN_m5_123_sample_5_example_sample
#========================================================================


#---------------------
# Visualise example galaxy (stars + gas + DM)
"""sgs = SWIFTGalaxies(sample_input['virtual_snapshot_file'], SOAP(sample_input['soap_catalogue_file'], soap_index=soap_indicies_sample, ), auto_recentre=False,
                    preload={"gas.coordinates", "gas.masses", "gas.velocities", "gas.smoothing_lengths", "dark_matter.coordinates", "dark_matter.masses", "stars.coordinates", "stars.masses", "stars.smoothing_lengths", },)
output_figures = sgs.map(_visualize_galaxy_example)     # Use map to apply `myvis` to each `sg` in `sgs` and give us the results in order.
"""

#---------------------
# Visualise example galaxy L25_m5, L100_m6 (stars + gas + gas_sf + gas_HII + gas_H1 + gas_H2)
sgs = SWIFTGalaxies(sample_input['virtual_snapshot_file'], SOAP(sample_input['soap_catalogue_file'], soap_index=soap_indicies_sample, ), auto_recentre=False,
                    preload={"gas.coordinates", "gas.masses", "gas.velocities", "gas.star_formation_rates", "gas.temperatures", "gas.metal_mass_fractions", "gas.element_mass_fractions.hydrogen", "gas.species_fractions.HII", "gas.species_fractions.HI", "gas.species_fractions.H2", "gas.smoothing_lengths", "stars.coordinates", "stars.masses", "stars.smoothing_lengths", },)       
output_figures = sgs.map(_visualize_galaxy_gas)


    















