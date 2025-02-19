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
data = load(colibre_base_path + simulation_dir + '/' + "SOAP/colibre_with_SOAP_membership_0%s.hdf5"%snapshot_no)
#data = load(colibre_base_path + simulation_dir + '/' + "snapshots/colibre_0%s/colibre_0%s.0.hdf5"%(snapshot_no, snapshot_no))


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
print(boxsize)


"""
An even easier way to convert your properties to physical is to use the built-in to_physical and convert_to_physical methods, as follows:

physical_rho_gas = rho_gas.to_physical()

# Convert in-place
rho_gas.convert_to_physical()
"""



"""
For instance, should you ever need to know what a dataset represents, you can ask for a description:
"""

#print(dir(data.metadata.gas_properties))

for particle_type_i in ('gas', 'stars', 'dark_matter', 'black_holes'):
    print('\n==========================================================')
    print('List of %s particle data:' %particle_type_i)
    
    temp_particle_data = getattr(data.metadata, '%s_properties'%particle_type_i)
    temp_column_data   = getattr(data.metadata, '%s_properties'%particle_type_i).named_columns
    
    for property_i, column_key_i, units_i, property_desc_i in zip(temp_particle_data.field_names, temp_particle_data.named_columns.keys(), temp_particle_data.field_units, temp_particle_data.field_descriptions):
        print('\n\t%s\t%s\t%s' %(property_i, units_i, property_desc_i))
        
        if temp_column_data[column_key_i] != None:
            print('\t\t%s   with following array and indexes:'%column_key_i)
            
            for index_i, species_i in enumerate(temp_column_data[column_key_i]):
                print('\t\t%s\t%s' %(index_i, species_i, ))
        
        
        


#print('\nList of gas properties: ')
#print(data.metadata.gas_properties.field_names)
#print('\nList of stars properties: ')
#print(data.metadata.stars_properties.field_names)
#print('\nList of DM properties: ')
#print(data.metadata.dark_matter_properties.field_names)
#print('\nList of BH properties: ')
#print(data.metadata.black_holes_properties.field_names)


"""
=============================================================
PARTICLE DATA SNAPSHOT:

data.gas, which contains all information about gas particles in the simulation.
data.dark_matter, likewise containing information about the dark matter particles in the simulation.
data.metadata, an instance of swiftsimio.metadata.objects.SWIFTSnapshotMetadata
data.units, an instance of swiftsimio.metadata.objects.SWIFTUnits

List of gas particle data:

	averaged_star_formation_rates		10227144.887961628 Msun/Gyr			Star formation rates of the particles averaged over 
																			the period set by the first two snapshot triggers 
																			Not masked.
																			
	compton_yparameters					1.0 Mpc**2							Compton y parameters in the physical frame computed 
																			based on the cooling tables. This is 0 for star-forming 
																			particles. Not masked.
																			
	coordinates							1.0 Mpc								Co-moving positions of the particles Not masked.

	densities							10000000000.0 Msun/Mpc**3			Co-moving mass densities of the particles Not masked.

	densities_at_last_agnevent			10000000000.0 Msun/Mpc**3			Physical density (not subgrid) of the gas at the last 
																			AGN feedback event that hit the particles. -1 if the 
																			particles have never been heated. Not masked.

	densities_at_last_supernova_event	10000000000.0 Msun/Mpc**3			Physical density (not subgrid) of the gas at the last 
																			SNII thermal feedback event that hit the particles. -1 
																			if the particles have never been heated. Not masked.

	diffusion_parameters				None								Diffusion coefficient (alpha_diff) of the particles 
																			Not masked.

	dust_mass_fractions					None								Fractions of the particles' masses that are in a given 
																			species of dust grain Not masked.
		PartType0/DustMassFractions   with following array and indexes:
		0	GraphiteLarge
		1	MgSilicatesLarge
		2	FeSilicatesLarge
		3	GraphiteSmall
		4	MgSilicatesSmall
		5	FeSilicatesSmall

	element_mass_fractions				None								Fractions of the particles' masses that are in the given 
																			element Not masked.
		PartType0/ElementMassFractions   with following array and indexes:
		0	hydrogen
		1	helium
		2	carbon
		3	nitrogen
		4	oxygen
		5	neon
		6	magnesium
		7	silicon
		8	iron
		9	strontium
		10	barium
		11	europium

	element_mass_fractions_diffuse		None								Fractions of the particles' masses that are in the 
																			given element and in the diffuse (non-dust) phase 
																			Not masked.

	energies_received_from_agnfeedback	10459.449255935968 Mpc**2*Msun/Gyr**2	Total amount of thermal energy from AGN feedback 
																				events received by the particles. Not masked.

	entropies							2.253422093363313e-13 Mpc**4/(Gyr**2*Msun**(2/3))	Co-moving entropies per unit mass of the 
																							particles Not masked.

	fofgroup_ids						None								Friends-Of-Friends ID of the group the particles
																			belong to Not masked.

	group_nr_bound						None								No description available Not masked.

	halo_catalogue_index				None								No description available Not masked.

	internal_energies					1.0459449255935967e-06 Mpc**2/Gyr**2	Co-moving thermal energies per unit mass of the 
																				particles Not masked.

	iron_mass_fractions_from_snia		None								Fractions of the particles' masses that are in iron 
																			produced by SNIa stars (incorporating both depleted 
																			and nebular phases)Fractions of the particles' masses 
																			that are in metals (incorporating both depleted and 
																			nebular phases) Not masked.

	iron_mass_fractions_from_snia_diffuse	None							Fractions of the particles' masses that are in iron 
																			originating from SNIa, but not depleted into dust 
																			Not masked.

	laplacian_internal_energies			1.0459449255935967e-06 Gyr**(-2)	Laplacian (del squared) of the Internal Energy per unit 
																			mass of the particles Not masked.

	last_agnfeedback_scale_factors		None								Scale-factors at which the particles were last hit 
																			by AGN feedback. -1 if a particle has never been 
																			hit by feedback Not masked.

	last_energies_received_from_agnfeedback	10459.449255935968 Mpc**2*Msun/Gyr**2	The energy the particles received the last time 
																					they were heated by AGN feedback. Not masked.

	last_fofhalo_masses					10000000000.0 Msun					Masses of the last FOF haloes the particles where 
																			part of. -1 if the particle has never been in a FOF 
																			group Not masked.

	last_fofhalo_masses_scale_factors	None								Scale-factors at which the particle was last in a FOF 
																			group Not masked.

	last_ismscale_factors				None								Scale-factor at which the particle was part of the ISM 
																			for the last time, i.e. its density was larger than 100 
																			times the mean density and (its neutral fraction larger 
																			than 50% OR be an HII region). -1 if the particle was 
																			never part of the ISM. Not masked.

	last_kinetic_early_feedback_scale_factors	None						Scale-factors at which the particles were last hit by 
																			kinetic early feedback. -1 if a particle has never been 
																			hit by feedback Not masked.

	last_sniikinetic_feedback_scale_factors		None						Scale-factors at which the particles were last hit by 
																			SNII kinetic feedback. -1 if a particle has never been 
																			hit by feedback Not masked.

	last_sniikinetic_feedbackvkick		0.0010227144887961627 Mpc/Gyr		Physical kick velocity the particles were kicked with 
																			at last SNII kinetic feedback event. -1 if a particle 
																			has never been hit by feedback Not masked.

	last_sniithermal_feedback_scale_factors		None						Scale-factors at which the particles were last hit by 
																			SNII thermal feedback. -1 if a particle has never been 
																			hit by feedback Not masked.

	last_snia_thermal_feedback_scale_factors	None						Scale-factors at which the particles were last hit by 
																			SNIa thermal feedback. -1 if a particle has never been 
																			hit by feedback Not masked.

	last_star_formation_scale_factors			None						Scale-factors at which the stars last had a non-zero 
																			star formation rate. Not masked.

	masses								10000000000.0 Msun					Masses of the particles Not masked.

	masses_from_agb						10000000000.0 Msun					Masses of gas that have been produced by AGN stars 
																			Not masked.

	masses_from_cejsn					10000000000.0 Msun					Mass of europium that have been produced by common-envelop 
																			jets SN events Not masked.

	masses_from_collapsar				10000000000.0 Msun					Mass of europium that have been produced by collapsar 
																			events Not masked.

	masses_from_nsm						10000000000.0 Msun					Mass of europium that have been produced by neutron 
																			star merger events Not masked.

	masses_from_snii					10000000000.0 Msun					Masses of gas that have been produced by SNII stars 
																			Not masked.

	masses_from_snia					10000000000.0 Msun					Masses of gas that have been produced by SNIa stars 
																			Not masked.

	maximal_sniikinetic_feedbackvkick	0.0010227144887961627 Mpc/Gyr		Maximal physical kick velocity the particles were kicked 
																			with in SNII kinetic feedback. -1 if a particle has never 
																			been hit by feedback Not masked.

	maximal_temperature_scale_factors	None								Scale-factors at which the maximal temperature was reached 
																			Not masked.

	maximal_temperatures				1.0 K								Maximal temperatures ever reached by the particles 
																			Not masked.

	mean_iron_weighted_redshifts		None								Mean redshift of SNIa events weighted by the iron mass 
																			imparted by each event. -1 if a particle has never been 
																			enriched by SNIa. Not masked.

	mean_metal_weighted_redshifts		None								Mean redshift of enrichment events weighted by the metal 
																			mass imparted by each event. -1 if a particle has never 
																			been enriched. Not masked.

	metal_mass_fractions				None								Fractions of the particles' masses that are in metals 
																			(incorporating both depleted and nebular phases) Not masked.

	metal_mass_fractions_from_agb		None								Fractions of the particles' masses that are in metals 
																			produced by AGB stars (incorporating both depleted and 
																			nebular phases) Not masked.

	metal_mass_fractions_from_snii		None								Fractions of the particles' masses that are in metals 
																			produced by SNII stars (incorporating both depleted and 
																			nebular phases) Not masked.

	metal_mass_fractions_from_snia		None								Fractions of the particles' masses that are in metals 
																			produced by SNIa stars (incorporating both depleted and 
																			nebular phases) Not masked.

	minimal_smoothing_length_scale_factors	None							Scale-factors at which the minimal smoothing length was 
																			reached Not masked.

	minimal_smoothing_lengths			1.0 Mpc								Maximal temperatures ever reached by the particles Not masked.

	particle_ids						None								Unique IDs of the particles Not masked.

	potentials							1.0459449255935967e-06 Mpc**2/Gyr**2	Co-moving gravitational potential at position of the 
																				particles Not masked.

	pressures							10459.449255935968 Msun/(Gyr**2*Mpc)	Co-moving pressures of the particles Not masked.

	progenitor_particle_ids				None								ID of the progenitor of this particle. If this particle 
																			is the result of one (or many) splitting events, this ID 
																			corresponds to the ID of the particle in the initial conditions 
																			that its lineage can be traced back to. If the particle was 
																			never split, this is the same as ParticleIDs. Not masked.

	rank_bound							None								No description available Not masked.

	smoothing_lengths					1.0 Mpc								Co-moving smoothing lengths (FWHM of the kernel) of the 
																			particles Not masked.

	species_fractions					None								Species fractions array for all ions and molecules in the 
																			CHIMES network. The fraction of species i is defined in 
																			terms of its number density relative to hydrogen, i.e. n_i 
																			/ n_H_tot. Not masked.
		PartType0/SpeciesFractions   with following array and indexes:
		0	elec
		1	HI
		2	HII
		3	Hm				To get mass fraction, need:		 mass_h2 = mass_gas_particle * element_mass_fractions[0] * species_fractions[7] * 2
		4	HeI																				(hydrogen)					(molecular)			  (2x H atoms)
		5	HeII
		6	HeIII
		7	H2					
		8	H2p																			
		9	H3p

	split_counts						None								Number of times this particle has been split. Note that 
																			both particles that take part in the splitting have counter 
																			incremented, so the number of splitting events in an entire 
																			simulation is half of the sum of all of these numbers. 
																			Not masked.

	split_trees							None								Binary tree describing splitting events. Particles that keep 
																			the original ID have a value of zero in a splitting event, 
																			whereasparticles given a new ID have a value of one. Not masked.

	star_formation_rates				10227144.887961628 Msun/Gyr			Star formation rates of the particles. Not masked.

	temperatures						1.0 K								Temperature of the particles Not masked.

	total_dust_mass_fractions			None								Fractions of the particles' masses that are in dust (sum 
																			of all grains species) Not masked.

	total_electron_number_densities		1.0 Mpc**(-3)						Total electron number densities in the physical frame 
																			computed as the sum of the electron number densities from 
																			the elements evolved with the non-equilibrium network 
																			CHIMES and the electron number densities from the other 
																			elements based on the equilibrium cooling tables. Not masked.

	velocities							0.0010227144887961627 Mpc/Gyr		Peculiar velocities of the stars. This is (a * dx/dt) where 
																			x is the co-moving positions of the particles Not masked.

	velocity_dispersions				1.0459449255935967e-06 Mpc**2/Gyr**2	Physical velocity dispersions (3D) squared, this is the 
																				velocity dispersion of the total velocity (peculiar velocity 
																				+ Hubble flow, a H x + a (dx/dt) ). Values of the Velocity 
																				dispersion that have the value of FLT_MAX are particles 
																				that do not have neighbours and therefore the velocity 
																				dispersion of these particles cannot be calculated Not masked.

	velocity_divergence_time_differentials	1.0459449255935967e-06 Gyr**(-2)	Time differential (over the previous step) of the velocity 
																				divergence field around the particles. Again, provided 
																				without cosmology as this includes a Hubble flow term. To 
																				get back to a peculiar velocity divergence time differential, 
																				x_pec = a^4 (x - a^{-2} n_D dH / dt) Not masked.

	velocity_divergences					0.0010227144887961627 1/Gyr			Local velocity divergence field around the particles. Provided 
																				without cosmology, as this includes the Hubble flow. To 
																				return to a peculiar velocity divergence, div . v_pec = 
																				a^2 (div . v - n_D H) Not masked.

	viscosity_parameters				None									Visosity coefficient (alpha_visc) of the particles, 
																				multiplied by the balsara switch Not masked.

	xray_luminosities					10.697030298873957 Mpc**2*Msun/Gyr**3	Intrinsic X-ray luminosities in various bands. This is 0 
																				for star-forming particles. Not masked.
		PartType0/XrayLuminosities   with following array and indexes:
		0	erosita_low
		1	erosita_high
		2	ROSAT

	xray_photon_luminosities			0.0010227144887961627 1/Gyr				Intrinsic X-ray photon luminosities in various bands. This 
																				is 0 for star-forming particles. Not masked.
		PartType0/XrayPhotonLuminosities   with following array and indexes:
		0	erosita_low
		1	erosita_high
		2	ROSAT

==========================================================
List of stars particle data:

	ages								977.79 Gyr							Ages of the stars. Not masked.

	averaged_star_formation_rates		10227144.887961628 Msun/Gyr			Star formation rates of the particles averaged over the 
																			period set by the first two snapshot triggers when the particle 
																			was still a gas particle. Not masked.

	birth_densities						10000000000.0 Msun/Mpc**3			Physical densities at the time of birth of the gas particles 
																			that turned into stars (note that we store the physical density 
																			at the birth redshift, no conversion is needed) Not masked.

	birth_scale_factors					None								Scale-factors at which the stars were born Not masked.

	birth_temperatures					1.0 K								Temperatures at the time of birth of the gas particles that
																			turned into stars Not masked.

	birth_velocity_dispersions			1.0459449255935967e-06 Mpc**2/Gyr**2	Physical velocity dispersions (3D) squared at the birth time 
																				of the gas particles that turned into stars, this is the 
																				velocity dispersion of the total velocity (peculiar velocity + 
																				Hubble flow, a H x + a (dx/dt) ). Not masked.

	coordinates							1.0 Mpc								Co-moving position of the particles Not masked.

	element_mass_fractions				None								Fractions of the particles' masses that are in the given 
																			element Not masked.
		PartType4/ElementMassFractions   with following array and indexes:
		0	hydrogen
		1	helium
		2	carbon
		3	nitrogen
		4	oxygen
		5	neon
		6	magnesium
		7	silicon
		8	iron
		9	strontium
		10	barium
		11	europium

	energies_received_from_agnfeedback	10459.449255935968 Mpc**2*Msun/Gyr**2	Total amount of thermal energy from AGN feedback events 
																				received by the particles when the particles were still gas 
																				particles. Not masked.

	fofgroup_ids						None								Friends-Of-Friends ID of the group the particles belong to Not masked.

	group_nr_bound						None								No description available Not masked.

	halo_catalogue_index				None								No description available Not masked.

	initial_masses						10000000000.0 Msun					Masses of the star particles at birth time Not masked.

	iron_mass_fractions_from_snia		None								Fractions of the particles' masses that are in iron 
																			produced by SNIa stars Not masked.

	last_agnfeedback_scale_factors		None								Scale-factors at which the particles were last hit by AGN 
																			feedback when they were still gas particles. -1 if a particle 
																			has never been hit by feedback Not masked.

	last_fofhalo_masses					10000000000.0 Msun					Masses of the last FOF haloes the particles where part of 
																			when they were still a gas particle. -1 if the particle has never 
																			been in a FOF group Not masked.

	last_fofhalo_masses_scale_factors	None								Scale-factors at which the particle was last in a FOF group 
																			when they were still a gas particle Not masked.

	last_kinetic_early_feedback_scale_factors	None						Scale-factors at which the particles were last hit by kinetic 
																			early feedback when they were still gas particles. -1 if a 
																			particle has never been hit by feedback Not masked.

	last_sniikinetic_feedback_scale_factors		None						Scale-factors at which the particles were last hit by SNII kinetic 
																			feedback when they were still gas particles. -1 if a particle has 
																			never been hit by feedback Not masked.

	last_sniithermal_feedback_scale_factors		None						Scale-factors at which the particles were last hit by SNII thermal 
																			feedback when they were still gas particles. -1 if a particle 
																			has never been hit by feedback Not masked.

	last_snia_thermal_feedback_scale_factors	None						Scale-factors at which the particles were last hit by SNIa thermal 
																			feedback when they were still gas particles. -1 if a particle 
																			has never been hit by feedback Not masked.

	luminosities						None								Rest-frame dust-free AB-luminosities of the star particles in 
																			the GAMA bands. These were computed using the BC03 (GALAXEV) 
																			models convolved with different filter bands and interpolated 
																			in log-log (f(log(Z), log(age)) = log(flux)) as used in the 
																			dust-free modelling of Trayford et al. (2015). The luminosities 
																			are given in dimensionless units. They have been divided by 
																			3631 Jy already, i.e. they can be turned into absolute 
																			AB-magnitudes (rest-frame absolute maggies) directly by 
																			applying -2.5 log10(L) without additional corrections. Not masked.
		PartType4/Luminosities   with following array and indexes:
		0	GAMA_u
		1	GAMA_g
		2	GAMA_r
		3	GAMA_i
		4	GAMA_z
		5	GAMA_Y
		6	GAMA_J
		7	GAMA_H
		8	GAMA_K

	masses								10000000000.0 Msun					Masses of the particles at the current point in time 
																			(i.e. after stellar losses) Not masked.

	masses_from_agb						10000000000.0 Msun					Masses of gas that have been produced by AGN stars Not masked.

	masses_from_cejsn					10000000000.0 Msun					Masses of gas that have been produced by common-envelop jets 
																			SN events Not masked.

	masses_from_collapsar				10000000000.0 Msun					Masses of gas that have been produced by collapsar events 
																			Not masked.

	masses_from_nsm						10000000000.0 Msun					Masses of gas that have been produced by neutron star merger 
																			events Not masked.

	masses_from_snii					10000000000.0 Msun					Masses of gas that have been produced by SNII stars Not masked.

	masses_from_snia					10000000000.0 Msun					Masses of gas that have been produced by SNIa stars Not masked.

	maximal_sniikinetic_feedbackvkick	0.0010227144887961627 Mpc/Gyr		Maximal physical kick velocity in SNII kinetic feedback when 
																			the stellar particles were kicked with when they were still gas 
																			particles. -1 if a particle has never been hit by feedback 
																			Not masked.

	maximal_temperature_scale_factors	None								Scale-factors at which the maximal temperature was reached 
																			Not masked.

	maximal_temperatures				1.0 K								Maximal temperatures ever reached by the particles before they 
																			got converted to stars Not masked.

	metal_mass_fractions				None								Fractions of the particles' masses that are in metals Not masked.

	metal_mass_fractions_from_agb		None								Fractions of the particles' masses that are in metals produced 
																			by AGB stars Not masked.

	metal_mass_fractions_from_snii		None								Fractions of the particles' masses that are in metals produced 
																			by SNII stars Not masked.

	metal_mass_fractions_from_snia		None								Fractions of the particles' masses that are in metals produced 
																			by SNIa stars Not masked.

	particle_ids						None								Unique ID of the particles Not masked.

	potentials							1.0459449255935967e-06 Mpc**2/Gyr**2	Gravitational potentials of the particles Not masked.

	progenitor_particle_ids				None								Progenitor ID of the gas particle that became this star. If this 
																			particle is the result of one (or many) splitting events, this ID 
																			corresponds to the ID of the particle in the initial conditions 
																			that its lineage can be traced back to. If the particle was 
																			never split, this is the same as ParticleIDs. Not masked.

	rank_bound							None								No description available Not masked.

	sniifeedback_energy_fractions		None								Fractions of the canonical feedback energy that was used 
																			for the stars' SNII feedback events Not masked.

	snia_rates							0.0010227144887961627 1/Gyr			SNIa rate averaged over the last enrichment timestep Not masked.

	smoothing_lengths					1.0 Mpc								Co-moving smoothing lengths (FWHM of the kernel) of the particles 
																			Not masked.

	split_counts						None								Number of times the gas particle that turned into this star particle 
																			was split. Note that both particles that take part in the splitting 
																			have this counter incremented, so the number of splitting events in 
																			an entire simulation is half of the sum of all of these numbers. 
																			Not masked.

	split_trees							None								Binary tree describing splitting events. Particles that keep the 
																			original ID have a value of zero in a splitting event, whereas 
																			particles given a new ID have a value of one. Not masked.

	velocities							0.0010227144887961627 Mpc/Gyr		Peculiar velocities of the particles. This is a * dx/dt where x 
																			is the co-moving position of the particles. Not masked.

==========================================================
List of dark_matter particle data:

	coordinates							1.0 Mpc								Co-moving position of the particles Not masked.

	fofgroup_ids						None								Friends-Of-Friends ID of the group the particles belong to 
																			Not masked.

	group_nr_bound						None								No description available Not masked.

	halo_catalogue_index				None								No description available Not masked.

	masses								10000000000.0 Msun					Masses of the particles Not masked.

	particle_ids						None								Unique ID of the particles Not masked.

	potentials							1.0459449255935967e-06 Mpc**2/Gyr**2	Gravitational potentials of the particles Not masked.

	rank_bound							None								No description available Not masked.

	velocities							0.0010227144887961627 Mpc/Gyr		Peculiar velocities of the particles. This is a * dx/dt where 
																			x is the co-moving position of the particles. Not masked.

==========================================================
List of black_holes particle data:

	agntotal_injected_energies			10459.449255935968 Mpc**2*Msun/Gyr**2	Total (cumulative) physical energies injected into gas particles 
																				in AGN feedback. Not masked.

	accreted_angular_momenta			10227144.887961628 Mpc**2*Msun/Gyr		Physical angular momenta that the black holes have accumulated 
																				through subgrid accretion. Not masked.

	accretion_boost_factors				None									Multiplicative factors by which the Bondi-Hoyle-Lyttleton 
																				accretion rates have been increased by the density-dependent 
																				Booth & Schaye (2009) accretion model. Not masked.

	accretion_limited_time_steps		977.79 Gyr								Accretion-limited time steps of black holes. The actual time 
																				step of the particles may differ due to the minimum allowed 
																				value. Not masked.

	accretion_rates						10227144.887961628 Msun/Gyr				Physical instantaneous accretion rates of the particles Not masked.

	averaged_accretion_rates			10227144.887961628 Msun/Gyr				Accretion rates of the black holes averaged over the period set 
																				by the first two snapshot triggers Not masked.

	coordinates							1.0 Mpc									Co-moving position of the particles Not masked.

	cumulative_number_of_seeds			None									Total number of BH seeds that have merged into this black 
																				hole Not masked.

	dust_masses							10000000000.0 Msun						Mass contents of the BH particles in a given dust grain species 
																				Not masked.

	dynamical_masses					10000000000.0 Msun						Dynamical masses of the particles Not masked.

	eddington_fractions					None									Accretion rates of black holes in units of their Eddington rates. 
																				This is based on the unlimited accretion rates, so these fractions 
																				can be above the limiting fEdd. Not masked.

	element_masses						10000000000.0 Msun						Mass contents of the BH particles in a given element Not masked.

	element_masses_diffuse				10000000000.0 Msun						Masses of the BH that are in the given element and were in the 
																				gas phase when accreted Not masked.

	energy_reservoirs					10459.449255935968 Mpc**2*Msun/Gyr**2	Physcial energy contained in the feedback reservoir of the 
																				particles Not masked.

	fofgroup_ids						None									Friends-Of-Friends ID of the group the particles belong to Not masked.

	formation_scale_factors				None									Scale-factors at which the BHs were formed Not masked.

	gwmass_losses						10000000000.0 Msun						Cumulative masses lost to GW via BH-BH mergers over the history of 
																				the black holes. This includes the mass loss from all the progenitors. 
																				Not masked.

	gas_circular_velocities				0.0010227144887961627 Mpc/Gyr			Circular velocities of the gas around the black hole at the smoothing 
																				radius. This is j / h_BH, where j is the smoothed, peculiar specific 
																				angular momentum of gas around the black holes, and h_BH is the smoothing 
																				length of each black hole. Not masked.

	gas_curl_velocities					0.0010227144887961627 Mpc/Gyr			Velocity curl (3D) of the gas particles around the black holes. Not masked.

	gas_densities						10000000000.0 Msun/Mpc**3				Co-moving densities of the gas around the particles Not masked.

	gas_relative_velocities				0.0010227144887961627 Mpc/Gyr			Peculiar relative velocities of the gas particles around the black holes. 
																				This is a * dx/dt where x is the co-moving position of the particles. 
																				Not masked.

	gas_sound_speeds					0.0010227144887961627 Mpc/Gyr			Co-moving sound-speeds of the gas around the particles Not masked.

	gas_temperatures					1.0 K									Temperature of the gas surrounding the black holes. Not masked.

	gas_velocity_dispersions			0.0010227144887961627 Mpc/Gyr			Velocity dispersion (3D) of the gas particles around the black holes. 
																				This is a * sqrt(<|dx/dt|^2> - <|dx/dt|>^2) where x is the co-moving 
																				position of the particles relative to the black holes. Not masked.

	group_nr_bound						None									No description available Not masked.

	halo_catalogue_index				None									No description available Not masked.

	iron_masses_from_snia				10000000000.0 Msun						Masses of the BH particles in iron that have been produced by SNIa 
																				stars Not masked.

	iron_masses_from_snia_diffuse		10000000000.0 Msun						Masses of the BH particles in iron that have been produced by SNIa 
																				stars and were in the gas phase when accreted Not masked.

	last_agnfeedback_scale_factors		None									Scale-factors at which the black holes last had an AGN event. 
																				Not masked.

	last_high_eddington_fraction_scale_factors	None							Scale-factors at which the black holes last reached a large Eddington 
																				ratio. -1 if never reached. Not masked.

	last_major_merger_scale_factors		None									Scale-factors at which the black holes last had a major merger. 
																				Not masked.

	last_minor_merger_scale_factors		None									Scale-factors at which the black holes last had a minor merger. 
																				Not masked.

	last_reposition_velocities			0.0010227144887961627 Mpc/Gyr			Physical speeds at which the black holes repositioned most recently. 
																				This is 0 for black holes that have never repositioned, or if the 
																				simulation has been run without prescribed repositioning speed.
																				Not masked.

	masses_from_agb						10000000000.0 Msun						Masses of the BH particles that have been produced by AGB stars 
																				Not masked.

	masses_from_cejsn					10000000000.0 Msun						Masses of the BH particles in europium that have been produced by 
																				common-envelop jets SN events Not masked.

	masses_from_collapsar				10000000000.0 Msun						Masses of the BH particles in europium that have been produced by 
																				collapsar events Not masked.

	masses_from_nsm						10000000000.0 Msun						Masses of the BH particles in europium that have been produced by 
																				neutron star merger events Not masked.

	masses_from_snii					10000000000.0 Msun						Masses of the BH particles that have been produced by SNII stars 
																				Not masked.

	masses_from_snia					10000000000.0 Msun						Masses of the BH particles that have been produced by SNIa stars 
																				Not masked.

	metal_masses						10000000000.0 Msun						Mass contents of the BH particles in a metals Not masked.

	metal_masses_from_agb				10000000000.0 Msun						Masses of the BH particles in metals that have been produced by AGB stars 
																				Not masked.

	metal_masses_from_snii				10000000000.0 Msun						Masses of the BH particles in metals that have been produced by SNII stars 
																				Not masked.

	metal_masses_from_snia				10000000000.0 Msun						Masses of the BH particles in metals that have been produced by SNIa stars 
																				Not masked.

	minimal_time_bin_scale_factors		None									Scale-factors at which the minimal time-bin was reached Not masked.

	minimal_time_bins					None									Minimal Time Bins reached by the black holes Not masked.

	number_of_agnevents					None									Integer number of AGN events the black hole has had so far (the number 
																				of time steps in which the BH did AGN feedback). Not masked.

	number_of_gas_neighbours			None									Integer number of gas neighbour particles within the black hole kernels. 
																				Not masked.

	number_of_heating_events			None									Integer number of (thermal) energy injections the black hole has had so 
																				far. This counts each heated gas particle separately, and so can increase 
																				by more than one during a single time step. Not masked.

	number_of_mergers					None									Number of mergers the black holes went through. This does not include the 
																				number of mergers accumulated by any merged black hole. Not masked.

	number_of_repositions				None									Number of repositioning events the black holes went through. This does 
																				not include the number of reposition events accumulated by any merged 
																				black holes. Not masked.

	number_of_time_steps				None									Total number of time steps at which the black holes were active. Not masked.

	particle_ids						None									Unique ID of the particles Not masked.

	potentials							1.0459449255935967e-06 Mpc**2/Gyr**2	Gravitational potentials of the particles Not masked.

	progenitor_particle_ids				None									Progenitor ID of the gas particle that became the seed BH. If this particle 
																				is the result of one (or many) splitting events, this ID corresponds to the 
																				ID of the particle in the initial conditions that its lineage can be traced 
																				back to. If the particle was never split, this is the same as ParticleIDs. 
																				Not masked.

	rank_bound							None									No description available Not masked.

	smoothing_lengths					1.0 Mpc									Co-moving smoothing lengths (FWHM of the kernel) of the particles Not masked.

	split_counts						None									Number of times the gas particle that became this BH seed was split. Note 
																				that both particles that take part in the splitting have this counter 
																				incremented, so the number of splitting events in an entire simulation is 
																				half of the sum of all of these numbers. Not masked.

	split_trees							None									Binary tree describing splitting events prior to BH seeding. Particles that 
																				keep the original ID have a value of zero in a splitting event, whereas 
																				particles given a new ID have a value of one. Not masked.

	subgrid_masses						10000000000.0 Msun						Subgrid masses of the particles Not masked.

	swallowed_angular_momenta			10227144.887961628 Mpc**2*Msun/Gyr		Physical angular momenta that the black holes have accumulated by swallowing 
																				gas particles. Not masked.

	total_accreted_masses				10000000000.0 Msun						Total mass accreted onto the main progenitor of the black holes since birth. 
																				This does not include any mass accreted onto any merged black holes. Not masked.

	velocities							0.0010227144887961627 Mpc/Gyr			Peculiar velocities of the particles. This is a * dx/dt where x is the co-moving 
																				position of the particles. Not masked.
"""















