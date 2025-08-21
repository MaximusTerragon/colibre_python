import h5py
import numpy as np
import pandas as pd
import math
from read_dataset_directories_colibre import _assign_directories

#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================


"""
Online resource: https://plotdigitizer.com/app

"""


#--------------
# GSMF
def _create_Driver2022_new():
    # Create and write
    with h5py.File("%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metdata")
        grp.attrs["bibcode"]      = "2022MNRAS.513..439D"
        grp.attrs["citation"]     = "Driver et al. (2022) (GAMA-DR4)"
        grp.attrs["comment"]      = "Data obtained assuming a Chabrier IMF and h = 0.7. h-corrected for ** COLIBRE by me ** using cosmology: h=0.681, and includes correction down to z=0. Ignoring the mass bins for which GAMA is systematically incomplete."
        grp.attrs["name"]         = "GSMF from GAMA-DR4"
        grp.attrs["plot_as"]      = "points" 
        grp.attrs["redshift"]     = 0.0
        grp.attrs["redshift_lower"] = 0.0
        grp.attrs["redshift_upper"] = 0.0
        
        # Creating cosmology metadata
        grp = f.create_group("cosmology")
        grp.attrs["H0"]         = 68.1
        grp.attrs["Neff"]       = 3.046
        grp.attrs["Ob0"]        = 0.0486
        grp.attrs["Ode0"]       = 0.693922
        grp.attrs["Om0"]        = 0.306
        grp.attrs["Tcmb0"]      = 2.7255
        grp.attrs["m_nu"]       = np.array([0.0, 0.0, 0.0])
        grp.attrs["m_nu_units"] = 'eV'
        grp.attrs["name"]       = 'Abbott22'
        
        #-------------------------------
        # Creating dataset + cosmology
        
        #-----------
        # y-values: total
        grp = f.create_group("data/total/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/total/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.arange(6.875, 11.626, 0.250)
        log_y      = 0.0807 + np.array([-0.691, -1.084, -1.011, -1.349, -1.287, -1.544, -1.669, -1.688, -1.795, -1.886, -2.055, -2.142, -2.219, -2.274, -2.292, -2.361, -2.561, -2.922, -3.414, -4.704])
        log_y_scat          = np.array([ 0.176,  0.125,  0.071,  0.092,  0.079,  0.071,  0.045,  0.032,  0.024,  0.020,  0.014,  0.010,  0.009,  0.009,  0.009,  0.010,  0.013,  0.019,  0.032,  0.138])
        lower = (10**log_y) - (10**(log_y - log_y_scat))
        upper = (10**(log_y + log_y_scat)) - (10**log_y)
        y_scat = np.stack((lower, upper), axis=0)
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = (10**log_y) * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/total/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/total/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/total/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: early type morphology
        grp = f.create_group("data/etg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/etg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF) of galaxies with early-type morphology'
        
                # Data
        log_x      = np.arange(8.625, 11.626, 0.250)
        log_y      = 0.0866 + np.array([-4.022, -3.157, -3.334, -3.491, -3.323, -3.202, -3.077, -3.008, -2.992, -3.074, -3.164, -3.483, -4.623])
        log_y_scat          = np.array([ 0.301,  0.097,  0.087,  0.075,  0.041,  0.035,  0.031,  0.029,  0.029,  0.031,  0.035,  0.048,  0.146])
        lower = (10**log_y) - (10**(log_y - log_y_scat))
        upper = (10**(log_y + log_y_scat)) - (10**log_y)
        y_scat = np.stack((lower, upper), axis=0)
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = (10**log_y) * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/etg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/etg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/etg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: disky morphology
        grp = f.create_group("data/ltg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/ltg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF) of galaxies with discy morphology'
        
        # Data
        log_x      = np.arange(6.625, 11.126, 0.250)
        log_y      = 0.0866 + np.array([-1.080, -1.098, -0.938, -0.838, -1.348, -1.591, -1.919, -1.837, -1.878, -2.129, -2.222, -2.401, -2.537, -2.824, -3.039, -3.318, -3.766, -4.322, -4.845])
        log_y_scat          = np.array([ 0.301,  0.222,  0.114,  0.077,  0.089,  0.074,  0.071,  0.043,  0.030,  0.026,  0.019,  0.016,  0.017,  0.024,  0.031,  0.041,  0.067,  0.114,  0.222])
        lower = (10**log_y) - (10**(log_y - log_y_scat))
        upper = (10**(log_y + log_y_scat)) - (10**log_y)
        y_scat = np.stack((lower, upper), axis=0)
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = (10**log_y) * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/ltg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/ltg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/ltg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        print("Successfully created: %s/GalaxyStellarMassFunction/Driver2022_complete.hdf5"%obs_dir)
#--------------
def _create_Kelvin2014():
    # Create and write
    with h5py.File("%s/GalaxyStellarMassFunction/Kelvin2014.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = "2014MNRAS.444.1647K"
        grp.attrs["citation"]     = "Kelvin et al. (2014) (GAMA-DR1)"
        grp.attrs["comment"]      = "Data obtained assuming a Chabrier IMF and h = 0.7. h-corrected for ** COLIBRE by me ** using cosmology: h=0.681, and includes correction down to z=0. Ignoring the mass bins for which GAMA is systematically incomplete."
        grp.attrs["name"]         = "GSMF from GAMA-DR1"
        grp.attrs["plot_as"]      = "points" 
        grp.attrs["redshift"]     = 0.0
        grp.attrs["redshift_lower"] = 0.0
        grp.attrs["redshift_upper"] = 0.0
        
        # Creating cosmology metadata
        grp = f.create_group("cosmology")
        grp.attrs["H0"]         = 68.1
        grp.attrs["Neff"]       = 3.046
        grp.attrs["Ob0"]        = 0.0486
        grp.attrs["Ode0"]       = 0.693922
        grp.attrs["Om0"]        = 0.306
        grp.attrs["Tcmb0"]      = 2.7255
        grp.attrs["m_nu"]       = np.array([0.0, 0.0, 0.0])
        grp.attrs["m_nu_units"] = 'eV'
        grp.attrs["name"]       = 'Abbott22'
        
        #-------------------------------
        # Creating dataset + cosmology
        
        
        #-----------
        # y-values: ETG plus little blue spheroids
        grp = f.create_group("data/etg_plus_lbs/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass of ETGs including little blue spheroids'
        grp = f.create_group("data/etg_plus_lbs/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.2523, 8.3505, 8.4518, 8.5506, 8.6493, 8.7491, 8.8493, 8.9506, 9.0479, 9.1507, 9.2519, 9.3496, 9.4524, 9.5514, 9.6517, 9.7520, 9.8524, 9.9517, 10.0518, 10.1513, 10.2521, 10.3525, 10.4537, 10.5521, 10.6540, 10.7534, 10.8511, 10.9521, 11.0534, 11.1546, 11.2525, 11.3550, 11.4533])
        y          = np.array([4.7614e-05, 0.00034391, 0.00054860, 0.0010827, 0.0015224, 0.0015757, 0.0019260, 0.0021491, 0.0022389, 0.0019188, 0.0016852, 0.0022021, 0.0019533, 0.0020185, 0.0018772, 0.0019037, 0.0024307, 0.0027640, 0.0026351, 0.0026703, 0.0032626, 0.0033716, 0.0046404, 0.0036393, 0.0030433, 0.0021983, 0.0022464, 0.0016224, 0.0014129, 0.00066731, 0.00047775, 0.00023947, 4.7918e-05])
        y_upper_y    = np.array([9.5018e-05, 0.00047325, 0.00070355, 0.0013023, 0.0018041, 0.0018358, 0.0022173, 0.0025139, 0.0025730, 0.0022294, 0.0019721, 0.0025234, 0.0022404, 0.0023314, 0.0021730, 0.0021831, 0.0027511, 0.0031754, 0.0029976, 0.0030709, 0.0036262, 0.0037502, 0.0051079, 0.0039939, 0.0034393, 0.0025623, 0.0025874, 0.0019495, 0.0016609, 0.00087361, 0.00062822, 0.00034298, 9.4481e-05])
        y_lower_y    = np.array([2.4423e-05, 0.00021307, 0.00039134, 0.00085652, 0.0012481, 0.0013048, 0.0016248, 0.0018147, 0.0019031, 0.0016160, 0.0013954, 0.0018676, 0.0016340, 0.0017047, 0.0015680, 0.0015905, 0.0020833, 0.0023863, 0.0022796, 0.0022733, 0.0028755, 0.0029349, 0.0041808, 0.0032111, 0.0026305, 0.0018599, 0.0019042, 0.0013437, 0.0011991, 0.00048147, 0.00032379, 0.00013043, 2.2605e-05])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/etg_plus_lbs/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/etg_plus_lbs/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/etg_plus_lbs/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        
        #-----------
        # y-values: LTG 
        grp = f.create_group("data/ltg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass of disc-dominated galaxies'
        grp = f.create_group("data/ltg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.0495, 8.1489, 8.2497, 8.3512, 8.4506, 8.5494, 8.6521, 8.7499, 8.8512, 8.9507, 9.0508, 9.1533, 9.2494, 9.3503, 9.4517, 9.5502, 9.6511, 9.7527, 9.8524, 9.9514, 10.0545, 10.1517, 10.2520, 10.3513, 10.4530, 10.5531, 10.6527, 10.7531, 10.8526, 10.9531, 11.0527])
        y          = np.array([0.00011535, 0.00021782, 0.00080060, 0.0012954, 0.0041227, 0.0056334, 0.0069398, 0.0084666, 0.0095890, 0.0091491, 0.0086275, 0.0082589, 0.0087465, 0.0072058, 0.0062965, 0.0062897, 0.0044314, 0.0046205, 0.0039598, 0.0035704, 0.0026603, 0.0030454, 0.0029052, 0.0021108, 0.0016286, 0.0010511, 0.0012986, 0.00066833, 0.00038625, 0.00038416, 0.00019172])
        y_upper_y    = np.array([0.00018769, 0.00031851, 0.00099362, 0.0015315, 0.0047608, 0.0062231, 0.0074579, 0.0089904, 0.010219, 0.0097741, 0.0093353, 0.0089462, 0.0095706, 0.0077410, 0.0067943, 0.0068208, 0.0048637, 0.0051001, 0.0043370, 0.0039532, 0.0030014, 0.0034391, 0.0033716, 0.0023930, 0.0018875, 0.0012730, 0.0015386, 0.00084945, 0.00050665, 0.00051488, 0.00028562])
        y_lower_y    = np.array([4.0691e-05, 0.00011566, 0.00060044, 0.0010441, 0.0036339, 0.0051065, 0.0062941, 0.0077437, 0.0088650, 0.0083148, 0.0079424, 0.0075971, 0.0080321, 0.0065633, 0.0056980, 0.0056666, 0.0039512, 0.0041355, 0.0034714, 0.0031613, 0.0023318, 0.0026490, 0.0025314, 0.0017899, 0.0013390, 0.00082654, 0.0010384, 0.00052017, 0.00024769, 0.00024625, 9.4878e-05])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/ltg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/ltg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/ltg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: total 
        grp = f.create_group("data/total/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/total/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.0470, 8.1491, 8.2490, 8.3511, 8.4489, 8.5490, 8.6497, 8.7494, 8.8495, 8.9488, 9.0482, 9.1496, 9.2499, 9.3499, 9.4512, 9.5504, 9.6501, 9.7505, 9.8533, 9.9514, 10.0517, 10.1519, 10.2525, 10.3526, 10.4542, 10.5513, 10.6540, 10.7512, 10.8504, 10.9498, 11.0533, 11.1509, 11.2533, 11.3474, 11.4485])
        y          = np.array([0.00011617, 0.00026878, 0.00085626, 0.0016338, 0.0046905, 0.0067443, 0.0085648, 0.010060, 0.011574, 0.011607, 0.010801, 0.010342, 0.010606, 0.0093815, 0.0082090, 0.0083403, 0.0064105, 0.0066110, 0.0063195, 0.0065175, 0.0053107, 0.0058054, 0.0062167, 0.0055491, 0.0063919, 0.0047088, 0.0043853, 0.0028315, 0.0026247, 0.0019791, 0.0016315, 0.00067091, 0.00046248, 0.00024150, 0.00004772])
        y_upper_y    = np.array([0.00018322, 0.00037372, 0.0010433, 0.0019014, 0.0051565, 0.0072732, 0.0090326, 0.010648, 0.012217, 0.012605, 0.011802, 0.011093, 0.011429, 0.010161, 0.0087980, 0.0089211, 0.0069453, 0.0070515, 0.0069286, 0.0069566, 0.0058039, 0.0062767, 0.0067274, 0.0059896, 0.0069337, 0.0051319, 0.0048089, 0.0032065, 0.0029488, 0.0022957, 0.0019029, 0.00087305, 0.00062630, 0.00033668, 0.00009515])
        y_lower_y    = np.array([4.2482e-05, 0.00015200, 0.00064868, 0.0012123, 0.0041012, 0.0057783, 0.0077289, 0.0094501, 0.010847, 0.010695, 0.0099878, 0.0095790, 0.0097726, 0.0086881, 0.0077060, 0.0077621, 0.0057689, 0.0060122, 0.0058301, 0.0059070, 0.0047600, 0.0054678, 0.0056566, 0.0049646, 0.0057505, 0.0041795, 0.0039424, 0.0024198, 0.0023656, 0.0016680, 0.0013929, 0.00050894, 0.00033185, 0.00013281, 2.0211e-05])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/total/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/total/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/total/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: total 
        grp = f.create_group("data/etg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/etg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.8451, 8.9478, 9.0477, 9.1474, 9.2477, 9.3463, 9.4469, 9.5482, 9.6489, 9.7469, 9.8484, 9.9507, 10.049, 10.151, 10.252, 10.352, 10.453, 10.552, 10.651, 10.753, 10.854, 10.954, 11.056, 11.154, 11.251, 11.355, 11.457])
        y          = np.array([0.00014470, 0.00024172, 0.00036293, 0.00049595, 0.0011275, 0.0016990, 0.0017333, 0.0014795, 0.0017604, 0.0018388, 0.0023696, 0.0027501, 0.0026527, 0.0027129, 0.0032973, 0.0033910, 0.0047480, 0.0036921, 0.0030915, 0.0022234, 0.0022732, 0.0016449, 0.0014489, 0.00067314, 0.0004840, 0.00024277, 0.00004702])
        y_upper_y    = np.array([0.0002293, 0.00034876, 0.00049489, 0.00064819, 0.0013558, 0.0019876, 0.0020254, 0.0017527, 0.0020428, 0.0021423, 0.0026844, 0.0031016, 0.0030763, 0.0031334, 0.0036669, 0.0038131, 0.0052646, 0.0040964, 0.0034813, 0.0025892, 0.0026528, 0.0019701, 0.0017681, 0.00085149, 0.0006308, 0.00034815, 0.00008841])
        y_lower_y    = np.array([6.2559e-05, 0.00013650, 0.00023522, 0.00034921, 0.00091988, 0.0014446, 0.0014607, 0.0012255, 0.0015012, 0.0015498, 0.0020655, 0.0023973, 0.0023289, 0.0023356, 0.0028818, 0.0030044, 0.0042675, 0.0031957, 0.0027057, 0.0019102, 0.0019568, 0.0013832, 0.0012533, 0.00050150, 0.00034036, 0.00013583, 0.000019])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/etg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/etg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/etg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        print("Successfully created: %s/GalaxyStellarMassFunction/Kelvin2014.hdf5"%obs_dir)
              

#--------------
# H1 mass func and mass fraction func
def _create_Lagos2014_H1():
    # Create and write
    with h5py.File("%s/GalaxyHIMassFunction/Lagos2014_H1.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = "2014MNRAS.443.1002L"
        grp.attrs["citation"]     = "Lagos et al. (2014)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed."
        grp.attrs["name"]         = "HI mass function and mass function fraction from ATLAS3D"
        grp.attrs["plot_as"]      = "points" 
        grp.attrs["redshift"]     = 0.0
        grp.attrs["redshift_lower"] = 0.0
        grp.attrs["redshift_upper"] = 0.0
        
        # Creating cosmology metadata
        grp = f.create_group("cosmology")
        grp.attrs["H0"]         = 68.1
        grp.attrs["Neff"]       = 3.046
        grp.attrs["Ob0"]        = 0.0486
        grp.attrs["Ode0"]       = 0.693922
        grp.attrs["Om0"]        = 0.306
        grp.attrs["Tcmb0"]      = 2.7255
        grp.attrs["m_nu"]       = np.array([0.0, 0.0, 0.0])
        grp.attrs["m_nu_units"] = 'eV'
        grp.attrs["name"]       = 'Abbott22'
        
        #-------------------------------
        # Creating dataset + cosmology
        
        
        #-----------
        # y-values: ATLAS3D 1/V corrected
        grp = f.create_group("data/massfunc/1Vcorr/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass function 1/V corrected'
        grp = f.create_group("data/massfunc/1Vcorr/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = 
        y          = 
        y_upper_y    = 
        y_lower_y    = 
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
            
        # Create dataset
        dset = f.create_dataset("data/massfunc/1Vcorr/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        
        
        
        print("Successfully created: %s/GalaxyHIMassFunction/Lagos2014_H1.hdf5"%obs_dir)




#pd.read_csv('file.csv', delimiter=' ', delim_whitespace=True)

      
#================================   
# Run: 

#_create_Driver2022_new()
#_create_Kelvin2014()
        
_create_Lagos2014_H1()
_create_Lagos2014_H2()
        
        
        
        
        

