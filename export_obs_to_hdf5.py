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
        grp.attrs["name"]         = "HI mass function and mass function fraction from ATLAS3D ready to be plotted"
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
        # y-values: ATLAS3D uncorrected mass function
        grp = f.create_group("data/massfunc/uncorr/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass function uncorrected'
        grp = f.create_group("data/massfunc/uncorr/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (H1MF)'
        
        # Data
        log_x      = np.array([6.3705, 6.6187, 6.8696, 7.1201, 7.3697, 7.6227, 7.8690, 8.1228, 8.3732, 8.6220, 8.8718, 9.1232, 9.3746, 9.6232, 9.8744])
        x          = 10**log_x
        y            = 10**np.array([-3.8819, -3.8704, -3.1764, -3.8765, -3.8680, -3.5780, -3.1822, -3.1736, -3.0329, -3.0392, -3.1097, -3.4006, -3.0342, -3.8781, -3.8833])
        y_upper_y    = 10**np.array([-3.4503, -3.4486, -2.9884, -3.4476, -3.4460, -3.2750, -2.9915, -2.9947, -2.8776, -2.8750, -2.9285, -3.1580, -2.8804, -3.4530, -3.4486])
        y_lower_y    = 10**np.array([-4.3139, -4.3038, -3.3728, -4.3061, -4.3084, -3.8790, -3.3820, -3.3700, -3.2017, -3.1992, -3.2821, -3.6550, -3.2023, -4.3141, -4.3190])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Create dataset
        dset = f.create_dataset("data/massfunc/uncorr/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/massfunc/uncorr/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfunc/uncorr/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: ATLAS3D 1/V corrected mass function
        grp = f.create_group("data/massfunc/1Vcorr/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass function 1/V corrected'
        grp = f.create_group("data/massfunc/1Vcorr/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (H1MF)'
        
        # Data
        log_x      = np.array([6.3705, 6.6187, 6.8696, 7.1201, 7.3697, 7.6227, 7.8690, 8.1228, 8.3732, 8.6220, 8.8718, 9.1232, 9.3746, 9.6232, 9.8744])
        x          = 10**log_x
        y            = 10**np.array([-1.9976, -2.3729, -2.0529, -3.1306, -3.5021, -3.5780, -3.1822, -3.1736, -3.0329, -3.0392, -3.1097, -3.4006, -3.0342, -3.8781, -3.8833])
        y_upper_y    = 10**np.array([-1.5712, -1.9505, -1.8618, -2.6953, -3.0720, -3.2750, -2.9915, -2.9947, -2.8776, -2.8750, -2.9285, -3.1580, -2.8804, -3.4530, -3.4486])
        y_lower_y    = 10**np.array([-2.4372, -2.8106, -2.2528, -3.5604, -3.9403, -3.8790, -3.3820, -3.3700, -3.2017, -3.1992, -3.2821, -3.6550, -3.2023, -4.3141, -4.3190])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Create dataset
        dset = f.create_dataset("data/massfunc/1Vcorr/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: ATLAS3D non-1/V corrected mass FRACTION function
        grp = f.create_group("data/massfracfunc/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass fraction, not 1/V corrected, taken from HI/L_K distribution with conversions already applied'
        grp = f.create_group("data/massfracfunc/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass fraction function, not 1/V corrected, taken from HI/L_K distribution function with conversions already applied'
        
        # Data
        log_x      = np.array([-4.1236, -3.8745, -3.6268, -3.3784, -3.1290, -2.8775, -2.6269, -2.3756, -2.1283, -1.8792, -1.6309, -1.3787, -1.1305, -0.88018, -0.37902])
        x          = 10**log_x
        y            = 10**np.array([-3.8781, -3.5823, -3.2749, -3.3948, -3.5706, -3.4017, -3.1747, -3.4025, -3.1708, -3.0969, -2.9185, -3.1816, -3.5793, -3.5786, -3.8797])
        y_upper_y    = 10**np.array([-3.4462, -3.2696, -3.0628, -3.1508, -3.2674, -3.1495, -2.9833, -3.1469, -2.9889, -2.9224, -2.7776, -2.9817, -3.2694, -3.2704, -3.4409])
        y_lower_y    = 10**np.array([-4.3150, -3.8849, -3.4909, -3.6460, -3.8802, -3.6419, -3.3684, -3.6461, -3.3737, -3.2766, -3.0663, -3.3711, -3.8790, -3.8817, -4.3111])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Unit conversions -> given as M*/L -> * L/M* to get M*/M* | mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.82 M*/L* for ETGs
        x      = x * (1/0.82)
        #y      = 
        #y_scat = 
        
        # Create dataset
        dset = f.create_dataset("data/massfracfunc/x/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        dset = f.create_dataset("data/massfracfunc/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfracfunc/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        print("Successfully created: %s/GalaxyHIMassFunction/Lagos2014_H1.hdf5"%obs_dir)
# H1 mass func and mass fraction func
def _create_Lagos2014_H2():
    # Create and write
    with h5py.File("%s/GalaxyH2MassFunction/Lagos2014_H2.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = "2014MNRAS.443.1002L"
        grp.attrs["citation"]     = "Lagos et al. (2014)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed."
        grp.attrs["name"]         = "H2 mass function and mass function fraction from ATLAS3D ready to be plotted, divided by 1.36 to remove He"
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
        # y-values: ATLAS3D uncorrected mass function
        grp = f.create_group("data/massfunc/uncorr/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'H2 mass function uncorrected, divided by 1.36 for He'
        grp = f.create_group("data/massfunc/uncorr/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (H2MF)'
        
        # Data
        log_x      = np.array([7.0929, 7.2954, 7.4958, 7.6934, 7.8960, 8.0955, 8.2971, 8.4990, 8.6996, 8.8991, 9.0979, 9.2980])
        x          = 10**log_x
        y            = 10**np.array([-3.8958, -2.8994, -3.4183, -3.2996, -3.1221, -2.9406, -2.8971, -3.0543, -2.9409, -3.8937, -3.5951, -3.8995])
        y_upper_y    = 10**np.array([-3.4624, -2.7557, -3.1654, -3.0729, -2.9348, -2.7971, -2.7642, -2.8928, -2.8028, -3.4681, -3.2932, -3.4711])
        y_lower_y    = 10**np.array([-4.3361, -3.0333, -3.6647, -3.5079, -3.2934, -3.0813, -3.0370, -3.2202, -3.0873, -4.3301, -3.9023, -4.3271])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Unit conversions -> given as M*/L -> * L/M* to get M*/M* | mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.5 M*/L* for ETGs, divided by 1.36 for He
        x      = x / 1.36
        
        # Create dataset
        dset = f.create_dataset("data/massfunc/uncorr/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/massfunc/uncorr/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfunc/uncorr/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        #-----------
        # y-values: ATLAS3D 1/V corrected mass function
        grp = f.create_group("data/massfunc/1Vcorr/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'H2 mass function 1/V corrected, divided by 1.36 for He'
        grp = f.create_group("data/massfunc/1Vcorr/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (H2MF)'
        
        # Data
        log_x      = np.array([7.0929, 7.2954, 7.4958, 7.6934, 7.8960, 8.0955, 8.2971, 8.4990, 8.6996, 8.8991, 9.0979, 9.2980])
        x          = 10**log_x
        y            = 10**np.array([-2.6063, -1.9020, -2.7252, -2.8998, -3.0271, -2.9406, -2.8971, -3.0543, -2.9409, -3.8937, -3.5951, -3.8995])
        y_upper_y    = 10**np.array([-2.1748, -1.7718, -2.4802, -2.6863, -2.8508, -2.7971, -2.7642, -2.8928, -2.8028, -3.4681, -3.2932, -3.4711])
        y_lower_y    = 10**np.array([-3.0360, -2.0460, -2.9777, -3.1211, -3.2110, -3.0813, -3.0370, -3.2202, -3.0873, -4.3301, -3.9023, -4.3271])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Unit conversions -> given as M*/L -> * L/M* to get M*/M* | mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.5 M*/L* for ETGs, divided by 1.36 for He
        x      = x / 1.36
        
        # Create dataset
        dset = f.create_dataset("data/massfunc/1Vcorr/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: ATLAS3D 1/V uncorrected mass FRACTION function
        grp = f.create_group("data/massfracfunc/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'H2 mass fraction, not 1/V corrected, taken from HI/L_K distribution with conversions already applied, divided by 1.36'
        grp = f.create_group("data/massfracfunc/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'H2 mass fraction function, not 1/V corrected, taken from HI/L_K distribution function with conversions already applied'
        
        # Data
        log_x      = np.array([-3.5051, -3.3019, -3.1039, -2.9018, -2.7025, -2.5027, -2.3029, -2.1024, -1.9027, -1.7029, -1.5041, -1.2996])
        x          = 10**log_x
        y            = 10**np.array([-3.1985, -3.2931, -3.8958, -2.9894, -3.2928, -2.8905, -3.4152, -3.0510, -2.9930, -3.2941, -3.2911, -3.1944])
        y_upper_y    = 10**np.array([-2.9971, -3.0758, -3.4584, -2.8369, -3.0733, -2.7578, -3.1684, -2.8875, -2.8355, -3.0714, -3.0754, -3.0001])
        y_lower_y    = 10**np.array([-3.3898, -3.5090, -4.3300, -3.1455, -3.5111, -3.0317, -3.6673, -3.2095, -3.1377, -3.5041, -3.5089, -3.3912])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Unit conversions -> mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.82 M*/L* for ETGs, divided by 1.36 for He
        x      = (x * (1/0.82))/1.36        
        #y      = 
        #y_scat = 
        
        # Create dataset
        dset = f.create_dataset("data/massfracfunc/x/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        dset = f.create_dataset("data/massfracfunc/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfracfunc/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        print("Successfully created: %s/GalaxyH2MassFunction/Lagos2014_H2.hdf5"%obs_dir)


#--------------
# Read ATLAS3D results from Davis+19 from csv, export as hdf5
def _convert_Davis2019_ATLAS3D():
    # Read CSV from Tim
    csv_dict = pd.read_csv('%s/A3D_mk_size_sigma_mh2.csv'%obs_dir,  )
    
    # print(csv_dict['logM(H2)_err'])
    
    
    # Create and write
    with h5py.File("%s/Davis2019_ATLAS3D.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = ""
        grp.attrs["citation"]     = "Davis et al. (2019)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed, 56 detections"
        grp.attrs["name"]         = "ATLAS3D points from Davis2019, all adjusted for direct use"
                
        #-------------------------------
        # Creating dataset + cosmology
        
        grp = f.create_group("data/Galaxy")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Galaxy name'
        x = list(csv_dict['Galaxy'])
        dset = f.create_dataset("data/Galaxy/values", data=x, dtype=h5py.string_dtype())
        dset.attrs["units"]    = ''
        print(len(x))
        
        
        grp = f.create_group("data/Virgo")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'True/false if it lies within a sphere of radius = 3.5 Mpc of centre of virgo cluster'
        x = np.array(csv_dict['Virgo'])
        dset = f.create_dataset("data/Virgo/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        grp = f.create_group("data/BCG")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'True/false if it is a BCG'
        x = np.array(csv_dict['BCG'])
        dset = f.create_dataset("data/BCG/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        
        grp = f.create_group("data/Distance")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Distance to object, NOT cosmology adjusted to COLIBRE'
        x    = np.array(csv_dict['D'])
        h_old = 0.720
        h_new = 0.681
        #x = x * (h_old/h_new)
        dset = f.create_dataset("data/Distance/values", data=x)
        dset.attrs["units"]    = 'Mpc'
        print(len(x))
        
        
        
        grp = f.create_group("data/M_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Total galaxy absolute magnitude in K-T band'
        x_mag    = np.array(csv_dict['M_K'])
        dset = f.create_dataset("data/M_K/values", data=x_mag)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        grp = f.create_group("data/L_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 Total galaxy luminosity in K-T band, and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        x    = 10**(0.4*(3.28 - np.array(csv_dict['M_K'])))
        x    = np.log10(x)
        dset = f.create_dataset("data/L_K/values", data=x)
        dset.attrs["units"]    = 'Lsun'
        print(len(x))
        
        grp = f.create_group("data/log_Mstar")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 derived Mstar from mass-to-light ratio using L_K, with M*/L_K=0.82 (Bell et al. 2003), and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        x    = 10**(0.4*(3.28 - np.array(csv_dict['M_K'])))
        x_mass    = x * 0.82      # assuming mass-to-light of 0.82 for K-band
        x_mass_log = np.log10(x_mass)
        dset = f.create_dataset("data/log_Mstar/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        
        grp = f.create_group("data/log_MJAM")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'MJAM is approx Mstar, fit using Cappellari+13a fit: log M* = 10.58 - 0.44(M_K + 23)'
        x_mass_log = 10.58 - 0.44*(x_mag + 23)
        dset = f.create_dataset("data/log_MJAM/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        print(x_mass_log)
        
        
        
        grp = f.create_group("data/Re")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Effective radii in kpc, not log'
        r    = 10**np.array(csv_dict['log(Re)'])
        x    = np.array(csv_dict['D']) * r * np.pi/648
        dset = f.create_dataset("data/Re/values", data=x)
        dset.attrs["units"]    = 'kpc'
        print(len(x))
        
        grp = f.create_group("data/Sig_e")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Stellar velocity dispersion km/s, not log'
        x_sig    = 10**np.array(csv_dict['logSig_e'])
        dset = f.create_dataset("data/Sig_e/values", data=x_sig)
        dset.attrs["units"]    = 'km/s'
        print(len(x_sig))
        
        grp = f.create_group("data/Eta_kin")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Eta_kin in Msun**1/3 km^-1 s'
        x = (x_mass**1/3)/x_sig
        dset = f.create_dataset("data/Eta_kin/values", data=x)
        dset.attrs["units"]    = 'Msun**(1/3) * s * km**(-1)'
        print(len(x))
        
        
        
        grp = f.create_group("data/log_H2")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H2 mass in log10, adjusted by dividing by 1.36'
        x    = 10**np.array(csv_dict['logM(H2)'])
        x    = x/1.36
        x    = np.log10(x)
        x[csv_dict['logM(H2)'] == 0] = math.nan
        dset = f.create_dataset("data/log_H2/values", data=x)
        dset.attrs["units"]    = 'Msun'
        print(len(x))
        
        grp = f.create_group("data/det_mask")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H2 mass determinate limit < / > / ='
        x    = np.array([0,  0,  1,  1,  0,  1,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,  0, 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0, 0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0, 0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  1,  0,  0,  1,  0,  0,  0, 0, 0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  0,  1])
        x.astype(np.bool)
        dset = f.create_dataset("data/det_mask/values", data=x)
        dset.attrs["units"]    = ''
        print(len(x))
        
        grp = f.create_group("data/log_H2_err")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H2 mass error in log10'
        x     = np.array(csv_dict['logM(H2)_err'])
        dset = f.create_dataset("data/log_H2_err/values", data=x)
        dset.attrs["units"]    = 'Msun'
        print(len(x))
        
        
        
        print("Successfully created: %s/Davis2019_ATLAS3D.hdf5"%obs_dir)
# MASSIVE results, export as hdf5
def _create_Davis2019_MASSIVE():
    
    # Create and write
    with h5py.File("%s/Davis2019_MASSIVE.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = ""
        grp.attrs["citation"]     = "Davis et al. (2019)"
        grp.attrs["comment"]      = "Original data from MASSIVE. No cosmology correction needed."
        grp.attrs["name"]         = "MASSIVE points from Davis2019, all adjusted for direct use"
                
        #-------------------------------
        # Creating dataset + cosmology
        
        grp = f.create_group("data/Galaxy")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Galaxy name'
        x = ["IC 0310", "NGC 0057", "NGC 0227", "NGC 0315", "NGC 0383", "NGC 0410", "NGC 0467", "NGC 0499", "NGC 0507", "NGC 0533", "NGC 0547", "NGC 0665", "NGC 0708", "NGC 0741", "NGC 0890", "NGC 0910", "NGC 0997", "NGC 1060", "NGC 1129", "NGC 1132", "NGC 1167", "NGC 1453", "NGC 1497", "NGC 1573", "NGC 1600", "NGC 1684", "NGC 1700", "NGC 2256", "NGC 2258", "NGC 2274", "NGC 2320", "NGC 2418", "NGC 2513", "NGC 2672", "NGC 2693", "NGC 2783", "NGC 2832", "NGC 2892", "NGC 3158", "NGC 3805", "NGC 3816", "NGC 3842", "NGC 3862", "NGC 4055", "NGC 4073", "NGC 4472", "NGC 4486", "NGC 4649",    "NGC 4839", "NGC 4874", "NGC 4889", "NGC 4914", "NGC 5208", "NGC 5252", "NGC 5322", "NGC 5353", "NGC 5490", "NGC 5557", "NGC 6482", "NGC 7052","NGC 7265", "NGC 7274", "NGC 7550", "NGC 7556", "NGC 7618", "NGC 7619", "NGC 7626"]
        dset = f.create_dataset("data/Galaxy/values", data=x, shape=len(x), dtype=h5py.string_dtype())
        dset.attrs["units"]    = ''
        print(len(x))
        
        
        grp = f.create_group("data/Cluster")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'True/false if it is within a cluster'
        x = np.array([1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        dset = f.create_dataset("data/Cluster/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        
        grp = f.create_group("data/BCG")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'True/false if it is a BCG'
        x = np.array([0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0])
        dset = f.create_dataset("data/BCG/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        
        
        
        grp = f.create_group("data/Distance")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Distance to object, NOT cosmology adjusted to COLIBRE'
        x    = np.array([77.5, 76.3, 75.9, 70.3, 71.3, 71.3, 75.8, 69.8, 69.8, 77.9, 74.0, 74.6, 69.0, 73.9, 55.6, 79.8, 90.4, 67.4, 73.9, 97.6, 70.2, 56.4, 87.8, 65.0, 63.8, 63.5, 54.4, 79.4, 59.0, 73.8, 89.4, 74.1, 70.8, 61.5, 74.4, 101.4, 105.2, 101.1, 103.4, 99.4, 99.4, 99.4, 99.4, 107.2, 91.5, 16.7, 16.7, 16.5, 102.0, 102.0, 102.0, 74.5, 105.0, 103.8, 34.2, 41.1, 78.6, 51.0, 61.4, 69.3, 82.8, 82.8, 72.7, 103.0, 76.3, 54.0, 54.0])
        h_old = 0.720
        h_new = 0.681
        #x = x * (h_old/h_new)
        dset = f.create_dataset("data/Distance/values", data=x)
        dset.attrs["units"]    = 'Mpc'
        print(len(x))
        
        
        
        
        grp = f.create_group("data/M_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Total galaxy absolute magnitude in K-S band'
        x_mag    = np.array([ -25.35, -25.75, -25.32, -26.30, -25.81, -25.90, -25.40, -25.50, -25.93, -26.05, -25.83, -25.51, -25.65, -26.06, -25.50, -25.33, -25.40, -26.00, -26.14, -25.70, -25.64, -25.67, -25.31, -25.55, -25.99, -25.34, -25.60, -25.87, -25.66, -25.69, -25.93, -25.42, -25.52, -25.60, -25.76, -25.72, -26.42, -25.70, -26.28, -25.69, -25.40, -25.91, -25.50, -25.40, -26.33, -25.72, -25.31, -25.36, -25.85, -26.18, -26.64, -25.72, -25.61, -25.32, -25.51, -25.45, -25.57, -25.46, -25.60, -25.67, -25.93, -25.39, -25.43, -25.83, -25.44, -25.65, -25.65])
        dset = f.create_dataset("data/M_K/values", data=x_mag)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        grp = f.create_group("data/L_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 Total galaxy luminosity in K-T band, and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        L_K    = np.array([11.45, 11.61, 11.44, 11.83, 11.64, 11.67, 11.47, 11.51, 11.68, 11.73, 11.64, 11.52, 11.57, 11.74, 11.51, 11.44, 11.47, 11.71, 11.77, 11.59, 11.57, 11.58, 11.44, 11.53, 11.71, 11.45, 11.55, 11.66, 11.58, 11.59, 11.68, 11.48, 11.52, 11.55, 11.62, 11.60, 11.88, 11.59, 11.82, 11.59, 11.47, 11.68, 11.51, 11.47, 11.84, 11.60, 11.44, 11.46, 11.65, 11.78, 11.97, 11.60, 11.56, 11.44, 11.52, 11.49, 11.54, 11.50, 11.55, 11.58, 11.68, 11.47, 11.48, 11.64, 11.49, 11.57, 11.57])
        dset = f.create_dataset("data/L_K/values", data=L_K)
        dset.attrs["units"]    = 'Lsun'
        print(len(L_K))
        
        grp = f.create_group("data/log_Mstar")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 derived Mstar from mass-to-light ratio using L_K with M*/L_K=0.5, and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        x_mass     = (10**L_K) * 0.82      # assuming mass-to-light of 0.82 for K-band
        x_mass_log = np.log10(x_mass)
        dset = f.create_dataset("data/log_Mstar/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        
        grp = f.create_group("data/log_MJAM")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'MJAM is approx Mstar, fit using Cappellari+13a fit: log M* = 10.58 - 0.44(M_K + 23)'
        x_mass_log = 10.58 - 0.44*(x_mag + 23)
        dset = f.create_dataset("data/log_MJAM/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        print(x_mass_log)
        
        
        
        grp = f.create_group("data/Sig_e")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Stellar velocity dispersion km/s, not log'
        x_sig    = np.array([205., 251., 262., 341., 257., 247., 247., 266., 257., 258., 232., 164., 219., 289., 194., 219., 215., 271., 259., 218., 172., 272., 190., 264., 293., 262., 223., 259., 254., 259., 298., 247., 253., 262., 296., 264., 291., 234., 289., 225., 207., 231., 232., 270., 292., 258., 336., 340., 275., 258., 337., 225., 235., 196., 239., 290., 282., 223., 291., 266., 206., 244., 224., 243., 265., 277., 250.])
        dset = f.create_dataset("data/Sig_e/values", data=x_sig)
        dset.attrs["units"]    = 'km/s'
        print(len(x_sig))
        
        grp = f.create_group("data/Eta_kin")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Eta_kin in Msun**1/3 km^-1 s'
        x = (x_mass**1/3)/x_sig
        dset = f.create_dataset("data/Eta_kin/values", data=x)
        dset.attrs["units"]    = 'Msun**(1/3) * s * km**(-1)'
        print(len(x))
        
        
        
        grp = f.create_group("data/Re")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Effective radii in kpc, not log'
        x = np.array([ 5.7, 10.0, 5.4, 8.5, 7.1, 10.9, 9.0, 5.3, 13.0, 15.3, 7.1, 4.9, 16.5, 9.6, 8.2, 9.9, 10.3, 12.0, 10.8, 14.6, 10.1, 7.9, 7.9, 7.9, 12.7, 9.0, 6.0, 16.8, 10.1, 10.1, 8.4, 7.1, 8.2, 4.3, 5.5, 18.7, 10.8, 11.4, 8.1, 7.9, 8.8, 11.6, 19.2, 7.1, 10.2, 14.3, 5.7, 5.4, 14.4, 15.8, 16.3, 11.3, 9.3, 7.9, 3.3, 4.8, 7.4, 3.6, 4.9, 9.2, 12.7, 9.4, 9.9, 13.2, 6.2, 9.0, 7.0])        
        dset = f.create_dataset("data/Re/values", data=x)
        dset.attrs["units"]    = 'kpc'
        print(len(x))
        
        
        
        grp = f.create_group("data/log_H2")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H2 mass in log10, adjusted by dividing by 1.36'
        x    = 10**np.array([9.02, 8.60, 8.52, 8.81, 9.23, 8.66, 8.77, 8.62, 8.68, 8.67, 8.76, 9.18, 8.83, 8.66, 8.79, 8.12, 9.26, 7.89, 8.81, 8.61, 8.52, 8.91, 9.10, 8.66, 8.41, 9.20, 8.31, 8.54, 8.81, 8.58, 8.41, 8.68, 8.49, 8.65, 8.40, 8.85, 8.59, 8.84, 8.81, 8.30, 8.67, 8.15, 8.49, 8.72, 8.90, 7.25, 6.70, 7.83, 8.88, 8.86, 8.93, 8.79, 9.49, 8.86, 7.76, 8.28, 8.73, 7.92, 8.62, 9.63, 8.65, 8.55, 8.85, 8.56, 8.60, 7.52, 7.90])
        x    = x/1.36
        x    = np.log10(x)
        dset = f.create_dataset("data/log_H2/values", data=x)
        dset.attrs["units"]    = 'Msun'
        print(len(x))
        
        grp = f.create_group("data/det_mask")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H2 mass determinate limit < / > / ='
        x    = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        x.astype(np.bool)
        dset = f.create_dataset("data/det_mask/values", data=x)
        dset.attrs["units"]    = ''
        print(len(x))
        
        
        print("Successfully created: %s/Davis2019_MASSIVE.hdf5"%obs_dir)


#-------------
# ATLAS3D H2 extent results from Davis+13
def _create_Davis2013_ATLAS3D_CO_extent():
    # Create and write
    with h5py.File("%s/Davis2013_ATLAS3D_CO_extent.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = ""
        grp.attrs["citation"]     = "Davis et al. (2013)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed. Contains H2 extents"
        grp.attrs["name"]         = "ATLAS3D points from Davis2013, all adjusted for direct use"
                
        #-------------------------------
        # Creating dataset + cosmology
        
        grp = f.create_group("data/Galaxy")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Galaxy name'
        x = ["IC 0676", "IC 0719", "IC 1024", "NGC 0524", "NGC 1222", "NGC 1266", "NGC 2697", "NGC 2764", "NGC 2768", "NGC 2824", "NGC 3032", "NGC 3182", "NGC 3489", "NGC 3607", "NGC 3619", "NGC 3626", "NGC 3665", "NGC 4119", "NGC 4150", "NGC 4292", "NGC 4324", "NGC 4429", "NGC 4435", "NGC 4459", "NGC 4476", "NGC 4477", "NGC 4526", "NGC 4550", "NGC 4694", "NGC 4710", "NGC 4753", "NGC 5379", "NGC 5866", "NGC 6014", "NGC 7465", "PGC 029321", "PGC 058114", "UGC 05408", "UGC 06176", "UGC 09519"]
        dset = f.create_dataset("data/Galaxy/values", data=x, shape=len(x), dtype=h5py.string_dtype())
        dset.attrs["units"]    = ''
        print(len(x))
        
        
        grp = f.create_group("data/R_CO")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'maximum CO extent, beam corrected and converted to a linear size using the distance to the galaxy'
        x    = np.array([1.22, 1.85, 1.21, 1.05, 1.90, 0.62, 2.02, 3.42, 0.64, 1.36, 0.50, 1.16, 0.54, 1.49, 0.46, 1.22, 1.58, 0.63, 0.91, 0.31, 1.90, 0.48, 0.28, 0.51, 0.78, 0.21, 0.44, 0.30, 1.01, 2.41, 1.59, 2.95, 2.26, 0.46, 1.67, 0.84, 0.77, 0.66, 0.32, 0.69])
        #h_old = 0.720
        #h_new = 0.681
        #x = x * (h_old/h_new)
        dset = f.create_dataset("data/R_CO/values", data=x)
        dset.attrs["units"]    = 'kpc'
        print(len(x))
        
        
        grp = f.create_group("data/R_CO_Re_ratio")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'ratio of the CO extent to the effective radius of the galaxy (tabulated in Paper I)'
        x    = np.array([0.46, 0.48, 1.08, 0.21, 1.00, 0.22, 0.64, 1.45, 0.10, 1.03, 0.54, 0.34, 0.43, 0.36, 0.13, 0.50, 0.32, 0.20, 0.78, 0.15, 1.19, 0.14, 0.11, 0.19, 0.60, 0.07, 0.13, 0.26, 0.43, 0.96, 0.29, 0.97, 0.86, 0.12, 1.75, 0.55, 0.72, 0.43, 0.16, 0.77])
        #h_old = 0.720
        #h_new = 0.681
        #x = x * (h_old/h_new)
        dset = f.create_dataset("data/R_CO_Re_ratio/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        
        grp = f.create_group("data/log_R_CO_L_Ks")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'ratio of the CO extent to the effective radius of the galaxy (tabulated in Paper I)'
        x    = np.array([-10.13, -10.13, -9.84, -11.17, -10.09, -10.65, -10.08, -10.05, -11.38, -10.31, -10.41, -10.52, -10.77, -11.03, -11.08, -10.55, -11.08, -10.55, -10.01, -10.44, -10.08, -11.36, -11.39, -11.16, -10.13, -11.50, -11.51, -10.74, -10.17, -10.34, -11.15, -9.67, -10.56, -10.83, -10.17, -10.05, -10.05, -10.31, -10.87, -10.21])
        #h_old = 0.720
        #h_new = 0.681
        #x = x * (h_old/h_new)
        dset = f.create_dataset("data/log_R_CO_L_Ks/values", data=x)
        dset.attrs["units"]    = 'kpc * Lsun**(-1)'
        print(len(x))
        
        
        
        
        print("Successfully created: %s/Davis2013_ATLAS3D_CO_extent.hdf5"%obs_dir)

#-------------
# ATLAS3D ellipticities in projection and uncertainties as measured by Krajnovic+11
def _create_Krajnovic2011_ATLAS3D_ellip():
    # Create and write
    with h5py.File("%s/Krajnovic2011_ellip.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = ""
        grp.attrs["citation"]     = "Krajnovic et al. (2011)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed. Contains projected ellipticities and uncertainties"
        grp.attrs["name"]         = "ATLAS3D points from Krajnovic2011, all adjusted for direct use"
                
        #-------------------------------
        # Creating dataset + cosmology
        
        grp = f.create_group("data/Galaxy")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Galaxy name'
        x = ["IC 0560", "IC 0598", "IC 0676", "IC 0719", "IC 0782", "IC 1024", "IC 3631", "NGC 0448", "NGC 0474", "NGC 0502", "NGC 0509", "NGC 0516", "NGC 0524", "NGC 0525", "NGC 0661", "NGC 0680", "NGC 0770", "NGC 0821", "NGC 0936", "NGC 1023", "NGC 1121", "NGC 1222", "NGC 1248", "NGC 1266", "NGC 1289", "NGC 1665", "NGC 2481", "NGC 2549", "NGC 2577", "NGC 2592", "NGC 2594", "NGC 2679", "NGC 2685", "NGC 2695", "NGC 2698", "NGC 2699", "NGC 2764", "NGC 2768", "NGC 2778", "NGC 2824", "NGC 2852", "NGC 2859", "NGC 2880", "NGC 2950", "NGC 2962", "NGC 2974", "NGC 3032", "NGC 3073", "NGC 3098", "NGC 3156", "NGC 3182", "NGC 3193", "NGC 3226", "NGC 3230", "NGC 3245", "NGC 3248", "NGC 3301", "NGC 3377", "NGC 3379", "NGC 3384", "NGC 3400", "NGC 3412", "NGC 3414", "NGC 3457", "NGC 3458", "NGC 3489", "NGC 3499", "NGC 3522", "NGC 3530", "NGC 3595", "NGC 3599", "NGC 3605", "NGC 3607", "NGC 3608", "NGC 3610", "NGC 3613", "NGC 3619", "NGC 3626", "NGC 3630", "NGC 3640", "NGC 3641", "NGC 3648", "NGC 3658", "NGC 3665", "NGC 3674", "NGC 3694", "NGC 3757", "NGC 3796", "NGC 3838", "NGC 3941", "NGC 3945", "NGC 3998", "NGC 4026", "NGC 4036", "NGC 4078", "NGC 4111", "NGC 4119", "NGC 4143", "NGC 4150", "NGC 4168", "NGC 4179", "NGC 4191", "NGC 4203", "NGC 4215", "NGC 4233", "NGC 4249", "NGC 4251", "NGC 4255", "NGC 4259", "NGC 4261", "NGC 4262", "NGC 4264", "NGC 4267", "NGC 4268", "NGC 4270", "NGC 4278", "NGC 4281", "NGC 4283", "NGC 4324", "NGC 4339", "NGC 4340", "NGC 4342", "NGC 4346", "NGC 4350", "NGC 4365", "NGC 4371", "NGC 4374", "NGC 4377", "NGC 4379", "NGC 4382", "NGC 4387", "NGC 4406", "NGC 4417", "NGC 4425", "NGC 4429", "NGC 4434", "NGC 4435", "NGC 4442", "NGC 4452", "NGC 4458", "NGC 4459", "NGC 4461", "NGC 4472", "NGC 4473", "NGC 4474", "NGC 4476", "NGC 4477", "NGC 4478", "NGC 4483", "NGC 4486", "NGC 4486A", "NGC 4489", "NGC 4494", "NGC 4503", "NGC 4521", "NGC 4526", "NGC 4528", "NGC 4546", "NGC 4550", "NGC 4551", "NGC 4552", "NGC 4564", "NGC 4570", "NGC 4578", "NGC 4596", "NGC 4608", "NGC 4612", "NGC 4621", "NGC 4623", "NGC 4624", "NGC 4636", "NGC 4638", "NGC 4643", "NGC 4649", "NGC 4660", "NGC 4684", "NGC 4690", "NGC 4694", "NGC 4697", "NGC 4710", "NGC 4733", "NGC 4753", "NGC 4754", "NGC 4762", "NGC 4803", "NGC 5103", "NGC 5173", "NGC 5198", "NGC 5273", "NGC 5308", "NGC 5322", "NGC 5342", "NGC 5353", "NGC 5355", "NGC 5358", "NGC 5379", "NGC 5422", "NGC 5473", "NGC 5475", "NGC 5481", "NGC 5485", "NGC 5493", "NGC 5500", "NGC 5507", "NGC 5557", "NGC 5574", "NGC 5576", "NGC 5582", "NGC 5611", "NGC 5631", "NGC 5638", "NGC 5687", "NGC 5770", "NGC 5813", "NGC 5831", "NGC 5838", "NGC 5839", "NGC 5845", "NGC 5846", "NGC 5854", "NGC 5864", "NGC 5866", "NGC 5869", "NGC 6010", "NGC 6014", "NGC 6017", "NGC 6149", "NGC 6278", "NGC 6547", "NGC 6548", "NGC 6703", "NGC 6798", "NGC 7280", "NGC 7332", "NGC 7454", "NGC 7457", "NGC 7465", "NGC 7693", "NGC 7710", "PGC 016060", "PGC 028887", "PGC 029321", "PGC 035754", "PGC 042549", "PGC 044433", "PGC 050395", "PGC 051753", "PGC 054452", "PGC 056772", "PGC 058114", "PGC 061468", "PGC 071531", "PGC 170172", "UGC 03960", "UGC 04551", "UGC 05408", "UGC 06062", "UGC 06176", "UGC 08876", "UGC 09519"]
        dset = f.create_dataset("data/Galaxy/values", data=x, shape=len(x), dtype=h5py.string_dtype())
        dset.attrs["units"]    = ''
        print(len(x))
        
        
        grp = f.create_group("data/ellip")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Global ellipticity, measured within 2.5 - 3 half-light radii.'
        x    = np.array([0.56, 0.67, 0.25, 0.71, 0.28, 0.64, 0.43, 0.57, 0.12, 0.10, 0.64, 0.66, 0.05, 0.47, 0.31, 0.22, 0.29, 0.35, 0.22, 0.63, 0.51, 0.28, 0.15, 0.25, 0.41, 0.41, 0.46, 0.69, 0.41, 0.21, 0.32, 0.07, 0.40, 0.28, 0.54, 0.14, 0.49, 0.57, 0.20, 0.24, 0.14, 0.15, 0.36, 0.41, 0.45, 0.37, 0.17, 0.12, 0.77, 0.50, 0.20, 0.09, 0.17, 0.61, 0.46, 0.40, 0.69, 0.33, 0.13, 0.50, 0.44, 0.44, 0.22, 0.01, 0.29, 0.45, 0.13, 0.48, 0.53, 0.46, 0.08, 0.40, 0.13, 0.20, 0.19, 0.46, 0.09, 0.33, 0.66, 0.15, 0.11, 0.44, 0.16, 0.22, 0.64, 0.18, 0.15, 0.40, 0.56, 0.25, 0.35, 0.22, 0.75, 0.60, 0.56, 0.79, 0.65, 0.40, 0.33, 0.17, 0.71, 0.26, 0.11, 0.64, 0.55, 0.05, 0.48, 0.49, 0.58, 0.16, 0.12, 0.19, 0.08, 0.55, 0.55, 0.09, 0.51, 0.04, 0.56, 0.07, 0.42, 0.58, 0.64, 0.60, 0.24, 0.48, 0.05, 0.18, 0.16, 0.25, 0.37, 0.31, 0.65, 0.67, 0.52, 0.06, 0.32, 0.60, 0.73, 0.08, 0.21, 0.61, 0.19, 0.43, 0.42, 0.28, 0.14, 0.17, 0.51, 0.16, 0.15, 0.09, 0.14, 0.54, 0.73, 0.76, 0.41, 0.52, 0.68, 0.25, 0.11, 0.53, 0.73, 0.29, 0.25, 0.07, 0.32, 0.32, 0.67, 0.06, 0.23, 0.39, 0.12, 0.16, 0.30, 0.63, 0.29, 0.52, 0.32, 0.75, 0.06, 0.50, 0.48, 0.83, 0.37, 0.35, 0.13, 0.17, 0.16, 0.80, 0.36, 0.54, 0.48, 0.32, 0.62, 0.66, 0.79, 0.21, 0.70, 0.27, 0.26, 0.20, 0.20, 0.47, 0.16, 0.48, 0.31, 0.35, 0.55, 0.07, 0.10, 0.37, 0.06, 0.27, 0.10, 0.62, 0.12, 0.31, 0.08, 0.68, 0.68, 0.58, 0.32, 0.75, 0.12, 0.11, 0.32, 0.45, 0.67, 0.11, 0.03, 0.47, 0.36, 0.74, 0.26, 0.47, 0.33, 0.24, 0.59, 0.72, 0.33, 0.12, 0.33, 0.39, 0.64, 0.27, 0.51, 0.16, 0.45, 0.20, 0.28, 0.29, 0.09, 0.28, 0.61, 0.12, 0.45, 0.49, 0.63, 0.25])
        #h_old = 0.720
        #h_new = 0.681
        #x = x * (h_old/h_new)
        dset = f.create_dataset("data/ellip/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        
        grp = f.create_group("data/ellip_err")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Global ellipticity uncertainty, measured within 2.5 - 3 half-light radii.'
        x    = np.array([0.20, 0.02, 0.09, 0.01, 0.06, 0.02, 0.04, 0.06, 0.06, 0.03,    0.07, 0.09, 0.03, 0.05, 0.01, 0.01, 0.01, 0.10, 0.01, 0.03,    0.04, 0.08, 0.01, 0.04, 0.02, 0.21, 0.17, 0.03, 0.12, 0.01,    0.05, 0.06, 0.05, 0.01, 0.25, 0.03, 0.11, 0.06, 0.02, 0.10,    0.01, 0.01, 0.01, 0.03, 0.04, 0.03, 0.10, 0.01, 0.04, 0.01,    0.02, 0.02, 0.05, 0.03, 0.03, 0.01, 0.01, 0.12, 0.01, 0.03,    0.01, 0.01, 0.06, 0.01, 0.02, 0.04, 0.16, 0.03, 0.04, 0.02,    0.01, 0.13, 0.08, 0.04, 0.04, 0.04, 0.08, 0.05, 0.05, 0.02,    0.01, 0.03, 0.01, 0.01, 0.02, 0.04, 0.02, 0.01, 0.04, 0.04,    0.17, 0.06, 0.02, 0.03, 0.09, 0.02, 0.01, 0.04, 0.01, 0.05,    0.02, 0.04, 0.03, 0.01, 0.01, 0.01, 0.05, 0.06, 0.03, 0.03,    0.01, 0.01, 0.01, 0.04, 0.04, 0.01, 0.04, 0.01, 0.03, 0.01,    0.08, 0.09, 0.02, 0.12, 0.02, 0.10, 0.01, 0.02, 0.00, 0.07,    0.03, 0.06, 0.09, 0.04, 0.04, 0.01, 0.05, 0.00, 0.04, 0.02,    0.03, 0.01, 0.03, 0.03, 0.16, 0.03, 0.01, 0.01, 0.04, 0.06,    0.01, 0.00, 0.02, 0.02, 0.01, 0.05, 0.02, 0.04, 0.01, 0.02,    0.01, 0.04, 0.03, 0.01, 0.02, 0.20, 0.04, 0.11, 0.04, 0.06,    0.06, 0.04, 0.15, 0.01, 0.12, 0.00, 0.03, 0.08, 0.04, 0.03,    0.00, 0.03, 0.01, 0.10, 0.01, 0.09, 0.01, 0.02, 0.02, 0.04,    0.03, 0.05, 0.04, 0.01, 0.01, 0.01, 0.03, 0.01, 0.03, 0.07,   0.04, 0.14, 0.04, 0.02, 0.04, 0.04, 0.02, 0.05, 0.09, 0.02,    0.04, 0.05, 0.09, 0.03, 0.02, 0.06, 0.04, 0.09, 0.03, 0.01,    0.02, 0.08, 0.07, 0.05, 0.02, 0.08, 0.01, 0.05, 0.02, 0.18,    0.01, 0.03, 0.01, 0.04, 0.06, 0.00, 0.02, 0.02, 0.02, 0.04,    0.02, 0.01, 0.02, 0.01, 0.03, 0.03, 0.03, 0.03, 0.02, 0.09,    0.06, 0.06, 0.00, 0.02, 0.01, 0.01, 0.05, 0.02, 0.04, 0.08])
        #h_old = 0.720
        #h_new = 0.681
        #x = x * (h_old/h_new)
        dset = f.create_dataset("data/ellip_err/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        
        print("Successfully created: %s/Krajnovic2011_ellip.hdf5"%obs_dir)
# ATLAS3D ellipticities in projection and uncertainties as measured by Krajnovic+11
def _create_Cappellari2011_ATLAS3D_masses():
    # Create and write
    with h5py.File("%s/Cappellari2011_masses.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = ""
        grp.attrs["citation"]     = "Capellari et al. (2011)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed. Contains K-band luminosities and converted stellar masses"
        grp.attrs["name"]         = "ATLAS3D points from Cappellari+11, all adjusted for direct use"
                
        #-------------------------------
        # Creating dataset + cosmology
        
        grp = f.create_group("data/Galaxy")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Galaxy name'
        x = [ "IC 0560", "IC 0598", "IC 0676", "IC 0719", "IC 0782", "IC 1024", "IC 3631", "NGC 0448", "NGC 0474", "NGC 0502", "NGC 0509", "NGC 0516", "NGC 0524", "NGC 0525", "NGC 0661", "NGC 0680", "NGC 0770", "NGC 0821", "NGC 0936", "NGC 1023", "NGC 1121", "NGC 1222", "NGC 1248", "NGC 1266", "NGC 1289", "NGC 1665", "NGC 2481", "NGC 2549", "NGC 2577", "NGC 2592", "NGC 2594", "NGC 2679", "NGC 2685", "NGC 2695", "NGC 2698", "NGC 2699", "NGC 2764", "NGC 2768", "NGC 2778", "NGC 2824", "NGC 2852", "NGC 2859", "NGC 2880", "NGC 2950", "NGC 2962", "NGC 2974", "NGC 3032", "NGC 3073", "NGC 3098", "NGC 3156", "NGC 3182", "NGC 3193", "NGC 3226", "NGC 3230", "NGC 3245", "NGC 3248", "NGC 3301", "NGC 3377", "NGC 3379", "NGC 3384", "NGC 3400", "NGC 3412", "NGC 3414", "NGC 3457", "NGC 3458", "NGC 3489", "NGC 3499", "NGC 3522", "NGC 3530", "NGC 3595", "NGC 3599", "NGC 3605", "NGC 3607", "NGC 3608", "NGC 3610", "NGC 3613", "NGC 3619", "NGC 3626", "NGC 3630", "NGC 3640", "NGC 3641", "NGC 3648", "NGC 3658", "NGC 3665", "NGC 3674", "NGC 3694", "NGC 3757", "NGC 3796", "NGC 3838", "NGC 3941", "NGC 3945", "NGC 3998", "NGC 4026", "NGC 4036", "NGC 4078", "NGC 4111", "NGC 4119", "NGC 4143", "NGC 4150", "NGC 4168", "NGC 4179", "NGC 4191", "NGC 4203", "NGC 4215", "NGC 4233", "NGC 4249", "NGC 4251", "NGC 4255", "NGC 4259", "NGC 4261", "NGC 4262", "NGC 4264", "NGC 4267", "NGC 4268", "NGC 4270", "NGC 4278", "NGC 4281", "NGC 4283", "NGC 4324", "NGC 4339", "NGC 4340", "NGC 4342", "NGC 4346", "NGC 4350", "NGC 4365", "NGC 4371", "NGC 4374", "NGC 4377", "NGC 4379", "NGC 4382", "NGC 4387", "NGC 4406", "NGC 4417", "NGC 4425", "NGC 4429", "NGC 4434", "NGC 4435", "NGC 4442", "NGC 4452", "NGC 4458", "NGC 4459", "NGC 4461", "NGC 4472", "NGC 4473", "NGC 4474", "NGC 4476", "NGC 4477", "NGC 4478", "NGC 4483", "NGC 4486", "NGC 4486A", "NGC 4489", "NGC 4494", "NGC 4503", "NGC 4521", "NGC 4526", "NGC 4528", "NGC 4546", "NGC 4550", "NGC 4551", "NGC 4552", "NGC 4564", "NGC 4570", "NGC 4578", "NGC 4596", "NGC 4608", "NGC 4612", "NGC 4621", "NGC 4623", "NGC 4624", "NGC 4636", "NGC 4638", "NGC 4643", "NGC 4649", "NGC 4660", "NGC 4684", "NGC 4690", "NGC 4694", "NGC 4697", "NGC 4710", "NGC 4733", "NGC 4753", "NGC 4754", "NGC 4762", "NGC 4803", "NGC 5103", "NGC 5173", "NGC 5198", "NGC 5273", "NGC 5308", "NGC 5322", "NGC 5342", "NGC 5353", "NGC 5355", "NGC 5358", "NGC 5379", "NGC 5422", "NGC 5473", "NGC 5475", "NGC 5481", "NGC 5485", "NGC 5493", "NGC 5500", "NGC 5507", "NGC 5557", "NGC 5574", "NGC 5576", "NGC 5582", "NGC 5611", "NGC 5631", "NGC 5638", "NGC 5687", "NGC 5770", "NGC 5813", "NGC 5831", "NGC 5838", "NGC 5839", "NGC 5845", "NGC 5846", "NGC 5854", "NGC 5864", "NGC 5866", "NGC 5869", "NGC 6010", "NGC 6014", "NGC 6017", "NGC 6149", "NGC 6278", "NGC 6547", "NGC 6548", "NGC 6703", "NGC 6798", "NGC 7280", "NGC 7332", "NGC 7454", "NGC 7457", "NGC 7465", "NGC 7693", "NGC 7710", "PGC 016060", "PGC 028887", "PGC 029321", "PGC 035754", "PGC 042549", "PGC 044433", "PGC 050395", "PGC 051753", "PGC 054452", "PGC 056772", "PGC 058114", "PGC 061468", "PGC 071531", "PGC 170172", "UGC 03960", "UGC 04551", "UGC 05408", "UGC 06062", "UGC 06176", "UGC 08876", "UGC 09519"]
        dset = f.create_dataset("data/Galaxy/values", data=x, shape=len(x), dtype=h5py.string_dtype())
        dset.attrs["units"]    = ''
        print(len(x))
        
        grp = f.create_group("data/Virgo")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'True/false if it lies within a sphere of radius = 3.5 Mpc of centre of virgo cluster'
        x = np.array([ 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,0, 0,0,1,0,0,0,0,0,0,0, 0,0,0,1,0,1,0,0,0,0, 0,0,1,1,1,1,0,1,0,1, 1,1,1,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,0,1,0,1, 1,0,1,1,1,1,1,1,1,1, 1,1,1,1,0,1,1,1,1,0, 0,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0 ])
        dset = f.create_dataset("data/Virgo/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        grp = f.create_group("data/BCG")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'True/false if it is a BCG'
        x = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        dset = f.create_dataset("data/BCG/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x))
        
        grp = f.create_group("data/M_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Total galaxy absolute magnitude in K-S band'
        x_mag    = np.array([ -22.10, -22.60, -22.27, -22.70, -22.02, -21.85, -22.01, -23.02, -23.91, -23.05, -21.89, -22.21, -24.71, -21.86, -23.19, -24.17, -22.57, -23.99, -24.85, -24.01, -22.70, -22.71, -22.90, -22.93, -23.46, -23.63, -23.38, -22.43, -23.41, -22.88, -22.36, -22.81, -22.78, -23.64, -23.32, -22.72, -23.19, -24.71, -22.23, -22.93, -22.18, -24.13, -22.98, -22.93, -24.01, -23.62, -22.01, -21.78, -22.72, -22.15, -23.19, -24.63, -23.24, -24.18, -23.69, -22.43, -23.28, -22.76, -23.80, -23.52, -21.82, -22.55, -23.98, -21.89, -23.12, -22.99, -21.88, -21.67, -22.00, -23.28, -22.22, -21.83, -24.74, -23.65, -23.69, -24.26, -23.57, -23.30, -23.16, -24.60, -21.85, -23.06, -23.45, -24.92, -23.23, -22.35, -22.15, -21.84, -22.52, -23.06, -24.31, -23.33, -23.03, -24.40, -22.99, -23.27, -22.60, -23.10, -21.65, -24.03, -23.18, -23.10, -23.44, -23.43, -23.88, -21.98, -23.68, -22.99, -22.19, -25.18, -22.60, -23.00, -23.18, -23.05, -23.69, -23.80, -24.01, -21.80, -22.61, -22.49, -23.01, -22.07, -22.55, -23.13, -25.21, -23.45, -25.12, -22.43, -22.24, -25.13, -22.13, -25.04, -22.86, -22.09, -24.32,-22.55, -23.83, -23.63, -21.88, -21.76, -23.89, -23.08, -25.78, -23.77, -22.28, -21.78, -23.75, -22.80, -21.84, -25.38, -21.82, -21.59, -24.11, -23.22, -23.92, -24.62, -22.05, -23.30, -22.27, -22.18, -24.29, -23.08, -23.48, -22.66, -23.63, -22.94, -22.55, -24.14, -21.74, -23.67, -24.36, -23.01, -23.69, -25.46, -22.69, -22.21, -22.96, -22.15, -23.93, -23.53, -21.80, -25.09, -23.64, -24.48, -22.28, -22.36, -22.88, -24.10, -22.37, -24.13, -25.26, -22.61, -25.11, -22.40, -22.01, -22.08, -23.69, -24.25, -22.88, -22.68, -23.61, -24.49, -21.93, -23.19, -24.87, -22.30, -24.15, -23.28, -22.20, -23.70, -23.80, -23.22, -22.15, -25.09, -23.69, -24.13, -22.53, -22.92, -25.01, -23.30, -23.62, -24.00, -23.27, -23.53, -22.99, -22.52, -22.60, -24.19, -23.60, -23.19, -23.85, -23.52, -22.83, -23.75, -23.00, -22.38, -22.82, -21.58, -21.99, -22.64, -22.26, -21.66, -21.90, -22.71, -22.25, -21.92, -21.92, -21.59, -22.06, -21.57, -21.68, -21.74, -21.89, -21.89,-22.92, -22.03, -22.82, -22.66, -22.37, -21.98])
        dset = f.create_dataset("data/M_K/values", data=x_mag)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x_mag))
        
        
        grp = f.create_group("data/L_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 Total galaxy luminosity in K-T band, and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        L_K    = 10**(0.4*(3.28 - x_mag))
        L_K    = np.log10(L_K)
        dset = f.create_dataset("data/L_K/values", data=L_K)
        dset.attrs["units"]    = 'Lsun'
        print(len(L_K))
        
        grp = f.create_group("data/log_Mstar")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 derived Mstar from mass-to-light ratio using L_K with M*/L_K=0.5, and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        x_mass     = (10**L_K) * 0.82      # assuming mass-to-light of 0.82 for K-band
        x_mass_log = np.log10(x_mass)
        dset = f.create_dataset("data/log_Mstar/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        
        
        grp = f.create_group("data/log_MJAM")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'MJAM is approx Mstar, fit using Cappellari+13a fit: log M* = 10.58 - 0.44(M_K + 23)'
        x_mass_log = 10.58 - 0.44*(x_mag + 23)
        dset = f.create_dataset("data/log_MJAM/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        print(x_mass_log)
        
        
        print("Successfully created: %s/Cappellari2011_masses.hdf5"%obs_dir)


#-------------
# ATLAS3D H1 results from Serra+12
def _create_Serra2012_ATLAS3D_HI():
    # Create and write
    with h5py.File("%s/Serra2012_ATLAS3D_HI.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = ""
        grp.attrs["citation"]     = "Serra et al. (2012)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed. Contains H1 masses and mass fractions"
        grp.attrs["name"]         = "166 ATLAS3D points from Serra2012, all adjusted for direct use"
                
        #-------------------------------
        # Creating dataset + cosmology
        
        grp = f.create_group("data/Galaxy")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Galaxy name'
        x = [ "IC 0598", "IC 3631", "NGC 0661", "NGC 0680", "NGC 0770", "NGC 0821", "NGC 1023", "NGC 2481", "NGC 2549", "NGC 2577", "NGC 2592", "NGC 2594", "NGC 2679", "NGC 2685", "NGC 2764", "NGC 2768", "NGC 2778", "NGC 2824", "NGC 2852", "NGC 2859", "NGC 2880", "NGC 2950", "NGC 3032", "NGC 3073", "NGC 3098", "NGC 3182", "NGC 3193", "NGC 3226", "NGC 3230", "NGC 3245", "NGC 3248", "NGC 3301", "NGC 3377", "NGC 3379", "NGC 3384", "NGC 3400", "NGC 3412", "NGC 3414", "NGC 3457", "NGC 3458", "NGC 3489", "NGC 3499", "NGC 3522", "NGC 3530", "NGC 3595", "NGC 3599", "NGC 3605", "NGC 3607", "NGC 3608", "NGC 3610", "NGC 3613", "NGC 3619", "NGC 3626", "NGC 3648", "NGC 3658", "NGC 3665", "NGC 3674", "NGC 3694", "NGC 3757", "NGC 3796", "NGC 3838", "NGC 3941", "NGC 3945", "NGC 3998", "NGC 4026", "NGC 4036", "NGC 4078", "NGC 4111", "NGC 4119", "NGC 4143", "NGC 4150", "NGC 4168", "NGC 4203", "NGC 4251", "NGC 4262", "NGC 4267", "NGC 4278", "NGC 4283", "NGC 4340", "NGC 4346", "NGC 4350", "NGC 4371", "NGC 4374", "NGC 4377", "NGC 4379", "NGC 4382", "NGC 4387", "NGC 4406", "NGC 4425", "NGC 4429", "NGC 4435", "NGC 4452", "NGC 4458", "NGC 4459", "NGC 4461", "NGC 4473", "NGC 4474", "NGC 4477", "NGC 4489", "NGC 4494", "NGC 4503", "NGC 4521", "NGC 4528", "NGC 4550", "NGC 4551", "NGC 4552", "NGC 4564", "NGC 4596", "NGC 4608", "NGC 4621", "NGC 4638", "NGC 4649", "NGC 4660", "NGC 4694", "NGC 4710", "NGC 4733", "NGC 4754", "NGC 4762", "NGC 5103", "NGC 5173", "NGC 5198", "NGC 5273", "NGC 5308", "NGC 5322", "NGC 5342", "NGC 5353", "NGC 5355", "NGC 5358", "NGC 5379", "NGC 5422", "NGC 5473", "NGC 5475", "NGC 5481", "NGC 5485", "NGC 5500", "NGC 5557", "NGC 5582", "NGC 5611", "NGC 5631", "NGC 5687", "NGC 5866", "NGC 6149", "NGC 6278", "NGC 6547", "NGC 6548", "NGC 6703", "NGC 6798", "NGC 7280", "NGC 7332", "NGC 7454", "NGC 7457", "NGC 7465", "PGC 028887", "PGC 029321", "PGC 035754", "PGC 044433", "PGC 050395", "PGC 051753", "PGC 061468", "PGC 071531", "UGC 03960", "UGC 04551", "UGC 05408", "UGC 06176", "UGC 08876", "UGC 09519"]
        dset = f.create_dataset("data/Galaxy/values", data=x, shape=len(x), dtype=h5py.string_dtype())
        dset.attrs["units"]    = ''
        print(len(x))
        
        
        grp = f.create_group("data/log_H1")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H1 mass in log units'
        x    = np.array([ 7.45, 7.71, 7.37,  9.47, 7.56, 6.91,  9.29, 7.42, 6.51, 7.35, 7.18,  8.91, 7.35,  9.33,  9.28,  7.81, 7.06,  7.59, 7.27,  8.46, 7.03, 6.69,  8.04,  8.56, 7.12,  6.92,  8.19, 7.10, 7.71, 7.00, 7.22, 7.13, 6.52, 6.49,  7.25, 7.19, 6.55,  8.28,  8.07, 7.35,  6.87,  6.81,  8.47, 7.37, 7.43, 7.03, 6.83, 6.92,  7.16, 7.02, 7.28,  9.00,  8.94, 7.38, 7.42, 7.43, 7.41, 7.49, 7.10, 7.10,  8.38,  8.73,  8.85,  8.45,  8.50,  8.41, 7.64,  8.81, 7.10, 6.80,  6.26, 7.46,  9.15, 6.97,  8.69, 7.17,  8.80, 6.36, 7.03, 6.66, 6.88, 7.10, 7.26, 7.16, 7.04, 6.97, 7.03,  8.00, 6.71, 7.12, 7.23, 7.27, 6.91, 6.91, 7.33, 6.86, 7.08, 6.95, 6.74, 6.84, 7.14,  7.75, 7.18, 6.89, 7.39, 6.87, 6.91, 7.13, 7.22, 6.86, 7.12, 7.18, 6.88,  8.21,  6.84, 7.12, 7.18, 7.40,  8.57,  9.33,  8.49, 6.81, 7.63, 7.34, 7.50, 7.45, 7.50, 7.52, 7.36,  7.87, 7.40, 7.28, 7.21, 7.17, 7.36,  8.57,  9.65, 7.15,  8.89, 7.32,  6.96, 7.56, 7.67, 7.63, 7.12, 7.18,  9.38,  7.92,  6.62, 7.16, 6.61,  9.98,  7.65, 7.68, 7.58, 7.66, 7.51, 7.52, 7.54, 7.37,  7.79, 7.25,  8.52,  9.02, 7.43,  9.27])
        dset = f.create_dataset("data/log_H1/values", data=x)
        dset.attrs["units"]    = 'Msun'
        print(len(x))
        
        grp = f.create_group("data/det_mask")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H2 mass determinate limit < / > / ='
        x    = np.array([0,   0,  0,  1,   0,   0,   1,   0,   0,   0,   0,   1,   0,   1,   1,   1,   0,   1,   0,   1,   0,   0,   1,   1,   0,   1,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   1,   1,   0,   1,   1,   1,   0,   0,   0,   0,   0,   1,   0,   0,   1,   1,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   0,   1,   0,   0,   1,   0,   1,   0,  1,  0,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   0,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   1,   1,   0,   1,   0,   1,   0,   0,   0,   0,   0,   1,   1,   1,   0,   0,   1,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   1,   1,   0,   1])
        x.astype(np.bool)
        dset = f.create_dataset("data/det_mask/values", data=x)
        dset.attrs["units"]    = ''
        print(len(x))
        
        
        grp = f.create_group("data/log_H1_Lk_frac")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H1 mass/Lk magnitude in log units'
        x_frac    = np.array([-2.90, -2.40, -3.22, -1.51, -2.78, -4.00, -1.63, -3.25, -3.78, -3.33, -3.28, -1.35, -3.08, -1.10, -1.31, -3.38, -3.14, -2.89, -2.91, -2.50, -3.47, -3.79, -2.08, -1.46, -3.28, -3.67, -2.98, -3.51, -3.28, -3.79, -3.07, -3.49, -3.89, -4.35, -3.47, -2.85, -3.78, -2.63, -2.00, -3.21, -3.63, -3.25, -1.51, -2.74, -3.19, -3.17, -3.21, -4.29, -3.61, -3.77, -3.74, -1.74, -1.70, -3.16, -3.27, -3.85, -3.19, -2.76, -3.07, -2.95, -1.94, -1.80, -2.18, -2.19, -2.02, -2.66, -2.86, -1.81, -3.25, -3.75, -3.72, -3.46, -1.53, -3.82, -1.66, -3.42, -2.04, -3.67, -3.48, -3.67, -3.68, -3.59, -4.10, -3.13, -3.17, -4.39, -3.14, -3.33, -3.44, -3.92, -3.62, -2.80, -3.10, -3.95, -3.22, -3.96, -3.14, -3.86, -3.21, -4.12, -3.46, -3.13, -2.95, -3.33, -2.80, -4.16, -3.63, -3.64, -3.27, -4.11, -3.39, -4.32, -3.51, -1.96, -3.88, -2.92, -3.59, -3.70, -1.69, -1.13, -2.46, -3.45, -3.34, -4.08, -2.85, -3.90, -2.77, -2.60, -2.79, -2.92, -3.61, -3.19, -3.18, -3.59, -2.73, -2.69, -0.97, -3.04, -1.90, -3.28, -3.95, -2.79, -3.32, -3.12, -3.47, -3.67, -1.34, -2.52, -4.19, -3.35, -3.66, -0.46, -2.57, -2.30, -2.49, -2.55, -2.57, -2.56, -2.45, -2.64, -2.28, -3.23, -1.60, -1.36, -2.83, -0.83])
        dset = f.create_dataset("data/log_H1_Lk_frac/values", data=x_frac)
        dset.attrs["units"]    = 'Msun * Lsun**(-1)'
        print(len(x_frac))
        
        
        grp = f.create_group("data/H1_frac")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'H1 mass/M* mass converted assuming mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.82 M*/L* for ETGs'
        x_frac    = 10**x_frac
        # Unit conversions -> given as M*/L -> * L/M* to get M*/M* | mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.82 M*/L* for ETGs
        x_frac    = x_frac * (1/0.82)
        
        dset = f.create_dataset("data/H1_frac/values", data=x_frac)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x_frac))
        
        
        
        print("Successfully created: %s/Serra2012_ATLAS3D_HI.hdf5"%obs_dir)


# MASSIVE results, export as hdf5
def _create_Veale2017_MASSIVE():
    
    # Create and write
    with h5py.File("%s/Veale2017_MASSIVE.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = ""
        grp.attrs["citation"]     = "Davis et al. (2019)"
        grp.attrs["comment"]      = "Original data from MASSIVE. No cosmology correction needed."
        grp.attrs["name"]         = "MASSIVE points from Davis2019, all adjusted for direct use"
                
        #-------------------------------
        # Creating dataset + cosmology
        
        grp = f.create_group("data/Galaxy")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Galaxy name'
        x = ["NGC 0057","NGC 0080","NGC 0128","NGC 0227","NGC 0315","NGC 0383","NGC 0393","NGC 0410","NGC 0467","PGC 004829","NGC 0499","NGC 0507","NGC 0533","NGC 0545","NGC 0547","NGC 0665","UGC 01332","NGC 0708","UGC 01389","NGC 0741","NGC 0777","NGC 0890","NGC 0910","NGC 0997","NGC 1016","NGC 1060",           "NGC 1066","NGC 1132","NGC 1129","NGC 1167","NGC 1226","IC 0310","NGC 1272","UGC 02783","NGC 1453","NGC 1497","NGC 1600","NGC 1573","NGC 1684","NGC 1700","NGC 2208","NGC 2256","NGC 2274","NGC 2258","NGC 2320","UGC 03683","NGC 2332","NGC 2340","UGC 03894","NGC 2418","NGC 2456","NGC 2492","NGC 2513","NGC 2672","NGC 2693","NGC 2783","NGC 2832","NGC 2892","NGC 2918","NGC 3158","NGC 3209","NGC 3332","NGC 3343","NGC 3462","NGC 3562","NGC 3615","NGC 3805","NGC 3816","NGC 3842","NGC 3862","NGC 3937","NGC 4055","NGC 4065","NGC 4066",           "NGC 4059","NGC 4073","NGC 4213","NGC 4472",           "NGC 4486","NGC 4555","NGC 4649","NGC 4816",           "NGC 4839","NGC 4874","NGC 4889","NGC 4914","NGC 5129","NGC 5208","PGC 047776","NGC 5252","NGC 5322","NGC 5353","NGC 5490","NGC 5557","IC 1143","UGC 10097","NGC 6223","NGC 6364","NGC 6375","UGC 10918","NGC 6442","NGC 6482","NGC 6575","NGC 7052","NGC 7242","NGC 7265", "NGC 7274","NGC 7386","NGC 7426","NGC 7436","NGC 7550","NGC 7556","NGC 7618","NGC 7619","NGC 7626"]
        dset = f.create_dataset("data/Galaxy/values", data=x, shape=len(x), dtype=h5py.string_dtype())
        dset.attrs["units"]    = ''
        print(len(x))
        
        

        grp = f.create_group("data/M_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Total galaxy absolute magnitude in K-S band'
        x_mag    = np.array([-25.75,-25.66,-25.35,-25.32,-26.30,-25.81,-25.44,-25.90,-25.40,-25.30,-25.50,-25.93,-26.05,-25.83,-25.83,-25.51,-25.57,-25.65,-25.41,-26.06,-25.94,-25.50,-25.33,-25.40,-26.33,-26.00,-25.31,-25.70,-26.14,-25.64,-25.51,-25.35,-25.80,-25.44,-25.67,-25.31,-25.99,-25.55,-25.34,-25.60,-25.63,-25.87,-25.69,-25.66,-25.93,-25.52,-25.39,-25.90,-25.58,-25.42,-25.33,-25.36,-25.52,-25.60,-25.76,-25.72,-26.42,-25.70,-25.49,-26.28,-25.55,-25.38,-25.33,-25.62,-25.65,-25.58,-25.69,-25.40,-25.91,-25.50,-25.62,-25.40,-25.47,-25.35,-25.41,-26.33,-25.44,-25.72,-25.31,-25.92,-25.36,-25.33,-25.85,-26.18,-26.64,-25.72,-25.92,-25.61,-25.36,-25.32,-25.51,-25.45,-25.57,-25.46,-25.45,-25.43,-25.59,-25.38,-25.53,-25.75,-25.40,-25.60,-25.58,-25.67,-26.34,-25.93,-25.39,-25.58,-25.74,-26.16,-25.43,-25.83,-25.44,-25.65,-25.65])
        dset = f.create_dataset("data/M_K/values", data=x_mag)
        dset.attrs["units"]    = 'dimensionless'
        print(len(x_mag))
        
        
        grp = f.create_group("data/L_K")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 Total galaxy luminosity in K-T band, and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        L_K    = 10**(0.4*(3.28 - x_mag))
        L_K    = np.log10(L_K)
        dset = f.create_dataset("data/L_K/values", data=L_K)
        dset.attrs["units"]    = 'Lsun'
        print(len(L_K))
        
        grp = f.create_group("data/log_Mstar")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'Log10 derived Mstar from mass-to-light ratio using L_K with M*/L_K=0.5, and a Sun K-band magnitude of 3.28 (Binney & Merrifield+98)'
        x_mass     = (10**L_K) * 0.82      # assuming mass-to-light of 0.82 for K-band
        x_mass_log = np.log10(x_mass)
        dset = f.create_dataset("data/log_Mstar/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        
        
        grp = f.create_group("data/log_MJAM")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'MJAM is approx Mstar, fit using Cappellari+13a fit: log M* = 10.58 - 0.44(M_K + 23)'
        x_mass_log = np.array([11.79, 11.75, 11.61, 11.60, 12.03, 11.82, 11.65, 11.86, 11.64, 11.59, 11.68, 11.87, 11.92, 11.83, 11.83, 11.68, 11.71, 11.75, 11.64, 11.93, 11.87, 11.68, 11.61, 11.64, 12.05, 11.90, 11.60, 11.77, 11.96, 11.74, 11.68, 11.61, 11.81, 11.65, 11.75, 11.60, 11.90, 11.70, 11.61, 11.72, 11.74, 11.84, 11.76, 11.75, 11.87, 11.69, 11.63, 11.86, 11.72, 11.64, 11.61, 11.62, 11.69, 11.72, 11.79, 11.78, 12.08, 11.77, 11.68, 12.02, 11.70, 11.63, 11.61, 11.73, 11.75, 11.72, 11.76, 11.64, 11.86, 11.68, 11.73, 11.64, 11.67, 11.61, 11.64, 12.05, 11.65, 11.78, 11.60, 11.86, 11.62, 11.61, 11.83, 11.98, 12.18, 11.78, 11.86, 11.73, 11.62, 11.60, 11.68, 11.66, 11.71, 11.66, 11.66, 11.65, 11.72, 11.63, 11.69, 11.79, 11.64, 11.72, 11.72, 11.75, 12.05, 11.87, 11.63, 11.72, 11.79, 11.97, 11.65, 11.83, 11.65, 11.75, 11.75])
        dset = f.create_dataset("data/log_MJAM/values", data=x_mass_log)
        dset.attrs["units"]    = 'Msun'
        print(len(x_mass_log))
        
        
        grp = f.create_group("data/log_Mhalo")
        grp.attrs["comoving"]    = False
        grp.attrs["description"] = 'True/false if it is within a cluster'
        x = np.array([ math.nan, 14.1, math.nan, 13.5, 13.5, 14.4, math.nan, 14.4, math.nan, math.nan, 14.4, 14.4, 13.5, 14.5, 14.5, 13.7, 13.8, 14.5, 13.8, 13.8, 13.5, math.nan, 14.8, 13.0, 13.9, 14.0, 14.0, 13.6, 14.8, 13.1, 13.2, 14.8, 14.8, 12.6, 13.9, math.nan, 14.2, 14.1, 13.7, 12.7, math.nan, 13.7, 13.3, 12.2, 14.2, 13.6, 14.2, 14.2, 13.7, math.nan, math.nan, 13.0, 13.6, 13.0, math.nan, 12.8, 13.7, math.nan, math.nan, 13.3, 11.8, math.nan, math.nan, math.nan, 13.5, 13.6, 14.8, 14.8, 14.8, 14.8, 14.2, 14.3, 14.3, 14.3, 14.3, 13.9, 13.4, 14.7, 14.7, math.nan, 14.7, 15.3, 15.3, 15.3, 15.3, math.nan, math.nan, 13.0, 14.1, 14.1, 13.7, 13.6, math.nan, 13.3, 13.0, 12.7, 13.5, math.nan, math.nan, math.nan, math.nan, 13.1, math.nan, math.nan,  14.0, 14.7, 14.7, 13.9, 13.8, 14.4, 11.9, 14.0, 13.7, 14.0, 14.0])
        dset = f.create_dataset("data/log_Mhalo/values", data=x)
        dset.attrs["units"]    = 'Msun'
        print(len(x))
        
        
        print("Successfully created: %s/Veale2017_MASSIVE.hdf5"%obs_dir)



def _test_load(file_name = 'GalaxyH2MassFunction/Lagos2014_H2'):
    # Load the observational data, specify the units we want, from RobMcGibbon's COLIBRE_Introduction
    with h5py.File('%s/%s.hdf5'%(obs_dir, file_name), 'r') as file:
        
        """# A description of the file and data is in the metadata
        print(f'File keys: {file.keys()}')
        for k, v in file['metadata'].attrs.items():
            print(f'{k}: {v}')
        # Data
        print(file['x'].keys())
        print(file['y'].keys())
        print(' ')"""
        
        obs_x     = file['data/massfunc/1Vcorr/x/values'][:] * u.Unit(file['data/massfunc/1Vcorr/x/values'].attrs['units'])
        obs_y = file['data/massfunc/1Vcorr/y/values'][:] * u.Unit(file['data/massfunc/1Vcorr/y/values'].attrs['units'])
        obs_y_err = file['data/massfunc/1Vcorr/y/scatter'][:] * u.Unit(file['data/massfunc/1Vcorr/y/scatter'].attrs['units'])
        
    obs_x = obs_mass.to('Msun')
    obs_y = obs_fraction.to('Mpc**(-3)')
    obs_y_err = obs_fraction_scatter.to('Mpc**(-3)')
    
    print(obs_mass)

#=============================================================  
# Run: 

#_create_Driver2022_new()
#_create_Kelvin2014()
        
#_create_Lagos2014_H1()
#_create_Lagos2014_H2()
        
_convert_Davis2019_ATLAS3D()
_create_Davis2019_MASSIVE()

#_create_Davis2013_ATLAS3D_CO_extent()
#_create_Krajnovic2011_ATLAS3D_ellip()

#_create_Serra2012_ATLAS3D_HI()
_create_Cappellari2011_ATLAS3D_masses()

_create_Veale2017_MASSIVE()
        
#==========================================================  
# Test load
#_test_load(file_name='GalaxyH2MassFunction/Lagos2014_H2')
        

